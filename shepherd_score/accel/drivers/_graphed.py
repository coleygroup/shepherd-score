"""Generic CUDA-graph fine loop: capture one Adam optimisation step, replay it N times.

The per-pose fine loop is launch-bound at small and medium batch. One in-place step is captured
into a CUDA graph and replayed ``steps`` times; loop state lives in persistent buffers, so N
replays equal N eager steps with no per-step launch overhead. The base class owns capture and
replay; a subclass supplies its buffers, the step body (``_step``), bucket loading (``_load``),
the reset (``_reset``) and the result (``_result``). Kernels may allocate their outputs inside
the capture; what breaks it is a host sync or data-dependent control flow inside the step and
Python rebinding of loop-carried tensors. Autotune/JIT is warmed before capture in a side stream.
"""
from __future__ import annotations

from collections import OrderedDict
import torch

from .._stats import record as _record_steps


# --- compute-aware P-cap ----------------------------------------------------------------
# The graph's per-step launch saving is roughly fixed while its cost grows with per-row kernel
# work, so the P at which it stops beating eager scales as budget / work; graph_cap() clamps
# that to [MIN, CEIL]. CEIL bounds one graph's persistent buffers (about 1 KB per row).
_GRAPH_WORK_BUDGET = 300_000_000
_GRAPH_CAP_CEIL = 262144
_GRAPH_CAP_MIN = 2000


def graph_cap(work, budget=None):
    """Max pose-rows P to graph for a bucket whose per-row kernel work is ``work`` (e.g.
    N_pad*M_pad). Below this the graph wins; above it the gating falls back to eager."""
    b = _GRAPH_WORK_BUDGET if budget is None else budget
    return max(_GRAPH_CAP_MIN, min(_GRAPH_CAP_CEIL, int(b) // max(int(work), 1)))
# Replays between convergence tests in the blocked early-stop (one host sync per block). The
# test is per pair, like the eager loop's, so one converged pair cannot halt the bucket. The
# replay loop uses the caller's patience unchanged, so it stops where the eager loop does.
_GRAPH_ES_BLOCK = 5

# Process-wide LRU cache of captured graphs keyed by (device, mode, shapes, P, steps, params).
# Bounded because each graph pins persistent GPU buffers; unbounded growth ends in OOM.
_FINE_GRAPH_CACHE: "OrderedDict" = OrderedDict()
_GRAPH_CACHE_MAX = 24


def _evict_lru():
    """Drop the least-recently-used cached graph and return its buffers to the allocator."""
    _, old = _FINE_GRAPH_CACHE.popitem(last=False)
    del old
    torch.cuda.empty_cache()


def reset_graph_cache():
    """Free every cached graph and its persistent buffers and return the blocks to the CUDA
    allocator, so independent runs start from an unfragmented GPU. The Triton autotune cache
    is left intact."""
    _FINE_GRAPH_CACHE.clear()
    torch.cuda.empty_cache()


class _GraphedFineBase:
    """Capture one in-place fine step into a CUDA graph; ``run`` replays it ``steps`` times.

    Subclasses allocate all persistent buffers in their ``__init__`` (loop-carried state
    ``q``/``t``/Adam-moments/``best*`` plus the bucket inputs and per-step temporaries) and
    then call ``super().__init__(steps)``. They implement:

      * ``_step()``    -- one in-place optimisation step over the persistent buffers. Must use
                          ``out=``/in-place ops only (no Python rebinding of loop-carried
                          tensors, no host sync). Kernels may allocate their own outputs.
      * ``_load(*x)``  -- copy this bucket's inputs into the persistent input buffers.
      * ``_reset()``   -- reset q/t to the seeds, zero Adam moments, best=-inf, best*=seeds.
      * ``_result()``  -- return (best_score, best_q, best_t).

    The warmup -> capture -> replay logic below is identical for every mode.
    """

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.graph = None
        # Blocked early-stop schedule (set by run_graphed from the driver's eager params).
        # es_patience == 0 -> disabled (fixed-step replay).
        self.es_patience = 0
        self.es_tol = 1e-5
        self.es_block = _GRAPH_ES_BLOCK
        # Seed rows per pair, so the early-stop can reduce `best` per pair; set by run_graphed.
        # 0 means the whole buffer is treated as one row.
        self.es_seeds = 0

    # --- subclass hooks (no-ops here so a mis-specified subclass fails loudly) ---
    def _step(self):        raise NotImplementedError
    def _load(self, *x):    raise NotImplementedError
    def _reset(self):       raise NotImplementedError
    def _result(self):      raise NotImplementedError

    def capture(self, *inputs):
        """Warm autotune/JIT off the capture stream, then capture exactly one ``_step``."""
        self._load(*inputs); self._reset()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                       # warmup -> compile/autotune
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self._step()

    def run(self, *inputs):
        """Load this bucket, reset loop state, replay the captured step (with blocked
        early-stop when enabled), and return the best pose."""
        self._load(*inputs); self._reset()
        if self.es_patience:
            # Mirror the eager early-stop schedule: eager checks at steps 0, 5, 10, ... and
            # seeds its baseline with the step-0 best, so replay one step, seed `prev` from it,
            # then check every es_block replays. The test is per pair: `best` holds S seed rows
            # per pair, pair-major, so .view(-1, S) row k is pair k, and the loop may break only
            # once every pair has stalled. Everything here runs between replays, outside the
            # capture; `improved.any()` is the one host sync per block.
            S = self.es_seeds or self.best.numel()
            self.graph.replay()
            done = 1
            prev = self.best.view(-1, S).amax(dim=1); no_improve = 0
            while done < self.steps:
                k = min(self.es_block, self.steps - done)
                for _ in range(k):
                    self.graph.replay()
                done += k
                cur = self.best.view(-1, S).amax(dim=1)
                improved = (cur - prev) > self.es_tol
                if not improved.any():                 # one host sync per block, not per step
                    no_improve += 1
                    if no_improve >= self.es_patience:
                        break
                else:
                    no_improve = 0
                # Advance a pair's baseline only where that pair improved, as the eager rule does.
                prev = torch.where(improved, cur, prev)
        else:
            for _ in range(self.steps):
                self.graph.replay()
            done = self.steps
        # Replays executed == value+grad evaluations. Record the execution path separately
        # from its stopping policy. No-op unless _stats recording is enabled.
        _record_steps(done, self.steps, done < self.steps, graphed=True)
        return self._result()


def _set_schedule(gf, es_patience, es_tol, es_seeds):
    gf.es_patience = int(es_patience)
    gf.es_tol = float(es_tol)
    gf.es_seeds = int(es_seeds)


def run_graphed(make, key, inputs, *, es_patience=0, es_tol=1e-5, es_seeds=0):
    """Fetch (or build+capture) the graph for ``key`` and run this bucket through it.

    ``make`` is a zero-arg factory for the subclass instance (called only on cache miss);
    ``inputs`` is the tuple forwarded to ``capture``/``run`` (and thence ``_load``). One
    captured graph serves every bucket of the same key (same shapes/P/steps/params).
    ``es_patience``/``es_tol`` set the replay-loop blocked early-stop, with the same patience
    and tolerance as the driver's eager early-stop (0 -> run a fixed ``steps`` replays), and
    ``es_seeds`` is the driver's seed count per pair, which that early-stop needs to reduce
    ``best`` per pair rather than over the whole bucket. All three are properties of this
    call, not of the cached graph, so they are set on every call, cache hit or miss.
    """
    gf = _FINE_GRAPH_CACHE.get(key)
    if gf is not None:
        _FINE_GRAPH_CACHE.move_to_end(key)       # mark most-recently-used
        _set_schedule(gf, es_patience, es_tol, es_seeds)
        return gf.run(*inputs)
    gf = make()
    _set_schedule(gf, es_patience, es_tol, es_seeds)
    # Capture; if it OOMs, free LRU graphs and retry (rather than failing -> eager forever).
    while True:
        try:
            gf.capture(*inputs)
            break
        except Exception as e:
            if "out of memory" not in str(e).lower() or not _FINE_GRAPH_CACHE:
                raise                             # non-OOM, or nothing left to free -> propagate
            _evict_lru()
    _FINE_GRAPH_CACHE[key] = gf
    while len(_FINE_GRAPH_CACHE) > _GRAPH_CACHE_MAX:
        _evict_lru()
    return gf.run(*inputs)
