"""Persistent single-threaded process pool for the CPU (``numba``) batched aligner.

The in-process numba path parallelises one batch across threads behind a per-step barrier and
scales sub-linearly on many-core hosts. This pool shards the pairs across persistent worker
processes instead (``align_with_*(backend="numba", num_workers=N)``), each running the unchanged
batched aligner single-threaded. Pairs are independent, so sharding changes no pair's problem;
agreement with one big call is to convergence tolerance, not bitwise, because the fine loop's
early stop depends on which pairs share a batch. Only small numpy arrays cross the process
boundary. Uses ``spawn``, so the importing program must be ``if __name__ == "__main__"``-guarded.
"""
from __future__ import annotations

import atexit
import os

import numpy as np

# Modes with a ``_MODE_SPEC`` entry; the others use the single-process path. Legacy aliases
# (esp -> surf_esp, esp_combo -> vol_and_surf_esp) are normalised in align_pairs.
from ._modes import PROCESS_MODES as POOL_MODES, LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES
from ._procs import collect, shutdown, start_worker

# Worker side (runs in each persistent child process)

def _run_shard(bm, torch, mode, rows, kwargs):
    """Run the batched aligner on one shard. ``rows`` holds one tuple of numpy arrays per pair,
    ordered as ``_MODE_SPEC[mode]['tensors']``; returns ``(scores (k,), transforms (k,4,4))``."""
    mode = _LEGACY_MODE_ALIASES.get(mode, mode)
    spec = bm._MODE_SPEC[mode]
    tnames = spec["tensors"]
    pair_attrs = {t for (side, _), (t, _) in zip(spec["extract"], tnames) if side == "pair"}
    standins = []
    shared = {}                     # an array pickled once for many rows arrives as one object
    for row in rows:
        s = bm._ProcStandIn(torch.device("cpu"))
        for (tname, dt), arr in zip(tnames, row):
            if tname in pair_attrs:
                key = (tname, id(arr))
                if key not in shared:
                    shared[key] = torch.as_tensor(np.asarray(arr), dtype=dt)
                setattr(s, tname, shared[key])
            else:
                setattr(s, tname, torch.as_tensor(np.asarray(arr), dtype=dt))
        standins.append(s)
    if standins:
        getattr(bm, "_align_batch_" + mode)(standins, **kwargs)
    tf_attr, sc_attr = spec["out"]
    scores = np.array([float(getattr(s, sc_attr)) for s in standins], dtype=np.float64)
    if standins:
        transforms = np.stack([
            torch.as_tensor(getattr(s, tf_attr)).detach().cpu().numpy().astype(np.float32)
            for s in standins])
    else:
        transforms = np.zeros((0, 4, 4), dtype=np.float32)
    return scores, transforms


def _worker_loop(task_q, res_conn):
    """Persistent single-threaded worker serving ``(mode, rows, kwargs)`` tasks until a ``None``
    sentinel; replies on ``res_conn`` are ``(scores, transforms)`` or ``("__ERR__", traceback)``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # pure CPU; set before torch import
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)                          # one thread per worker
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass
    from shepherd_score.accel import batch as bm
    while True:
        task = task_q.get()
        if task is None:
            break
        mode, rows, kwargs = task
        try:
            res_conn.send(_run_shard(bm, torch, mode, rows, kwargs))
        except Exception:                             # noqa: BLE001 - relayed to parent
            import traceback
            res_conn.send(("__ERR__", traceback.format_exc()))

# Pool (parent side)

class CpuAlignPool:
    """A fixed-size pool of persistent single-threaded worker processes.

    One task queue and one result pipe per worker, so shard ``w`` always maps to worker ``w``
    (round-robin over pairs) and results reassemble in order. A worker that dies mid-call, or
    any other interruption of :meth:`align`, closes the pool (``get_pool`` then starts a new
    one), since replies still in flight would otherwise be read by the next call.
    """

    def __init__(self, num_workers: int):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.num_workers = int(num_workers)
        # Closable from the first started worker on, so a failed start stops the others.
        self._task_qs, self._res_conns, self._procs, self._closed = [], [], [], False
        try:
            for i in range(self.num_workers):
                q, conn, p = start_worker(ctx, _worker_loop, (), daemon=True,
                                          name=f"cpu-align-{i}")
                self._task_qs.append(q); self._res_conns.append(conn); self._procs.append(p)
        except BaseException:
            self.close()
            raise

    def align(self, mode, per_pair, kwargs):
        """Shard ``per_pair`` (list of tuples of numpy arrays) round-robin across workers and
        return ``(scores, transforms)`` reassembled in the original order."""
        K = len(per_pair)
        shards = [list(range(w, K, self.num_workers)) for w in range(self.num_workers)]

        def submit():
            for w, idxs in enumerate(shards):
                self._task_qs[w].put((mode, [per_pair[i] for i in idxs], dict(kwargs)))

        replies = collect(
            {w: (self._res_conns[w], self._procs[w]) for w in range(self.num_workers)},
            self.close, submit=submit, what="CPU align pool worker", fail_fast=False,
            # success res[0] is an array
            error_of=lambda res: res[1] if isinstance(res[0], str) and res[0] == "__ERR__" else None)
        scores = [0.0] * K
        transforms = [None] * K
        for w, (sc, tf) in replies.items():
            for j, i in enumerate(shards[w]):
                scores[i] = float(sc[j])
                transforms[i] = tf[j]
        return scores, transforms

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        shutdown(self._task_qs, self._procs, self._res_conns, grace=2.0)


_POOL: CpuAlignPool | None = None


def get_pool(num_workers: int) -> CpuAlignPool:
    """Return the module-level persistent pool, rebuilt only if the worker count changes."""
    global _POOL
    if _POOL is None or _POOL._closed or _POOL.num_workers != num_workers:
        if _POOL is not None:
            _POOL.close()
        _POOL = CpuAlignPool(num_workers)
    return _POOL


@atexit.register
def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None

# Public entry (parent side)

def align_pairs(mode, pairs, num_workers, align_kwargs):
    """Align ``pairs`` (``MoleculePair``) across the persistent CPU pool, writing
    ``sim_aligned_*`` / ``transform_*`` back in place. Also caches the per-pair input tensors
    (``_*_t``) on each pair, as the single-process path does for ``return_aligned``. Returns nothing.
    """
    import torch
    from shepherd_score.accel import batch as bm
    from shepherd_score.accel.batch.aligners import _pair_src_of
    mode = _LEGACY_MODE_ALIASES.get(mode, mode)    # accept legacy esp / esp_combo
    spec = bm._MODE_SPEC[mode]
    extract, tnames = spec["extract"], spec["tensors"]
    tf_attr, sc_attr = spec["out"]

    # A pair-level source (one avoid cloud attached to many pairs) is read once and the same
    # array object put in every row: pickling sends it once per shard, and the worker and the
    # write-back below keep it one tensor, so the bucket builder sees one shared cloud.
    pair_src = {}

    def _read(p, side, fn):
        if side != "pair":
            return np.asarray(fn(getattr(p, side)))
        key = (id(fn), id(_pair_src_of(p)))
        if key not in pair_src:
            pair_src[key] = np.asarray(fn(p))
        return pair_src[key]

    per_pair = [tuple(_read(p, side, fn) for (side, fn) in extract) for p in pairs]
    scores, transforms = get_pool(num_workers).align(mode, per_pair, align_kwargs)

    back = {}
    for i, p in enumerate(pairs):
        for (tname, dt), arr in zip(tnames, per_pair[i]):
            if id(arr) not in back:
                back[id(arr)] = torch.as_tensor(arr, dtype=dt)
            setattr(p, tname, back[id(arr)])
        setattr(p, tf_attr, torch.as_tensor(transforms[i], dtype=torch.float32))
        setattr(p, sc_attr, float(scores[i]))
