# shepherd_score/accel/batch/_pad.py
"""Size bucketing, GPU-memory-safe sub-batching, and batched scatter-fill
primitives shared by the batched aligners."""
from __future__ import annotations
import torch


# --- size bucketing ---
# Every point count is rounded up to a multiple of _BAND, so pairs in one band share a padded
# tensor size. The adaptive planner in _bucket.py snaps its pads to this grid.
_BAND = 16

def _band_key(n: int) -> int:
    "Upper bound of the _BAND-sized band that n falls into."
    return ((n + _BAND - 1) // _BAND) * _BAND

# Fine-loop footprint (bytes per pair) observed at run time, keyed by (mode, N_pad, M_pad,
# num_seeds); the sub-batcher sizes each bucket's chunk from it.
_PAIR_FOOTPRINT_BYTES: dict[tuple, int] = {}


#: Upper bound, in poses, on one fine-loop sub-batch, applied only where a caller passes
#: ``pose_cap``; it keeps a bucket's graph-vs-eager choice independent of allocator state.
_FINE_CHUNK_POSES = 81920


#: Pairs per call on the CPU path; bounds peak host RSS on an out-of-core screen, where the
#: default shard would otherwise hand the fine loop one 100k-pair batch.
_CPU_CHUNK_PAIRS = 10_000


def _concat_chunks(process, K: int, step: int):
    """Run ``process`` over ``K`` pairs in ``step``-sized chunks and concatenate.

    The fixed-size CPU twin of the GPU branch in ``_subbatched_align``; the memory-budget
    machinery there has no meaning off CUDA.
    """
    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    while s < K:
        k = min(step, K - s)
        sc, q, t = process(s, k)
        sc_parts.append(sc); q_parts.append(q); t_parts.append(t)
        s += k
    if len(sc_parts) == 1:
        return sc_parts[0], q_parts[0], t_parts[0]
    return (torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts))


def _subbatched_align(process, K: int, *, key: tuple, device: torch.device,
                      safety: float = 0.7, init_cap: int = 1024, pose_cap: int = 0,
                      seeds: int = 1):
    """Drive ``process(start, count) -> (scores, q, t)`` over ``K`` independent pairs in
    GPU-memory-safe sub-batches and concatenate the per-pair results.

    Chunking is score-identical only when every chunk runs the same number of fine steps: pairs
    are independent within a step, but the early-stop break is chunk-global (it fires once every
    pair in the chunk has stalled), so re-cutting the chunks can hand a pair a different step
    count. Any two schedules agree when early stopping cannot fire, i.e. a patience of at least
    the number of checks in the budget. ``early_stop_patience=0`` does not do that: the graph
    replay loop reads 0 as "no early stop", but the eager and fused CPU loops stop at the first
    check where no pair improved.

    Bytes per pair is read from the fine loop's peak allocation and cached per
    ``key=(mode, N_pad, M_pad, num_seeds)``; each chunk is sized to keep its peak under
    ``safety`` x (free device memory + torch's reusable cache). An unseen shape starts at
    ``init_cap`` pairs and grows once calibrated; an OOM halves the chunk and retries. Off CUDA
    ``process`` runs in fixed ``_CPU_CHUNK_PAIRS`` chunks.
    """
    if device.type != "cuda":
        # No device allocator to run out of, so no budget loop; the fixed chunk only bounds
        # peak host RSS. Batches at or below _CPU_CHUNK_PAIRS run in one call.
        if K <= _CPU_CHUNK_PAIRS:
            return process(0, K)
        return _concat_chunks(process, K, _CPU_CHUNK_PAIRS)

    key = (torch.cuda.current_device(),) + tuple(key)   # device-scope the footprint cache

    def _budget() -> float:
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return safety * (free + max(0, reusable))

    # Pose cap: only ever shrinks a chunk, so the memory safety below is untouched. Applied
    # before the memory sizing so ``need_resize``'s later growth respects it too.
    cap_pairs = max(1, int(pose_cap) // max(1, int(seeds))) if pose_cap > 0 else K

    fp = _PAIR_FOOTPRINT_BYTES.get(key)
    need_resize = fp is None
    K_sub = max(1, min(K, int(_budget() // fp))) if fp else min(K, init_cap)
    K_sub = max(1, min(K_sub, cap_pairs))

    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    while s < K:
        k = min(K_sub, K - s)
        try:
            torch.cuda.reset_peak_memory_stats()
            base = int(torch.cuda.memory_allocated())
            sc, q, t = process(s, k)
            peak = int(torch.cuda.max_memory_allocated())
            # Charge the chunk for its own growth, not the device-wide high-water mark:
            # reset_peak_memory_stats() rebases the peak to what is already resident, so
            # without ``- base`` a chunk is billed for every byte other objects hold.
            growth = max(0, peak - base)
            # Fold a chunk into the per-pair footprint only when it is large enough to amortise
            # the fixed workspace overhead (seed/autotune scratch, independent of k); a tiny
            # trailing remainder would otherwise lock in an inflated bytes/pair. The first chunk
            # has k == K_sub, so calibration is never starved.
            if k >= max(1, K_sub // 4):
                fp_meas = max(1, -(-growth // k))                # ceil bytes/pair
                # Keep the max() fold: the first chunk of a shape pays for the captured graph's
                # persistent buffers, which later chunks reuse, so a later reading would
                # under-size the chunk. The cache is never invalidated, so an over-estimate from
                # fixed in-call workspace persists for the process.
                _PAIR_FOOTPRINT_BYTES[key] = max(_PAIR_FOOTPRINT_BYTES.get(key, 0), fp_meas)
            sc_parts.append(sc); q_parts.append(q); t_parts.append(t)
            s += k
            if need_resize:   # first success -> we now know the real footprint
                fp = _PAIR_FOOTPRINT_BYTES[key]
                remaining = K - s
                if remaining > 0:
                    K_sub = max(1, min(remaining, int(_budget() // fp), cap_pairs))
                need_resize = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Some OOMs surface as a plain RuntimeError; only treat those as OOM.
            if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                    and "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            if k <= 1:
                raise
            K_sub = max(1, k // 2)
    return torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts)


def _scatter_fill(out: torch.Tensor, tensors: list[torch.Tensor], sizes: list[int]) -> None:
    """Fill a pre-zeroed padded workspace ``out`` of shape ``(K, P_pad, *feat)`` so that
    ``out[i, :sizes[i]] = tensors[i]`` for each of the ``K`` per-pair tensors.

    One batched ``torch.cat`` plus one vectorised scatter instead of ``K`` device copies; the
    padding rows are left untouched (the caller zeroes them).
    """
    K, P_pad = out.shape[0], out.shape[1]
    device = out.device
    # ``sizes`` is a host list; summing it on the host avoids a CUDA sync per scatter.
    S = sum(sizes)
    if S == 0:
        return
    n = torch.as_tensor(sizes, device=device, dtype=torch.long)
    flat = torch.cat(tensors, dim=0)                       # (S, *feat)
    starts = torch.cumsum(n, 0) - n                        # (K,) first flat-row of each pair
    seg = torch.repeat_interleave(starts, n)               # (S,) segment start per flat row
    local = torch.arange(S, device=device) - seg           # (S,) within-pair row index
    dst = torch.repeat_interleave(torch.arange(K, device=device) * P_pad, n) + local
    out.view(K * P_pad, *out.shape[2:])[dst] = flat
