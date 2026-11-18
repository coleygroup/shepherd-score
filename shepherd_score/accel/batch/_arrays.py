# shepherd_score/accel/batch/_arrays.py
"""Array-native screen path: the batched alignment with no per-molecule Python objects.

The object aligners take a list of pair objects; a streaming screen would otherwise build K
pair objects, bin them in a Python loop and ``cat`` their coordinates back into the padded
array the store already holds. Here bucket membership is a span over an index array, the
workspace is gathered straight from the store's contiguous buffer, the reference side is
broadcast rather than replicated, and the transforms come back as one (K,4,4) array. The
partition reuses ``_merge_group`` / ``_cap_upfront``; :func:`align_arrays` reads the mode's
:class:`ModeSpec`, so every registry mode takes this path. ``ENABLED`` is a test seam only.
"""
from __future__ import annotations

import numpy as np
import torch

from ._bucket import Bucket, _cap_upfront, _merge_group, _min_wave, PadSpec
from ._pad import _band_key, _BAND, _subbatched_align, _FINE_CHUNK_POSES
from .._modes import SPECS, MODE_SEEDS
from ..channels import CHANNELS

#: Test seam: the parity tests flip this to force the object path. Not a runtime switch.
ENABLED = True

#: Modes whose fine-loop sub-batch is capped at ``_FINE_CHUNK_POSES``; the cap makes the
#: graph-vs-eager choice a function of the band rather than of allocator state.
_POSE_CAP_MODES = ("vol",)


class Span:
    """A contiguous ``[lo, hi)`` slice of a precomputed ordering, standing in for a Bucket's
    member list.

    ``__len__``, ``__add__`` and slicing are all ``_merge_group`` needs. Merges are always
    adjacent: buckets fold into their predecessor in ascending band order, so ``a.hi == b.lo``.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int):
        self.lo = int(lo)
        self.hi = int(hi)

    def __len__(self) -> int:
        return self.hi - self.lo

    def __add__(self, other: "Span") -> "Span":
        if self.hi != other.lo:                      # never silently produce a wrong partition
            raise AssertionError(f"non-adjacent span merge {self.hi} != {other.lo}")
        return Span(self.lo, other.hi)

    def __getitem__(self, sl: slice) -> "Span":
        start = 0 if sl.start is None else int(sl.start)
        stop = len(self) if sl.stop is None else min(int(sl.stop), len(self))
        return Span(self.lo + start, self.lo + stop)

    def idx(self, order: np.ndarray) -> np.ndarray:
        return order[self.lo:self.hi]


class IdxSet:
    """Bucket members as an index array, for keys with several dimensions.

    ``_merge_group`` re-sorts after each fold once the key has several dims, so merges are no
    longer adjacent and :class:`Span` cannot serve; ``__add__`` concatenates instead.
    """

    __slots__ = ("arr",)

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __add__(self, other: "IdxSet") -> "IdxSet":
        return IdxSet(np.concatenate([self.arr, other.arr]))

    def __getitem__(self, sl: slice) -> "IdxSet":
        return IdxSet(self.arr[sl])

    def idx(self, order=None) -> np.ndarray:
        """Absolute shard indices; ``order`` is accepted and ignored to match :meth:`Span.idx`."""
        return self.arr


def plan_spans(m_sizes: np.ndarray, n_ref: int, seeds: int, device):
    """Partition K library molecules into padded buckets without touching them individually.

    Returns ``(order, buckets)``: ``order`` is a stable argsort by fit band and each bucket's
    ``.members`` is a :class:`Span` into it. The ref size is constant on the screen path, so
    the cell key is the fit band alone.
    """
    K = int(m_sizes.shape[0])
    bands = ((m_sizes.astype(np.int64) + 15) // 16) * 16          # == _band_key, vectorized
    order = np.argsort(bands, kind="stable")                      # stable => shard order within a cell
    sb = bands[order]
    cuts = np.flatnonzero(np.diff(sb)) + 1
    starts = np.concatenate(([0], cuts, [K]))
    ref_pad = _band_key(int(n_ref))
    cells = [Bucket(Span(int(starts[i]), int(starts[i + 1])),
                    {"ref": ref_pad, "fit": int(sb[starts[i]])})
             for i in range(len(starts) - 1)]
    spec = PadSpec(merge={"ref": None, "fit": None}, seeds=int(seeds))
    return order, _cap_upfront(_merge_group(cells, spec, _min_wave(device)), spec, device)


def plan_spans_multi(fit_dims: dict, const_dims: dict, spec, device, partition: dict = None):
    """Multi-dimensional twin of :func:`plan_spans`, for PadSpecs that key several dims.

    ``fit_dims``: name -> (K,) per-molecule sizes; ``const_dims``: name -> int, fixed for the
    whole screen; ``partition``: name -> exact value, uniform across the screen. Cells are keyed
    on the banded value of every merge dim in ``spec.merge`` order, then merged with
    ``_merge_group`` and ``_cap_upfront``.
    """
    names = list(spec.merge)
    K = int(next(iter(fit_dims.values())).shape[0]) if fit_dims else int(next(iter(const_dims.values())))
    cols = []
    for n in names:
        if n in fit_dims:
            cols.append(((np.asarray(fit_dims[n], dtype=np.int64) + 15) // 16) * 16)
        else:
            cols.append(np.full(K, _band_key(int(const_dims[n])), dtype=np.int64))
    key = np.stack(cols, axis=1)                                  # (K, nm)
    # lexsort takes the last key as primary; reverse so the sort follows names order, as
    # _merge_group does.
    order = np.lexsort(tuple(key[:, i] for i in range(len(names) - 1, -1, -1)))
    sk = key[order]
    cuts = np.flatnonzero((np.diff(sk, axis=0) != 0).any(axis=1)) + 1
    starts = np.concatenate(([0], cuts, [K]))
    cells = []
    for i in range(len(starts) - 1):
        lo, hi = int(starts[i]), int(starts[i + 1])
        pad = {names[j]: int(sk[lo, j]) for j in range(len(names))}
        for pn, pv in (partition or {}).items():
            pad[pn] = pv
        cells.append(Bucket(IdxSet(order[lo:hi]), pad))
    return _cap_upfront(_merge_group(cells, spec, _min_wave(device)), spec, device)


def gather_fill(out: torch.Tensor, src: torch.Tensor,
                src_start: torch.Tensor, counts: torch.Tensor) -> None:
    """Fill a pre-zeroed ``(k, P_pad, ...)`` workspace directly from the store's contiguous
    buffer.

    Same destination arithmetic as ``_scatter_fill``; the source rows are an ``index_select``
    from ``src`` instead of a ``cat`` of per-molecule views.
    """
    k, P_pad = out.shape[0], out.shape[1]
    tot = int(counts.sum())
    if tot == 0:
        return
    dev = out.device
    seg_start = torch.cumsum(counts, 0) - counts                  # (k,) first flat row per molecule
    seg = torch.repeat_interleave(seg_start, counts)              # (tot,)
    local = torch.arange(tot, device=dev) - seg                   # (tot,) row within its molecule
    srcrow = torch.repeat_interleave(src_start, counts) + local   # (tot,) row in the store buffer
    dst = torch.repeat_interleave(torch.arange(k, device=dev) * P_pad, counts) + local
    out.view(k * P_pad, *out.shape[2:])[dst] = src.index_select(0, srcrow)


def _const_seed_batch(const_seeds: torch.Tensor, k: int, device):
    """A canonical store's constant seed set, broadcast over a bucket of ``k`` molecules.

    Returns ``(quats (k,S,4), trans (k,S,3))`` in the ``batched_seeds_torch`` layout. The
    translations are zero: a canonical store and the query are both centered on the heavy-atom
    centroid, so the COM-aligning translation vanishes for every ``CONST_SEED_MODES`` mode. The
    expand is materialised because the engine reshapes the seeds into pose rows.
    """
    S = int(const_seeds.shape[0])
    return (const_seeds.unsqueeze(0).expand(k, -1, -1).contiguous(),
            torch.zeros(k, S, 3, device=device, dtype=torch.float32))

# the generic array aligner

def _counts_of(fit: dict, name: str, K: int, device):
    """``(counts (K,), starts (K,) or None, source tensor)`` for one channel's fit side."""
    flat, off = fit[name]
    if off is None:                                               # dense (K, S, ...) block
        S = int(flat.shape[1])
        return torch.full((K,), S, device=device, dtype=torch.long), None, flat
    return off[1:] - off[:-1], off, flat


def align_arrays(mode: str, ref: dict, fit: dict, *, params: dict, steps_fine: int,
                 num_seeds=None, const_seeds=None, avoid=None, early_stop_patience=None,
                 early_stop_tol: float = 1e-5):
    """Array-native alignment of one shard against one query, for any registry mode.

    ``ref``: channel name -> the query's (N, ...) device tensor. ``fit``: channel name ->
    ``(flat, off)`` (CSR) or ``(dense, None)``. ``avoid``: the query-side ``(K_a, 3)`` avoid
    cloud for ``vol_avoid``, else None. Returns ``(scores (K,) float64, SE3 (K,4,4) float32)``
    as numpy arrays in shard order.
    """
    from ..drivers.engine import Batch, align, is_shared_pair, term_self_overlaps
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

    spec = SPECS[mode]
    names = []
    for c in spec.channels:
        r = spec.resolve_channel(c, params)
        if r not in names and not CHANNELS[r].is_pair:
            names.append(r)
    device = next(iter(fit.values()))[0].device
    n_seeds = int(MODE_SEEDS[mode] if num_seeds is None else num_seeds)
    es_patience = int(spec.patience if early_stop_patience is None else early_stop_patience)

    any_ch = names[0]
    K = int(fit[any_ch][1].shape[0] - 1) if fit[any_ch][1] is not None else int(fit[any_ch][0].shape[0])
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    # per-channel fit counts / starts / source, and the host-side count vectors the pads need
    cnt, start, src, host = {}, {}, {}, {}
    for n in names:
        c, s, f = _counts_of(fit, n, K, device)
        cnt[n], start[n], src[n] = c, s, f
        host[n] = c.detach().cpu().numpy()

    # ---- bucket on the spec's cost dims -----------------------------------------------------
    bch = []
    for b in spec.bucket:
        r = spec.resolve_channel(b, params)
        if r not in bch:
            bch.append(r)
    if len(bch) == 1:
        order, buckets = plan_spans(host[bch[0]], int(ref[bch[0]].shape[0]), n_seeds, device)
        pad_of = lambda bk, b, side: int(bk.pad["ref" if side == "ref" else "fit"])   # noqa: E731
    else:
        merge = {}
        for b in bch:
            merge[f"n_{b}"] = None
            merge[f"m_{b}"] = None
        pspec = PadSpec(merge=merge, seeds=n_seeds)
        fit_dims = {f"m_{b}": host[b] for b in bch}
        const_dims = {f"n_{b}": int(ref[b].shape[0]) for b in bch}
        buckets = plan_spans_multi(fit_dims, const_dims, pspec, device)
        order = None
        pad_of = lambda bk, b, side: int(bk.pad[f"n_{b}" if side == "ref" else f"m_{b}"])  # noqa: E731

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)
    order_t = None if order is None else torch.as_tensor(order, device=device, dtype=torch.long)
    # the query's own channel widths are constant across the whole screen
    n_pads = {n: _band_key(int(ref[n].shape[0])) or _BAND for n in names}
    pose_cap = _FINE_CHUNK_POSES if mode in _POSE_CAP_MODES else 0

    for bk in buckets:
        if order_t is None:
            rows_np = bk.members.idx()
            rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
        else:
            rows = order_t[bk.members.lo:bk.members.hi]
            rows_np = bk.members.idx(order)
        k = bk.K
        chans = {}
        pads = []
        for n in names:
            ch = CHANNELS[n]
            dt = torch.int64 if ch.dtype == "int64" else torch.float32
            in_bucket = any(CHANNELS[b].basis == ch.basis for b in bch)
            if in_bucket:
                b = next(b for b in bch if CHANNELS[b].basis == ch.basis)
                n_pad, m_pad = pad_of(bk, b, "ref"), pad_of(bk, b, "fit")
            else:
                n_pad = n_pads[n]
                m_pad = _band_key(int(host[n][rows_np].max())) or _BAND
            N = int(ref[n].shape[0])
            feat = tuple(ref[n].shape[1:])
            if ch.pad == 0:
                r_pad = torch.zeros((k, n_pad) + feat, device=device, dtype=dt)
                f_pad = torch.zeros((k, m_pad) + feat, device=device, dtype=dt)
            else:
                r_pad = torch.full((k, n_pad) + feat, ch.pad, device=device, dtype=dt)
                f_pad = torch.full((k, m_pad) + feat, ch.pad, device=device, dtype=dt)
            r_pad[:, :N] = ref[n]                              # one query broadcast over k rows
            c = cnt[n].index_select(0, rows)
            if start[n] is None:                               # dense block: a row select
                S = int(src[n].shape[1])
                f_pad[:, :S] = src[n].index_select(0, rows)
            else:
                gather_fill(f_pad, src[n], start[n].index_select(0, rows), c)
            chans[n] = Batch(r_pad, f_pad,
                             torch.full((k,), N, dtype=torch.int32, device=device),
                             c.to(torch.int32))
            pads += [n_pad, m_pad]
        if avoid is not None:
            # the query's one cloud, as a single row every pose reads (not k padded copies)
            Ka = int(avoid.shape[0])
            a_pad = torch.zeros(1, _band_key(Ka) or _BAND, 3, device=device, dtype=torch.float32)
            a_pad[0, :Ka] = avoid
            chans["avoid"] = Batch(a_pad, None,
                                   torch.full((1,), Ka, dtype=torch.int32, device=device), None)
        cs = None if const_seeds is None else _const_seed_batch(const_seeds, k, device)
        so = term_self_overlaps(spec, chans, params, ref_shared=True)   # once per bucket

        def _proc(_s, _k, _ch=chans, _cs=cs, _so=so):
            sl = slice(_s, _s + _k)
            sub = {n: b if is_shared_pair(b) else
                   Batch(b.ref[sl], None if b.fit is None else b.fit[sl], b.n_real[sl],
                         None if b.m_real is None else b.m_real[sl])
                   for n, b in _ch.items()}
            return align(spec, sub, params=params, num_seeds=n_seeds, steps_fine=steps_fine,
                         lr=float(params["lr"]), early_stop_patience=es_patience,
                         early_stop_tol=early_stop_tol, ref_shared=True,
                         seeds=None if _cs is None else (_cs[0][sl], _cs[1][sl]),
                         self_overlaps=[None if p is None else (p[0][sl], p[1][sl])
                                        for p in _so])
        sc, qb, tb = _subbatched_align(_proc, k, key=(mode,) + tuple(pads) + (n_seeds,),
                                       device=device, pose_cap=pose_cap, seeds=n_seeds)
        out_scores[rows_np] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3
