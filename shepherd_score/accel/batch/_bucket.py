"""Mode-agnostic adaptive bucketer for the batched aligners.

The aligners group same-size pairs into one padded workspace per kernel launch; a fixed band
over-fragments a wide size distribution and over-pads a tight one, so :func:`plan_buckets`
plans per call from the :class:`PadSpec` a driver supplies. Kernels mask padding to the real
counts and seeds are keyed on real counts, so bucketing changes no result; it minimises padded
work plus a launch penalty, with pads snapped to ``_band_key`` multiples so the Triton autotune
cache stays small. ``merge`` dims are cost-driving and merged across; ``partition`` dims must
be uniform within a bucket and never merge across.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ._pad import _band_key

# --- tuning constants -------------------------------------------------------------------
_CTAS_PER_SM = 16    # CTAs per SM assumed when sizing one occupancy wave
# Cap on a merged bucket, in CTA waves. Merging only fills under-occupied buckets; past this it
# adds padding waste and grows extents the heavy kernels' indexing cannot safely address.
_MERGE_MAX_WAVES = 2.0
# Per-bucket cap on upfront held memory (padded clouds + hoisted seeds) as a fraction of the
# device's free memory at call time, so buckets shrink on a busy GPU and grow on an idle one.
_UPFRONT_FRAC = 0.25
# Floor on that cap, so the split cannot degenerate when the device is nearly full.
_UPFRONT_FLOOR_BYTES = 256 * (1024 ** 2)


def _max_upfront_bytes(device) -> float:
    """Per-bucket upfront-memory cap, derived from the device's current free memory."""
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free = 8 * (1024 ** 3)                                    # conservative if unavailable
    return max(_UPFRONT_FLOOR_BYTES, _UPFRONT_FRAC * float(free))


def _cap_upfront(buckets: list, spec: "PadSpec", device) -> list:
    """Split any bucket whose hoisted pad+seed footprint exceeds the adaptive upfront cap.

    The kernels mask padding, so a split changes no kernel result; the fine loop's early-stop
    break is chunk-global, though (see ``_subbatched_align``)."""
    if getattr(device, "type", None) != "cuda":
        return buckets
    max_bytes = _max_upfront_bytes(device)
    out: list = []
    for b in buckets:
        # Peak bytes per pair: the held padded coords (12 B/pt) and seed q/t (28 B) plus the
        # seed-generation transient, which upcasts to float64 over the whole bucket and is not
        # sub-batched; about 176 B per point of the largest cloud covers it.
        merge_pads = [int(b.pad[n]) for n in spec.merge]
        per_pair = sum(merge_pads) * 12 + int(spec.seeds) * 28 + max(merge_pads) * 176
        max_k = max(1, int(max_bytes // max(1, per_pair)))
        if b.K <= max_k:
            out.append(b)
        else:
            for s in range(0, b.K, max_k):
                out.append(Bucket(b.members[s:s + max_k], dict(b.pad)))
    return out


def _min_wave(device) -> int:
    """Pose count that fills one CTA wave on ``device``; a bucket below it is force-merged.
    On CPU it is 1, so occupancy merging is disabled."""
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        return max(1, sm * _CTAS_PER_SM)
    except Exception:
        return 2048


# --- spec + bucket ----------------------------------------------------------------------
@dataclass
class PadSpec:
    """Declarative description of how a mode pads and buckets.

    Parameters
    ----------
    merge : dict[str, Callable]
        name -> ``f(item) -> int``. Cost-driving cloud sizes; keyed and adaptively merged.
    seeds : int
        Poses per pair (``num_seeds``); used only for the occupancy floor (``K*seeds``).
    work : Callable | None
        ``f(pad: dict[str, int]) -> float`` estimating per-pose kernel cost from the pad sizes.
        Default: product of the ``merge`` pads (the single-channel ``N_pad*M_pad`` kernels).
    partition : dict[str, Callable]
        name -> ``f(item) -> int``. Must be uniform within a bucket; never merged across.
    """
    merge: dict
    seeds: int
    work: Callable | None = None
    partition: dict = field(default_factory=dict)

    def sizes(self, item) -> dict:
        out = {}
        for n, fn in self.merge.items():
            out[n] = int(fn(item))
        for n, fn in self.partition.items():
            out[n] = int(fn(item))
        return out

    def work_of(self, pad: dict) -> float:
        if self.work is not None:
            return float(self.work(pad))
        v = 1.0
        for n in self.merge:
            v *= max(1, int(pad[n]))
        return v


@dataclass
class Bucket:
    members: list
    pad: dict                       # field name -> padded size (a _band_key multiple)

    @property
    def K(self) -> int:
        return len(self.members)


# --- the planner ------------------------------------------------------------------------
def plan_buckets(items, spec: PadSpec, device) -> list[Bucket]:
    """Partition ``items`` into padded buckets minimising padded work plus a launch penalty.

    The per-item binning is O(N) on the host and written without per-item dicts; the merge runs
    on the small occupied-cell set. Merge dims key on their band; partition dims key on their
    exact value, since they must be uniform within a bucket.
    """
    if not items:
        return []

    mfns = list(spec.merge.values()); mnames = list(spec.merge)
    pfns = list(spec.partition.values()); pnames = list(spec.partition)
    nm = len(mfns)
    B = 16                                              # == _pad._BAND; inlined for the hot loop

    cells: dict = {}
    for it in items:
        msz = [fn(it) for fn in mfns]
        mk = tuple(((v + B - 1) // B) * B for v in msz)
        pk = tuple(fn(it) for fn in pfns)
        key = (mk, pk)
        c = cells.get(key)
        if c is None:
            cells[key] = [[it], msz, pk]
        else:
            c[0].append(it)
            mm = c[1]
            for i in range(nm):
                if msz[i] > mm[i]:
                    mm[i] = msz[i]

    def _pad_of(c):
        mm, pk = c[1], c[2]
        pad = {mnames[i]: _band_key(mm[i]) for i in range(nm)}
        for i in range(len(pnames)):
            pad[pnames[i]] = pk[i]
        return pad

    # group by partition signature (never merge across different partition values), then
    # greedy-merge within each group along the merge dims.
    groups: dict = {}
    for (mk, pk), c in cells.items():
        groups.setdefault(pk, []).append(Bucket(c[0], _pad_of(c)))

    mw = _min_wave(device)
    out: list[Bucket] = []
    for buckets in groups.values():
        out.extend(_merge_group(buckets, spec, mw))
    return _cap_upfront(out, spec, device)


def _merge_group(buckets: list[Bucket], spec: PadSpec, min_wave: int) -> list[Bucket]:
    """Repeatedly sort buckets by their merge-dim pads (so neighbours are shape-similar) and
    fold each into its predecessor while the merge is profitable, until nothing changes."""
    merge_names = list(spec.merge)
    changed = True
    while changed and len(buckets) > 1:
        changed = False
        buckets.sort(key=lambda b: tuple(b.pad[n] for n in merge_names))
        folded: list[Bucket] = [buckets[0]]
        for b in buckets[1:]:
            a = folded[-1]
            if _should_merge(a, b, spec, min_wave):
                folded[-1] = _merge(a, b, spec)
                changed = True
            else:
                folded.append(b)
        buckets = folded
    return buckets


def _should_merge(a: Bucket, b: Bucket, spec: PadSpec, min_wave: int) -> bool:
    a_poses = a.K * spec.seeds
    b_poses = b.K * spec.seeds
    # Merge only two under-occupied buckets, to fill the GPU toward a CTA wave; a full bucket
    # gains nothing from merging and would only add padding waste.
    if a_poses >= min_wave or b_poses >= min_wave:
        return False
    # Bound the merged size to _MERGE_MAX_WAVES waves. No pad-waste gate: both buckets are
    # under-occupied, so the occupancy win dominates, and the sort keeps the pad-up gradual.
    return a_poses + b_poses <= _MERGE_MAX_WAVES * min_wave


def _merge(a: Bucket, b: Bucket, spec: PadSpec) -> Bucket:
    return Bucket(a.members + b.members, {n: max(a.pad[n], b.pad[n]) for n in a.pad})
