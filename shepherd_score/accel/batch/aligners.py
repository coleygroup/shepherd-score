"""Batched (multi-GPU-aware) aligners for :class:`MoleculePair`: one body for every mode.

``_align_batch_<mode>(pairs, **kw)`` is generated for each registry mode from its
:class:`~shepherd_score.accel._modes.ModeSpec`: upload the spec's channels once per molecule,
bucket the pairs on the spec's cost dims, pad and scatter-fill every channel, run each bucket
through the generic engine in memory-safe sub-batches, then write ``transform_<mode>`` /
``sim_aligned_<mode>`` back onto the pairs. Every function is a free function over duck-typed
``MoleculePair`` objects, which bind them as static methods; there is no runtime dependency on
``_core``, so the module imports cheaply and the worker processes stay picklable.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

if TYPE_CHECKING:                     # annotations only; never imported at runtime
    from shepherd_score.container._core import MoleculePair

from ._pad import _band_key, _subbatched_align, _scatter_fill, _BAND
from ._bucket import plan_buckets, PadSpec
from .._modes import MODE_SEEDS as _MODE_SEEDS, MODE_STEPS as _MODE_STEPS, SPECS, canonical
from ..channels import CHANNELS

_DTYPES = {"float32": torch.float32, "int64": torch.int64}

#: The channel whose reference cloud supplies the legacy ``trans_init`` translation centers.
_TRANS_CHANNEL = {"pharm": "pharm_ancs", "pharm_tversky": "pharm_ancs"}


def _seeds_for(mode: str) -> int:
    """Per-mode default seed count from the mode registry."""
    return _MODE_SEEDS.get(canonical(mode), 50)


def _steps_for(mode: str) -> int:
    """Per-mode default fine-step count from the mode registry."""
    return _MODE_STEPS.get(canonical(mode), 50)


def _ref_molec_of(p):
    return p.ref_molec


def _fit_molec_of(p):
    return p.fit_molec


def _batch_upload(pairs, attr, src_fn, dtype, device, *, key_fn=None):
    """Set ``p.<attr>`` for every ``p`` with one host concat and one ``.to(device)`` view-split
    instead of one ``torch.as_tensor`` per pair.

    To match a per-pair ``torch.as_tensor(src, dtype=..., device=...)`` exactly: the dtype cast
    goes through torch, never numpy ``.astype``; ``np.concatenate`` keeps the source dtype; each
    per-molecule view is ``.clone()``d into its own allocation; and each ``src_fn`` yields one
    uniform-dtype attribute. Only pairs whose ``<attr>`` is None get the batched upload, and a
    wrong-device cached tensor is moved per pair. The upload is keyed per molecule, since an
    all-vs-all workload draws K pairs from far fewer molecules.
    """
    cold = [p for p in pairs if getattr(p, attr, None) is None]
    if cold:
        if key_fn is None:
            if attr.startswith("_ref"):
                key_fn = _ref_molec_of
            elif attr.startswith("_fit"):
                key_fn = _fit_molec_of
        if key_fn is None:
            reps = cold                                        # un-keyable attr: per-pair
        else:
            _first = {}
            for p in cold:
                _first.setdefault(id(key_fn(p)), p)            # molecules stay alive via ``pairs``
            reps = list(_first.values())
        arrs = [np.asarray(src_fn(p)) for p in reps]           # numpy, host, cheap
        sizes = [len(a) for a in arrs]
        flat = np.concatenate(arrs) if arrs else np.zeros((0,), np.float32)
        dev = torch.from_numpy(flat).to(device=device, dtype=dtype)  # torch does the cast
        if key_fn is None:
            for p, t in zip(reps, dev.split(sizes)):
                setattr(p, attr, t.clone())
        else:
            _ten = {}
            for p, t in zip(reps, dev.split(sizes)):
                _ten[id(key_fn(p))] = t.clone()                # one allocation per molecule
            for p in cold:
                setattr(p, attr, _ten[id(key_fn(p))])
    # Warm wrong-device path. Only a copy onto CUDA may be non-blocking: the kernels that read
    # it queue behind it on the stream. A copy off the GPU must block, since the numba kernels
    # read host memory directly and would race a copy still in flight.
    nb = device.type == "cuda"
    # A bare "cuda" never equals a tensor's "cuda:0", so every pair would re-run a no-op .to()
    # on every call; compare against the indexed current device instead.
    if nb and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for p in pairs:
        t = getattr(p, attr)
        if t.device != device:
            setattr(p, attr, t.to(device, non_blocking=nb))

# the generic aligner

#: Keywords every mode accepts that are not objective parameters: the optimiser schedule, the
#: legacy translation-seeded grid, and the screen front end's constant seeds and avoid cloud.
_LOOP_KW = ("steps_fine", "num_repeats", "trans_init", "num_repeats_per_trans", "topk",
            "early_stop_patience", "early_stop_tol", "const_seeds", "avoid_points",
            "avoid_points_t")


def resolve_params(spec, kw: dict, fn_name: str) -> dict:
    """Mode parameters for this call: the spec defaults overlaid with ``kw``; a required
    parameter (spec default ``None``) must be supplied; an unknown keyword is a TypeError."""
    params = dict(spec.params)
    for k, v in kw.items():
        if k in params:
            params[k] = v
        elif k not in _LOOP_KW:
            raise TypeError(f"{fn_name}() got an unexpected keyword argument {k!r}")
    for k, v in params.items():
        if v is None:
            raise TypeError(f"{fn_name}() missing required keyword-only argument: {k!r}")
    if spec.lam_scaling:
        # Surface convention: the caller's ``lam`` is raw and is scaled once by LAM_SCALING here,
        # for both the cross-overlap and the self-overlaps; the atom-centered ESP modes take it
        # raw (see score/constants.py).
        from ...score.constants import LAM_SCALING
        params["lam"] = LAM_SCALING * float(params["lam"])
    return params


def concrete_channels(spec, params, *, trans_init=False):
    """The concrete channel names this call reads (switches resolved), in spec order."""
    out = []
    for c in spec.channels:
        r = spec.resolve_channel(c, params)
        if r not in out:
            out.append(r)
    if trans_init:
        tc = _TRANS_CHANNEL.get(spec.name, "atoms")
        if tc not in out:
            out.append(tc)
    return out


def upload_channels(pairs, names, device):
    """Ensure every pair holds the cached device tensors of ``names`` (cold pairs are read off
    their molecules; the reader raises a clear ``ValueError`` when the data is missing)."""
    for n in names:
        ch = CHANNELS[n]
        dt = _DTYPES[ch.dtype]
        if ch.is_pair:
            _batch_upload(pairs, ch.ref_attr, ch.read, dt, device)
        else:
            _batch_upload(pairs, ch.ref_attr, lambda p, r=ch.read: r(p.ref_molec), dt, device)
            _batch_upload(pairs, ch.fit_attr, lambda p, r=ch.read: r(p.fit_molec), dt, device)


def _pad_spec(spec, params, names, n_seeds, trans_init):
    """The :class:`PadSpec` for this call: one merge dim per side per bucket channel, an exact
    ``tc`` partition under ``trans_init``, and the spec's cost model."""
    bch = []
    for b in spec.bucket:
        r = spec.resolve_channel(b, params)
        if r not in bch:
            bch.append(r)
    merge = {}
    if len(bch) == 1:
        ch = CHANNELS[bch[0]]
        merge["ref"] = lambda p, a=ch.ref_attr: getattr(p, a).shape[0]
        merge["fit"] = lambda p, a=ch.fit_attr: getattr(p, a).shape[0]
    else:
        for b in bch:
            ch = CHANNELS[b]
            merge[f"n_{b}"] = lambda p, a=ch.ref_attr: getattr(p, a).shape[0]
            merge[f"m_{b}"] = lambda p, a=ch.fit_attr: getattr(p, a).shape[0]
    partition = {}
    if trans_init:
        ta = CHANNELS[_TRANS_CHANNEL.get(spec.name, "atoms")].ref_attr
        partition["tc"] = lambda p, a=ta: int(getattr(p, a).shape[0])
    work = None
    if spec.work == "combo":
        cent = spec.resolve_channel("centers", params)

        def work(pad, c=cent):
            return (pad[f"n_{c}"] * pad[f"m_{c}"] + pad["n_surf"] * pad["m_cwh"]
                    + pad["m_surf"] * pad["n_cwh"])
    return PadSpec(merge=merge, seeds=n_seeds, partition=partition, work=work), bch


def _pad_width(bk, bch, name, side, sizes):
    """Padded width of channel ``name`` in bucket ``bk``: the planned pad for a bucket channel
    (any channel sharing its basis), else the bucket max banded, floored at one band."""
    ch = CHANNELS[name]
    for b in bch:
        if CHANNELS[b].basis == ch.basis:
            if len(bch) == 1:
                return int(bk.pad["ref" if side == "ref" else "fit"])
            return int(bk.pad[f"n_{b}" if side == "ref" else f"m_{b}"])
    return _band_key(max(sizes)) or _BAND


def build_bucket(spec, bucket, bk, bch, names, device):
    """Padded :class:`Batch` per channel for one bucket of pairs, plus the per-basis sizes."""
    from ..drivers.engine import Batch
    K = len(bucket)
    sizes_ref, sizes_fit = {}, {}                              # basis -> list[int]
    chans = {}
    for n in names:
        ch = CHANNELS[n]
        dt = _DTYPES[ch.dtype]
        ts_ref = [getattr(p, ch.ref_attr) for p in bucket]
        if ch.basis not in sizes_ref:
            sizes_ref[ch.basis] = [int(t.shape[0]) for t in ts_ref]
        n_list = sizes_ref[ch.basis]
        n_pad = _pad_width(bk, bch, n, "ref", n_list)
        feat = tuple(ts_ref[0].shape[1:])
        if ch.pad == 0:
            ref = torch.zeros((K, n_pad) + feat, device=device, dtype=dt)
        else:
            ref = torch.full((K, n_pad) + feat, ch.pad, device=device, dtype=dt)
        _scatter_fill(ref, ts_ref, n_list)
        n_real = torch.tensor(n_list, device=device, dtype=torch.int32)
        if ch.is_pair:
            chans[n] = Batch(ref, None, n_real, None)
            continue
        ts_fit = [getattr(p, ch.fit_attr) for p in bucket]
        if ch.basis not in sizes_fit:
            sizes_fit[ch.basis] = [int(t.shape[0]) for t in ts_fit]
        m_list = sizes_fit[ch.basis]
        m_pad = _pad_width(bk, bch, n, "fit", m_list)
        if ch.pad == 0:
            fit = torch.zeros((K, m_pad) + feat, device=device, dtype=dt)
        else:
            fit = torch.full((K, m_pad) + feat, ch.pad, device=device, dtype=dt)
        _scatter_fill(fit, ts_fit, m_list)
        m_real = torch.tensor(m_list, device=device, dtype=torch.int32)
        chans[n] = Batch(ref, fit, n_real, m_real)
    return chans


def _slice_chans(chans, sl):
    from ..drivers.engine import Batch
    return {n: Batch(b.ref[sl], None if b.fit is None else b.fit[sl], b.n_real[sl],
                     None if b.m_real is None else b.m_real[sl]) for n, b in chans.items()}


def _run_bucket(spec, chans, K, key, device, *, params, n_seeds, steps_fine, es_patience,
                es_tol, ref_shared, trans_centers, trans_centers_real, nrpt, topk, seeds=None,
                pose_cap=0):
    """One bucket through the engine in memory-safe sub-batches; returns ``(scores, q, t)``."""
    from ..drivers import engine

    so = engine.term_self_overlaps(spec, chans, params, ref_shared=ref_shared)   # per bucket

    def _proc(_s, _k):
        sl = slice(_s, _s + _k)
        return engine.align(
            spec, _slice_chans(chans, sl), params=params, num_seeds=n_seeds,
            steps_fine=steps_fine, lr=float(params["lr"]), early_stop_patience=es_patience,
            early_stop_tol=es_tol, ref_shared=ref_shared,
            self_overlaps=[None if p is None else (p[0][sl], p[1][sl]) for p in so],
            seeds=None if seeds is None else (seeds[0][sl], seeds[1][sl]),
            trans_centers=None if trans_centers is None else trans_centers[sl],
            trans_centers_real=None if trans_centers_real is None else trans_centers_real[sl],
            num_repeats_per_trans=nrpt, topk=topk)
    return _subbatched_align(_proc, K, key=key, device=device, pose_cap=pose_cap, seeds=n_seeds)


def _align_batch(spec, pairs, _fn, **kw) -> None:
    if not pairs:
        return
    name = _fn.__name__
    params = resolve_params(spec, kw, name)
    steps_fine = int(kw.get("steps_fine", _steps_for(spec.name)))
    nr = kw.get("num_repeats")
    n_seeds = int(nr) if nr is not None else _seeds_for(spec.name)
    trans_init = bool(kw.get("trans_init", False))
    nrpt = int(kw.get("num_repeats_per_trans", 10))
    topk = int(kw.get("topk", 30))
    es_patience = int(kw.get("early_stop_patience", spec.patience))
    es_tol = float(kw.get("early_stop_tol", 1e-5))

    # The pharmacophore family's ``extended_points`` objective has no kernel; it runs the
    # eager autograd driver.
    if spec.pharm_style and params.get("extended_points"):
        from .aligners_legacy import _align_batch_pharm_extended
        return _align_batch_pharm_extended(spec, pairs, params, n_seeds=n_seeds,
                                           steps_fine=steps_fine, trans_init=trans_init,
                                           nrpt=nrpt, topk=topk)

    device = pairs[0].device
    if device.type != "cuda":
        try:
            import numba  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"batched {spec.name} alignment on CPU requires numba; install numba "
                "(pip install numba) or run this mode on CUDA") from e

    names = concrete_channels(spec, params, trans_init=trans_init)
    upload_channels(pairs, names, device)
    pspec, bch = _pad_spec(spec, params, names, n_seeds, trans_init)
    ref_attrs = [CHANNELS[n].ref_attr for n in names]
    tcn = _TRANS_CHANNEL.get(spec.name, "atoms")

    all_pairs, all_scores, all_q, all_t = [], [], [], []
    for _bk in plan_buckets(pairs, pspec, device):
        bucket = _bk.members
        K = _bk.K
        chans = build_bucket(spec, bucket, _bk, bch, names, device)
        # A shared reference (one query object on every pair, as the screen sets it) lets the
        # engine solve the reference self-overlaps and seed frame once. Object identity, not
        # value: a pure copy of one source object gives identical rows.
        ref_shared = K > 1 and all(getattr(p, a) is getattr(bucket[0], a)
                                   for p in bucket for a in ref_attrs)
        tcb = tcr = None
        if trans_init:
            tc = int(_bk.pad["tc"])
            tcb = chans[tcn].ref[:, :tc].contiguous()
            tcr = torch.full((K,), tc, device=device, dtype=torch.int32)
        pads = tuple(int(_bk.pad[n]) for n in sorted(_bk.pad))
        scores, q_b, t_b = _run_bucket(
            spec, chans, K, (spec.name,) + pads + (n_seeds,), device, params=params,
            n_seeds=n_seeds, steps_fine=steps_fine, es_patience=es_patience, es_tol=es_tol,
            ref_shared=ref_shared, trans_centers=tcb, trans_centers_real=tcr, nrpt=nrpt,
            topk=topk)
        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_b)
        all_t.append(t_b)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()
    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu).detach().numpy()        # batched
    scores_list = scores_cpu.tolist()                                        # one C call
    tf_attr, sc_attr = spec.attrs
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        setattr(p, tf_attr, S)
        setattr(p, sc_attr, s)


def _make_aligner(mode: str):
    spec = SPECS[mode]

    def fn(pairs, **kw):
        return _align_batch(spec, pairs, fn, **kw)
    fn.__name__ = fn.__qualname__ = f"_align_batch_{mode}"
    fn.__doc__ = (f"Batched ``{mode}`` alignment over duck-typed MoleculePairs. Keywords: "
                  f"{', '.join(spec.params)}, steps_fine, num_repeats, trans_init, "
                  "num_repeats_per_trans, topk. Writes ``{}`` / ``{}`` onto each pair."
                  .format(*spec.attrs))
    return fn


for _m in SPECS:
    globals()[f"_align_batch_{_m}"] = _make_aligner(_m)

# --- legacy mode aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp) ----------
# Same function objects, so ``__name__`` stays canonical.
_align_batch_esp = _align_batch_surf_esp          # noqa: F821  (generated above)
_align_batch_esp_combo = _align_batch_vol_and_surf_esp   # noqa: F821

__all__ = ["_batch_upload", "_seeds_for", "_steps_for", "resolve_params", "concrete_channels", "upload_channels", "build_bucket"] + \
          [f"_align_batch_{m}" for m in SPECS] + ["_align_batch_esp", "_align_batch_esp_combo"]
