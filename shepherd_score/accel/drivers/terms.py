"""Objective-term evaluators: one kernel launch per :class:`~shepherd_score.accel._modes.Term`.

Each evaluator takes the padded per-pose tensors of its channels plus the current pose and
returns ``(value, dQ, dT)``, the overlap and its gradient in unit-quaternion space as the
dispatched kernels emit them, or ``(value, None, None)`` for a value-only term. The
pose-invariant self-overlaps a reduction needs come from :func:`self_overlap`. Shape-family
launches are sliced at the int32 pointer-offset ceiling ``_launch_step`` derives from the pads;
the pharmacophore and color kernels take the whole batch in one 1-D grid.
"""
from __future__ import annotations

import math

import torch

from ..kernels.dispatch import (
    overlap_score_grad_se3_batch, overlap_score_grad_esp_se3_batch,
    overlap_score_grad_avoid_se3_batch, pharm_color_score_grad_se3_batch,
    pharm_grad_dq_se3_batch, _batch_self_overlap, _batch_self_overlap_esp,
    esp_agreement_grad_se3_batch,
)

# Poses per kernel launch. The kernels launch a one-dimensional grid, so the bound that applies
# is the int32 pointer offset formed as ``mol * N_pad * 3``, which ``_launch_step`` derives from
# the pads at call time. ``_CHUNK`` is an upper bound that can be lowered to force slicing.
_CHUNK = 2 ** 31 - 1
_PID_MAX = 2 ** 31 - 1


def _launch_step(S, args_mol):
    """Poses per launch: the int32 offset ceiling from the largest pad in ``args_mol``, capped by
    ``_CHUNK``, and rounded down to a whole seed group so a slice never splits one molecule's
    seeds (the kernel's own ``pid // S`` would then address the wrong molecule)."""
    pad = max((int(a.shape[1]) for a in args_mol if a.dim() >= 2), default=1)
    ceil = max(1, min(_CHUNK, _PID_MAX // (3 * max(pad, 1))))
    return ceil if S == 1 else max(S, (ceil // S) * S)

# lookup tables for the typed kernels

def build_element_tables(device, dtype, max_z: int = 100, alpha: float = 0.81):
    """Element-indexed ``(alphas, Ks, cats)`` for the color kernel: every element an isotropic
    point Gaussian of width ``alpha`` (category 0); the pad label (Z=0) category 3 (skipped).
    ``K = (pi/(2a))^1.5`` mirrors ``build_lookup_tables``."""
    n = max_z + 1
    alphas = torch.full((n,), float(alpha), device=device, dtype=dtype)
    Ks = torch.full((n,), (math.pi / (2.0 * alpha)) ** 1.5, device=device, dtype=dtype)
    cats = torch.zeros(n, device=device, dtype=torch.long)
    cats[0] = 3
    return alphas, Ks, cats


def tables_for(term, device, dtype, params):
    """The lookup-table triple a typed term's kernel takes, or ``None``."""
    if term.tables is None:
        return None
    from ...score.analytical_gradients._torch import build_lookup_tables
    if term.tables == "color":
        return build_lookup_tables(device, dtype, directionless=True)
    if term.tables == "pharm":
        return build_lookup_tables(device, dtype, directionless=False)
    if term.tables == "element":
        return build_element_tables(device, dtype, alpha=float(params["alpha"]))
    raise KeyError(term.tables)

# chunked shape-family launches

#: ``kw`` entries that are per molecule, not per pose, and so must be sliced with ``args_mol``
#: in every chunk; passed whole, poses from the second chunk on read another molecule's counts.
_MOL_KW = ("N_real", "M_real")


def _chunked(fn, K, S, args_mol, args_pose, kw, extra):
    """Run ``fn`` over ``K`` poses in grid-safe slices; ``args_mol`` are per-molecule (K//S
    rows), ``args_pose`` per pose. Keeps each molecule's seed group whole in a chunk, and slices
    the per-molecule ``kw`` entries (``_MOL_KW``) with the molecules."""
    step = _launch_step(S, args_mol)
    if K <= step:                                   # one launch: nothing to slice, nothing to copy
        return fn(*args_mol, *args_pose, **kw, **extra)
    out_V = torch.empty(K, device=args_pose[0].device, dtype=args_mol[0].dtype)
    out_dQ = torch.empty_like(args_pose[0])
    out_dT = torch.empty_like(args_pose[1])
    for s in range(0, K, step):
        e = min(s + step, K)
        ms, me = s // S, e // S
        kw_c = {n: (v[ms:me] if n in _MOL_KW and v is not None else v) for n, v in kw.items()}
        V, dQ, dT = fn(*[a[ms:me] for a in args_mol], *[a[s:e] for a in args_pose],
                       **kw_c, **extra)
        out_V[s:e] = V
        out_dQ[s:e] = dQ
        out_dT[s:e] = dT
    return out_V, out_dQ, out_dT


def _avoid_chunked(AV, B, q, t, n_real, m_real, div, kw):
    """The avoid kernel over ``K`` poses in grid-safe slices. ``AV`` / ``n_real`` hold one row
    per ``div`` poses (a pair's shared cloud), ``B`` / ``m_real`` / ``q`` / ``t`` one per pose;
    slices are whole multiples of ``div`` so each keeps its avoid rows aligned. A single row
    (one cloud for the whole batch) is read by every pose: each slice passes it whole with
    ``A_DIV`` = the slice's pose count."""
    K = int(q.shape[0])
    if int(AV.shape[0]) == 1:
        fn = overlap_score_grad_avoid_se3_batch
        step = _launch_step(1, (AV, B))
        if K <= step:
            return fn(AV, B, q, t, N_real=n_real, M_real=m_real, A_DIV=K, **kw)
        out_V = torch.empty(K, device=q.device, dtype=AV.dtype)
        out_dQ = torch.empty_like(q)
        out_dT = torch.empty_like(t)
        for s in range(0, K, step):
            e = min(s + step, K)
            V, dQ, dT = fn(AV, B[s:e], q[s:e], t[s:e], N_real=n_real, M_real=m_real[s:e],
                           A_DIV=e - s, **kw)
            out_V[s:e] = V
            out_dQ[s:e] = dQ
            out_dT[s:e] = dT
        return out_V, out_dQ, out_dT
    step = _launch_step(div, (AV, B))
    fn = overlap_score_grad_avoid_se3_batch
    if K <= step:
        return fn(AV, B, q, t, N_real=n_real, M_real=m_real, A_DIV=div, **kw)
    out_V = torch.empty(K, device=q.device, dtype=AV.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)
    for s in range(0, K, step):
        e = min(s + step, K)
        V, dQ, dT = fn(AV[s // div:e // div], B[s:e], q[s:e], t[s:e],
                       N_real=n_real[s // div:e // div], M_real=m_real[s:e], A_DIV=div, **kw)
        out_V[s:e] = V
        out_dQ[s:e] = dQ
        out_dT[s:e] = dT
    return out_V, out_dQ, out_dT


def _layout_kw(seeds_per_mol, poses_per_cta):
    extra = {}
    if seeds_per_mol > 1:
        extra["seeds_per_mol"] = int(seeds_per_mol)
    if poses_per_cta > 1:
        extra["poses_per_cta"] = int(poses_per_cta)
    return extra

# term evaluation

class TermInputs:
    """The device tensors one term reads for a bucket of poses.

    ``ref`` / ``fit``: per-channel padded tensors in the term's channel order (per molecule row
    when ``seeds_per_mol > 1``, else per pose); ``n_real`` / ``m_real``: int32 real counts;
    ``tables``: the kernel's lookup triple or None; ``guard``: per-pose bool mask (``None`` when
    every pair has real points on both sides).
    """
    __slots__ = ("ref", "fit", "n_real", "m_real", "tables", "guard", "params")

    def __init__(self, ref, fit, n_real, m_real, tables=None, guard=None, params=None):
        self.ref = tuple(ref)
        self.fit = tuple(fit)
        self.n_real = n_real
        self.m_real = m_real
        self.tables = tables
        self.guard = guard
        self.params = params or {}


def evaluate(term, ti: TermInputs, q, t, *, need_grad=True, seeds_per_mol=1, poses_per_cta=1):
    """``(value, dQ, dT)`` of ``term`` at pose ``(q, t)``; ``(value, None, None)`` when value-only."""
    p = ti.params
    S = int(seeds_per_mol)
    K = int(q.shape[0])
    if term.kernel == "shape":
        A, = ti.ref
        B, = ti.fit
        kw = dict(alpha=float(p["alpha"]), N_real=ti.n_real, M_real=ti.m_real, NEED_GRAD=need_grad)
        return _chunked(overlap_score_grad_se3_batch, K, S, (A, B), (q, t), kw,
                        _layout_kw(S, poses_per_cta))
    if term.kernel == "esp":
        A, CA = ti.ref
        B, CB = ti.fit
        kw = dict(alpha=float(p["alpha"]), lam=float(p["lam"]), N_real=ti.n_real,
                  M_real=ti.m_real, NEED_GRAD=need_grad)
        # the fused ESP kernel takes (A, B, CA, CB, q, t)
        return _chunked(lambda a, b, ca, cb, qq, tt, **k: overlap_score_grad_esp_se3_batch(
            a, b, ca, cb, qq, tt, **k), K, S, (A, B, CA, CB), (q, t), kw,
            _layout_kw(S, poses_per_cta))
    if term.kernel == "color":
        A, TA = ti.ref
        B, TB = ti.fit
        al, Ks, cats = ti.tables
        return pharm_color_score_grad_se3_batch(A, B, q, t, TA, TB, al, Ks, cats,
                                                N_real=ti.n_real, M_real=ti.m_real,
                                                NEED_GRAD=need_grad)
    if term.kernel == "pharm":
        A, VA, TA = ti.ref
        B, VB, TB = ti.fit
        al, Ks, cats = ti.tables
        return pharm_grad_dq_se3_batch(q, t, TA, TB, A, B, VA, VB, al, Ks, cats,
                                       N_real=ti.n_real, M_real=ti.m_real, NEED_GRAD=need_grad)
    if term.kernel == "avoid":
        AV, = ti.ref
        B, = ti.fit
        return _avoid_chunked(AV, B, q, t, ti.n_real, ti.m_real, int(p.get("_a_div", 1)),
                              dict(min_dist=float(p["avoid_min_dist"]), NEED_GRAD=need_grad))
    if term.kernel == "esp_cmp":
        V, dQ, dT = esp_agreement(ti, q, t, need_grad=need_grad, seeds_per_mol=S)
        return (V, dQ, dT) if need_grad else (V, None, None)
    raise KeyError(term.kernel)


def evaluate_fused_pair(t0, t1, ti0: TermInputs, ti1: TermInputs, q, t, params):
    """Both channels of a shape+color mode in one launch, or ``None`` if it does not apply.

    ``vol_color_triton.vol_color_score_grad_se3_batch`` computes the Gaussian shape overlap and
    the directionless typed-anchor overlap in a single kernel, sharing the R(q) build and the
    tile loads. It is single-tile, so it is used only when every padded width fits
    ``VOL_COLOR_FUSED_MAX_PAD``; it has no numba twin, so it is imported directly rather than
    through the kernel dispatcher, and CPU tensors take the two-kernel path.

    Returns ``((V0, dQ0, dT0), (V1, dQ1, dT1))`` in the two terms' own order.
    """
    A, = ti0.ref
    B, = ti0.fit
    if not A.is_cuda:
        return None
    from ..kernels.vol_color_triton import (
        vol_color_score_grad_se3_batch, VOL_COLOR_FUSED_MAX_PAD)
    AN, TA = ti1.ref
    BN, TB = ti1.fit
    if max(A.shape[1], B.shape[1], AN.shape[1], BN.shape[1]) > VOL_COLOR_FUSED_MAX_PAD:
        return None
    al, Ks, cats = ti1.tables
    Vs, dQs, dTs, Oc, dQc, dTc = vol_color_score_grad_se3_batch(
        A, B, AN, BN, q, t, TA, TB, al, Ks, cats, alpha=float(params["alpha"]),
        N_real_cent=ti0.n_real, M_real_cent=ti0.m_real,
        N_real_anc=ti1.n_real, M_real_anc=ti1.m_real)
    return (Vs, dQs, dTs), (Oc, dQc, dTc)


def esp_agreement(ti: TermInputs, q, t, *, need_grad=False, seeds_per_mol=1):
    """The ShaEP surface-ESP agreement in [0, 1] and its SE(3) gradient: each molecule's surface
    ESP against the Coulomb field of the other's transformed atoms, masked by vdW+probe,
    averaged over both surfaces. Channels (per side): cwh, partial, radii, surf, surf_esp; real
    counts: with-H atoms (``n_real``/``m_real``) and surface points (in ``params``). Returns
    ``(V, dQ, dT)``; ``dQ``/``dT`` are zeros when ``need_grad`` is False."""
    if int(seeds_per_mol) != 1:
        raise ValueError("esp_cmp takes the replicated layout (one row per pose)")
    p = ti.params
    args_mol = (*ti.ref, *ti.fit, p["_n_surf"], p["_m_surf"], ti.n_real, ti.m_real)
    kw = dict(probe_radius=float(p["probe_radius"]), lam=float(p["lam"]), NEED_GRAD=bool(need_grad))
    return _chunked(esp_agreement_grad_se3_batch, int(q.shape[0]), 1, args_mol, (q, t), kw, {})

# self-overlaps (pose-invariant)

def self_overlap(term, side, n_real, tables, params):
    """``V_XX`` of one side of ``term`` (identity pose), per row. ``side`` is that side's channel
    tuple. Returns ``None`` for a term without a normalising self-overlap."""
    if term.kernel == "shape":
        A, = side
        return _in_chunks(lambda a, n: _batch_self_overlap(a, n, float(params["alpha"])), A, n_real)
    if term.kernel == "esp":
        A, CA = side
        return _in_chunks(lambda a, ca, n: _batch_self_overlap_esp(
            a, ca, n, float(params["alpha"]), float(params["lam"])), A, CA, n_real)
    if term.kernel == "color":
        A, TA = side
        al, Ks, cats = tables
        K = A.shape[0]
        eye = torch.tensor([[1., 0., 0., 0.]], device=A.device, dtype=A.dtype).expand(K, 4)
        zero = torch.zeros(K, 3, device=A.device, dtype=A.dtype)
        O, _, _ = pharm_color_score_grad_se3_batch(A, A, eye, zero, TA, TA, al, Ks, cats,
                                                   N_real=n_real, M_real=n_real, NEED_GRAD=False)
        return O
    if term.kernel == "pharm":
        A, VA, TA = side
        al, Ks, cats = tables
        K = A.shape[0]
        q0 = torch.zeros(K, 4, device=A.device, dtype=A.dtype)
        q0[:, 0] = 1.0
        z = torch.zeros(K, 3, device=A.device, dtype=A.dtype)
        O, _, _ = pharm_grad_dq_se3_batch(q0, z, TA, TA, A, A, VA, VA, al, Ks, cats,
                                          N_real=n_real, M_real=n_real, NEED_GRAD=False)
        return O
    return None


def _in_chunks(fn, *arrs):
    K = arrs[0].shape[0]
    if K <= _CHUNK:
        return fn(*arrs)
    out = torch.empty(K, device=arrs[0].device, dtype=arrs[0].dtype)
    for s in range(0, K, _CHUNK):
        e = min(s + _CHUNK, K)
        out[s:e] = fn(*[a[s:e] for a in arrs])
    return out
