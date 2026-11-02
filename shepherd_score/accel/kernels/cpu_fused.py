"""Fused CPU (numba) fine loop: a first-class CPU path for every mode.

No torch in the hot loop: inputs are marshalled to numpy once, each step chains the per-term
overlap+grad njit kernels with two njit ``prange`` tails (score / best / blended descent
gradient, then the tangent-projected Adam), and the per-pair early-stop check is a numpy
``.max(axis=1)``. The tails reproduce the torch fp32 arithmetic of ``drivers/engine.py`` op for
op (same operand order, float32 constants, eps inside the sqrt, no bias correction). With SVML
the SoA fp32 kernels of ``cpu_soa.py`` run; otherwise the fp64 AoS kernels of ``cpu.py``.
Seeds, step count and early-stop schedule are the caller's.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from ._svml import enable_svml

from .._stats import record as _record_steps

enable_svml()   # before any kernel compiles; pip's icc_rt needs the explicit load

# Adam constants as float32, matching what torch computes in float32 from the double literals.
_B1 = np.float32(0.9)
_A1 = np.float32(1.0 - 0.9)
_B2 = np.float32(0.999)
_A2 = np.float32(1.0 - 0.999)
_EPS = np.float32(1e-8)
_F1 = np.float32(1.0)
_F0 = np.float32(0.0)

try:
    import numba.core.config as _nbcfg
    _SVML = bool(_nbcfg.USING_SVML)
except Exception:
    _SVML = False
_USE_SOA = _SVML

_SVML_WARNED = False


def _warn_if_no_svml():
    global _SVML_WARNED
    if not _SVML and not _SVML_WARNED:
        _SVML_WARNED = True
        import warnings
        warnings.warn(
            "CPU alignment is running WITHOUT numba SVML "
            "(USING_SVML=False; numba>=0.62 dropped it), so the overlap kernels are unvectorized "
            "and ~3-6x slower than they should be. For full CPU speed, pip install "
            "'shepherd-score[cpu]' (numba<0.62 + icc_rt) or build the conda environment.yml. "
            "Suppress via warnings filters.",
            RuntimeWarning, stacklevel=3)

# njit tails

@njit(parallel=True, fastmath=False, cache=True)
def _tail_blend(Vg, dQg, dTg, kind, kc, cst, guard, useg, gpos, sims, wt, q, t, best, bq, bt,
                gq, gt):
    """Per pose: reduce every gradient term (Tanimoto / Tversky / raw, guarded), blend the
    similarities in term order (value-only rows of ``sims`` prefilled by the host), track the
    best pre-Adam pose, and build the blended descent gradient into ``gq``/``gt``."""
    P = q.shape[0]
    Tg = Vg.shape[0]
    T = sims.shape[0]
    for p in prange(P):
        # ---- gradient terms: similarity + d(sim)/dV, then their gradient contribution -------
        for g in range(Tg):
            V = Vg[g, p]
            k = kind[g]
            if k == 0:
                denom = cst[g, p] - V
                if useg[g] and not guard[g, p]:
                    denom = _F1
                sim = V / denom
                scale = cst[g, p] / (denom * denom)
            elif k == 1:
                denom = kc[g] * V + cst[g, p]
                if useg[g] and not guard[g, p]:
                    denom = _F1
                sim = V / denom
                scale = cst[g, p] / (denom * denom)
            else:
                sim = V
                scale = _F1
            if useg[g] and not guard[g, p]:
                sim = _F0
                scale = _F0
            j = gpos[g]
            sims[j, p] = sim
            w = wt[j]
            if g == 0:
                x0 = dQg[g, p, 0] * scale; x1 = dQg[g, p, 1] * scale
                x2 = dQg[g, p, 2] * scale; x3 = dQg[g, p, 3] * scale
                y0 = dTg[g, p, 0] * scale; y1 = dTg[g, p, 1] * scale; y2 = dTg[g, p, 2] * scale
                x0 = -x0; x1 = -x1; x2 = -x2; x3 = -x3
                y0 = -y0; y1 = -y1; y2 = -y2
                if w != _F1:
                    x0 = x0 * w; x1 = x1 * w; x2 = x2 * w; x3 = x3 * w
                    y0 = y0 * w; y1 = y1 * w; y2 = y2 * w
                gq[p, 0] = x0; gq[p, 1] = x1; gq[p, 2] = x2; gq[p, 3] = x3
                gt[p, 0] = y0; gt[p, 1] = y1; gt[p, 2] = y2
            else:
                nw = -w
                x0 = dQg[g, p, 0] * scale; x1 = dQg[g, p, 1] * scale
                x2 = dQg[g, p, 2] * scale; x3 = dQg[g, p, 3] * scale
                y0 = dTg[g, p, 0] * scale; y1 = dTg[g, p, 1] * scale; y2 = dTg[g, p, 2] * scale
                if w != _F1:
                    x0 = x0 * nw; x1 = x1 * nw; x2 = x2 * nw; x3 = x3 * nw
                    y0 = y0 * nw; y1 = y1 * nw; y2 = y2 * nw
                else:
                    x0 = -x0; x1 = -x1; x2 = -x2; x3 = -x3
                    y0 = -y0; y1 = -y1; y2 = -y2
                gq[p, 0] += x0; gq[p, 1] += x1; gq[p, 2] += x2; gq[p, 3] += x3
                gt[p, 0] += y0; gt[p, 1] += y1; gt[p, 2] += y2
        # ---- blend in term order: score = w0*s0 (+ w1*s1 ...) -------------------------------
        s = sims[0, p]
        if wt[0] != _F1:
            s = s * wt[0]
        for j in range(1, T):
            c = sims[j, p]
            if wt[j] != _F1:
                c = c * wt[j]
            s = s + c
        if s > best[p]:
            best[p] = s
            bq[p, 0] = q[p, 0]; bq[p, 1] = q[p, 1]; bq[p, 2] = q[p, 2]; bq[p, 3] = q[p, 3]
            bt[p, 0] = t[p, 0]; bt[p, 1] = t[p, 1]; bt[p, 2] = t[p, 2]


@njit(parallel=True, fastmath=False, cache=True)
def _tail_pharm(O, dQr, dTr, norm, C, tanimoto, q, t, best, bq, bt, gq, gt):
    """Pharm-style score / best / gradient: the kernel saw the unit-normalised ``q``; apply the
    guarded (clamped for Tversky) similarity, the normalisation Jacobian and leave the raw
    (not yet tangent-projected) gradient in ``gq``/``gt``."""
    P = q.shape[0]
    for p in prange(P):
        q0 = q[p, 0]; q1 = q[p, 1]; q2 = q[p, 2]; q3 = q[p, 3]
        qn = np.float32(math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
        qnc = qn
        if qnc < np.float32(1e-12):
            qnc = np.float32(1e-12)
        u0 = q0 / qnc; u1 = q1 / qnc; u2 = q2 / qnc; u3 = q3 / qnc
        Op = O[p]
        if tanimoto:
            denom = norm[p] - Op
            if denom > np.float32(1e-8):
                score = Op / denom
            else:
                score = _F0
            scale = -norm[p] / (denom * denom)
        else:
            D = C[p]
            if D > np.float32(1e-8):
                score = Op / D
            else:
                score = _F0
            if score > _F1:
                score = _F1
            active = _F1 if Op < D else _F0
            scale = -active / D
        s0 = scale * dQr[p, 0]; s1 = scale * dQr[p, 1]; s2 = scale * dQr[p, 2]; s3 = scale * dQr[p, 3]
        gt[p, 0] = scale * dTr[p, 0]; gt[p, 1] = scale * dTr[p, 1]; gt[p, 2] = scale * dTr[p, 2]
        dot = u0 * s0 + u1 * s1 + u2 * s2 + u3 * s3
        gq[p, 0] = (s0 - u0 * dot) / qnc
        gq[p, 1] = (s1 - u1 * dot) / qnc
        gq[p, 2] = (s2 - u2 * dot) / qnc
        gq[p, 3] = (s3 - u3 * dot) / qnc
        if score > best[p]:
            best[p] = score
            bq[p, 0] = q0; bq[p, 1] = q1; bq[p, 2] = q2; bq[p, 3] = q3
            bt[p, 0] = t[p, 0]; bt[p, 1] = t[p, 1]; bt[p, 2] = t[p, 2]


@njit(parallel=True, fastmath=False, cache=True)
def _tail_adam(q, t, gq, gt, mq, vq, mt, vt, lr):
    """Tangent-projected Adam on ``q`` + plain Adam on ``t`` + unit renorm, float32, in the
    operand order of ``kernels/cpu.py::fused_adam_qt_with_tangent_proj``."""
    P = q.shape[0]
    nlr = -lr
    for p in prange(P):
        radial = gq[p, 0] * q[p, 0] + gq[p, 1] * q[p, 1] + gq[p, 2] * q[p, 2] + gq[p, 3] * q[p, 3]
        d0 = gq[p, 0] - q[p, 0] * radial; d1 = gq[p, 1] - q[p, 1] * radial
        d2 = gq[p, 2] - q[p, 2] * radial; d3 = gq[p, 3] - q[p, 3] * radial
        mq[p, 0] = mq[p, 0] * _B1 + d0 * _A1; mq[p, 1] = mq[p, 1] * _B1 + d1 * _A1
        mq[p, 2] = mq[p, 2] * _B1 + d2 * _A1; mq[p, 3] = mq[p, 3] * _B1 + d3 * _A1
        vq[p, 0] = vq[p, 0] * _B2 + (_A2 * d0) * d0; vq[p, 1] = vq[p, 1] * _B2 + (_A2 * d1) * d1
        vq[p, 2] = vq[p, 2] * _B2 + (_A2 * d2) * d2; vq[p, 3] = vq[p, 3] * _B2 + (_A2 * d3) * d3
        q[p, 0] = q[p, 0] + (nlr * mq[p, 0]) / np.float32(math.sqrt(vq[p, 0] + _EPS))
        q[p, 1] = q[p, 1] + (nlr * mq[p, 1]) / np.float32(math.sqrt(vq[p, 1] + _EPS))
        q[p, 2] = q[p, 2] + (nlr * mq[p, 2]) / np.float32(math.sqrt(vq[p, 2] + _EPS))
        q[p, 3] = q[p, 3] + (nlr * mq[p, 3]) / np.float32(math.sqrt(vq[p, 3] + _EPS))
        e0 = gt[p, 0]; e1 = gt[p, 1]; e2 = gt[p, 2]
        mt[p, 0] = mt[p, 0] * _B1 + e0 * _A1; mt[p, 1] = mt[p, 1] * _B1 + e1 * _A1
        mt[p, 2] = mt[p, 2] * _B1 + e2 * _A1
        vt[p, 0] = vt[p, 0] * _B2 + (_A2 * e0) * e0; vt[p, 1] = vt[p, 1] * _B2 + (_A2 * e1) * e1
        vt[p, 2] = vt[p, 2] * _B2 + (_A2 * e2) * e2
        t[p, 0] = t[p, 0] + (nlr * mt[p, 0]) / np.float32(math.sqrt(vt[p, 0] + _EPS))
        t[p, 1] = t[p, 1] + (nlr * mt[p, 1]) / np.float32(math.sqrt(vt[p, 1] + _EPS))
        t[p, 2] = t[p, 2] + (nlr * mt[p, 2]) / np.float32(math.sqrt(vt[p, 2] + _EPS))
        qn = np.float32(math.sqrt(q[p, 0] * q[p, 0] + q[p, 1] * q[p, 1] + q[p, 2] * q[p, 2]
                                  + q[p, 3] * q[p, 3]))
        q[p, 0] = q[p, 0] / qn; q[p, 1] = q[p, 1] / qn; q[p, 2] = q[p, 2] / qn; q[p, 3] = q[p, 3] / qn

# marshalling: torch -> numpy once; per-term kernel closures

def _f32c(x):
    return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)


def _f64c(x):
    return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float64)


def _i64(x):
    return np.ascontiguousarray(x.detach().cpu().numpy()).astype(np.int64)


def _cast3(V, dQ, dT):
    return V.astype(np.float32), dQ.astype(np.float32), dT.astype(np.float32)


def _term_closure(tm, params):
    """``(q_np, t_np) -> (V, dQ, dT)`` float32 for one term, marshalled once."""
    ti = tm.inputs
    kind = tm.spec.kernel
    Nr, Mr = _i64(ti.n_real), _i64(ti.m_real)
    if kind == "shape":
        a_f = float(params["alpha"])
        if _USE_SOA:
            from .cpu_soa import _overlap_grad_kernel_soa, to_soa
            A = to_soa(_f32c(ti.ref[0])); B = to_soa(_f32c(ti.fit[0]))
            return lambda q, t: _overlap_grad_kernel_soa(A, B, q, t, Nr, Mr, a_f, True)
        from .cpu import _overlap_grad_kernel
        A = _f32c(ti.ref[0]); B = _f32c(ti.fit[0])
        return lambda q, t: _cast3(*_overlap_grad_kernel(A, B, q, t, Nr, Mr, a_f, True))
    if kind == "esp":
        a_f = float(params["alpha"]); inv_lam = 1.0 / float(params["lam"])
        CA = _f32c(ti.ref[1]); CB = _f32c(ti.fit[1])
        if _USE_SOA:
            from .cpu_soa import _overlap_grad_esp_kernel_soa, to_soa
            A = to_soa(_f32c(ti.ref[0])); B = to_soa(_f32c(ti.fit[0]))
            return lambda q, t: _overlap_grad_esp_kernel_soa(A, B, CA, CB, q, t, Nr, Mr, a_f,
                                                             inv_lam, True)
        from .cpu import _overlap_grad_esp_kernel
        A = _f32c(ti.ref[0]); B = _f32c(ti.fit[0])
        return lambda q, t: _cast3(*_overlap_grad_esp_kernel(A, B, CA, CB, q, t, Nr, Mr, a_f,
                                                             inv_lam, True))
    if kind == "color":
        from .cpu import _pharm_color_grad_kernel
        A = _f32c(ti.ref[0]); B = _f32c(ti.fit[0]); At = _i64(ti.ref[1]); Bt = _i64(ti.fit[1])
        al, Ks, cats = ti.tables
        aln, Ksn, cn = _f64c(al), _f64c(Ks), _i64(cats)
        return lambda q, t: _cast3(*_pharm_color_grad_kernel(A, B, q, t, At, Bt, aln, Ksn, cn,
                                                             Nr, Mr, True))
    if kind == "pharm":
        from .cpu import _pharm_grad_dq_kernel
        A, VA, TA = _f32c(ti.ref[0]), _f32c(ti.ref[1]), _i64(ti.ref[2])
        B, VB, TB = _f32c(ti.fit[0]), _f32c(ti.fit[1]), _i64(ti.fit[2])
        al, Ks, cats = ti.tables
        aln, Ksn, cn = _f64c(al), _f64c(Ks), _i64(cats)
        return lambda q, t: _cast3(*_pharm_grad_dq_kernel(A, B, q, t, TA, TB, VA, VB, aln, Ksn,
                                                          cn, Nr, Mr, True))
    if kind == "avoid":
        from .cpu import _avoid_grad_kernel
        AV = _f32c(ti.ref[0]); B = _f32c(ti.fit[0]); d0 = float(params["avoid_min_dist"])
        return lambda q, t: _cast3(*_avoid_grad_kernel(AV, B, q, t, Nr, Mr, d0, True))
    if kind == "esp_cmp":
        from .cpu import _esp_agreement_grad_kernel
        from ...score.constants import COULOMB_SCALING, LAM_SCALING
        cwh1, pc1, rad1, pts1, ptc1 = (_f32c(x) for x in ti.ref)
        cwh2, pc2, rad2, pts2, ptc2 = (_f32c(x) for x in ti.fit)
        n_surf = _i64(ti.params["_n_surf"]); m_surf = _i64(ti.params["_m_surf"])
        inv_lam = 1.0 / (LAM_SCALING * float(params["lam"]))
        coul = float(COULOMB_SCALING); probe = float(params["probe_radius"])
        need = bool(tm.spec.grad)
        Abuf = np.empty((cwh2.shape[0], cwh2.shape[1], 3), dtype=np.float64)
        Fbuf = np.empty_like(Abuf)

        def _ev(q, t):
            V, dQ, dT = _esp_agreement_grad_kernel(pts1, ptc1, n_surf, cwh1, pc1, rad1, Nr,
                                                   pts2, ptc2, m_surf, cwh2, pc2, rad2, Mr,
                                                   q, t, inv_lam, coul, probe, Abuf, Fbuf, need)
            return _cast3(V, dQ, dT) if need else (V.astype(np.float32), None, None)
        return _ev
    raise KeyError(kind)

# the loop

def run_fused(pr, steps, lr, es_patience, es_tol):
    """Run ``pr`` (an assembled :class:`~drivers.engine.Problem`, replicated layout, CPU fp32)
    through the fused loop. Returns torch ``(best, bq, bt)`` on the problem's device."""
    import torch
    _warn_if_no_svml()
    spec = pr.spec
    P = int(pr.q.shape[0])
    S = int(pr.P) or P
    q = _f32c(pr.q); t = _f32c(pr.t)
    mq = np.zeros((P, 4), np.float32); vq = np.zeros((P, 4), np.float32)
    mt = np.zeros((P, 3), np.float32); vt = np.zeros((P, 3), np.float32)
    gq = np.zeros((P, 4), np.float32); gt = np.zeros((P, 3), np.float32)
    best = np.full(P, -np.inf, np.float32)
    bq = q.copy(); bt = t.copy()
    prev = np.full(P // S, -np.inf, np.float32)
    no_improve = 0
    lr32 = np.float32(lr)
    evals = [_term_closure(tm, pr.params) for tm in pr.terms]

    if spec.pharm_style:
        tm = pr.terms[0]
        tanimoto = pr.similarity == "tanimoto"
        norm = _f32c(tm.norm) if tm.norm is not None else np.zeros(P, np.float32)
        C = _f32c(tm.C) if tm.C is not None else np.zeros(P, np.float32)
        ev = evals[0]
        step = -1
        for step in range(steps):
            n = np.sqrt((q * q).sum(1, keepdims=True)).astype(np.float32)
            qu = (q / np.maximum(n, np.float32(1e-12))).astype(np.float32)
            O, dQr, dTr = ev(qu, t)
            _tail_pharm(O, dQr, dTr, norm, C, tanimoto, q, t, best, bq, bt, gq, gt)
            if step % 5 == 0:
                cur = best.reshape(-1, S).max(axis=1)
                improved = (cur - prev) > es_tol
                if not improved.any():
                    no_improve += 1
                    if no_improve >= es_patience:
                        break
                else:
                    no_improve = 0
                prev = np.where(improved, cur, prev)
            _tail_adam(q, t, gq, gt, mq, vq, mt, vt, lr32)
    else:
        grad_ix = [i for i, tm in enumerate(pr.terms) if tm.spec.grad]
        val_ix = [i for i, tm in enumerate(pr.terms) if not tm.spec.grad]
        Tg, T = len(grad_ix), len(pr.terms)
        Vg = np.zeros((Tg, P), np.float32); dQg = np.zeros((Tg, P, 4), np.float32)
        dTg = np.zeros((Tg, P, 3), np.float32)
        kind = np.zeros(Tg, np.int64); kc = np.zeros(Tg, np.float32)
        cst = np.zeros((Tg, P), np.float32); guard = np.ones((Tg, P), np.bool_)
        useg = np.zeros(Tg, np.bool_); gpos = np.zeros(Tg, np.int64)
        sims = np.zeros((T, P), np.float32)
        wt = np.array([np.float32(tm.weight) for tm in pr.terms], np.float32)
        for g, i in enumerate(grad_ix):
            tm = pr.terms[i]
            gpos[g] = i
            red = tm.spec.reduction
            if red == "tanimoto":
                kind[g] = 0; cst[g] = _f32c(tm.norm)
            elif red == "tversky":
                kind[g] = 1; kc[g] = np.float32(tm.k); cst[g] = _f32c(tm.C)
            else:
                kind[g] = 2
            if tm.guard is not None:
                useg[g] = True; guard[g] = tm.guard.detach().cpu().numpy().astype(np.bool_)
        step = -1
        for step in range(steps):
            for g, i in enumerate(grad_ix):
                V, dQ, dT = evals[i](q, t)
                Vg[g] = V; dQg[g] = dQ; dTg[g] = dT
            for i in val_ix:
                V, _, _ = evals[i](q, t)
                tm = pr.terms[i]
                sims[i] = V if tm.guard is None else np.where(
                    tm.guard.detach().cpu().numpy(), V, np.float32(0.0))
            _tail_blend(Vg, dQg, dTg, kind, kc, cst, guard, useg, gpos, sims, wt, q, t, best,
                        bq, bt, gq, gt)
            if step % 5 == 0:
                cur = best.reshape(-1, S).max(axis=1)
                improved = (cur - prev) > es_tol
                if not improved.any():
                    no_improve += 1
                    if no_improve >= es_patience:
                        break
                else:
                    no_improve = 0
                prev = np.where(improved, cur, prev)
            _tail_adam(q, t, gq, gt, mq, vq, mt, vt, lr32)
    ran = (step + 1) if steps else 0
    _record_steps(ran, steps, ran < steps)
    dev = pr.device
    return (torch.from_numpy(best).to(dev), torch.from_numpy(bq).to(dev),
            torch.from_numpy(bt).to(dev))
