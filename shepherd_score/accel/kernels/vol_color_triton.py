"""Fused vol_color kernel: shape (volume) overlap and directionless color overlap in one launch.

The two channels (``overlap_score_grad_se3_batch`` for shape, ``pharm_color_score_grad_se3_batch``
for color) share the R(q) build, the isotropic-Gaussian overlap and the dV/dq tail; shape uses
a fixed alpha and no type gate, color a per-fit-type alpha/K from the lookup tables gated on
same-type and not-dummy. This kernel builds R(q) once and emits (Vs, dQs, dTs, Oc, dQc, dTc)
per pose. Single-tile (one BLOCK covers each cloud); the driver falls back to the two separate
kernels above ``VOL_COLOR_FUSED_MAX_PAD``.
"""
from __future__ import annotations

import math
import torch
import triton
from .tuning import autotune
import triton.language as tl


@autotune(
    configs=[triton.Config({}, num_warps=_w, num_stages=_s)
             for _w in (1, 2, 4) for _s in (1, 2, 3)],
    key=['Ns_pad', 'Ms_pad', 'Na_pad', 'Ma_pad'], cache_results=True)
@triton.jit
def _vol_color_fused_kernel(
    Ash_ptr, Bsh_ptr,                          # shape atoms (P*Ns*3), (P*Ms*3)
    Aanc_ptr, Banc_ptr,                        # color anchors (P*Na*3), (P*Ma*3)
    Atyp_ptr, Btyp_ptr,                        # color types (P*Na), (P*Ma) int32
    Nsh_ptr, Msh_ptr, Nanc_ptr, Manc_ptr,      # real counts (P,)
    Q_ptr, T_ptr,                              # (P*4), (P*3)
    alphas_ptr, Ks_ptr, cats_ptr,              # color type tables (T,)
    P, Ns_pad, Ms_pad, Na_pad, Ma_pad,         # ints
    half_alpha, k_const,                       # shape scalars
    Vs_ptr, dQs_ptr, dTs_ptr,                  # shape outputs (P,), (P*4), (P*3)
    Oc_ptr, dQc_ptr, dTc_ptr,                  # color outputs (P,), (P*4), (P*3)
    BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- quaternion / translation + R(q) (shared by both channels) ----
    Qb = Q_ptr + pid * 4
    Tb = T_ptr + pid * 3
    qr = tl.load(Qb + 0); qi = tl.load(Qb + 1); qj = tl.load(Qb + 2); qk = tl.load(Qb + 3)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)
    two = 2.0; four = 4.0
    r00 = 1 - two * (qj * qj + qk * qk); r01 = two * (qi * qj - qk * qr); r02 = two * (qi * qk + qj * qr)
    r10 = two * (qi * qj + qk * qr); r11 = 1 - two * (qi * qi + qk * qk); r12 = two * (qj * qk - qi * qr)
    r20 = two * (qi * qk - qj * qr); r21 = two * (qj * qk + qi * qr); r22 = 1 - two * (qi * qi + qj * qj)
    inv_ln2 = 1.4426950408889634
    wq = qr; xq = qi; yq = qj; zq = qk

    # ============================ SHAPE channel ============================
    realN = tl.load(Nsh_ptr + pid); realM = tl.load(Msh_ptr + pid)
    Ash = Ash_ptr + pid * Ns_pad * 3
    Bsh = Bsh_ptr + pid * Ms_pad * 3
    on = tl.arange(0, BLOCK); mn = on < realN; ni = tl.where(mn, on, 0)
    ax = tl.load(Ash + ni * 3 + 0, mask=mn, other=0.0)
    ay = tl.load(Ash + ni * 3 + 1, mask=mn, other=0.0)
    az = tl.load(Ash + ni * 3 + 2, mask=mn, other=0.0)
    om = tl.arange(0, BLOCK); mm = om < realM; mi = tl.where(mm, om, 0)
    bx0 = tl.load(Bsh + mi * 3 + 0, mask=mm, other=0.0)
    by0 = tl.load(Bsh + mi * 3 + 1, mask=mm, other=0.0)
    bz0 = tl.load(Bsh + mi * 3 + 2, mask=mm, other=0.0)
    bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
    by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
    bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
    dx = ax[:, None] - bx[None, :]; dy = ay[:, None] - by[None, :]; dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
    g = tl.where(mn[:, None] & mm[None, :], g, 0.0)
    Vs = tl.sum(g)
    if NEED_GRAD:
        coeff = (2.0 * half_alpha) * g
        fx = tl.sum(coeff * dx, 0); fy = tl.sum(coeff * dy, 0); fz = tl.sum(coeff * dz, 0)
        dTsx = tl.sum(fx); dTsy = tl.sum(fy); dTsz = tl.sum(fz)
        dw = (fx * (-two * zq * by0 + two * yq * bz0) +
              fy * (two * zq * bx0 - two * xq * bz0) +
              fz * (-two * yq * bx0 + two * xq * by0))
        dxq = (fx * (two * yq * by0 + two * zq * bz0) +
               fy * (two * yq * bx0 - four * xq * by0 - two * wq * bz0) +
               fz * (two * zq * bx0 + two * wq * by0 - four * xq * bz0))
        dyq = (fx * (-four * yq * bx0 + two * xq * by0 + two * wq * bz0) +
               fy * (two * xq * bx0 + two * zq * bz0) +
               fz * (-two * wq * bx0 + two * zq * by0 - four * yq * bz0))
        dzq = (fx * (-four * zq * bx0 - two * wq * by0 + two * xq * bz0) +
               fy * (two * wq * bx0 - four * zq * by0 + two * yq * bz0) +
               fz * (two * xq * bx0 + two * yq * by0))
        dw = tl.where(mm, dw, 0.0); dxq = tl.where(mm, dxq, 0.0)
        dyq = tl.where(mm, dyq, 0.0); dzq = tl.where(mm, dzq, 0.0)
        dQsw = tl.sum(dw); dQsx = tl.sum(dxq); dQsy = tl.sum(dyq); dQsz = tl.sum(dzq)

    # ============================ COLOR channel ============================
    realNa = tl.load(Nanc_ptr + pid); realMa = tl.load(Manc_ptr + pid)
    Aanc = Aanc_ptr + pid * Na_pad * 3
    Banc = Banc_ptr + pid * Ma_pad * 3
    Atyp = Atyp_ptr + pid * Na_pad
    Btyp = Btyp_ptr + pid * Ma_pad
    cn = tl.arange(0, BLOCK); cmn = cn < realNa; cni = tl.where(cmn, cn, 0)
    cax = tl.load(Aanc + cni * 3 + 0, mask=cmn, other=0.0)
    cay = tl.load(Aanc + cni * 3 + 1, mask=cmn, other=0.0)
    caz = tl.load(Aanc + cni * 3 + 2, mask=cmn, other=0.0)
    ntyp = tl.load(Atyp + cni, mask=cmn, other=0)
    cm = tl.arange(0, BLOCK); cmm = cm < realMa; cmi = tl.where(cmm, cm, 0)
    cbx0 = tl.load(Banc + cmi * 3 + 0, mask=cmm, other=0.0)
    cby0 = tl.load(Banc + cmi * 3 + 1, mask=cmm, other=0.0)
    cbz0 = tl.load(Banc + cmi * 3 + 2, mask=cmm, other=0.0)
    mtyp = tl.load(Btyp + cmi, mask=cmm, other=0)
    malpha = tl.load(alphas_ptr + mtyp); mK = tl.load(Ks_ptr + mtyp); mcat = tl.load(cats_ptr + mtyp)
    cbx = r00 * cbx0 + r01 * cby0 + r02 * cbz0 + tx
    cby = r10 * cbx0 + r11 * cby0 + r12 * cbz0 + ty
    cbz = r20 * cbx0 + r21 * cby0 + r22 * cbz0 + tz
    cdx = cax[:, None] - cbx[None, :]; cdy = cay[:, None] - cby[None, :]; cdz = caz[:, None] - cbz[None, :]
    cr2 = cdx * cdx + cdy * cdy + cdz * cdz
    half_a = malpha[None, :] * 0.5
    gc = tl.exp2((-half_a * cr2) * inv_ln2) * mK[None, :]
    cpair = cmn[:, None] & cmm[None, :] & (ntyp[:, None] == mtyp[None, :]) & (mcat[None, :] != 3)
    gc = tl.where(cpair, gc, 0.0)
    Oc = tl.sum(gc)
    if NEED_GRAD:
        ccoeff = malpha[None, :] * gc
        cfx = tl.sum(ccoeff * cdx, 0); cfy = tl.sum(ccoeff * cdy, 0); cfz = tl.sum(ccoeff * cdz, 0)
        dTcx = tl.sum(cfx); dTcy = tl.sum(cfy); dTcz = tl.sum(cfz)
        cdw = (cfx * (-two * zq * cby0 + two * yq * cbz0) +
               cfy * (two * zq * cbx0 - two * xq * cbz0) +
               cfz * (-two * yq * cbx0 + two * xq * cby0))
        cdxq = (cfx * (two * yq * cby0 + two * zq * cbz0) +
                cfy * (two * yq * cbx0 - four * xq * cby0 - two * wq * cbz0) +
                cfz * (two * zq * cbx0 + two * wq * cby0 - four * xq * cbz0))
        cdyq = (cfx * (-four * yq * cbx0 + two * xq * cby0 + two * wq * cbz0) +
                cfy * (two * xq * cbx0 + two * zq * cbz0) +
                cfz * (-two * wq * cbx0 + two * zq * cby0 - four * yq * cbz0))
        cdzq = (cfx * (-four * zq * cbx0 - two * wq * cby0 + two * xq * cbz0) +
                cfy * (two * wq * cbx0 - four * zq * cby0 + two * yq * cbz0) +
                cfz * (two * xq * cbx0 + two * yq * cby0))
        cdw = tl.where(cmm, cdw, 0.0); cdxq = tl.where(cmm, cdxq, 0.0)
        cdyq = tl.where(cmm, cdyq, 0.0); cdzq = tl.where(cmm, cdzq, 0.0)
        dQcw = tl.sum(cdw); dQcx = tl.sum(cdxq); dQcy = tl.sum(cdyq); dQcz = tl.sum(cdzq)

    # ---------------------------- stores ----------------------------
    tl.store(Vs_ptr + pid, Vs)
    tl.store(Oc_ptr + pid, Oc)
    if NEED_GRAD:
        dQsb = dQs_ptr + pid * 4
        tl.store(dQsb + 0, dQsw); tl.store(dQsb + 1, dQsx); tl.store(dQsb + 2, dQsy); tl.store(dQsb + 3, dQsz)
        dTsb = dTs_ptr + pid * 3
        tl.store(dTsb + 0, dTsx); tl.store(dTsb + 1, dTsy); tl.store(dTsb + 2, dTsz)
        dQcb = dQc_ptr + pid * 4
        tl.store(dQcb + 0, dQcw); tl.store(dQcb + 1, dQcx); tl.store(dQcb + 2, dQcy); tl.store(dQcb + 3, dQcz)
        dTcb = dTc_ptr + pid * 3
        tl.store(dTcb + 0, dTcx); tl.store(dTcb + 1, dTcy); tl.store(dTcb + 2, dTcz)


# Largest padded cloud edge for which the fused kernel is used: it holds both channels'
# BLOCKxBLOCK accumulators in registers, so larger tiles lose occupancy to the separate kernels.
VOL_COLOR_FUSED_MAX_PAD = 32


def vol_color_score_grad_se3_batch(
    centers_1, centers_2, anchors_1, anchors_2, q, t, ref_types, fit_types,
    alphas, Ks, cats, *, alpha=0.81,
    N_real_cent=None, M_real_cent=None, N_real_anc=None, M_real_anc=None, NEED_GRAD=True,
):
    """One fused launch computing both channels for vol_color. Returns
    (VAB_shape, dQ_shape, dT_shape, O_color, dQ_color, dT_color), the same six tensors the
    two separate kernels produce, in the same conventions (dQ = dO/dq).

    Shapes: centers_1/2 (P,Ns/Ms,3) shape atoms; anchors_1/2 (P,Na/Ma,3) color anchors;
    ref/fit_types (P,Na)/(P,Ma); q (P,4); t (P,3). The caller must ensure max pad <=
    VOL_COLOR_FUSED_MAX_PAD (else use the separate kernels)."""
    P, Ns_pad, _ = centers_1.shape
    _, Ms_pad, _ = centers_2.shape
    _, Na_pad, _ = anchors_1.shape
    _, Ma_pad, _ = anchors_2.shape
    dev = centers_1.device; dtype = centers_1.dtype
    BLOCK = triton.next_power_of_2(max(Ns_pad, Ms_pad, Na_pad, Ma_pad, 16))

    def _cnt(x, n, pad):
        return (torch.full((P,), pad, device=dev, dtype=torch.int32) if x is None
                else x.to(torch.int32).contiguous())
    Nsh = _cnt(N_real_cent, P, Ns_pad); Msh = _cnt(M_real_cent, P, Ms_pad)
    Nanc = _cnt(N_real_anc, P, Na_pad); Manc = _cnt(M_real_anc, P, Ma_pad)

    half_alpha = 0.5 * alpha
    k_const = math.pi ** 1.5 / ((2.0 * alpha) ** 1.5)

    # The kernel stores every score and, when NEED_GRAD, every gradient, so no pre-zeroing;
    # without NEED_GRAD the gradient buffers are left unwritten and keep their zeros.
    Vs = torch.empty(P, device=dev, dtype=dtype)
    Oc = torch.empty(P, device=dev, dtype=dtype)
    _g = torch.empty if NEED_GRAD else torch.zeros
    dQs = _g(P, 4, device=dev, dtype=dtype); dTs = _g(P, 3, device=dev, dtype=dtype)
    dQc = _g(P, 4, device=dev, dtype=dtype); dTc = _g(P, 3, device=dev, dtype=dtype)

    _vol_color_fused_kernel[(P,)](
        centers_1.contiguous().view(-1), centers_2.contiguous().view(-1),
        anchors_1.contiguous().view(-1), anchors_2.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        Nsh, Msh, Nanc, Manc,
        q.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, Ns_pad, Ms_pad, Na_pad, Ma_pad,
        half_alpha, k_const,
        Vs, dQs.view(-1), dTs.view(-1),
        Oc, dQc.view(-1), dTc.view(-1),
        BLOCK=BLOCK, NEED_GRAD=NEED_GRAD,
    )
    return Vs, dQs, dTs, Oc, dQc, dTc
