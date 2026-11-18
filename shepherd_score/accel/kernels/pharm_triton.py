"""Triton value + SE(3) gradient kernels for the typed/directional pharmacophore overlap.

One CTA per pose (grid=(P,)); a single BLOCK x BLOCK tile covers the feature counts.
"""
import torch
import triton
from .tuning import autotune
import triton.language as tl


# BLOCK is derived from the feature count (next power of two); autotune picks only the
# warp and stage counts.

#  Directionless "color" kernel for vol_color: the same same-type-only typed
#  Gaussian, but isotropic (w=1, no vectors, no weight gradient). Takes the
#  quaternion q (assumed unit; Adam renormalises each step) and emits dV/dq
#  in-register with the shape kernel's tail, so the driver needs no
#  rotation->quaternion projection. A = ref anchors (axis 0), B = fit anchors
#  (axis 1, rotated); dx = A - rot(B), the shape-kernel sign convention.

@autotune(configs=[triton.Config({}, num_warps=_w, num_stages=_s)
                          for _w in (1, 2, 4, 8) for _s in (1, 2, 3, 4)],
                 key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _pharm_color_grad_kernel(
    Aanc_ptr, Banc_ptr,            # ref/fit anchors  (P*N*3) / (P*M*3)
    Atyp_ptr, Btyp_ptr,            # ref/fit type idx (P*N) / (P*M)  int32
    Nreal_ptr, Mreal_ptr,          # (P,) ref / fit real counts
    Q_ptr, T_ptr,                  # (P*4) quaternion (w,x,y,z), (P*3) t
    alphas_ptr, Ks_ptr, cats_ptr,  # (T,) type lookup
    P, N_pad, M_pad,
    O_ptr, dQ_ptr, dT_ptr,         # outputs: (P,), (P*4), (P*3)
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)   # ref count
    realM = tl.load(Mreal_ptr + pid)   # fit count

    Aanc = Aanc_ptr + pid * N_pad * 3
    Banc = Banc_ptr + pid * M_pad * 3
    Atyp = Atyp_ptr + pid * N_pad
    Btyp = Btyp_ptr + pid * M_pad
    Qb = Q_ptr + pid * 4
    Tb = T_ptr + pid * 3

    qr = tl.load(Qb + 0); qi = tl.load(Qb + 1); qj = tl.load(Qb + 2); qk = tl.load(Qb + 3)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)
    two = 2.0; four = 4.0
    r00 = 1 - two * (qj * qj + qk * qk); r01 = two * (qi * qj - qk * qr); r02 = two * (qi * qk + qj * qr)
    r10 = two * (qi * qj + qk * qr); r11 = 1 - two * (qi * qi + qk * qk); r12 = two * (qj * qk - qi * qr)
    r20 = two * (qi * qk - qj * qr); r21 = two * (qj * qk + qi * qr); r22 = 1 - two * (qi * qi + qj * qj)
    inv_ln2 = 1.4426950408889634

    # ---- REF (axis 0, n): fixed anchors + types ----
    on = tl.arange(0, BLOCK)
    mn = on < realN
    ni = tl.where(mn, on, 0)
    ax = tl.load(Aanc + ni * 3 + 0, mask=mn, other=0.0)
    ay = tl.load(Aanc + ni * 3 + 1, mask=mn, other=0.0)
    az = tl.load(Aanc + ni * 3 + 2, mask=mn, other=0.0)
    ntyp = tl.load(Atyp + ni, mask=mn, other=0)

    # ---- FIT (axis 1, m): anchors rotated by (R(q), t); per-type alpha/K/cat ----
    om = tl.arange(0, BLOCK)
    mm = om < realM
    mIdx = tl.where(mm, om, 0)
    bx0 = tl.load(Banc + mIdx * 3 + 0, mask=mm, other=0.0)
    by0 = tl.load(Banc + mIdx * 3 + 1, mask=mm, other=0.0)
    bz0 = tl.load(Banc + mIdx * 3 + 2, mask=mm, other=0.0)
    mtyp = tl.load(Btyp + mIdx, mask=mm, other=0)
    malpha = tl.load(alphas_ptr + mtyp)
    mK = tl.load(Ks_ptr + mtyp)
    mcat = tl.load(cats_ptr + mtyp)
    bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
    by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
    bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz

    # ---- (n,m) tile: same-type isotropic Gaussian ----
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    half_a = malpha[None, :] * 0.5
    g = tl.exp2((-half_a * r2) * inv_ln2) * mK[None, :]
    pair = mn[:, None] & mm[None, :] & (ntyp[:, None] == mtyp[None, :]) & (mcat[None, :] != 3)
    g = tl.where(pair, g, 0.0)
    O = tl.sum(g)

    if NEED_GRAD:
        coeff = malpha[None, :] * g                 # alpha_m * g  (N x M)
        fx = tl.sum(coeff * dx, 0)                  # per-fit force (M,)
        fy = tl.sum(coeff * dy, 0)
        fz = tl.sum(coeff * dz, 0)
        dTx = tl.sum(fx); dTy = tl.sum(fy); dTz = tl.sum(fz)
        # dV/dq via the body-frame fit coords (bx0,by0,bz0); the shape-kernel tail
        wq = qr; xq = qi; yq = qj; zq = qk
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
        dQw = tl.sum(dw); dQx = tl.sum(dxq); dQy = tl.sum(dyq); dQz = tl.sum(dzq)

    tl.store(O_ptr + pid, O)
    if NEED_GRAD:
        dQb = dQ_ptr + pid * 4
        tl.store(dQb + 0, dQw); tl.store(dQb + 1, dQx); tl.store(dQb + 2, dQy); tl.store(dQb + 3, dQz)
        dTb = dT_ptr + pid * 3
        tl.store(dTb + 0, dTx); tl.store(dTb + 1, dTy); tl.store(dTb + 2, dTz)


def pharm_color_score_grad_se3_batch(
    A, B, q, t, ref_types, fit_types, alphas, Ks, cats, *,
    N_real=None, M_real=None, NEED_GRAD=True, BLOCK=None, num_warps=None, num_stages=None,
):
    """Directionless-color value + quaternion-gradient kernel for vol_color.
    A = ref anchors (P,N,3), B = fit anchors (P,M,3), q=(P,4), t=(P,3);
    ref/fit_types (P,N)/(P,M) int. Returns (O, dQ, dT) with dQ = dO/dq, like the shape
    kernel, so no rotation->quaternion projection is needed downstream."""
    P, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev = A.device
    if BLOCK is None:
        BLOCK = triton.next_power_of_2(max(N_pad, M_pad, 16))
    if N_real is None:
        N_real = torch.full((P,), N_pad, device=dev, dtype=torch.int32)
    if M_real is None:
        M_real = torch.full((P,), M_pad, device=dev, dtype=torch.int32)

    # The kernel stores every score and, when NEED_GRAD, every gradient, so no pre-zeroing;
    # without NEED_GRAD the gradient buffers are left unwritten and keep their zeros.
    O = torch.empty(P, device=dev, dtype=A.dtype)
    _g = torch.empty if NEED_GRAD else torch.zeros
    dQ = _g(P, 4, device=dev, dtype=A.dtype)
    dT = _g(P, 3, device=dev, dtype=A.dtype)

    _pharm_color_grad_kernel[(P,)](
        A.contiguous().view(-1), B.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        N_real.to(torch.int32).contiguous(), M_real.to(torch.int32).contiguous(),
        q.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, N_pad, M_pad,
        O, dQ.view(-1), dT.view(-1),
        BLOCK, NEED_GRAD,
    )
    return O, dQ, dT

#  Directional pharm value + quaternion gradient (pharm mode, in-register dQ).
#  The typed/directional Gaussian and weight of score.analytical_gradients._torch
#  .compute_overlap_and_grad_pharm (extended_points=False). Takes q (assumes
#  |q|=1) and emits dV/dq directly: the shape dR/dq tail applied to (positional
#  force, fit anchor) plus (weight force, fit vector).
#  Layout: REF = axis 0 (n), FIT = axis 1 (m); dx = ref - rot(fit).

@autotune(configs=[triton.Config({}, num_warps=_w, num_stages=_s)
                          for _w in (1, 2, 4, 8) for _s in (1, 2, 3, 4)],
                 key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _pharm_grad_dq_kernel(
    Aanc_ptr, Banc_ptr, Avec_ptr, Bvec_ptr,
    Atyp_ptr, Btyp_ptr, Nreal_ptr, Mreal_ptr,
    Q_ptr, T_ptr, alphas_ptr, Ks_ptr, cats_ptr,
    P, N_pad, M_pad,
    O_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)   # ref
    realM = tl.load(Mreal_ptr + pid)   # fit
    Aanc = Aanc_ptr + pid * N_pad * 3; Banc = Banc_ptr + pid * M_pad * 3
    Avec = Avec_ptr + pid * N_pad * 3; Bvec = Bvec_ptr + pid * M_pad * 3
    Atyp = Atyp_ptr + pid * N_pad; Btyp = Btyp_ptr + pid * M_pad
    Qb = Q_ptr + pid * 4; Tb = T_ptr + pid * 3

    qr = tl.load(Qb + 0); qi = tl.load(Qb + 1); qj = tl.load(Qb + 2); qk = tl.load(Qb + 3)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)
    two = 2.0; four = 4.0
    r00 = 1 - two * (qj * qj + qk * qk); r01 = two * (qi * qj - qk * qr); r02 = two * (qi * qk + qj * qr)
    r10 = two * (qi * qj + qk * qr); r11 = 1 - two * (qi * qi + qk * qk); r12 = two * (qj * qk - qi * qr)
    r20 = two * (qi * qk - qj * qr); r21 = two * (qj * qk + qi * qr); r22 = 1 - two * (qi * qi + qj * qj)
    inv_ln2 = 1.4426950408889634

    # REF (axis 0, n): anchors + normalised vectors + types
    on = tl.arange(0, BLOCK); mn = on < realN; ni = tl.where(mn, on, 0)
    ax = tl.load(Aanc + ni * 3 + 0, mask=mn, other=0.0)
    ay = tl.load(Aanc + ni * 3 + 1, mask=mn, other=0.0)
    az = tl.load(Aanc + ni * 3 + 2, mask=mn, other=0.0)
    rvx = tl.load(Avec + ni * 3 + 0, mask=mn, other=0.0)
    rvy = tl.load(Avec + ni * 3 + 1, mask=mn, other=0.0)
    rvz = tl.load(Avec + ni * 3 + 2, mask=mn, other=0.0)
    rnorm = tl.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    rinv = 1.0 / tl.where(rnorm > 1e-12, rnorm, 1e-12)
    rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
    ntyp = tl.load(Atyp + ni, mask=mn, other=0)

    # FIT (axis 1, m): anchors + normalised vectors + per-type alpha/K/cat
    om = tl.arange(0, BLOCK); mm = om < realM; mIdx = tl.where(mm, om, 0)
    bx0 = tl.load(Banc + mIdx * 3 + 0, mask=mm, other=0.0)
    by0 = tl.load(Banc + mIdx * 3 + 1, mask=mm, other=0.0)
    bz0 = tl.load(Banc + mIdx * 3 + 2, mask=mm, other=0.0)
    fvx = tl.load(Bvec + mIdx * 3 + 0, mask=mm, other=0.0)
    fvy = tl.load(Bvec + mIdx * 3 + 1, mask=mm, other=0.0)
    fvz = tl.load(Bvec + mIdx * 3 + 2, mask=mm, other=0.0)
    fnorm = tl.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
    finv = 1.0 / tl.where(fnorm > 1e-12, fnorm, 1e-12)
    fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv
    mtyp = tl.load(Btyp + mIdx, mask=mm, other=0)
    malpha = tl.load(alphas_ptr + mtyp); mK = tl.load(Ks_ptr + mtyp); mcat = tl.load(cats_ptr + mtyp)
    bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
    by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
    bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
    fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
    fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
    fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn

    # (n,m) tile: dx = ref - rot(fit)
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    al_m = malpha[None, :]
    E = tl.exp2((-al_m * 0.5 * r2) * inv_ln2)
    pair = mn[:, None] & mm[None, :] & (ntyp[:, None] == mtyp[None, :]) & (mcat[None, :] != 3)
    E = tl.where(pair, E, 0.0)
    D = fvtx[None, :] * rvxn[:, None] + fvty[None, :] * rvyn[:, None] + fvtz[None, :] * rvzn[:, None]
    Dcl = tl.minimum(tl.maximum(D, 0.0), 1.0)
    cat_m = mcat[None, :]
    w = tl.where(cat_m == 1, (Dcl + 2.0) / 3.0, tl.where(cat_m == 2, (tl.abs(D) + 2.0) / 3.0, 1.0))
    KwE = mK[None, :] * w * E
    O = tl.sum(KwE)

    if NEED_GRAD:
        # positional force per fit m (sum over ref n): coeff = alpha_m * KwE
        cpos = al_m * KwE
        fx = tl.sum(cpos * dx, 0); fy = tl.sum(cpos * dy, 0); fz = tl.sum(cpos * dz, 0)
        dTx = tl.sum(fx); dTy = tl.sum(fy); dTz = tl.sum(fz)
        # weight force per fit m: coeff = (1/3) K_m E c ; wforce = sum_n coeff * ref_vn_n
        c = tl.where(cat_m == 1, tl.where((D > 0.0) & (D < 1.0), 1.0, 0.0),
                     tl.where(cat_m == 2, tl.where(D > 0.0, 1.0, tl.where(D < 0.0, -1.0, 0.0)), 0.0))
        cwt = (1.0 / 3.0) * mK[None, :] * E * c
        wfx = tl.sum(cwt * rvxn[:, None], 0); wfy = tl.sum(cwt * rvyn[:, None], 0); wfz = tl.sum(cwt * rvzn[:, None], 0)
        # dV/dq = shape dR/dq tail with (positional force, fit anchor) + (weight force, fit vector)
        dw = (fx * (-two * qk * by0 + two * qj * bz0) + fy * (two * qk * bx0 - two * qi * bz0) + fz * (-two * qj * bx0 + two * qi * by0)
              + wfx * (-two * qk * fvyn + two * qj * fvzn) + wfy * (two * qk * fvxn - two * qi * fvzn) + wfz * (-two * qj * fvxn + two * qi * fvyn))
        dxq = (fx * (two * qj * by0 + two * qk * bz0) + fy * (two * qj * bx0 - four * qi * by0 - two * qr * bz0) + fz * (two * qk * bx0 + two * qr * by0 - four * qi * bz0)
               + wfx * (two * qj * fvyn + two * qk * fvzn) + wfy * (two * qj * fvxn - four * qi * fvyn - two * qr * fvzn) + wfz * (two * qk * fvxn + two * qr * fvyn - four * qi * fvzn))
        dyq = (fx * (-four * qj * bx0 + two * qi * by0 + two * qr * bz0) + fy * (two * qi * bx0 + two * qk * bz0) + fz * (-two * qr * bx0 + two * qk * by0 - four * qj * bz0)
               + wfx * (-four * qj * fvxn + two * qi * fvyn + two * qr * fvzn) + wfy * (two * qi * fvxn + two * qk * fvzn) + wfz * (-two * qr * fvxn + two * qk * fvyn - four * qj * fvzn))
        dzq = (fx * (-four * qk * bx0 - two * qr * by0 + two * qi * bz0) + fy * (two * qr * bx0 - four * qk * by0 + two * qj * bz0) + fz * (two * qi * bx0 + two * qj * by0)
               + wfx * (-four * qk * fvxn - two * qr * fvyn + two * qi * fvzn) + wfy * (two * qr * fvxn - four * qk * fvyn + two * qj * fvzn) + wfz * (two * qi * fvxn + two * qj * fvyn))
        dw = tl.where(mm, dw, 0.0); dxq = tl.where(mm, dxq, 0.0); dyq = tl.where(mm, dyq, 0.0); dzq = tl.where(mm, dzq, 0.0)
        dQw = tl.sum(dw); dQx = tl.sum(dxq); dQy = tl.sum(dyq); dQz = tl.sum(dzq)

    tl.store(O_ptr + pid, O)
    if NEED_GRAD:
        dQb = dQ_ptr + pid * 4
        tl.store(dQb + 0, dQw); tl.store(dQb + 1, dQx); tl.store(dQb + 2, dQy); tl.store(dQb + 3, dQz)
        dTb = dT_ptr + pid * 3
        tl.store(dTb + 0, dTx); tl.store(dTb + 1, dTy); tl.store(dTb + 2, dTz)


def pharm_grad_dq_se3_batch(
    q, t, ref_types, fit_types, ref_anchors, fit_anchors, ref_vectors, fit_vectors,
    alphas, Ks, cats, *, N_real=None, M_real=None, NEED_GRAD=True, BLOCK=None, num_warps=None, num_stages=None,
):
    """Directional pharm value + quaternion-gradient kernel (in-register dQ). Takes q (not R);
    returns (O, dQ, dT) with dQ = dO/dq, so no R->q projection is needed downstream."""
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev = ref_anchors.device
    if BLOCK is None:
        BLOCK = triton.next_power_of_2(max(N_pad, M_pad, 16))
    if N_real is None:
        N_real = torch.full((P,), N_pad, device=dev, dtype=torch.int32)
    if M_real is None:
        M_real = torch.full((P,), M_pad, device=dev, dtype=torch.int32)
    # The kernel stores every score and, when NEED_GRAD, every gradient, so no pre-zeroing;
    # without NEED_GRAD the gradient buffers are left unwritten and keep their zeros.
    O = torch.empty(P, device=dev, dtype=ref_anchors.dtype)
    _g = torch.empty if NEED_GRAD else torch.zeros
    dQ = _g(P, 4, device=dev, dtype=ref_anchors.dtype)
    dT = _g(P, 3, device=dev, dtype=ref_anchors.dtype)
    _pharm_grad_dq_kernel[(P,)](
        ref_anchors.contiguous().view(-1), fit_anchors.contiguous().view(-1),
        ref_vectors.contiguous().view(-1), fit_vectors.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        N_real.to(torch.int32).contiguous(), M_real.to(torch.int32).contiguous(),
        q.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, N_pad, M_pad, O, dQ.view(-1), dT.view(-1), BLOCK, NEED_GRAD,
    )
    return O, dQ, dT
