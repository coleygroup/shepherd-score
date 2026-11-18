"""Fused forward+backward Gaussian-Tanimoto overlap kernels in Triton.

Provides the value + SE(3) gradient (dV/dq, dV/dt) kernel used by the batched
coarse-to-fine aligners on CUDA tensors, plus the fused Adam quaternion/translation
update. The numba CPU twins live in :mod:`~shepherd_score.accel.kernels.cpu`.
"""
from __future__ import annotations

import math
import triton
from .tuning import autotune
import triton.language as tl
import torch


# Autotuned per (N_pad, M_pad) on the actual device. Candidates are small tiles (BLOCK <= 64,
# <= 4 warps): the kernel runs one CTA per pair, so occupancy comes from the pair count.
_OVERLAP_CONFIGS = [
    triton.Config({'BLOCK': _b}, num_warps=_w, num_stages=_s)
    for _b in (16, 32, 64) for _w in (1, 2, 4) for _s in (1, 2, 3, 4)
]


# Shared SE(3) device functions. Triton inlines a @triton.jit callee, so factoring the
# quaternion->rotation build and the force->quaternion-gradient tail here costs nothing.
@triton.jit
def _quat_to_rotmat(qr, qi, qj, qk):
    """Rotation matrix (row-major r00..r22) from a unit quaternion (w,x,y,z). Seeds are
    normalised and the Adam step renormalises q after every update, so q is unit here."""
    two = 2.0
    r00 = 1 - two*(qj*qj + qk*qk); r01 = two*(qi*qj - qk*qr); r02 = two*(qi*qk + qj*qr)
    r10 = two*(qi*qj + qk*qr);     r11 = 1 - two*(qi*qi + qk*qk); r12 = two*(qj*qk - qi*qr)
    r20 = two*(qi*qk - qj*qr);     r21 = two*(qj*qk + qi*qr);     r22 = 1 - two*(qi*qi + qj*qj)
    return r00, r01, r02, r10, r11, r12, r20, r21, r22


@triton.jit
def _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk):
    """Per-point quaternion-gradient contributions (dw,dx,dy,dz) from the translation-force
    components (fx,fy,fz) and the body-frame fit coords (bx0,by0,bz0). Analytic d(R b)/dq
    contracted with the force; caller masks padding and reduces. Works elementwise, so fx..bz0
    may be scalars or per-lane vectors."""
    two = 2.0; four = 4.0
    dw = (fx * (-two*qk*by0 + two*qj*bz0)
          + fy * ( two*qk*bx0 - two*qi*bz0)
          + fz * (-two*qj*bx0 + two*qi*by0))
    dxq = (fx * ( two*qj*by0 + two*qk*bz0)
           + fy * ( two*qj*bx0 - four*qi*by0 - two*qr*bz0)
           + fz * ( two*qk*bx0 + two*qr*by0 - four*qi*bz0))
    dyq = (fx * (-four*qj*bx0 + two*qi*by0 + two*qr*bz0)
           + fy * ( two*qi*bx0                 + two*qk*bz0)
           + fz * (-two*qr*bx0 + two*qk*by0 - four*qj*bz0))
    dzq = (fx * (-four*qk*bx0 - two*qr*by0 + two*qi*bz0)
           + fy * ( two*qr*bx0 - four*qk*by0 + two*qj*bz0)
           + fz * ( two*qi*bx0 + two*qj*by0))
    return dw, dxq, dyq, dzq


# cache_results persists the chosen config per (N_pad, M_pad), so the autotune sweep runs
# once per machine rather than once per process.
@autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _gauss_overlap_se3_tiled(
    A_ptr, B_ptr,                 # flat (B * N_pad * 3), (B * M_pad * 3)
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B,)
    BATCH, M_pad, N_pad,          # ints
    half_alpha, k_const,          # scalars
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,), dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # tile edge (chosen by autotune)
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr           # poses per molecule; 1 == one molecule per CTA (legacy)
):
    pid = tl.program_id(0)
    # One CTA per pose; coordinates belong to a molecule. With SEEDS > 1 the caller passes the
    # molecule blocks unreplicated and the SEEDS consecutive CTAs of a molecule share one copy.
    mol = pid // SEEDS
    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)

    # base pointers: coords by molecule, pose state by CTA
    A_ptr  = A_ptr  + mol * N_pad * 3
    B_ptr  = B_ptr  + mol * M_pad * 3
    Q_ptr  = Q_ptr  + pid * 4
    T_ptr  = T_ptr  + pid * 3
    dQ_ptr = dQ_ptr + pid * 4
    dT_ptr = dT_ptr + pid * 3
    S_ptr  = S_ptr  + pid

    qr = tl.load(Q_ptr + 0); qi = tl.load(Q_ptr + 1)
    qj = tl.load(Q_ptr + 2); qk = tl.load(Q_ptr + 3)
    tx = tl.load(T_ptr + 0); ty = tl.load(T_ptr + 1); tz = tl.load(T_ptr + 2)

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

    Vab_acc = 0.0
    dTx = 0.0; dTy = 0.0; dTz = 0.0
    dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0

    inv_ln2 = 1.4426950408889634

    # outer loop over A tiles, inner loop over B tiles
    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN

        a_idx = tl.where(mask_n, offs_n, 0)
        ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
        ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
        az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM

            b_idx = tl.where(mask_m, offs_m, 0)
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            # rotate + translate B tile
            bx = r00*bx0 + r01*by0 + r02*bz0 + tx
            by = r10*bx0 + r11*by0 + r12*bz0 + ty
            bz = r20*bx0 + r21*by0 + r22*bz0 + tz

            # broadcast differences (BLOCK x BLOCK)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            pair_mask = mask_n[:, None] & mask_m[None, :]
            g = tl.where(pair_mask, g, 0.0)

            Vab_acc += tl.sum(g)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                # forces: sum over i for each j (axis 0)
                fx = tl.sum(coeff * dx, 0)
                fy = tl.sum(coeff * dy, 0)
                fz = tl.sum(coeff * dz, 0)

                dTx += tl.sum(fx)
                dTy += tl.sum(fy)
                dTz += tl.sum(fz)

                # quaternion grads from the body-frame coords (bx0, by0, bz0)
                dw, dxq, dyq, dzq = _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk)

                # mask padding lanes
                dw  = tl.where(mask_m, dw,  0.0)
                dxq = tl.where(mask_m, dxq, 0.0)
                dyq = tl.where(mask_m, dyq, 0.0)
                dzq = tl.where(mask_m, dzq, 0.0)

                dQw += tl.sum(dw)
                dQx += tl.sum(dxq)
                dQy += tl.sum(dyq)
                dQz += tl.sum(dzq)

    # single final write per CTA; no atomics needed
    tl.store(S_ptr, Vab_acc)

    if NEED_GRAD:
        tl.store(dT_ptr + 0, dTx)
        tl.store(dT_ptr + 1, dTy)
        tl.store(dT_ptr + 2, dTz)
        tl.store(dQ_ptr + 0, dQw)
        tl.store(dQ_ptr + 1, dQx)
        tl.store(dQ_ptr + 2, dQy)
        tl.store(dQ_ptr + 3, dQz)


# Multi-pose variant: POSES poses of one molecule per CTA, so the tile loads, the
# quaternion->rotmat build and the loop overhead amortise over POSES poses.
@autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'])
@triton.jit
def _gauss_overlap_se3_multipose(
    A_ptr, B_ptr,
    Q_ptr, T_ptr,
    Nreal_ptr, Mreal_ptr,
    BATCH, M_pad, N_pad,
    half_alpha, k_const,
    S_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr,
    POSES: tl.constexpr,
    POSES_PAD: tl.constexpr,      # next power of two >= POSES (tl.arange extent)
):
    """POSES poses of one molecule per CTA, vectorised over the pose axis.

    The A and B tiles load once and the pose axis broadcasts over them, so every tile load,
    the quaternion->rotmat build and the loop overhead amortise over POSES poses. Poses are a
    tensor dimension (Triton has no ``__setitem__``), so POSES is a ``tl.arange`` extent padded
    to a power of two, and it must divide SEEDS so a CTA never straddles two molecules. The
    per-pose accumulation order matches the single-pose kernel.
    """
    pid = tl.program_id(0)
    base = pid * POSES
    mol = base // SEEDS                    # SEEDS % POSES == 0 => all POSES share this molecule

    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)
    A_ptr = A_ptr + mol * N_pad * 3
    B_ptr = B_ptr + mol * M_pad * 3

    # pose state: (POSES,) vectors, built once. tl.arange needs a power-of-two extent, so run
    # POSES_PAD lanes and mask the tail.
    p_off = tl.arange(0, POSES_PAD)
    mask_p = p_off < POSES
    qo = (base + p_off) * 4
    to = (base + p_off) * 3
    qr = tl.load(Q_ptr + qo + 0, mask=mask_p, other=0.0)
    qi = tl.load(Q_ptr + qo + 1, mask=mask_p, other=0.0)
    qj = tl.load(Q_ptr + qo + 2, mask=mask_p, other=0.0)
    qk = tl.load(Q_ptr + qo + 3, mask=mask_p, other=0.0)
    tx = tl.load(T_ptr + to + 0, mask=mask_p, other=0.0)
    ty = tl.load(T_ptr + to + 1, mask=mask_p, other=0.0)
    tz = tl.load(T_ptr + to + 2, mask=mask_p, other=0.0)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

    Vab_acc = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTx = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTy = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTz = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQw = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQx = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQy = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQz = tl.zeros([POSES_PAD], dtype=tl.float32)

    inv_ln2 = 1.4426950408889634

    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN
        a_idx = tl.where(mask_n, offs_n, 0)
        ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
        ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
        az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM
            b_idx = tl.where(mask_m, offs_m, 0)
            # one load, reused by every pose via the broadcast below
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            # (POSES, BLOCK): each pose's rotation applied to the shared body-frame tile
            bx = r00[:, None]*bx0[None, :] + r01[:, None]*by0[None, :] + r02[:, None]*bz0[None, :] + tx[:, None]
            by = r10[:, None]*bx0[None, :] + r11[:, None]*by0[None, :] + r12[:, None]*bz0[None, :] + ty[:, None]
            bz = r20[:, None]*bx0[None, :] + r21[:, None]*by0[None, :] + r22[:, None]*bz0[None, :] + tz[:, None]

            # (POSES, BLOCK_n, BLOCK_m)
            dx = ax[None, :, None] - bx[:, None, :]
            dy = ay[None, :, None] - by[:, None, :]
            dz = az[None, :, None] - bz[:, None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            pair_mask = mask_n[None, :, None] & mask_m[None, None, :]
            g = tl.where(pair_mask, g, 0.0)

            # Reduce the (n, m) plane in one pass over a flattened row so the summation tree
            # matches the single-pose kernel's tl.sum over a flat (BLOCK, BLOCK); a two-stage
            # tl.sum(tl.sum(g, 2), 1) would round differently.
            Vab_acc += tl.sum(tl.reshape(g, [POSES_PAD, BLOCK * BLOCK]), 1)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                fx = tl.sum(coeff * dx, 1)            # (POSES, BLOCK_m): sum over i
                fy = tl.sum(coeff * dy, 1)
                fz = tl.sum(coeff * dz, 1)

                dTx += tl.sum(fx, 1)
                dTy += tl.sum(fy, 1)
                dTz += tl.sum(fz, 1)

                dw, dxq, dyq, dzq = _quat_grad_tail(
                    fx, fy, fz,
                    bx0[None, :], by0[None, :], bz0[None, :],
                    qr[:, None], qi[:, None], qj[:, None], qk[:, None])
                mm = mask_m[None, :]
                dQw += tl.sum(tl.where(mm, dw, 0.0), 1)
                dQx += tl.sum(tl.where(mm, dxq, 0.0), 1)
                dQy += tl.sum(tl.where(mm, dyq, 0.0), 1)
                dQz += tl.sum(tl.where(mm, dzq, 0.0), 1)

    tl.store(S_ptr + base + p_off, Vab_acc, mask=mask_p)
    if NEED_GRAD:
        tl.store(dT_ptr + to + 0, dTx, mask=mask_p)
        tl.store(dT_ptr + to + 1, dTy, mask=mask_p)
        tl.store(dT_ptr + to + 2, dTz, mask=mask_p)
        tl.store(dQ_ptr + qo + 0, dQw, mask=mask_p)
        tl.store(dQ_ptr + qo + 1, dQx, mask=mask_p)
        tl.store(dQ_ptr + qo + 2, dQy, mask=mask_p)
        tl.store(dQ_ptr + qo + 3, dQz, mask=mask_p)


def overlap_score_grad_se3_batch(
    A, B, q, t, *,
    alpha: float = 0.81,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    seeds_per_mol: int = 1,
    poses_per_cta: int = 1,
):
    """Gaussian overlap value + SE(3) gradient; one CTA per pose, tile loops over A and B.

    Shapes with ``seeds_per_mol == 1``: A (K, N_pad, 3), B (K, M_pad, 3), q (K, 4), t (K, 3).
    With ``seeds_per_mol = S > 1`` the coordinate blocks are unreplicated: A (K // S, N_pad, 3),
    N_real / M_real (K // S,), q (K, 4), t (K, 3); CTA ``i`` reads molecule ``i // S``.
    Returns (V (K,), dQ (K, 4), dT (K, 3)).
    """
    K = q.shape[0]                       # POSES == CTAs
    S = int(seeds_per_mol)
    n_mol, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    if S < 1 or K % S != 0 or n_mol != K // S:
        raise ValueError(
            f"seeds_per_mol={S} inconsistent: q has {K} poses, A has {n_mol} molecules "
            f"(expected {K // S if S else 0})")
    device = A.device
    dtype  = A.dtype

    if N_real is None:
        N_real = torch.full((n_mol,), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((n_mol,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    # Both kernels store every element of out_S and, when NEED_GRAD, of out_dQ / out_dT (the
    # multi-pose mask only hides tl.arange padding lanes), so no pre-zeroing; without NEED_GRAD
    # the gradient buffers are left unwritten and keep their zeros.
    out_S  = torch.empty(K, device=device, dtype=dtype)
    if NEED_GRAD:
        out_dQ = torch.empty_like(q)
        out_dT = torch.empty_like(t)
    else:
        out_dQ = torch.zeros_like(q)
        out_dT = torch.zeros_like(t)

    POSES = int(poses_per_cta)
    if POSES > 1:
        # POSES poses of one molecule per CTA: every pose in a CTA must share a molecule, so
        # POSES must divide SEEDS.
        if S <= 1 or S % POSES != 0 or K % POSES != 0:
            raise ValueError(
                f"poses_per_cta={POSES} needs the deduped layout and SEEDS % POSES == 0 "
                f"(got seeds_per_mol={S}, K={K})")
        POSES_PAD = 1 << (POSES - 1).bit_length()      # tl.arange extent; tail lanes masked
        _gauss_overlap_se3_multipose[(K // POSES,)](
            A.contiguous().view(-1), B.contiguous().view(-1),
            q.contiguous().view(-1), t.contiguous().view(-1),
            N_real.contiguous(), M_real.contiguous(),
            K, M_pad, N_pad, half_alpha, k_const,
            out_S, out_dQ.view(-1), out_dT.view(-1),
            NEED_GRAD=NEED_GRAD, SEEDS=S, POSES=POSES, POSES_PAD=POSES_PAD,
        )
        return out_S, out_dQ, out_dT

    grid = (K,)    # 1-D launch: one CTA per alignment

    _gauss_overlap_se3_tiled[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad,
        half_alpha, k_const,
        out_S, out_dQ.view(-1), out_dT.view(-1),
        NEED_GRAD=NEED_GRAD,
        SEEDS=S,
    )
    return out_S, out_dQ, out_dT


# Fused Adam update for (q, t); one CTA handles up to BLOCK poses.
@triton.jit
def _adam_qt(
    Q_ptr, T_ptr,
    dQ_ptr, dT_ptr,
    Mq_ptr, Vq_ptr,
    Mt_ptr, Vt_ptr,
    K,
    lr: tl.constexpr,
    beta1: tl.constexpr = 0.9,
    beta2: tl.constexpr = 0.999,
    eps:   tl.constexpr = 1e-8,
    BLOCK: tl.constexpr = 256,     # threads per CTA (must divide 1024)
    PROJECT: tl.constexpr = False, # if True, tangent-project dQ: dQ -= q*(dQ.q)
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < K               # lanes beyond K are masked-out NOPs

    # ---------------- flat loads (4 components) ---------------------------
    # q           – parameters
    q0 = tl.load(Q_ptr + offs*4 + 0, mask=mask)
    q1 = tl.load(Q_ptr + offs*4 + 1, mask=mask)
    q2 = tl.load(Q_ptr + offs*4 + 2, mask=mask)
    q3 = tl.load(Q_ptr + offs*4 + 3, mask=mask)

    # dq          – gradients
    dq0 = tl.load(dQ_ptr + offs*4 + 0, mask=mask)
    dq1 = tl.load(dQ_ptr + offs*4 + 1, mask=mask)
    dq2 = tl.load(dQ_ptr + offs*4 + 2, mask=mask)
    dq3 = tl.load(dQ_ptr + offs*4 + 3, mask=mask)

    if PROJECT:                      # tangent-space projection: dQ -= q*(dQ.q)
        radial = dq0*q0 + dq1*q1 + dq2*q2 + dq3*q3
        dq0 = dq0 - q0 * radial
        dq1 = dq1 - q1 * radial
        dq2 = dq2 - q2 * radial
        dq3 = dq3 - q3 * radial

    # first-moment & second-moment for q
    mq0 = tl.load(Mq_ptr + offs*4 + 0, mask=mask)
    mq1 = tl.load(Mq_ptr + offs*4 + 1, mask=mask)
    mq2 = tl.load(Mq_ptr + offs*4 + 2, mask=mask)
    mq3 = tl.load(Mq_ptr + offs*4 + 3, mask=mask)

    vq0 = tl.load(Vq_ptr + offs*4 + 0, mask=mask)
    vq1 = tl.load(Vq_ptr + offs*4 + 1, mask=mask)
    vq2 = tl.load(Vq_ptr + offs*4 + 2, mask=mask)
    vq3 = tl.load(Vq_ptr + offs*4 + 3, mask=mask)

    # ---------------- flat loads (3 components) ---------------------------
    t0 = tl.load(T_ptr + offs*3 + 0, mask=mask)
    t1 = tl.load(T_ptr + offs*3 + 1, mask=mask)
    t2 = tl.load(T_ptr + offs*3 + 2, mask=mask)

    dt0 = tl.load(dT_ptr + offs*3 + 0, mask=mask)
    dt1 = tl.load(dT_ptr + offs*3 + 1, mask=mask)
    dt2 = tl.load(dT_ptr + offs*3 + 2, mask=mask)

    mt0 = tl.load(Mt_ptr + offs*3 + 0, mask=mask)
    mt1 = tl.load(Mt_ptr + offs*3 + 1, mask=mask)
    mt2 = tl.load(Mt_ptr + offs*3 + 2, mask=mask)

    vt0 = tl.load(Vt_ptr + offs*3 + 0, mask=mask)
    vt1 = tl.load(Vt_ptr + offs*3 + 1, mask=mask)
    vt2 = tl.load(Vt_ptr + offs*3 + 2, mask=mask)

    # ---------------- Adam update (each component) ------------------------
    mq0 = beta1*mq0 + (1-beta1)*dq0;  vq0 = beta2*vq0 + (1-beta2)*dq0*dq0
    mq1 = beta1*mq1 + (1-beta1)*dq1;  vq1 = beta2*vq1 + (1-beta2)*dq1*dq1
    mq2 = beta1*mq2 + (1-beta1)*dq2;  vq2 = beta2*vq2 + (1-beta2)*dq2*dq2
    mq3 = beta1*mq3 + (1-beta1)*dq3;  vq3 = beta2*vq3 + (1-beta2)*dq3*dq3

    q0 = q0 - lr * mq0 / tl.sqrt(vq0 + eps)
    q1 = q1 - lr * mq1 / tl.sqrt(vq1 + eps)
    q2 = q2 - lr * mq2 / tl.sqrt(vq2 + eps)
    q3 = q3 - lr * mq3 / tl.sqrt(vq3 + eps)

    mt0 = beta1*mt0 + (1-beta1)*dt0; vt0 = beta2*vt0 + (1-beta2)*dt0*dt0
    mt1 = beta1*mt1 + (1-beta1)*dt1; vt1 = beta2*vt1 + (1-beta2)*dt1*dt1
    mt2 = beta1*mt2 + (1-beta1)*dt2; vt2 = beta2*vt2 + (1-beta2)*dt2*dt2

    t0 = t0 - lr * mt0 / tl.sqrt(vt0 + eps)
    t1 = t1 - lr * mt1 / tl.sqrt(vt1 + eps)
    t2 = t2 - lr * mt2 / tl.sqrt(vt2 + eps)

    # ---------------- renormalise quaternion -----------------------------
    inv_norm = 1.0 / tl.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    q0 *= inv_norm; q1 *= inv_norm; q2 *= inv_norm; q3 *= inv_norm

    # ---------------- stores ---------------------------------------------
    tl.store(Q_ptr + offs*4 + 0, q0, mask=mask)
    tl.store(Q_ptr + offs*4 + 1, q1, mask=mask)
    tl.store(Q_ptr + offs*4 + 2, q2, mask=mask)
    tl.store(Q_ptr + offs*4 + 3, q3, mask=mask)

    tl.store(Mq_ptr + offs*4 + 0, mq0, mask=mask)
    tl.store(Mq_ptr + offs*4 + 1, mq1, mask=mask)
    tl.store(Mq_ptr + offs*4 + 2, mq2, mask=mask)
    tl.store(Mq_ptr + offs*4 + 3, mq3, mask=mask)

    tl.store(Vq_ptr + offs*4 + 0, vq0, mask=mask)
    tl.store(Vq_ptr + offs*4 + 1, vq1, mask=mask)
    tl.store(Vq_ptr + offs*4 + 2, vq2, mask=mask)
    tl.store(Vq_ptr + offs*4 + 3, vq3, mask=mask)

    tl.store(T_ptr + offs*3 + 0, t0, mask=mask)
    tl.store(T_ptr + offs*3 + 1, t1, mask=mask)
    tl.store(T_ptr + offs*3 + 2, t2, mask=mask)

    tl.store(Mt_ptr + offs*3 + 0, mt0, mask=mask)
    tl.store(Mt_ptr + offs*3 + 1, mt1, mask=mask)
    tl.store(Mt_ptr + offs*3 + 2, mt2, mask=mask)

    tl.store(Vt_ptr + offs*3 + 0, vt0, mask=mask)
    tl.store(Vt_ptr + offs*3 + 1, vt1, mask=mask)
    tl.store(Vt_ptr + offs*3 + 2, vt2, mask=mask)

def fused_adam_qt(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr):
    _warmup_adam_qt()                       # lazy one-time PTX build (must not run at import)
    K = q.shape[0]
    grid = (triton.cdiv(K, 256),)

    _adam_qt[grid](
        q.contiguous().view(-1),  t.contiguous().view(-1),
        dQ.contiguous().view(-1), dT.contiguous().view(-1),
        m_q.view(-1), v_q.view(-1), m_t.view(-1), v_t.view(-1),
        K, lr=lr
    )


def fused_adam_qt_with_tangent_proj(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr):
    """Fused Adam update with in-kernel tangent-space projection of the raw quaternion gradient.

    Args:
        q, t: quaternion (K,4) and translation (K,3) parameters (updated in-place)
        dQ: raw quaternion gradients (K,4), not yet tangent-projected
        dT: translation gradients (K,3)
        m_q, v_q, m_t, v_t: Adam moment tensors (updated in-place)
        lr: learning rate
    """
    K = q.shape[0]
    grid = (triton.cdiv(K, 256),)

    _adam_qt[grid](
        q.contiguous().view(-1),  t.contiguous().view(-1),
        dQ.contiguous().view(-1), dT.contiguous().view(-1),
        m_q.view(-1), v_q.view(-1), m_t.view(-1), v_t.view(-1),
        K, lr=lr, PROJECT=True
    )


# One-time _adam_qt warm-up so the first real call does not pay the PTX build. Must stay lazy:
# importing this module must not initialise CUDA, because the fork-based multi-GPU pool
# (shepherd_score.accel.multi_gpu) can only fork its workers before the parent touches CUDA.
_ADAM_QT_WARMED = False


def _warmup_adam_qt():
    global _ADAM_QT_WARMED
    if _ADAM_QT_WARMED or not torch.cuda.is_available():
        return
    _ADAM_QT_WARMED = True
    dummy = torch.zeros(512, 4, device="cuda", dtype=torch.float32)
    _adam_qt[(2,)](                         # 512 // 256 = 2 blocks
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        512, lr=0.001,                    # K=512, any lr
    )
    torch.cuda.synchronize()

# Batched self-overlap VPP(P,P) for a padded tensor via the identity pose.
@torch.no_grad()
def _batch_self_overlap(P_pad: torch.Tensor,
                              N_real: torch.Tensor,
                              alpha: float = 0.81) -> torch.Tensor:
    K, N_pad, _ = P_pad.shape
    q_id = torch.tensor([1.,0.,0.,0.], device=P_pad.device, dtype=P_pad.dtype).expand(K,4)
    t_0  = torch.zeros(K,3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_se3_batch(
        P_pad, P_pad, q_id, t_0,
        alpha=alpha,
        N_real=N_real, M_real=N_real,
        NEED_GRAD=False)
    return V

