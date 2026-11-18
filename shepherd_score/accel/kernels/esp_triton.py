"""Fused (forward + backward) ESP-weighted Gaussian-Tanimoto kernels in Triton.

Extends the base volumetric kernel (:mod:`~shepherd_score.accel.kernels.shape_triton`)
with electrostatic-potential weighting.
"""
from __future__ import annotations

import math
import triton
from .tuning import autotune
import triton.language as tl
import torch

from .shape_triton import _OVERLAP_CONFIGS, _quat_to_rotmat, _quat_grad_tail
from ...score.constants import COULOMB_SCALING, LAM_SCALING


# Autotuned per (N_pad, M_pad); cache_results persists the choice so the sweep runs once
# per machine.
@autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _gauss_overlap_esp_se3_tiled(
    A_ptr, B_ptr,                 # coordinates: flat (B * N_pad * 3), (B * M_pad * 3)
    CA_ptr, CB_ptr,               # charges: flat (B * N_pad), (B * M_pad)
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B,)
    BATCH, M_pad, N_pad,          # ints
    half_alpha, k_const,          # Gaussian parameters
    inv_lam,                      # 1/lam for ESP weighting
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,), dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # single tile edge (e.g. 64)
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr = 1       # poses per molecule; 1 == one molecule per CTA (legacy)
):
    """ESP-weighted Gaussian overlap with SE(3) gradients, one CTA per pose.

    V = sum_ij k_const * exp(-alpha/2 * R_ij^2) * exp(-(C_i - C_j)^2 / lam), with R_ij the
    spatial distance and C_i / C_j the ESP values at points i / j. The charge weight does not
    depend on the pose, so the gradient is the shape gradient scaled by it.
    """
    pid = tl.program_id(0)
    # Coordinates and charges belong to a molecule; pose state is per CTA. SEEDS == 1 gives
    # mol == pid, the one-molecule-per-CTA layout.
    mol = pid // SEEDS
    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)

    # base pointers: coords/charges by molecule, pose state by CTA
    A_ptr  = A_ptr  + mol * N_pad * 3
    B_ptr  = B_ptr  + mol * M_pad * 3
    CA_ptr = CA_ptr + mol * N_pad
    CB_ptr = CB_ptr + mol * M_pad
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

        ca = tl.load(CA_ptr + a_idx, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM

            b_idx = tl.where(mask_m, offs_m, 0)
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            cb = tl.load(CB_ptr + b_idx, mask=mask_m, other=0.0)

            # rotate + translate B tile
            bx = r00*bx0 + r01*by0 + r02*bz0 + tx
            by = r10*bx0 + r11*by0 + r12*bz0 + ty
            bz = r20*bx0 + r21*by0 + r22*bz0 + tz

            # broadcast differences (BLOCK x BLOCK)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            r2 = dx*dx + dy*dy + dz*dz

            dc = ca[:, None] - cb[None, :]
            c2 = dc * dc

            g_spatial = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const

            # exp(-c2 / lam) = exp2(-c2 * inv_lam / ln2)
            g_charge = tl.exp2((-c2 * inv_lam) * inv_ln2)

            g = g_spatial * g_charge

            pair_mask = mask_n[:, None] & mask_m[None, :]
            g = tl.where(pair_mask, g, 0.0)

            Vab_acc += tl.sum(g)

            if NEED_GRAD:
                # g_charge does not depend on the pose, so d(g)/dR = g_charge * d(g_spatial)/dR
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


@autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _gauss_overlap_esp_se3_multipose(
    A_ptr, B_ptr,
    CA_ptr, CB_ptr,
    Q_ptr, T_ptr,
    Nreal_ptr, Mreal_ptr,
    BATCH, M_pad, N_pad,
    half_alpha, k_const,
    inv_lam,
    S_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr,
    POSES: tl.constexpr,
    POSES_PAD: tl.constexpr,
):
    """POSES poses of one molecule per CTA, ESP variant of the shape multi-pose kernel.

    The charge weighting ``exp(-(Ci-Cj)^2/lam)`` does not depend on the pose, so ``dc``, ``c2``
    and their ``exp2`` are computed once per tile and reused by every pose. POSES is a
    ``tl.arange`` extent padded to POSES_PAD (a power of two) and must divide SEEDS.
    """
    pid = tl.program_id(0)
    base = pid * POSES
    mol = base // SEEDS

    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)
    A_ptr = A_ptr + mol * N_pad * 3
    B_ptr = B_ptr + mol * M_pad * 3
    CA_ptr = CA_ptr + mol * N_pad
    CB_ptr = CB_ptr + mol * M_pad

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
        ca = tl.load(CA_ptr + a_idx, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM
            b_idx = tl.where(mask_m, offs_m, 0)
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)
            cb = tl.load(CB_ptr + b_idx, mask=mask_m, other=0.0)
            pair_mask = mask_n[None, :, None] & mask_m[None, None, :]

            # charges do not rotate, so this exp2 is paid once per tile
            dc = ca[:, None] - cb[None, :]
            g_charge = tl.exp2((-(dc * dc) * inv_lam) * inv_ln2)

            bx = r00[:, None]*bx0[None, :] + r01[:, None]*by0[None, :] + r02[:, None]*bz0[None, :] + tx[:, None]
            by = r10[:, None]*bx0[None, :] + r11[:, None]*by0[None, :] + r12[:, None]*bz0[None, :] + ty[:, None]
            bz = r20[:, None]*bx0[None, :] + r21[:, None]*by0[None, :] + r22[:, None]*bz0[None, :] + tz[:, None]

            dx = ax[None, :, None] - bx[:, None, :]
            dy = ay[None, :, None] - by[:, None, :]
            dz = az[None, :, None] - bz[:, None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g_spatial = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            g = g_spatial * g_charge[None, :, :]
            g = tl.where(pair_mask, g, 0.0)

            Vab_acc += tl.sum(tl.sum(g, 2), 1)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                fx = tl.sum(coeff * dx, 1)
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




def overlap_score_grad_esp_se3_batch(
    A, B,
    charges_A, charges_B,
    q, t, *,
    alpha: float = 0.81,
    lam: float = 0.3,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    seeds_per_mol: int = 1,
    poses_per_cta: int = 1,
):
    """ESP-weighted overlap with SE(3) gradients; one CTA per pose, tile loops over A and B.

    Shapes:
      A : (K, N_pad, 3) - coordinates of molecule A (reference)
      B : (K, M_pad, 3) - coordinates of molecule B (fit)
      charges_A : (K, N_pad) - ESP values at A points
      charges_B : (K, M_pad) - ESP values at B points
      q : (K, 4) - quaternions
      t : (K, 3) - translations
    With ``seeds_per_mol > 1`` the coordinate blocks are unreplicated (see the shape kernel).

    Returns:
      VAB : (K,) - ESP-weighted overlap scores
      dQ : (K, 4) - quaternion gradients
      dT : (K, 3) - translation gradients
    """
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    device = A.device
    dtype  = A.dtype

    if N_real is None:
        N_real = torch.full((K,), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((K,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    S_seeds = int(seeds_per_mol)
    if S_seeds > 1:
        K = q.shape[0]
        n_mol = A.shape[0]
        if K % S_seeds != 0 or n_mol != K // S_seeds:
            raise ValueError(f"seeds_per_mol={S_seeds} inconsistent: q has {K} poses, "
                             f"A has {n_mol} molecules")

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)
    inv_lam    = 1.0 / lam

    # The kernel stores every score and, when NEED_GRAD, every gradient, so no pre-zeroing;
    # without NEED_GRAD the gradient buffers are left unwritten and keep their zeros.
    out_S  = torch.empty(K, device=device, dtype=dtype)
    out_dQ = torch.empty_like(q) if NEED_GRAD else torch.zeros_like(q)
    out_dT = torch.empty_like(t) if NEED_GRAD else torch.zeros_like(t)

    POSES = int(poses_per_cta)
    if POSES > 1:
        if S_seeds <= 1 or S_seeds % POSES != 0 or K % POSES != 0:
            raise ValueError(
                f"poses_per_cta={POSES} needs the deduped layout and SEEDS % POSES == 0 "
                f"(got seeds_per_mol={S_seeds}, K={K})")
        POSES_PAD = 1 << (POSES - 1).bit_length()
        _gauss_overlap_esp_se3_multipose[(K // POSES,)](
            A.contiguous().view(-1), B.contiguous().view(-1),
            charges_A.contiguous().view(-1), charges_B.contiguous().view(-1),
            q.contiguous().view(-1), t.contiguous().view(-1),
            N_real.contiguous(), M_real.contiguous(),
            K, M_pad, N_pad, half_alpha, k_const, inv_lam,
            out_S, out_dQ.view(-1), out_dT.view(-1),
            NEED_GRAD=NEED_GRAD, SEEDS=S_seeds, POSES=POSES, POSES_PAD=POSES_PAD,
        )
        return out_S, out_dQ, out_dT

    grid = (K,)    # 1-D launch: one CTA per alignment

    _gauss_overlap_esp_se3_tiled[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        charges_A.contiguous().view(-1),
        charges_B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad,
        half_alpha, k_const, inv_lam,
        out_S, out_dQ.view(-1), out_dT.view(-1),
        NEED_GRAD=NEED_GRAD, SEEDS=S_seeds,
    )
    return out_S, out_dQ, out_dT

#  ShaEP surface-ESP agreement with its SE(3) gradient (vol_and_surf_esp and its Tversky
#  variant). One CTA per pose. The fit molecule's atoms and surface points arrive in its own
#  frame and are moved by R(q) x + t in-kernel (no torch transform, so no TF32 matmul), and
#  both directions of the comparison are summed:
#      V = (sum_i g_i + sum_j g_j) / (n_surf + m_surf),  g = keep * exp(-(pe - esp)^2 / lam)
#  i over the reference surface (field from the moved fit atoms), j over the moved fit
#  surface (field from the reference atoms). The vdW+probe mask is piecewise constant, so it
#  contributes no gradient. Forces on the moved points are contracted into dV/dq with the
#  shape kernel's _quat_grad_tail.

@autotune(configs=_OVERLAP_CONFIGS, key=["NS1", "NA1", "NS2", "NA2"], cache_results=True)
@triton.jit
def _esp_agreement_grad_kernel(
    P1, PE1, A1, Q1, R1,          # reference: surface (K*NS1*3), surface ESP, with-H atoms, charges, radii
    P2, PE2, A2, Q2, R2,          # fit, in its own frame (moved here by q, t)
    Qp, Tp,                       # (K*4), (K*3)
    NS1r, NA1r, NS2r, NA2r,       # (K,) real counts
    NS1, NA1, NS2, NA2,           # pads
    inv_lam, coul, probe,
    V_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ns1 = tl.load(NS1r + pid); na1 = tl.load(NA1r + pid)
    ns2 = tl.load(NS2r + pid); na2 = tl.load(NA2r + pid)
    P1 = P1 + pid * NS1 * 3; PE1 = PE1 + pid * NS1
    A1 = A1 + pid * NA1 * 3; Q1 = Q1 + pid * NA1; R1 = R1 + pid * NA1
    P2 = P2 + pid * NS2 * 3; PE2 = PE2 + pid * NS2
    A2 = A2 + pid * NA2 * 3; Q2 = Q2 + pid * NA2; R2 = R2 + pid * NA2
    qr = tl.load(Qp + pid * 4 + 0); qi = tl.load(Qp + pid * 4 + 1)
    qj = tl.load(Qp + pid * 4 + 2); qk = tl.load(Qp + pid * 4 + 3)
    tx = tl.load(Tp + pid * 3 + 0); ty = tl.load(Tp + pid * 3 + 1); tz = tl.load(Tp + pid * 3 + 2)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)
    inv_ln2 = 1.4426950408889634

    total = 0.0
    gw = 0.0; gx = 0.0; gy = 0.0; gz = 0.0
    fTx = 0.0; fTy = 0.0; fTz = 0.0

    # reference surface points vs moved fit atoms
    for n0 in range(0, NS1, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < ns1
        n_idx = tl.where(mask_n, offs_n, 0)
        px = tl.load(P1 + n_idx * 3 + 0, mask=mask_n, other=0.0)
        py = tl.load(P1 + n_idx * 3 + 1, mask=mask_n, other=0.0)
        pz = tl.load(P1 + n_idx * 3 + 2, mask=mask_n, other=0.0)
        pe = tl.load(PE1 + n_idx, mask=mask_n, other=0.0)
        esp = tl.zeros([BLOCK], dtype=tl.float32)
        cnt = tl.zeros([BLOCK], dtype=tl.float32)
        for m0 in range(0, NA2, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < na2
            m_idx = tl.where(mask_m, offs_m, 0)
            bx = tl.load(A2 + m_idx * 3 + 0, mask=mask_m, other=0.0)
            by = tl.load(A2 + m_idx * 3 + 1, mask=mask_m, other=0.0)
            bz = tl.load(A2 + m_idx * 3 + 2, mask=mask_m, other=0.0)
            qc = tl.load(Q2 + m_idx, mask=mask_m, other=0.0)
            rad = tl.load(R2 + m_idx, mask=mask_m, other=0.0)
            ax = r00 * bx + r01 * by + r02 * bz + tx
            ay = r10 * bx + r11 * by + r12 * bz + ty
            az = r20 * bx + r21 * by + r22 * bz + tz
            dx = px[:, None] - ax[None, :]
            dy = py[:, None] - ay[None, :]
            dz = pz[:, None] - az[None, :]
            d = tl.sqrt(dx * dx + dy * dy + dz * dz)
            d = tl.where(d < 1e-6, 1e-6, d)
            pair_m = mask_m[None, :]
            esp += tl.sum(tl.where(pair_m, qc[None, :] / d, 0.0), axis=1)
            blocked = (d < (rad[None, :] + probe)) & pair_m
            cnt += tl.sum(tl.where(blocked, 1.0, 0.0), axis=1)
        esp = esp * coul
        diff = pe - esp
        keep = mask_n & (cnt == 0.0)
        g = tl.where(keep, tl.exp2((-(diff * diff) * inv_lam) * inv_ln2), 0.0)
        total += tl.sum(g)
        if NEED_GRAD:
            c = tl.where(keep, 2.0 * inv_lam * g * diff * coul, 0.0)
            for m0 in range(0, NA2, BLOCK):
                offs_m = m0 + tl.arange(0, BLOCK)
                mask_m = offs_m < na2
                m_idx = tl.where(mask_m, offs_m, 0)
                bx = tl.load(A2 + m_idx * 3 + 0, mask=mask_m, other=0.0)
                by = tl.load(A2 + m_idx * 3 + 1, mask=mask_m, other=0.0)
                bz = tl.load(A2 + m_idx * 3 + 2, mask=mask_m, other=0.0)
                qc = tl.load(Q2 + m_idx, mask=mask_m, other=0.0)
                ax = r00 * bx + r01 * by + r02 * bz + tx
                ay = r10 * bx + r11 * by + r12 * bz + ty
                az = r20 * bx + r21 * by + r22 * bz + tz
                dx = px[:, None] - ax[None, :]
                dy = py[:, None] - ay[None, :]
                dz = pz[:, None] - az[None, :]
                d = tl.sqrt(dx * dx + dy * dy + dz * dz)
                d = tl.where(d < 1e-6, 1e-6, d)
                w = tl.where(mask_m[None, :], c[:, None] * qc[None, :] / (d * d * d), 0.0)
                fx = tl.sum(w * dx, axis=0)          # per fit atom: force on the moved atom
                fy = tl.sum(w * dy, axis=0)
                fz = tl.sum(w * dz, axis=0)
                dw_, dx_, dy_, dz_ = _quat_grad_tail(fx, fy, fz, bx, by, bz, qr, qi, qj, qk)
                gw += tl.sum(dw_); gx += tl.sum(dx_); gy += tl.sum(dy_); gz += tl.sum(dz_)
                fTx += tl.sum(fx); fTy += tl.sum(fy); fTz += tl.sum(fz)

    # moved fit surface points vs reference atoms
    for n0 in range(0, NS2, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < ns2
        n_idx = tl.where(mask_n, offs_n, 0)
        sx = tl.load(P2 + n_idx * 3 + 0, mask=mask_n, other=0.0)
        sy = tl.load(P2 + n_idx * 3 + 1, mask=mask_n, other=0.0)
        sz = tl.load(P2 + n_idx * 3 + 2, mask=mask_n, other=0.0)
        pe = tl.load(PE2 + n_idx, mask=mask_n, other=0.0)
        px = r00 * sx + r01 * sy + r02 * sz + tx
        py = r10 * sx + r11 * sy + r12 * sz + ty
        pz = r20 * sx + r21 * sy + r22 * sz + tz
        esp = tl.zeros([BLOCK], dtype=tl.float32)
        cnt = tl.zeros([BLOCK], dtype=tl.float32)
        Gx = tl.zeros([BLOCK], dtype=tl.float32)
        Gy = tl.zeros([BLOCK], dtype=tl.float32)
        Gz = tl.zeros([BLOCK], dtype=tl.float32)
        for m0 in range(0, NA1, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < na1
            m_idx = tl.where(mask_m, offs_m, 0)
            ax = tl.load(A1 + m_idx * 3 + 0, mask=mask_m, other=0.0)
            ay = tl.load(A1 + m_idx * 3 + 1, mask=mask_m, other=0.0)
            az = tl.load(A1 + m_idx * 3 + 2, mask=mask_m, other=0.0)
            qc = tl.load(Q1 + m_idx, mask=mask_m, other=0.0)
            rad = tl.load(R1 + m_idx, mask=mask_m, other=0.0)
            dx = px[:, None] - ax[None, :]
            dy = py[:, None] - ay[None, :]
            dz = pz[:, None] - az[None, :]
            d = tl.sqrt(dx * dx + dy * dy + dz * dz)
            d = tl.where(d < 1e-6, 1e-6, d)
            pair_m = mask_m[None, :]
            qd = tl.where(pair_m, qc[None, :] / d, 0.0)
            esp += tl.sum(qd, axis=1)
            if NEED_GRAD:
                w = qd / (d * d)
                Gx += tl.sum(w * dx, axis=1); Gy += tl.sum(w * dy, axis=1); Gz += tl.sum(w * dz, axis=1)
            blocked = (d < (rad[None, :] + probe)) & pair_m
            cnt += tl.sum(tl.where(blocked, 1.0, 0.0), axis=1)
        esp = esp * coul
        diff = pe - esp
        keep = mask_n & (cnt == 0.0)
        g = tl.where(keep, tl.exp2((-(diff * diff) * inv_lam) * inv_ln2), 0.0)
        total += tl.sum(g)
        if NEED_GRAD:
            c = tl.where(keep, -2.0 * inv_lam * g * diff * coul, 0.0)
            fx = c * Gx; fy = c * Gy; fz = c * Gz
            dw_, dx_, dy_, dz_ = _quat_grad_tail(fx, fy, fz, sx, sy, sz, qr, qi, qj, qk)
            gw += tl.sum(dw_); gx += tl.sum(dx_); gy += tl.sum(dy_); gz += tl.sum(dz_)
            fTx += tl.sum(fx); fTy += tl.sum(fy); fTz += tl.sum(fz)

    inv_n = 1.0 / (ns1 + ns2).to(tl.float32)
    tl.store(V_ptr + pid, total * inv_n)
    if NEED_GRAD:
        tl.store(dQ_ptr + pid * 4 + 0, gw * inv_n); tl.store(dQ_ptr + pid * 4 + 1, gx * inv_n)
        tl.store(dQ_ptr + pid * 4 + 2, gy * inv_n); tl.store(dQ_ptr + pid * 4 + 3, gz * inv_n)
        tl.store(dT_ptr + pid * 3 + 0, fTx * inv_n); tl.store(dT_ptr + pid * 3 + 1, fTy * inv_n)
        tl.store(dT_ptr + pid * 3 + 2, fTz * inv_n)


def esp_agreement_grad_se3_batch(ref_atoms, ref_charges, ref_radii, ref_points, ref_point_esp,
                                 fit_atoms, fit_charges, fit_radii, fit_points, fit_point_esp,
                                 n_surf, m_surf, n_atoms, m_atoms, q, t, *,
                                 probe_radius: float = 1.0, lam: float = 0.001,
                                 NEED_GRAD: bool = True):
    """The ShaEP surface-ESP agreement ``V`` (K,) of the reference with the fit moved by
    ``(q, t)``, and ``dV/dq`` (K, 4), ``dV/dt`` (K, 3) (zeros when ``NEED_GRAD`` is False).

    Shapes (padded, one row per pose): ``*_atoms`` (K, A_pad, 3) with-H atoms, ``*_charges`` and
    ``*_radii`` (K, A_pad), ``*_points`` (K, S_pad, 3) surface points, ``*_point_esp`` (K, S_pad)
    ESP at those points. The fit tensors are in the fit's own frame. ``n_surf``/``m_surf`` and
    ``n_atoms``/``m_atoms`` are the real surface-point and atom counts. ``lam`` is the raw
    weighting parameter; ``LAM_SCALING`` is applied here.
    """
    K = int(q.shape[0])
    dev = q.device
    i32 = lambda x: x.to(device=dev, dtype=torch.int32).contiguous()
    c = lambda x: x.contiguous()
    V = torch.empty(K, device=dev, dtype=ref_points.dtype)
    dQ = torch.zeros(K, 4, device=dev, dtype=q.dtype)
    dT = torch.zeros(K, 3, device=dev, dtype=t.dtype)
    _esp_agreement_grad_kernel[(K,)](
        c(ref_points), c(ref_point_esp), c(ref_atoms), c(ref_charges), c(ref_radii),
        c(fit_points), c(fit_point_esp), c(fit_atoms), c(fit_charges), c(fit_radii),
        c(q), c(t), i32(n_surf), i32(n_atoms), i32(m_surf), i32(m_atoms),
        int(ref_points.shape[1]), int(ref_atoms.shape[1]), int(fit_points.shape[1]),
        int(fit_atoms.shape[1]),
        1.0 / (LAM_SCALING * float(lam)), float(COULOMB_SCALING), float(probe_radius),
        V, dQ, dT, NEED_GRAD=bool(NEED_GRAD))
    return V, dQ, dT


@torch.no_grad()
def _batch_self_overlap_esp(
    P_pad: torch.Tensor,
    charges_pad: torch.Tensor,
    N_real: torch.Tensor,
    alpha: float = 0.81,
    lam: float = 0.3
) -> torch.Tensor:
    """
    Batched self-overlap VPP(P,P) for ESP-weighted Gaussian overlap.

    Parameters
    ----------
    P_pad : (K, N_pad, 3) - padded coordinates
    charges_pad : (K, N_pad) - padded ESP values
    N_real : (K,) int32 - true point counts

    Returns
    -------
    V : (K,) - self-overlap values
    """
    K, N_pad, _ = P_pad.shape
    q_id = torch.tensor([1., 0., 0., 0.], device=P_pad.device, dtype=P_pad.dtype).expand(K, 4)
    t_0  = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_esp_se3_batch(
        P_pad, P_pad,
        charges_pad, charges_pad,
        q_id, t_0,
        alpha=alpha,
        lam=lam,
        N_real=N_real, M_real=N_real,
        NEED_GRAD=False)
    return V
