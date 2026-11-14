"""Linear hard-sphere excluded-volume ("avoid") penalty kernel in Triton.

Value + SE(3) gradient of the piecewise-linear penalty

    A_pen = sum_a sum_b relu((d0 - ||A_a - B'_b||)/d0),   B'_b = R(q) B_b + t

with A the fixed avoid points (ref frame, not transformed) and B the fit points. Structurally
the shape kernel (``shape_triton._gauss_overlap_se3_tiled``) with the Gaussian replaced by a
``rsqrt``-based distance + hinge, and the loops swapped (fit tiles outermost); the numba twin
is ``cpu.overlap_score_grad_avoid_se3_batch``.
"""
from __future__ import annotations

import triton
from .tuning import autotune
import triton.language as tl
import torch

from .shape_triton import _quat_to_rotmat, _quat_grad_tail, _OVERLAP_CONFIGS


@autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _avoid_penalty_se3_tiled(
    A_ptr, B_ptr,                 # flat (B/A_DIV * N_pad * 3), (B * M_pad * 3): avoid, fit points
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B/A_DIV,) avoid counts, (B,) fit counts
    BATCH, M_pad, N_pad, A_DIV,   # ints; A_DIV pose rows share one avoid row
    min_dist, inv_d0,             # scalars: d0 and 1/d0
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,) penalty, dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # tile edge (chosen by autotune)
    NEED_GRAD: tl.constexpr
):
    pid = tl.program_id(0)
    arow = pid // A_DIV           # every seed of a pair reads one avoid row (L2-resident)
    realN = tl.load(Nreal_ptr + arow)
    realM = tl.load(Mreal_ptr + pid)

    A_ptr  = A_ptr  + arow * N_pad * 3
    B_ptr  = B_ptr  + pid * M_pad * 3
    Q_ptr  = Q_ptr  + pid * 4
    T_ptr  = T_ptr  + pid * 3
    dQ_ptr = dQ_ptr + pid * 4
    dT_ptr = dT_ptr + pid * 3
    S_ptr  = S_ptr  + pid

    qr = tl.load(Q_ptr + 0); qi = tl.load(Q_ptr + 1)
    qj = tl.load(Q_ptr + 2); qk = tl.load(Q_ptr + 3)
    tx = tl.load(T_ptr + 0); ty = tl.load(T_ptr + 1); tz = tl.load(T_ptr + 2)

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

    Vacc = 0.0
    dTx = 0.0; dTy = 0.0; dTz = 0.0
    dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0

    # Fit tiles outermost: each fit tile is rotated once, its force is accumulated over every
    # avoid tile, and the quaternion tail then runs once per fit tile (not once per tile pair).
    # Avoid clouds are typically much larger than the fit (K >> M).
    for m0 in range(0, M_pad, BLOCK):
        offs_m = m0 + tl.arange(0, BLOCK)
        mask_m = offs_m < realM
        b_idx = tl.where(mask_m, offs_m, 0)
        bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
        by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
        bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

        bx = r00*bx0 + r01*by0 + r02*bz0 + tx
        by = r10*bx0 + r11*by0 + r12*bz0 + ty
        bz = r20*bx0 + r21*by0 + r22*bz0 + tz

        fx = tl.zeros((BLOCK,), dtype=tl.float32)
        fy = tl.zeros((BLOCK,), dtype=tl.float32)
        fz = tl.zeros((BLOCK,), dtype=tl.float32)

        # Bounding sphere of this posed fit tile. An avoid point farther than radius + d0 from
        # its center is beyond d0 of every atom in the tile, so it adds neither value nor
        # gradient; an avoid tile with no point within reach is skipped (exact, not a cutoff).
        # The cloud arrives Z-ordered (channels.z_order), so its tiles are spatially compact.
        n_m = tl.maximum(tl.sum(mask_m.to(tl.float32)), 1.0)
        cx = tl.sum(tl.where(mask_m, bx, 0.0)) / n_m
        cy = tl.sum(tl.where(mask_m, by, 0.0)) / n_m
        cz = tl.sum(tl.where(mask_m, bz, 0.0)) / n_m
        ex = bx - cx; ey = by - cy; ez = bz - cz
        rad = tl.sqrt(tl.max(tl.where(mask_m, ex*ex + ey*ey + ez*ez, 0.0)))
        reach = rad + min_dist + 1e-3                    # slack absorbs the fp32 rounding
        reach2 = reach * reach

        for n0 in range(0, N_pad, BLOCK):
            offs_n = n0 + tl.arange(0, BLOCK)
            mask_n = offs_n < realN
            a_idx = tl.where(mask_n, offs_n, 0)
            ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
            ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
            az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)
            gx = ax - cx; gy = ay - cy; gz = az - cz
            # tiles wholly past realM (padding) have a degenerate sphere at the origin: skip them
            near = mask_n & (gx*gx + gy*gy + gz*gz < reach2) & (m0 < realM)
            if tl.max(near.to(tl.int32)) > 0:
                dx = ax[:, None] - bx[None, :]
                dy = ay[:, None] - by[None, :]
                dz = az[:, None] - bz[None, :]
                r2 = dx*dx + dy*dy + dz*dz
                # one rsqrt serves both d and the force's 1/d (the clamp keeps r2 = 0 finite: d = 0)
                ir = tl.math.rsqrt(tl.maximum(r2, 1e-30))
                d = r2 * ir

                pair_mask = mask_n[:, None] & mask_m[None, :]
                active = pair_mask & (d < min_dist)             # inside the hard-sphere shell
                hinge = tl.where(active, (min_dist - d) * inv_d0, 0.0)
                Vacc += tl.sum(hinge)

                if NEED_GRAD:
                    # coeff = (1/d0)/d on the active band (d>0), else 0; f += coeff*dx (== +dA_pen/dB')
                    coeff = tl.where(active & (d > 1e-8), inv_d0 * ir, 0.0)
                    fx += tl.sum(coeff * dx, 0)
                    fy += tl.sum(coeff * dy, 0)
                    fz += tl.sum(coeff * dz, 0)

        if NEED_GRAD:
            dTx += tl.sum(fx)
            dTy += tl.sum(fy)
            dTz += tl.sum(fz)

            dw, dxq, dyq, dzq = _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk)
            dw  = tl.where(mask_m, dw,  0.0)
            dxq = tl.where(mask_m, dxq, 0.0)
            dyq = tl.where(mask_m, dyq, 0.0)
            dzq = tl.where(mask_m, dzq, 0.0)
            dQw += tl.sum(dw)
            dQx += tl.sum(dxq)
            dQy += tl.sum(dyq)
            dQz += tl.sum(dzq)

    tl.store(S_ptr, Vacc)
    if NEED_GRAD:
        tl.store(dT_ptr + 0, dTx); tl.store(dT_ptr + 1, dTy); tl.store(dT_ptr + 2, dTz)
        tl.store(dQ_ptr + 0, dQw); tl.store(dQ_ptr + 1, dQx)
        tl.store(dQ_ptr + 2, dQy); tl.store(dQ_ptr + 3, dQz)


def overlap_score_grad_avoid_se3_batch(
    A, B, q, t, *,
    min_dist: float = 2.0,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    BLOCK: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    A_DIV: int = 1,
):
    """Linear hard-sphere avoid penalty value + SE(3) gradient; one CTA per pose.
    Shapes: A (K/A_DIV, N_pad, 3) fixed avoid points (pose row k reads avoid row k // A_DIV),
    B (K, M_pad, 3) fit points, q (K,4), t (K,3); N_real is per avoid row, M_real per pose.
    Returns (S (K,), dQ (K,4), dT (K,3)); same signature as the numba twin."""
    K, M_pad, _ = B.shape
    N_pad = A.shape[1]
    A_DIV = int(A_DIV)
    if A.shape[0] * A_DIV != K:
        raise ValueError(f"avoid rows {A.shape[0]} x A_DIV {A_DIV} != poses {K}")
    device = A.device
    dtype = A.dtype

    if N_real is None:
        N_real = torch.full((A.shape[0],), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((K,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    inv_d0 = 1.0 / float(min_dist)
    out_S = torch.zeros(K, device=device, dtype=dtype)
    out_dQ = torch.zeros_like(q)
    out_dT = torch.zeros_like(t)

    grid = (K,)
    _avoid_penalty_se3_tiled[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad, A_DIV,
        float(min_dist), inv_d0,
        out_S, out_dQ.view(-1), out_dT.view(-1),
        NEED_GRAD=NEED_GRAD,
    )
    return out_S, out_dQ, out_dT
