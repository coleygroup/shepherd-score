"""
Analytical gradients for pharmacophore alignment scoring.

Replaces PyTorch autograd with hand-derived gradients for the pharmacophore
Tanimoto similarity objective. This module computes gradients of the overlap
O_AB w.r.t. SE(3) parameters (quaternion + translation) analytically,
then applies the Tanimoto chain rule.
"""
import math
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from shepherd_score.score.constants import P_TYPES, P_ALPHAS

P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))

# Pharmacophore type categories
_NONDIRECTIONAL = {'hydrophobe', 'znbinder', 'anion', 'cation'}
_DIRECTIONAL = {'acceptor', 'donor', 'halogen'}
_AROMATIC = {'aromatic'}


def _rotation_matrix_from_unit_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix from unit quaternion using the standard formula.

    Preserves dtype of q (unlike quaternions_to_rotation_matrix which may cast to float32).
    Supports single (4,) and batched (B,4) inputs.

    Parameters
    ----------
    q : torch.Tensor (4,) or (B,4)
        Unit quaternion(s) in (w, x, y, z) order.

    Returns
    -------
    R : torch.Tensor (3,3) or (B,3,3)
    """
    if q.dim() == 1:
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
        ]).reshape(3, 3)
        return R
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R


def rotation_matrix_jacobians_quat(q: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the four 3x3 Jacobians dR/dq_k for k in {w, x, y, z}.

    Assumes q is a unit quaternion (or batch of unit quaternions).

    Parameters
    ----------
    q : torch.Tensor (4,) or (B, 4)
        Unit quaternion(s) in (w, x, y, z) order.

    Returns
    -------
    dR_dqw, dR_dqx, dR_dqy, dR_dqz : each torch.Tensor (3,3) or (B,3,3)
    """
    batched = q.dim() == 2

    if batched:
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        zero = torch.zeros_like(qw)

        # dR/dqw
        dR_dqw = torch.stack([
            zero, -2*qz, 2*qy,
            2*qz, zero, -2*qx,
            -2*qy, 2*qx, zero
        ], dim=-1).reshape(-1, 3, 3)

        # dR/dqx
        dR_dqx = torch.stack([
            zero, 2*qy, 2*qz,
            2*qy, -4*qx, -2*qw,
            2*qz, 2*qw, -4*qx
        ], dim=-1).reshape(-1, 3, 3)

        # dR/dqy
        dR_dqy = torch.stack([
            -4*qy, 2*qx, 2*qw,
            2*qx, zero, 2*qz,
            -2*qw, 2*qz, -4*qy
        ], dim=-1).reshape(-1, 3, 3)

        # dR/dqz
        dR_dqz = torch.stack([
            -4*qz, -2*qw, 2*qx,
            2*qw, -4*qz, 2*qy,
            2*qx, 2*qy, zero
        ], dim=-1).reshape(-1, 3, 3)
    else:
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        z = torch.tensor(0.0, device=q.device, dtype=q.dtype)

        dR_dqw = torch.stack([
            z, -2*qz, 2*qy,
            2*qz, z, -2*qx,
            -2*qy, 2*qx, z
        ]).reshape(3, 3)

        dR_dqx = torch.stack([
            z, 2*qy, 2*qz,
            2*qy, -4*qx, -2*qw,
            2*qz, 2*qw, -4*qx
        ]).reshape(3, 3)

        dR_dqy = torch.stack([
            -4*qy, 2*qx, 2*qw,
            2*qx, z, 2*qz,
            -2*qw, 2*qz, -4*qy
        ]).reshape(3, 3)

        dR_dqz = torch.stack([
            -4*qz, -2*qw, 2*qx,
            2*qw, -4*qz, 2*qy,
            2*qx, 2*qy, z
        ]).reshape(3, 3)

    return dR_dqw, dR_dqx, dR_dqy, dR_dqz


def project_grad_R_to_quaternion(G: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Project gradient w.r.t. rotation matrix R onto quaternion parameters.

    dL/dq_k = Tr(G^T @ dR/dq_k) = sum_{ij} G_ij * (dR/dq_k)_ij

    Parameters
    ----------
    G : torch.Tensor (3,3) or (B,3,3)
        Gradient w.r.t. rotation matrix.
    q : torch.Tensor (4,) or (B,4)
        Unit quaternion.

    Returns
    -------
    grad_q : torch.Tensor (4,) or (B,4)
    """
    dR_dqw, dR_dqx, dR_dqy, dR_dqz = rotation_matrix_jacobians_quat(q)

    if G.dim() == 2:
        grad_qw = (G * dR_dqw).sum()
        grad_qx = (G * dR_dqx).sum()
        grad_qy = (G * dR_dqy).sum()
        grad_qz = (G * dR_dqz).sum()
        return torch.stack([grad_qw, grad_qx, grad_qy, grad_qz])
    else:
        # Batched: (B,3,3) element-wise multiply then sum over (3,3)
        grad_qw = (G * dR_dqw).sum(dim=(-2, -1))
        grad_qx = (G * dR_dqx).sum(dim=(-2, -1))
        grad_qy = (G * dR_dqy).sum(dim=(-2, -1))
        grad_qz = (G * dR_dqz).sum(dim=(-2, -1))
        return torch.stack([grad_qw, grad_qx, grad_qy, grad_qz], dim=-1)


def compute_overlap_and_grad_pharm(
    R: torch.Tensor,
    t: torch.Tensor,
    ref_pharms: torch.Tensor,
    fit_pharms: torch.Tensor,
    ref_anchors: torch.Tensor,
    fit_anchors_orig: torch.Tensor,
    ref_vectors: torch.Tensor,
    fit_vectors_orig: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute overlap O_AB and its gradients w.r.t. R (3x3) and t (3,).

    Supports batched inputs: R (B,3,3), t (B,3), pharms (B,N)/(B,M), etc.
    Also supports single instance: R (3,3), t (3,), pharms (N,)/(M,), etc.

    Parameters
    ----------
    R : torch.Tensor (3,3) or (B,3,3) — rotation matrix
    t : torch.Tensor (3,) or (B,3) — translation vector
    ref_pharms : torch.Tensor (N,) or (B,N) — pharmacophore type indices for reference
    fit_pharms : torch.Tensor (M,) or (B,M) — pharmacophore type indices for fit
    ref_anchors : torch.Tensor (N,3) or (B,N,3) — reference pharmacophore positions
    fit_anchors_orig : torch.Tensor (M,3) or (B,M,3) — original fit positions (before transform)
    ref_vectors : torch.Tensor (N,3) or (B,N,3) — reference unit vectors
    fit_vectors_orig : torch.Tensor (M,3) or (B,M,3) — original fit unit vectors

    Returns
    -------
    O_AB : torch.Tensor scalar or (B,)
    grad_R : torch.Tensor (3,3) or (B,3,3)
    grad_t : torch.Tensor (3,) or (B,3)
    """
    batched = R.dim() == 3

    if batched:
        B = R.shape[0]
        # P'_a = R @ P_a^T + t  => (B,M,3)
        fit_anchors_t = torch.bmm(R, fit_anchors_orig.permute(0, 2, 1)).permute(0, 2, 1) + t.unsqueeze(1)

        O_AB = torch.zeros(B, device=R.device, dtype=R.dtype)
        grad_R = torch.zeros(B, 3, 3, device=R.device, dtype=R.dtype)
        grad_t = torch.zeros(B, 3, device=R.device, dtype=R.dtype)

        # Normalize original vectors only; transformed vectors stay unit since R is orthogonal
        ref_vectors_n = F.normalize(ref_vectors, p=2, dim=-1)
        fit_vectors_orig_n = F.normalize(fit_vectors_orig, p=2, dim=-1)
        # R @ unit_vec = unit_vec when R is orthogonal, so no re-normalization needed
        fit_vectors_n = torch.bmm(R, fit_vectors_orig_n.permute(0, 2, 1)).permute(0, 2, 1)

        # Get unique pharmacophore types (same across batch since data is replicated)
        unique_ref = torch.unique(ref_pharms[0])
        unique_fit = torch.unique(fit_pharms[0])
        unique_ptypes = torch.cat([unique_ref, unique_fit]).unique()

        for ptype_idx in unique_ptypes:
            ptype_idx_int = ptype_idx.long().item()
            ptype_name = P_TYPES_LWRCASE[ptype_idx_int]
            alpha_m = P_ALPHAS[ptype_name]
            K_m = (math.pi / (2.0 * alpha_m)) ** 1.5

            # Type masks (same across batch) — use first batch element
            ref_idx = (ref_pharms[0] == ptype_idx_int).nonzero(as_tuple=True)[0]
            fit_idx = (fit_pharms[0] == ptype_idx_int).nonzero(as_tuple=True)[0]

            if len(ref_idx) == 0 or len(fit_idx) == 0:
                continue

            n_ref = len(ref_idx)
            n_fit = len(fit_idx)

            # Extract type-m data for all batch elements at once
            P_b_ref = ref_anchors[:, ref_idx]       # (B, n_ref, 3)
            P_a_fit_t = fit_anchors_t[:, fit_idx]    # (B, n_fit, 3)
            P_a_orig = fit_anchors_orig[:, fit_idx]   # (B, n_fit, 3)

            dist_sq = torch.cdist(P_a_fit_t, P_b_ref, p=2.0) ** 2  # (B, n_fit, n_ref)
            E_ab = torch.exp(-alpha_m / 2.0 * dist_sq)  # (B, n_fit, n_ref)

            if ptype_name in _NONDIRECTIONAL:
                O_AB += K_m * E_ab.sum(dim=(1, 2))

                sum_E_b = E_ab.sum(dim=2)  # (B, n_fit)
                sum_E_a = E_ab.sum(dim=1)  # (B, n_ref)
                term1 = (sum_E_b.unsqueeze(-1) * P_a_fit_t).sum(dim=1)
                term2 = (sum_E_a.unsqueeze(-1) * P_b_ref).sum(dim=1)
                grad_t += -alpha_m * K_m * (term1 - term2)

                term_Z = torch.bmm(E_ab, P_b_ref)
                E_delta_sum_ref = sum_E_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R += -alpha_m * K_m * torch.bmm(E_delta_sum_ref.transpose(1, 2), P_a_orig)

            elif ptype_name in _DIRECTIONAL:
                V_ref = ref_vectors_n[:, ref_idx]         # (B, n_ref, 3)
                V_fit_t = fit_vectors_n[:, fit_idx]        # (B, n_fit, 3)
                V_fit_orig = fit_vectors_orig_n[:, fit_idx]  # (B, n_fit, 3)

                D_ab = torch.bmm(V_fit_t, V_ref.permute(0, 2, 1))
                D_clamped = torch.clamp(D_ab, 0.0, 1.0)
                w_ab = (D_clamped + 2.0) / 3.0

                wE = w_ab * E_ab
                O_AB += K_m * wE.sum(dim=(1, 2))

                sum_wE_b = wE.sum(dim=2)
                sum_wE_a = wE.sum(dim=1)
                term1 = (sum_wE_b.unsqueeze(-1) * P_a_fit_t).sum(dim=1)
                term2 = (sum_wE_a.unsqueeze(-1) * P_b_ref).sum(dim=1)
                grad_t += -alpha_m * K_m * (term1 - term2)

                term_Z = torch.bmm(wE, P_b_ref)
                wE_delta_sum_ref = sum_wE_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R_spatial = -alpha_m * K_m * torch.bmm(wE_delta_sum_ref.transpose(1, 2), P_a_orig)

                dw_mask = ((D_ab > 0.0) & (D_ab < 1.0)).float()
                coeff = (1.0 / 3.0) * K_m * (E_ab * dw_mask)
                grad_R_weight = torch.bmm(V_ref.transpose(1, 2), coeff.transpose(1, 2))
                grad_R_weight = torch.bmm(grad_R_weight, V_fit_orig)

                grad_R += grad_R_spatial + grad_R_weight

            elif ptype_name in _AROMATIC:
                V_ref = ref_vectors_n[:, ref_idx]
                V_fit_t = fit_vectors_n[:, fit_idx]
                V_fit_orig = fit_vectors_orig_n[:, fit_idx]

                D_ab = torch.bmm(V_fit_t, V_ref.permute(0, 2, 1))
                abs_D = torch.abs(D_ab)
                w_ab = (abs_D + 2.0) / 3.0

                wE = w_ab * E_ab
                O_AB += K_m * wE.sum(dim=(1, 2))

                sum_wE_b = wE.sum(dim=2)
                sum_wE_a = wE.sum(dim=1)
                term1 = (sum_wE_b.unsqueeze(-1) * P_a_fit_t).sum(dim=1)
                term2 = (sum_wE_a.unsqueeze(-1) * P_b_ref).sum(dim=1)
                grad_t += -alpha_m * K_m * (term1 - term2)

                term_Z = torch.bmm(wE, P_b_ref)
                wE_delta_sum_ref = sum_wE_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R_spatial = -alpha_m * K_m * torch.bmm(wE_delta_sum_ref.transpose(1, 2), P_a_orig)

                sgn_D = torch.sign(D_ab)
                coeff = (1.0 / 3.0) * K_m * (E_ab * sgn_D)
                grad_R_weight = torch.bmm(V_ref.transpose(1, 2), coeff.transpose(1, 2))
                grad_R_weight = torch.bmm(grad_R_weight, V_fit_orig)

                grad_R += grad_R_spatial + grad_R_weight
    else:
        # Single instance
        # P'_a = R @ P_a^T + t  => (M, 3)
        fit_anchors_t = (R @ fit_anchors_orig.T).T + t
        fit_vectors_t = (R @ fit_vectors_orig.T).T

        O_AB = torch.tensor(0.0, device=R.device, dtype=R.dtype)
        grad_R = torch.zeros(3, 3, device=R.device, dtype=R.dtype)
        grad_t = torch.zeros(3, device=R.device, dtype=R.dtype)

        ref_vectors_n = F.normalize(ref_vectors, p=2, dim=-1)
        fit_vectors_orig_n = F.normalize(fit_vectors_orig, p=2, dim=-1)
        # R @ unit_vec = unit_vec when R is orthogonal, so no re-normalization needed
        fit_vectors_n = (R @ fit_vectors_orig_n.T).T

        unique_ref = torch.unique(ref_pharms)
        unique_fit = torch.unique(fit_pharms)
        unique_ptypes = torch.cat([unique_ref, unique_fit]).unique()

        for ptype_idx in unique_ptypes:
            ptype_idx_int = ptype_idx.long().item()
            ptype_name = P_TYPES_LWRCASE[ptype_idx_int]
            alpha_m = P_ALPHAS[ptype_name]
            K_m = (math.pi / (2.0 * alpha_m)) ** 1.5

            ref_idx = (ref_pharms == ptype_idx_int).nonzero(as_tuple=True)[0]
            fit_idx = (fit_pharms == ptype_idx_int).nonzero(as_tuple=True)[0]

            if len(ref_idx) == 0 or len(fit_idx) == 0:
                continue

            P_b_ref = ref_anchors[ref_idx]       # (n_ref, 3)
            P_a_fit_t = fit_anchors_t[fit_idx]    # (n_fit, 3)
            P_a_orig = fit_anchors_orig[fit_idx]   # (n_fit, 3)

            dist_sq = torch.cdist(P_a_fit_t, P_b_ref, p=2.0) ** 2  # (n_fit, n_ref)
            E_ab = torch.exp(-alpha_m / 2.0 * dist_sq)

            if ptype_name in _NONDIRECTIONAL:
                O_AB = O_AB + K_m * E_ab.sum()

                sum_E_b = E_ab.sum(dim=1)  # (n_fit,)
                sum_E_a = E_ab.sum(dim=0)  # (n_ref,)
                term1 = (sum_E_b.unsqueeze(-1) * P_a_fit_t).sum(dim=0)
                term2 = (sum_E_a.unsqueeze(-1) * P_b_ref).sum(dim=0)
                grad_t = grad_t + (-alpha_m * K_m) * (term1 - term2)

                term_Z = torch.mm(E_ab, P_b_ref)
                E_delta_sum_ref = sum_E_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R = grad_R + (-alpha_m * K_m) * torch.mm(E_delta_sum_ref.T, P_a_orig)

            elif ptype_name in _DIRECTIONAL:
                V_ref = ref_vectors_n[ref_idx]
                V_fit_t = fit_vectors_n[fit_idx]
                V_fit_orig = fit_vectors_orig_n[fit_idx]

                D_ab = torch.mm(V_fit_t, V_ref.T)
                D_clamped = torch.clamp(D_ab, 0.0, 1.0)
                w_ab = (D_clamped + 2.0) / 3.0

                wE = w_ab * E_ab
                O_AB = O_AB + K_m * wE.sum()

                sum_wE_b = wE.sum(dim=1)
                sum_wE_a = wE.sum(dim=0)
                term1 = (sum_wE_b.unsqueeze(-1) * P_a_fit_t).sum(dim=0)
                term2 = (sum_wE_a.unsqueeze(-1) * P_b_ref).sum(dim=0)
                grad_t = grad_t + (-alpha_m * K_m) * (term1 - term2)

                term_Z = torch.mm(wE, P_b_ref)
                wE_delta_sum_ref = sum_wE_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R_spatial = (-alpha_m * K_m) * torch.mm(wE_delta_sum_ref.T, P_a_orig)

                dw_mask = ((D_ab > 0.0) & (D_ab < 1.0)).float()
                coeff = (1.0 / 3.0) * K_m * (E_ab * dw_mask)
                
                grad_R_weight = torch.mm(V_ref.T, coeff.T)
                grad_R_weight = torch.mm(grad_R_weight, V_fit_orig)

                grad_R = grad_R + grad_R_spatial + grad_R_weight

            elif ptype_name in _AROMATIC:
                V_ref = ref_vectors_n[ref_idx]
                V_fit_t = fit_vectors_n[fit_idx]
                V_fit_orig = fit_vectors_orig_n[fit_idx]

                D_ab = torch.mm(V_fit_t, V_ref.T)
                abs_D = torch.abs(D_ab)
                w_ab = (abs_D + 2.0) / 3.0

                wE = w_ab * E_ab
                O_AB = O_AB + K_m * wE.sum()

                sum_wE_b = wE.sum(dim=1)
                sum_wE_a = wE.sum(dim=0)
                term1 = (sum_wE_b.unsqueeze(-1) * P_a_fit_t).sum(dim=0)
                term2 = (sum_wE_a.unsqueeze(-1) * P_b_ref).sum(dim=0)
                grad_t = grad_t + (-alpha_m * K_m) * (term1 - term2)

                term_Z = torch.mm(wE, P_b_ref)
                wE_delta_sum_ref = sum_wE_b.unsqueeze(-1) * P_a_fit_t - term_Z
                grad_R_spatial = (-alpha_m * K_m) * torch.mm(wE_delta_sum_ref.T, P_a_orig)

                sgn_D = torch.sign(D_ab)
                coeff = (1.0 / 3.0) * K_m * (E_ab * sgn_D)
                
                grad_R_weight = torch.mm(V_ref.T, coeff.T)
                grad_R_weight = torch.mm(grad_R_weight, V_fit_orig)

                grad_R = grad_R + grad_R_spatial + grad_R_weight

    return O_AB, grad_R, grad_t


def compute_self_overlaps_pharm(
    ptype_1: torch.Tensor,
    ptype_2: torch.Tensor,
    anchors_1: torch.Tensor,
    anchors_2: torch.Tensor,
    vectors_1: torch.Tensor,
    vectors_2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute self-overlaps VAA and VBB (invariant to SE(3)).

    Parameters
    ----------
    ptype_1, ptype_2 : torch.Tensor (N,) and (M,)
    anchors_1, anchors_2 : torch.Tensor (N,3) and (M,3)
    vectors_1, vectors_2 : torch.Tensor (N,3) and (M,3)

    Returns
    -------
    VAA_total, VBB_total : torch.Tensor (scalar each)
    """
    from shepherd_score.score.pharmacophore_scoring import (
        get_volume_overlap_score,
        get_vector_volume_overlap_score,
    )

    VAA_total = torch.tensor(0.0, device=anchors_1.device, dtype=anchors_1.dtype)
    VBB_total = torch.tensor(0.0, device=anchors_1.device, dtype=anchors_1.dtype)

    unique_ptypes = torch.cat([torch.unique(ptype_1), torch.unique(ptype_2)]).unique()

    for ptype_idx in unique_ptypes:
        ptype_idx_int = ptype_idx.long().item()
        ptype_name = P_TYPES_LWRCASE[ptype_idx_int]

        if ptype_name in _NONDIRECTIONAL:
            _, VAA, VBB = get_volume_overlap_score(
                ptype_str=ptype_name,
                ptype_1=ptype_1,
                ptype_2=ptype_2,
                anchors_1=anchors_1,
                anchors_2=anchors_2
            )
        elif ptype_name in _DIRECTIONAL:
            _, VAA, VBB = get_vector_volume_overlap_score(
                ptype_str=ptype_name,
                ptype_1=ptype_1,
                ptype_2=ptype_2,
                anchors_1=anchors_1,
                anchors_2=anchors_2,
                vectors_1=vectors_1,
                vectors_2=vectors_2,
                allow_antiparallel=False
            )
        elif ptype_name in _AROMATIC:
            _, VAA, VBB = get_vector_volume_overlap_score(
                ptype_str=ptype_name,
                ptype_1=ptype_1,
                ptype_2=ptype_2,
                anchors_1=anchors_1,
                anchors_2=anchors_2,
                vectors_1=vectors_1,
                vectors_2=vectors_2,
                allow_antiparallel=True
            )
        else:
            continue

        if isinstance(VAA, (int, float)):
            VAA = torch.tensor(VAA, device=anchors_1.device, dtype=anchors_1.dtype)
        if isinstance(VBB, (int, float)):
            VBB = torch.tensor(VBB, device=anchors_1.device, dtype=anchors_1.dtype)

        VAA_total = VAA_total + VAA
        VBB_total = VBB_total + VBB

    return VAA_total, VBB_total


def apply_tanimoto_chain_rule(
    O_AB: torch.Tensor,
    U: torch.Tensor,
    grad_R: torch.Tensor,
    grad_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the Tanimoto chain rule to scale overlap gradients into loss gradients.

    loss = 1 - S = 1 - O_AB / (U - O_AB + O_AB) = 1 - O_AB / U ... wait no:
    S = O_AB / (VAA + VBB - O_AB) = O_AB / (U - O_AB)

    dS/d(O_AB) = U / (U - O_AB)^2

    loss = 1 - S, so d(loss)/d(O_AB) = -U / (U - O_AB)^2

    Parameters
    ----------
    O_AB : torch.Tensor scalar or (B,)
    U : torch.Tensor scalar (VAA + VBB, precomputed)
    grad_R : torch.Tensor (3,3) or (B,3,3)
    grad_t : torch.Tensor (3,) or (B,3)

    Returns
    -------
    loss : torch.Tensor scalar or (B,)
    scaled_grad_R : torch.Tensor (3,3) or (B,3,3)
    scaled_grad_t : torch.Tensor (3,) or (B,3)
    """
    denom = U - O_AB
    S = O_AB / denom
    loss = 1.0 - S
    # d(loss)/d(O_AB) = -U / (U - O_AB)^2
    scale = -U / (denom * denom)

    if grad_R.dim() == 3:
        # Batched
        scale_R = scale.unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        scale_t = scale.unsqueeze(-1)  # (B,1)
    else:
        scale_R = scale
        scale_t = scale

    scaled_grad_R = scale_R * grad_R
    scaled_grad_t = scale_t * grad_t

    return loss, scaled_grad_R, scaled_grad_t


def compute_analytical_grad_se3(
    se3_params: torch.Tensor,
    ref_pharms: torch.Tensor,
    fit_pharms: torch.Tensor,
    ref_anchors: torch.Tensor,
    fit_anchors: torch.Tensor,
    ref_vectors: torch.Tensor,
    fit_vectors: torch.Tensor,
    VAA_total: torch.Tensor,
    VBB_total: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss and analytical gradient of loss w.r.t. se3_params.

    Parameters
    ----------
    se3_params : torch.Tensor (7,) or (B, 7)
        [q_w, q_x, q_y, q_z, t_x, t_y, t_z]
    ref_pharms, fit_pharms : torch.Tensor (N,)/(M,) or (B,N)/(B,M)
    ref_anchors, fit_anchors : torch.Tensor (N,3)/(M,3) or (B,N,3)/(B,M,3)
    ref_vectors, fit_vectors : torch.Tensor (N,3)/(M,3) or (B,N,3)/(B,M,3)
    VAA_total, VBB_total : torch.Tensor scalar (precomputed self-overlaps)

    Returns
    -------
    loss : torch.Tensor scalar or (B,)
    grad_se3_params : torch.Tensor (7,) or (B, 7)
    """
    batched = se3_params.dim() == 2
    U = VAA_total + VBB_total

    if batched:
        B = se3_params.shape[0]
        q_raw = se3_params[:, :4]
        t = se3_params[:, 4:]

        # Normalize quaternion
        q_norm_val = torch.norm(q_raw, dim=1, keepdim=True)
        q_norm = q_raw / q_norm_val

        # Build rotation matrix
        R = _rotation_matrix_from_unit_quat(q_norm)  # (B, 3, 3)

        # Compute overlap and gradients
        O_AB, grad_R, grad_t = compute_overlap_and_grad_pharm(
            R, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors
        )

        # Apply Tanimoto chain rule
        loss, scaled_grad_R, scaled_grad_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        # Project R gradient to quaternion
        grad_q_norm = project_grad_R_to_quaternion(scaled_grad_R, q_norm)  # (B, 4)

        # Chain rule for quaternion normalization: d/dq_raw = (I - q_hat q_hat^T) / ||q_raw|| @ d/dq_norm
        # q_hat = q_raw / ||q_raw||, so this is the Jacobian of normalization
        # d(q_norm)/d(q_raw) = (I - q_norm q_norm^T) / ||q_raw||
        q_norm_expanded = q_norm.unsqueeze(-1)  # (B, 4, 1)
        I = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype).unsqueeze(0)
        proj = I - q_norm_expanded @ q_norm_expanded.transpose(-1, -2)  # (B, 4, 4)
        grad_q_raw = (proj @ grad_q_norm.unsqueeze(-1)).squeeze(-1) / q_norm_val  # (B, 4)

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t], dim=1)  # (B, 7)

        # Return mean loss and per-element gradient divided by B
        # (matching autograd behavior: loss = (1-score).mean(); loss.backward())
        return loss.mean(), grad_se3 / B

    else:
        q_raw = se3_params[:4]
        t = se3_params[4:]

        q_norm_val = torch.norm(q_raw)
        q_norm = q_raw / q_norm_val

        R = _rotation_matrix_from_unit_quat(q_norm)

        O_AB, grad_R, grad_t = compute_overlap_and_grad_pharm(
            R, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors
        )

        loss, scaled_grad_R, scaled_grad_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        grad_q_norm = project_grad_R_to_quaternion(scaled_grad_R, q_norm)

        # Normalization chain rule
        q_hat = q_norm.unsqueeze(-1)  # (4, 1)
        proj = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype) - q_hat @ q_hat.T
        grad_q_raw = (proj @ grad_q_norm) / q_norm_val

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t])
        return loss, grad_se3
