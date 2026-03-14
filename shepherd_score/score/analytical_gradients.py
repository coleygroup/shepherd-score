"""
Analytical gradients for shape and pharmacophore alignment scoring.

Replaces PyTorch autograd with hand-derived gradients for the shape and
pharmacophore Tanimoto similarity objectives (optionally with an avoid-points
penalty term). Gradients of the overlap O_AB w.r.t. SE(3) parameters
(quaternion + translation) are computed analytically, then the Tanimoto
chain rule is applied.
"""
import math
from typing import Tuple
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from shepherd_score.score.constants import P_TYPES, P_ALPHAS

P_TYPES_LWRCASE = tuple([p.lower() for p in P_TYPES])

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


@lru_cache(maxsize=4)
def build_lookup_tables_cached(device_str: str, dtype_str: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alphas = []
    Ks = []
    categories = []

    for p in P_TYPES:
        p_lower = p.lower()
        a = P_ALPHAS.get(p_lower, 1.0)
        alphas.append(a)
        Ks.append((math.pi / (2.0 * a)) ** 1.5)

        if p_lower in _NONDIRECTIONAL:
            categories.append(0)
        elif p_lower in _DIRECTIONAL:
            categories.append(1)
        elif p_lower in _AROMATIC:
            categories.append(2)
        else:
            categories.append(3)

    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str.split('.')[-1])
    return (
        torch.tensor(alphas, device=device, dtype=dtype),
        torch.tensor(Ks, device=device, dtype=dtype),
        torch.tensor(categories, device=device, dtype=torch.long)
    )

def build_lookup_tables(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build constant lookup tables for all P_TYPES.
    categories: 0=_NONDIRECTIONAL, 1=_DIRECTIONAL, 2=_AROMATIC, 3=Dummy
    """
    return build_lookup_tables_cached(str(device), str(dtype))

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
    Compute overlap O_AB and gradients fully vectorized across all types.
    """
    batched = R.dim() == 3
    if not batched:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        ref_pharms = ref_pharms.unsqueeze(0)
        fit_pharms = fit_pharms.unsqueeze(0)
        ref_anchors = ref_anchors.unsqueeze(0)
        fit_anchors_orig = fit_anchors_orig.unsqueeze(0)
        ref_vectors = ref_vectors.unsqueeze(0)
        fit_vectors_orig = fit_vectors_orig.unsqueeze(0)

    # Ensure pharms are integral for indexing
    ref_pharms = ref_pharms.to(torch.long)
    fit_pharms = fit_pharms.to(torch.long)

    fit_anchors_t = torch.bmm(R, fit_anchors_orig.permute(0, 2, 1)).permute(0, 2, 1) + t.unsqueeze(1)

    # Vectors
    ref_vectors_n = F.normalize(ref_vectors, p=2, dim=-1)
    fit_vectors_orig_n = F.normalize(fit_vectors_orig, p=2, dim=-1)
    fit_vectors_n = torch.bmm(R, fit_vectors_orig_n.permute(0, 2, 1)).permute(0, 2, 1)

    # Constants
    alphas, Ks, cats = build_lookup_tables(R.device, R.dtype)

    # Masking matches exactly between fit and ref
    same_type = (fit_pharms.unsqueeze(2) == ref_pharms.unsqueeze(1))
    valid_type = (cats[fit_pharms] != 3).unsqueeze(2)
    mask = same_type & valid_type  # (B, n_fit, n_ref)

    alpha_ab = alphas[fit_pharms].unsqueeze(2)
    K_ab = Ks[fit_pharms].unsqueeze(2)
    cat_ab = cats[fit_pharms].unsqueeze(2)

    dist_sq = torch.cdist(fit_anchors_t, ref_anchors, p=2.0) ** 2
    E_ab = torch.exp(-alpha_ab / 2.0 * dist_sq) * mask.to(R.dtype)

    D_ab = torch.bmm(fit_vectors_n, ref_vectors_n.permute(0, 2, 1))

    D_clamped = torch.clamp(D_ab, 0.0, 1.0)
    w_dir = (D_clamped + 2.0) / 3.0
    w_arom = (torch.abs(D_ab) + 2.0) / 3.0

    w_ab = torch.where(cat_ab == 1, w_dir,
           torch.where(cat_ab == 2, w_arom, 1.0)).to(R.dtype)

    wE_ab = w_ab * E_ab

    O_AB = (K_ab * wE_ab).sum(dim=(1, 2))

    alpha_K_wE = -alpha_ab * K_ab * wE_ab
    sum_aKwE_b = alpha_K_wE.sum(dim=2)
    sum_aKwE_a = alpha_K_wE.sum(dim=1)

    term1 = (sum_aKwE_b.unsqueeze(-1) * fit_anchors_t).sum(dim=1)
    term2 = (sum_aKwE_a.unsqueeze(-1) * ref_anchors).sum(dim=1)
    grad_t = term1 - term2

    term_Z = torch.bmm(alpha_K_wE, ref_anchors)
    wE_delta_sum_ref = sum_aKwE_b.unsqueeze(-1) * fit_anchors_t - term_Z
    grad_R_spatial = torch.bmm(wE_delta_sum_ref.transpose(1, 2), fit_anchors_orig)

    coeff_dir = (D_ab > 0.0) & (D_ab < 1.0)
    coeff_arom = torch.sign(D_ab)

    c_ab = torch.where(cat_ab == 1, coeff_dir.to(R.dtype),
           torch.where(cat_ab == 2, coeff_arom.to(R.dtype), 0.0)).to(R.dtype)

    coeff = (1.0 / 3.0) * K_ab * E_ab * c_ab

    gRw_tmp = torch.bmm(ref_vectors_n.transpose(1, 2), coeff.transpose(1, 2))
    grad_R_weight = torch.bmm(gRw_tmp, fit_vectors_orig_n)

    grad_R = grad_R_spatial + grad_R_weight

    if not batched:
        return O_AB[0], grad_R[0], grad_t[0]
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
    Compute self-overlaps VAA and VBB vectorially.
    """
    I = torch.eye(3, device=anchors_1.device, dtype=anchors_1.dtype)
    zero = torch.zeros(3, device=anchors_1.device, dtype=anchors_1.dtype)

    VAA, _, _ = compute_overlap_and_grad_pharm(I, zero, ptype_1, ptype_1, anchors_1, anchors_1, vectors_1, vectors_1)
    VBB, _, _ = compute_overlap_and_grad_pharm(I, zero, ptype_2, ptype_2, anchors_2, anchors_2, vectors_2, vectors_2)

    return VAA, VBB


def apply_tanimoto_chain_rule(
    O_AB: torch.Tensor,
    U: torch.Tensor,
    grad_R: torch.Tensor,
    grad_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = U - O_AB
    S = O_AB / denom
    loss = 1.0 - S
    scale = -U / (denom * denom)

    if grad_R.dim() == 3:
        scale_R = scale.unsqueeze(-1).unsqueeze(-1)
        scale_t = scale.unsqueeze(-1)
    else:
        scale_R = scale
        scale_t = scale

    scaled_grad_R = scale_R * grad_R
    scaled_grad_t = scale_t * grad_t

    return loss, scaled_grad_R, scaled_grad_t


def compute_overlap_and_grad_shape(
    R: torch.Tensor,
    t: torch.Tensor,
    ref_points: torch.Tensor,
    fit_points_orig: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute shape overlap O_AB and gradients w.r.t. rotation matrix R and translation t.

    Shape scoring uses uniform weight w=1 and a single alpha, so there is no weight
    gradient term (unlike pharmacophore scoring).

    Parameters
    ----------
    R : torch.Tensor (3,3) or (B,3,3)
    t : torch.Tensor (3,) or (B,3)
    ref_points : torch.Tensor (N,3) or (B,N,3)
    fit_points_orig : torch.Tensor (M,3) or (B,M,3)
    alpha : float

    Returns
    -------
    O_AB : torch.Tensor scalar or (B,)
    grad_R : torch.Tensor (3,3) or (B,3,3)
    grad_t : torch.Tensor (3,) or (B,3)
    """
    batched = R.dim() == 3
    if not batched:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        fit_points_orig = fit_points_orig.unsqueeze(0)

    K = (math.pi / (2.0 * alpha)) ** 1.5

    # Transform fit points: (B, n_fit, 3)
    fit_points_t = torch.bmm(R, fit_points_orig.permute(0, 2, 1)).permute(0, 2, 1) + t.unsqueeze(1)

    # Pairwise squared distances: (B, n_fit, n_ref)
    dist_sq = torch.cdist(fit_points_t, ref_points, p=2.0) ** 2

    # Gaussian terms: (B, n_fit, n_ref)
    E_ab = torch.exp(-alpha / 2.0 * dist_sq)

    # Overlap: O_AB = K * sum(E_ab)
    O_AB = K * E_ab.sum(dim=(1, 2))  # (B,)

    # Coefficient: -alpha * K * E_ab
    aKE = -alpha * K * E_ab  # (B, n_fit, n_ref)

    # Gradient w.r.t. t: grad_t = sum_{a,b} aKE_ab * (P'_a - P_b)
    sum_over_ref = aKE.sum(dim=2)                                           # (B, n_fit)
    sum_over_fit = aKE.sum(dim=1)                                           # (B, n_ref)
    term1 = (sum_over_ref.unsqueeze(-1) * fit_points_t).sum(dim=1)         # (B, 3)
    term2 = (sum_over_fit.unsqueeze(-1) * ref_points).sum(dim=1)           # (B, 3)
    grad_t = term1 - term2                                                  # (B, 3)

    # Gradient w.r.t. R: grad_R = sum_{a,b} aKE_ab * (P'_a - P_b) @ P_a^T
    # No weight gradient since w=1 always for shape scoring
    term_Z = torch.bmm(aKE, ref_points)                                     # (B, n_fit, 3)
    delta_sum = sum_over_ref.unsqueeze(-1) * fit_points_t - term_Z          # (B, n_fit, 3)
    grad_R = torch.bmm(delta_sum.transpose(1, 2), fit_points_orig)          # (B, 3, 3)

    if not batched:
        return O_AB[0], grad_R[0], grad_t[0]
    return O_AB, grad_R, grad_t


def compute_self_overlaps_shape(
    ref_points: torch.Tensor,
    fit_points: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute self-overlaps VAA and VBB for shape scoring. Invariant to SE(3).

    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
    fit_points : torch.Tensor (M,3)
    alpha : float

    Returns
    -------
    VAA, VBB : torch.Tensor scalars
    """
    I = torch.eye(3, device=ref_points.device, dtype=ref_points.dtype)
    zero = torch.zeros(3, device=ref_points.device, dtype=ref_points.dtype)
    VAA, _, _ = compute_overlap_and_grad_shape(I, zero, ref_points, ref_points, alpha)
    VBB, _, _ = compute_overlap_and_grad_shape(I, zero, fit_points, fit_points, alpha)
    return VAA, VBB


def compute_analytical_grad_se3_shape(
    se3_params: torch.Tensor,
    ref_points: torch.Tensor,
    fit_points: torch.Tensor,
    alpha: float,
    VAA_total: torch.Tensor,
    VBB_total: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss (1 - Tanimoto shape similarity) and its gradient w.r.t. SE(3) parameters
    using analytical gradients.

    Parameters
    ----------
    se3_params : torch.Tensor (7,) or (B,7)
        [q_w, q_x, q_y, q_z, t_x, t_y, t_z]
    ref_points : torch.Tensor (N,3) or (B,N,3)
    fit_points : torch.Tensor (M,3) or (B,M,3)
    alpha : float
    VAA_total : torch.Tensor scalar
        Precomputed self-overlap for ref_points.
    VBB_total : torch.Tensor scalar
        Precomputed self-overlap for fit_points.

    Returns
    -------
    loss : torch.Tensor scalar
    grad_se3 : torch.Tensor (7,) or (B,7)
    """
    batched = se3_params.dim() == 2
    U = VAA_total + VBB_total

    if batched:
        B = se3_params.shape[0]
        q_raw = se3_params[:, :4]
        t = se3_params[:, 4:]

        q_norm_val = torch.norm(q_raw, dim=1, keepdim=True)
        q_norm = q_raw / q_norm_val

        R = _rotation_matrix_from_unit_quat(q_norm)

        O_AB, grad_R, grad_t = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)

        loss, scaled_grad_R, scaled_grad_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        grad_q_norm = project_grad_R_to_quaternion(scaled_grad_R, q_norm)

        q_norm_expanded = q_norm.unsqueeze(-1)
        I_mat = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype).unsqueeze(0)
        proj = I_mat - q_norm_expanded @ q_norm_expanded.transpose(-1, -2)
        grad_q_raw = (proj @ grad_q_norm.unsqueeze(-1)).squeeze(-1) / q_norm_val

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t], dim=1)
        return loss.mean(), grad_se3 / B

    else:
        q_raw = se3_params[:4]
        t = se3_params[4:]

        q_norm_val = torch.norm(q_raw)
        q_norm = q_raw / q_norm_val

        R = _rotation_matrix_from_unit_quat(q_norm)

        O_AB, grad_R, grad_t = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)

        loss, scaled_grad_R, scaled_grad_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        grad_q_norm = project_grad_R_to_quaternion(scaled_grad_R, q_norm)

        q_hat = q_norm.unsqueeze(-1)
        proj = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype) - q_hat @ q_hat.T
        grad_q_raw = (proj @ grad_q_norm) / q_norm_val

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t])
        return loss, grad_se3


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
    batched = se3_params.dim() == 2
    U = VAA_total + VBB_total

    if batched:
        B = se3_params.shape[0]
        q_raw = se3_params[:, :4]
        t = se3_params[:, 4:]

        q_norm_val = torch.norm(q_raw, dim=1, keepdim=True)
        q_norm = q_raw / q_norm_val

        R = _rotation_matrix_from_unit_quat(q_norm)

        O_AB, grad_R, grad_t = compute_overlap_and_grad_pharm(
            R, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors
        )

        loss, scaled_grad_R, scaled_grad_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        grad_q_norm = project_grad_R_to_quaternion(scaled_grad_R, q_norm)

        q_norm_expanded = q_norm.unsqueeze(-1)
        I = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype).unsqueeze(0)
        proj = I - q_norm_expanded @ q_norm_expanded.transpose(-1, -2)
        grad_q_raw = (proj @ grad_q_norm.unsqueeze(-1)).squeeze(-1) / q_norm_val

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t], dim=1)

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

        q_hat = q_norm.unsqueeze(-1)
        proj = torch.eye(4, device=se3_params.device, dtype=se3_params.dtype) - q_hat @ q_hat.T
        grad_q_raw = (proj @ grad_q_norm) / q_norm_val

        grad_se3 = torch.cat([grad_q_raw, scaled_grad_t])
        return loss, grad_se3
