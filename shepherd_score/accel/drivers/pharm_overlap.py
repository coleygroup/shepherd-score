# shepherd_score/accel/drivers/pharm_overlap.py
# Batched pharmacophore overlap scoring in PyTorch rather than Triton: the overlap depends on
# type-specific alpha values and optional vector / extended-point logic. Coarse stages call
# these under `torch.no_grad()`; fine stages may enable autograd on (q, t).

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ...score.constants import P_TYPES, P_ALPHAS

# Lowercase type names for indexing
P_TYPES_LOWER = tuple(map(str.lower, P_TYPES))

# Pre-compute alpha array for GPU indexing.
# Built directly from P_TYPES so it stays in sync if pharmacophore types are
# added/removed upstream (e.g. the 'Dummy' type). Order matches P_TYPES.
ALPHA_VALUES = torch.tensor(
    [P_ALPHAS[name] for name in P_TYPES_LOWER],
    dtype=torch.float32,
)

# Types that use vector cosine weighting
VECTOR_TYPES = {2, 0, 1, 4}  # aromatic, acceptor, donor, halogen
# Types where antiparallel vectors are acceptable
ANTIPARALLEL_TYPES = {2}  # aromatic only
# Types that support extended-point scoring (HBA/HBD)
EXTENDED_POINT_TYPES = {0, 1}  # acceptor, donor


def _similarity_from_overlaps(
    VAB: torch.Tensor,
    VAA: torch.Tensor,
    VBB: torch.Tensor,
    *,
    similarity: str,
) -> torch.Tensor:
    similarity = similarity.lower()
    if similarity == "tanimoto":
        denom = VAA + VBB - VAB
        return torch.where(denom > 1e-8, VAB / denom, torch.zeros_like(VAB))
    if similarity == "tversky":
        sigma = 0.95
    elif similarity == "tversky_ref":
        sigma = 1.0
    elif similarity == "tversky_fit":
        sigma = 0.05
    else:
        raise ValueError(
            "similarity must be one of ('tanimoto','tversky','tversky_ref','tversky_fit')"
        )
    denom = sigma * VAA + (1.0 - sigma) * VBB
    score = torch.where(denom > 1e-8, VAB / denom, torch.zeros_like(VAB))
    return torch.clamp_max(score, max=1.0)


def pharm_similarity_from_overlaps(
    VAB: torch.Tensor,
    VAA: torch.Tensor,
    VBB: torch.Tensor,
    *,
    similarity: str,
) -> torch.Tensor:
    """
    Compute pharmacophore similarity from overlap terms.

    Exposed helper for fast coarse/fine optimizers so they can reuse precomputed self-overlaps.
    """
    return _similarity_from_overlaps(VAB, VAA, VBB, similarity=similarity)


def _gaussian_overlap_kernel(r2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian overlap: k * exp(-alpha/2 * r^2)
    where k = (pi / (2*alpha))^1.5

    Parameters
    ----------
    r2 : torch.Tensor (..., N, M)
        Squared distances
    alpha : torch.Tensor (..., N, M) or scalar
        Alpha values (can be type-specific)

    Returns
    -------
    overlap : torch.Tensor (..., N, M)
        Gaussian overlap values
    """
    k_const = (math.pi / (2.0 * alpha)) ** 1.5
    return k_const * torch.exp(-0.5 * alpha * r2)


def _batch_pharm_overlap_typed(
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        vectors_1: torch.Tensor,
        vectors_2: torch.Tensor,
        types_1: torch.Tensor,
        types_2: torch.Tensor,
        *,
        extended_points: bool = False,
        only_extended: bool = False,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batched pharmacophore overlaps with type matching and vector weighting.

    Parameters
    ----------
    anchors_1 : torch.Tensor (B, N, 3)
        Reference pharmacophore anchor positions
    anchors_2 : torch.Tensor (B, M, 3)
        Fit pharmacophore anchor positions
    vectors_1 : torch.Tensor (B, N, 3)
        Reference pharmacophore vectors (normalized)
    vectors_2 : torch.Tensor (B, M, 3)
        Fit pharmacophore vectors (normalized)
    types_1 : torch.Tensor (B, N) int
        Reference pharmacophore type indices
    types_2 : torch.Tensor (B, M) int
        Fit pharmacophore type indices
    N_real, M_real : torch.Tensor (B,) optional
        Real point counts for padded batches

    Returns
    -------
    VAB : torch.Tensor (B,)
        Cross-overlap scores
    VAA : torch.Tensor (B,)
        Reference self-overlap scores
    VBB : torch.Tensor (B,)
        Fit self-overlap scores
    """
    device = anchors_1.device
    dtype = anchors_1.dtype
    B, N, _ = anchors_1.shape
    _, M, _ = anchors_2.shape

    # Move alpha values to device
    alphas = ALPHA_VALUES.to(device=device, dtype=dtype)

    # Normalize vectors
    vectors_1 = F.normalize(vectors_1, p=2, dim=-1)
    vectors_2 = F.normalize(vectors_2, p=2, dim=-1)

    # Create masks for valid points
    if N_real is None:
        mask_1 = torch.ones(B, N, device=device, dtype=dtype)
    else:
        idx = torch.arange(N, device=device).unsqueeze(0)
        mask_1 = (idx < N_real.unsqueeze(1)).float()

    if M_real is None:
        mask_2 = torch.ones(B, M, device=device, dtype=dtype)
    else:
        idx = torch.arange(M, device=device).unsqueeze(0)
        mask_2 = (idx < M_real.unsqueeze(1)).float()

    # Initialize accumulators
    VAB = torch.zeros(B, device=device, dtype=dtype)
    VAA = torch.zeros(B, device=device, dtype=dtype)
    VBB = torch.zeros(B, device=device, dtype=dtype)

    # Process each pharmacophore type
    for ptype_idx in range(len(P_TYPES)):
        alpha = alphas[ptype_idx]
        use_vectors = ptype_idx in VECTOR_TYPES
        allow_antiparallel = ptype_idx in ANTIPARALLEL_TYPES
        use_extended = extended_points and (ptype_idx in EXTENDED_POINT_TYPES)

        # Type masks
        type_mask_1 = (types_1 == ptype_idx).float() * mask_1  # (B, N)
        type_mask_2 = (types_2 == ptype_idx).float() * mask_2  # (B, M)

        # Skip if no features of this type
        if type_mask_1.sum() == 0 and type_mask_2.sum() == 0:
            continue

        if use_extended:
            # Extended-point semantics (legacy): anchor overlap + (anchor+vector) overlap,
            # with no cosine weighting. `only_extended` drops the anchor term.
            if not only_extended:
                diff_12 = anchors_1.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, N, M, 3)
                r2_12 = (diff_12 ** 2).sum(dim=-1)  # (B, N, M)
                g_12 = _gaussian_overlap_kernel(r2_12, alpha)
                pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
                VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

                diff_11 = anchors_1.unsqueeze(2) - anchors_1.unsqueeze(1)  # (B, N, N, 3)
                r2_11 = (diff_11 ** 2).sum(dim=-1)
                g_11 = _gaussian_overlap_kernel(r2_11, alpha)
                pair_mask_11 = type_mask_1.unsqueeze(2) * type_mask_1.unsqueeze(1)
                VAA += (g_11 * pair_mask_11).sum(dim=(1, 2))

                diff_22 = anchors_2.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, M, M, 3)
                r2_22 = (diff_22 ** 2).sum(dim=-1)
                g_22 = _gaussian_overlap_kernel(r2_22, alpha)
                pair_mask_22 = type_mask_2.unsqueeze(2) * type_mask_2.unsqueeze(1)
                VBB += (g_22 * pair_mask_22).sum(dim=(1, 2))

            ext_1 = anchors_1 + vectors_1
            ext_2 = anchors_2 + vectors_2

            diff_12 = ext_1.unsqueeze(2) - ext_2.unsqueeze(1)  # (B, N, M, 3)
            r2_12 = (diff_12 ** 2).sum(dim=-1)  # (B, N, M)
            g_12 = _gaussian_overlap_kernel(r2_12, alpha)
            pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
            VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

            diff_11 = ext_1.unsqueeze(2) - ext_1.unsqueeze(1)
            r2_11 = (diff_11 ** 2).sum(dim=-1)
            g_11 = _gaussian_overlap_kernel(r2_11, alpha)
            pair_mask_11 = type_mask_1.unsqueeze(2) * type_mask_1.unsqueeze(1)
            VAA += (g_11 * pair_mask_11).sum(dim=(1, 2))

            diff_22 = ext_2.unsqueeze(2) - ext_2.unsqueeze(1)
            r2_22 = (diff_22 ** 2).sum(dim=-1)
            g_22 = _gaussian_overlap_kernel(r2_22, alpha)
            pair_mask_22 = type_mask_2.unsqueeze(2) * type_mask_2.unsqueeze(1)
            VBB += (g_22 * pair_mask_22).sum(dim=(1, 2))
            # Self overlaps already accumulated for this ptype; skip the generic self-overlap block.
            continue
        else:
            # Compute squared distances for cross-overlap (1 vs 2)
            diff_12 = anchors_1.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, N, M, 3)
            r2_12 = (diff_12 ** 2).sum(dim=-1)  # (B, N, M)

            # Compute Gaussian overlap
            g_12 = _gaussian_overlap_kernel(r2_12, alpha)

            # Apply vector weighting if needed
            if use_vectors:
                cos_sim = torch.einsum('bni,bmi->bnm', vectors_1, vectors_2)

                if allow_antiparallel:
                    weight = (torch.abs(cos_sim) + 2.0) / 3.0
                else:
                    weight = (cos_sim.clamp(min=0) + 2.0) / 3.0

                g_12 = g_12 * weight

            pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
            VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

        # Self-overlap for reference (1 vs 1)
        diff_11 = anchors_1.unsqueeze(2) - anchors_1.unsqueeze(1)  # (B, N, N, 3)
        r2_11 = (diff_11 ** 2).sum(dim=-1)
        g_11 = _gaussian_overlap_kernel(r2_11, alpha)

        if use_vectors:
            cos_sim_11 = torch.einsum('bni,bmi->bnm', vectors_1, vectors_1)
            if allow_antiparallel:
                weight_11 = (torch.abs(cos_sim_11) + 2.0) / 3.0
            else:
                weight_11 = (cos_sim_11.clamp(min=0) + 2.0) / 3.0
            g_11 = g_11 * weight_11

        pair_mask_11 = type_mask_1.unsqueeze(2) * type_mask_1.unsqueeze(1)
        VAA += (g_11 * pair_mask_11).sum(dim=(1, 2))

        # Self-overlap for fit (2 vs 2)
        diff_22 = anchors_2.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, M, M, 3)
        r2_22 = (diff_22 ** 2).sum(dim=-1)
        g_22 = _gaussian_overlap_kernel(r2_22, alpha)

        if use_vectors:
            cos_sim_22 = torch.einsum('bni,bmi->bnm', vectors_2, vectors_2)
            if allow_antiparallel:
                weight_22 = (torch.abs(cos_sim_22) + 2.0) / 3.0
            else:
                weight_22 = (cos_sim_22.clamp(min=0) + 2.0) / 3.0
            g_22 = g_22 * weight_22

        pair_mask_22 = type_mask_2.unsqueeze(2) * type_mask_2.unsqueeze(1)
        VBB += (g_22 * pair_mask_22).sum(dim=(1, 2))

    return VAB, VAA, VBB


def _batch_pharm_cross_overlap_typed(
    anchors_1: torch.Tensor,
    anchors_2: torch.Tensor,
    vectors_1: torch.Tensor,
    vectors_2: torch.Tensor,
    types_1: torch.Tensor,
    types_2: torch.Tensor,
    *,
    extended_points: bool = False,
    only_extended: bool = False,
    N_real: Optional[torch.Tensor] = None,
    M_real: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute batched pharmacophore *cross* overlaps only (VAB).

    This is significantly cheaper than computing (VAB,VAA,VBB) and is used by fast optimizers
    that reuse precomputed self overlaps.
    """
    device = anchors_1.device
    dtype = anchors_1.dtype
    B, N, _ = anchors_1.shape
    _, M, _ = anchors_2.shape

    alphas = ALPHA_VALUES.to(device=device, dtype=dtype)

    vectors_1 = F.normalize(vectors_1, p=2, dim=-1)
    vectors_2 = F.normalize(vectors_2, p=2, dim=-1)

    if N_real is None:
        mask_1 = torch.ones(B, N, device=device, dtype=dtype)
    else:
        idx = torch.arange(N, device=device).unsqueeze(0)
        mask_1 = (idx < N_real.unsqueeze(1)).float()

    if M_real is None:
        mask_2 = torch.ones(B, M, device=device, dtype=dtype)
    else:
        idx = torch.arange(M, device=device).unsqueeze(0)
        mask_2 = (idx < M_real.unsqueeze(1)).float()

    VAB = torch.zeros(B, device=device, dtype=dtype)

    for ptype_idx in range(len(P_TYPES)):
        alpha = alphas[ptype_idx]
        use_vectors = ptype_idx in VECTOR_TYPES
        allow_antiparallel = ptype_idx in ANTIPARALLEL_TYPES
        use_extended = extended_points and (ptype_idx in EXTENDED_POINT_TYPES)

        type_mask_1 = (types_1 == ptype_idx).float() * mask_1  # (B, N)
        type_mask_2 = (types_2 == ptype_idx).float() * mask_2  # (B, M)

        if type_mask_1.sum() == 0 and type_mask_2.sum() == 0:
            continue

        if use_extended:
            if not only_extended:
                diff_12 = anchors_1.unsqueeze(2) - anchors_2.unsqueeze(1)
                r2_12 = (diff_12 ** 2).sum(dim=-1)
                g_12 = _gaussian_overlap_kernel(r2_12, alpha)
                pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
                VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

            ext_1 = anchors_1 + vectors_1
            ext_2 = anchors_2 + vectors_2
            diff_12 = ext_1.unsqueeze(2) - ext_2.unsqueeze(1)
            r2_12 = (diff_12 ** 2).sum(dim=-1)
            g_12 = _gaussian_overlap_kernel(r2_12, alpha)
            pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
            VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))
            continue

        diff_12 = anchors_1.unsqueeze(2) - anchors_2.unsqueeze(1)
        r2_12 = (diff_12 ** 2).sum(dim=-1)
        g_12 = _gaussian_overlap_kernel(r2_12, alpha)

        if use_vectors:
            cos_sim = torch.einsum("bni,bmi->bnm", vectors_1, vectors_2)
            if allow_antiparallel:
                weight = (torch.abs(cos_sim) + 2.0) / 3.0
            else:
                weight = (cos_sim.clamp(min=0) + 2.0) / 3.0
            g_12 = g_12 * weight

        pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)
        VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

    return VAB


def batch_pharm_cross_overlap_with_transform(
    anchors_1: torch.Tensor,
    anchors_2: torch.Tensor,
    vectors_1: torch.Tensor,
    vectors_2: torch.Tensor,
    types_1: torch.Tensor,
    types_2: torch.Tensor,
    q: torch.Tensor,
    t: torch.Tensor,
    *,
    extended_points: bool = False,
    only_extended: bool = False,
    N_real: Optional[torch.Tensor] = None,
    M_real: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-overlap VAB after applying SE(3) transform to fit molecule.
    """
    from ._common import apply_se3_transform, apply_so3_transform

    anchors_2_t = apply_se3_transform(anchors_2, q, t)
    vectors_2_t = apply_so3_transform(vectors_2, q)

    return _batch_pharm_cross_overlap_typed(
        anchors_1,
        anchors_2_t,
        vectors_1,
        vectors_2_t,
        types_1,
        types_2,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=N_real,
        M_real=M_real,
    )


def batch_pharm_self_overlap(
        anchors: torch.Tensor,
        vectors: torch.Tensor,
        types: torch.Tensor,
        *,
        extended_points: bool = False,
        only_extended: bool = False,
        N_real: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute batched pharmacophore self-overlap.

    Returns
    -------
    V : torch.Tensor (B,)
        Self-overlap scores
    """
    _, V, _ = _batch_pharm_overlap_typed(
        anchors, anchors,
        vectors, vectors,
        types, types,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=N_real,
        M_real=N_real)

    return V


def batch_pharm_similarity(
    anchors_1: torch.Tensor,
    anchors_2: torch.Tensor,
    vectors_1: torch.Tensor,
    vectors_2: torch.Tensor,
    types_1: torch.Tensor,
    types_2: torch.Tensor,
    *,
    similarity: str = "tanimoto",
    extended_points: bool = False,
    only_extended: bool = False,
    N_real: Optional[torch.Tensor] = None,
    M_real: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute pharmacophore similarity (tanimoto/tversky variants) in batch.

    This matches the semantics of `score.pharmacophore_scoring.get_overlap_pharm` for:
    - similarity mode
    - extended_points / only_extended (acceptor/donor)
    """
    VAB, VAA, VBB = _batch_pharm_overlap_typed(
        anchors_1,
        anchors_2,
        vectors_1,
        vectors_2,
        types_1,
        types_2,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=N_real,
        M_real=M_real,
    )
    return _similarity_from_overlaps(VAB, VAA, VBB, similarity=similarity)
