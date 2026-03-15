"""
Alignment algorithms using analytical gradients (no PyTorch autograd).
These functions replace the autograd-based counterparts in _torch.py and
provide ~1.5-2.5x speedup by injecting hand-derived gradients into PyTorch's
Adam optimizer.
"""
from typing import Union, Tuple, Optional

import torch
from torch import optim

from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm, _SIM_TYPE
from shepherd_score.alignment.utils.se3 import get_SE3_transform, apply_SE3_transform, apply_SO3_transform
from shepherd_score.alignment._torch import (
    _initialize_se3_params,
    _initialize_se3_params_with_translations,
    score_ROCS_overlay_with_avoid,
)


def optimize_ROCS_overlay_analytical(ref_points: torch.Tensor,
                                      fit_points: torch.Tensor,
                                      alpha: float,
                                      *,
                                      fit_points_for_avoid: Optional[torch.Tensor] = None,
                                      avoid_points: Optional[torch.Tensor] = None,
                                      avoid_min_dist: float = 2.0,
                                      avoid_weight: float = 1.0,
                                      num_repeats: int = 50,
                                      trans_centers: Union[torch.Tensor, None] = None,
                                      lr: float = 0.1,
                                      max_num_steps: int = 200,
                                      verbose: bool = False
                                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize shape alignment using analytical gradients instead of autograd.

    Same interface and behavior as ``optimize_ROCS_overlay``, but uses hand-derived
    analytical gradients with a manual Adam optimizer, eliminating PyTorch autograd overhead.

    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
    fit_points : torch.Tensor (M,3)
    alpha : float
    fit_points_for_avoid : torch.Tensor (M2,3) or None
        Points to penalize for overlap with avoid_points. Defaults to fit_points if None.
    avoid_points : torch.Tensor (K,3) or None
        Fixed points to avoid overlapping with.
    avoid_min_dist : float
        Distance threshold for avoid penalty.
    avoid_weight : float
        Weight of the avoid penalty term.
    num_repeats : int
    trans_centers : torch.Tensor or None
    lr : float
    max_num_steps : int
    verbose : bool

    Returns
    -------
    tuple of (aligned_points, SE3_transform, score)
    """
    from shepherd_score.score.analytical_gradients import (
        compute_self_overlaps_shape,
        compute_analytical_grad_se3_shape,
        compute_analytical_grad_se3_shape_with_avoid,
    )

    # Initialize SE(3) parameters (requires_grad=True so PyTorch Adam can manage them)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_points,
            fit_points=fit_points,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    # Precompute self-overlaps (invariant to SE(3))
    VAA_total, VBB_total = compute_self_overlaps_shape(ref_points, fit_points, alpha)

    # Resolve avoid points
    if avoid_points is not None and fit_points_for_avoid is None:
        fit_points_for_avoid = fit_points

    # Replicate data for batched computation
    if num_repeats > 1:
        ref_points_rep = ref_points.repeat((num_repeats, 1, 1)).squeeze(0)
        fit_points_rep = fit_points.repeat((num_repeats, 1, 1)).squeeze(0)
        if avoid_points is not None:
            fit_pts_avoid_rep = fit_points_for_avoid.repeat((num_repeats, 1, 1)).squeeze(0)
    else:
        ref_points_rep = ref_points
        fit_points_rep = fit_points
        if avoid_points is not None:
            fit_pts_avoid_rep = fit_points_for_avoid

    if verbose:
        print(f'Initial shape similarity score: {get_overlap(ref_points, fit_points, alpha):.3f}')

    # Use PyTorch's Adam (same implementation as optimize_ROCS_overlay) with manually-set
    # gradients from the analytical computation.  This avoids float32 drift from a
    # hand-rolled Adam and keeps optimizer trajectories comparable.
    optimizer = optim.Adam([se3_params], lr=lr)

    last_loss = 1.0
    counter = 0

    for step in range(max_num_steps):
        with torch.no_grad():
            if avoid_points is not None:
                loss, grad = compute_analytical_grad_se3_shape_with_avoid(
                    se3_params, ref_points_rep, fit_points_rep, alpha, VAA_total, VBB_total,
                    fit_pts_avoid_rep, avoid_points, avoid_min_dist, avoid_weight
                )
            else:
                loss, grad = compute_analytical_grad_se3_shape(
                    se3_params, ref_points_rep, fit_points_rep, alpha, VAA_total, VBB_total
                )

        optimizer.zero_grad()
        se3_params.grad = grad
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1 - loss.item()}")

        # Early stopping
        if abs(loss.item() - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss.item()
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(se3_params)
    aligned_points = apply_SE3_transform(fit_points_rep, SE3_transform)

    if avoid_points is not None:
        aligned_pts_for_avoid = apply_SE3_transform(fit_pts_avoid_rep, SE3_transform)
        scores = score_ROCS_overlay_with_avoid(
            ref_points=ref_points_rep,
            fit_points=aligned_points,
            alpha=alpha,
            fit_points_for_avoid=aligned_pts_for_avoid,
            avoid_points=avoid_points,
            avoid_min_dist=avoid_min_dist,
            avoid_weight=avoid_weight,
        )
    else:
        scores = get_overlap(centers_1=ref_points_rep, centers_2=aligned_points, alpha=alpha)

    if num_repeats == 1:
        if verbose:
            print(f'Optimized shape similarity score: {scores:.3f}')
        best_alignment = aligned_points.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        if verbose:
            print(f'Optimized shape similarity score -- max: {scores.max():.3f} | mean: {scores.mean():.3f} | min: {scores.min():.3f}')
        best_idx = torch.argmax(scores.detach().cpu())
        best_alignment = aligned_points.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]

    return best_alignment, best_transform, best_score


def optimize_pharm_overlay_analytical(ref_pharms: torch.Tensor,
                                      fit_pharms: torch.Tensor,
                                      ref_anchors: torch.Tensor,
                                      fit_anchors: torch.Tensor,
                                      ref_vectors: torch.Tensor,
                                      fit_vectors: torch.Tensor,
                                      similarity: _SIM_TYPE = 'tanimoto',
                                      extended_points: bool = False,
                                      only_extended: bool = False,
                                      num_repeats: int = 50,
                                      trans_centers: Union[torch.Tensor, None] = None,
                                      lr: float = 0.1,
                                      max_num_steps: int = 200,
                                      verbose: bool = False
                                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize pharmacophore alignment using analytical gradients instead of autograd.

    Same interface and behavior as ``optimize_pharm_overlay``, but uses hand-derived
    analytical gradients with PyTorch's Adam optimizer, eliminating PyTorch autograd overhead.

    Supports ``similarity='tanimoto'``, ``'tversky'``, ``'tversky_ref'``, and
    ``'tversky_fit'``. Does not support ``extended_points=True``.

    Parameters
    ----------
    ref_pharms : torch.Tensor (N,)
    fit_pharms : torch.Tensor (M,)
    ref_anchors : torch.Tensor (N,3)
    fit_anchors : torch.Tensor (M,3)
    ref_vectors : torch.Tensor (N,3)
    fit_vectors : torch.Tensor (M,3)
    similarity : str
    extended_points : bool
    only_extended : bool
    num_repeats : int
    trans_centers : torch.Tensor or None
    lr : float
    max_num_steps : int
    verbose : bool

    Returns
    -------
    tuple of (aligned_anchors, aligned_vectors, SE3_transform, score)
    """
    from shepherd_score.score.analytical_gradients import (
        compute_self_overlaps_pharm,
        compute_analytical_grad_se3,
    )

    _SIGMA_MAP = {'tversky': 0.95, 'tversky_ref': 1.0, 'tversky_fit': 0.05}
    sim_lower = similarity.lower()
    if sim_lower in _SIGMA_MAP:
        sigma = _SIGMA_MAP[sim_lower]
    elif sim_lower == 'tanimoto':
        sigma = 0.5  # unused for tanimoto
    else:
        raise ValueError(
            f"Unknown similarity '{similarity}'. "
            "Expected one of: 'tanimoto', 'tversky', 'tversky_ref', 'tversky_fit'."
        )

    if extended_points:
        raise NotImplementedError(
            "Analytical gradients do not support extended_points=True."
        )

    # Initialize SE(3) parameters (without requires_grad)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_anchors, fit_points=fit_anchors, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_anchors,
            fit_points=fit_anchors,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    # Precompute self-overlaps (invariant to SE(3))
    VAA_total, VBB_total = compute_self_overlaps_pharm(
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors
    )

    # Replicate data for batched computation
    if num_repeats > 1:
        ref_pharms_rep = ref_pharms.repeat((num_repeats, 1)).squeeze(0)
        fit_pharms_rep = fit_pharms.repeat((num_repeats, 1)).squeeze(0)
        ref_anchors_rep = ref_anchors.repeat((num_repeats, 1, 1)).squeeze(0)
        fit_anchors_rep = fit_anchors.repeat((num_repeats, 1, 1)).squeeze(0)
        ref_vectors_rep = ref_vectors.repeat((num_repeats, 1, 1)).squeeze(0)
        fit_vectors_rep = fit_vectors.repeat((num_repeats, 1, 1)).squeeze(0)
    else:
        ref_pharms_rep = ref_pharms
        fit_pharms_rep = fit_pharms
        ref_anchors_rep = ref_anchors
        fit_anchors_rep = fit_anchors
        ref_vectors_rep = ref_vectors
        fit_vectors_rep = fit_vectors

    if verbose:
        init_score = get_overlap_pharm(
            ref_pharms, fit_pharms, ref_anchors, fit_anchors,
            ref_vectors, fit_vectors, similarity=similarity
        )
        print(f'Initial pharmacophore similarity score: {init_score:.3f}')

    # Use PyTorch's Adam (same implementation as optimize_pharm_overlay) with manually-set
    # gradients from the analytical computation.
    optimizer = optim.Adam([se3_params], lr=lr)

    last_loss = 1.0
    counter = 0

    for step in range(max_num_steps):
        with torch.no_grad():
            loss, grad = compute_analytical_grad_se3(
                se3_params, ref_pharms_rep, fit_pharms_rep,
                ref_anchors_rep, fit_anchors_rep, ref_vectors_rep, fit_vectors_rep,
                VAA_total, VBB_total,
                similarity=sim_lower, sigma=sigma,
            )

        optimizer.zero_grad()
        se3_params.grad = grad
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1 - loss.item()}")

        # Early stopping
        if abs(loss.item() - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss.item()
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(se3_params)
    aligned_anchors = apply_SE3_transform(fit_anchors_rep, SE3_transform)
    aligned_vectors = apply_SO3_transform(fit_vectors_rep, SE3_transform)
    scores = get_overlap_pharm(
        ptype_1=ref_pharms_rep,
        ptype_2=fit_pharms_rep,
        anchors_1=ref_anchors_rep,
        anchors_2=aligned_anchors,
        vectors_1=ref_vectors_rep,
        vectors_2=aligned_vectors,
        similarity=similarity
    )

    if num_repeats == 1:
        if verbose:
            print(f'Optimized pharmacophore similarity score: {scores:.3f}')
        best_alignment = aligned_anchors.cpu()
        best_aligned_vectors = aligned_vectors.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        if verbose:
            print(f'Optimized pharmacophore similarity score -- max: {scores.max():.3f} | mean: {scores.mean():.3f} | min: {scores.min():.3f}')
        best_idx = torch.argmax(scores.detach().cpu())
        best_alignment = aligned_anchors.cpu()[best_idx]
        best_aligned_vectors = aligned_vectors.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]

    return best_alignment, best_aligned_vectors, best_transform, best_score


def optimize_ROCS_esp_overlay_analytical(
    ref_points: torch.Tensor,
    fit_points: torch.Tensor,
    ref_charges: torch.Tensor,
    fit_charges: torch.Tensor,
    alpha: float,
    lam: float,
    *,
    num_repeats: int = 50,
    trans_centers: Union[torch.Tensor, None] = None,
    lr: float = 0.1,
    max_num_steps: int = 200,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize ESP alignment using analytical gradients instead of autograd.

    Same interface and behavior as ``optimize_ROCS_esp_overlay``, but uses hand-derived
    analytical gradients with a manual Adam optimizer.

    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
    fit_points : torch.Tensor (M,3)
    ref_charges : torch.Tensor (N,)
    fit_charges : torch.Tensor (M,)
    alpha : float
    lam : float
        Pre-scaled lam (e.g. LAM_SCALING * lam_user).
    num_repeats : int
    trans_centers : torch.Tensor or None
    lr : float
    max_num_steps : int
    verbose : bool

    Returns
    -------
    tuple of (aligned_points, SE3_transform, score)
    """
    from shepherd_score.score.analytical_gradients import (
        compute_self_overlaps_esp,
        compute_analytical_grad_se3_esp,
    )

    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_points,
            fit_points=fit_points,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    # Precompute self-overlaps (invariant to SE(3))
    VAA_total, VBB_total = compute_self_overlaps_esp(
        ref_points, fit_points, ref_charges, fit_charges, alpha, lam
    )

    # Replicate data for batched computation
    if num_repeats > 1:
        ref_points_rep = ref_points.repeat((num_repeats, 1, 1)).squeeze(0)
        fit_points_rep = fit_points.repeat((num_repeats, 1, 1)).squeeze(0)
    else:
        ref_points_rep = ref_points
        fit_points_rep = fit_points

    if verbose:
        print(f'Initial ESP similarity score: {get_overlap_esp(ref_points, fit_points, ref_charges, fit_charges, alpha, lam):.3f}')

    optimizer = optim.Adam([se3_params], lr=lr)

    last_loss = 1.0
    counter = 0

    for step in range(max_num_steps):
        with torch.no_grad():
            loss, grad = compute_analytical_grad_se3_esp(
                se3_params, ref_points_rep, fit_points_rep,
                ref_charges, fit_charges, alpha, lam, VAA_total, VBB_total
            )

        optimizer.zero_grad()
        se3_params.grad = grad
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1 - loss.item()}")

        if abs(loss.item() - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss.item()
        if counter > 10:
            break

    se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(se3_params)
    aligned_points = apply_SE3_transform(fit_points_rep, SE3_transform)

    # Replicate charges for batched scoring
    if num_repeats > 1:
        fit_charges_rep = fit_charges.unsqueeze(0).expand(num_repeats, -1)
    else:
        fit_charges_rep = fit_charges

    scores = get_overlap_esp(
        centers_1=ref_points,
        charges_1=ref_charges,
        centers_2=aligned_points,
        charges_2=fit_charges_rep,
        alpha=alpha,
        lam=lam,
    )

    if num_repeats == 1:
        if verbose:
            print(f'Optimized ESP similarity score: {scores.item():.3f}')
        best_alignment = aligned_points.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        best_idx = torch.argmax(scores.detach().cpu())
        if verbose:
            print(f'Optimized ESP similarity score -- max: {scores[best_idx].item():.3f} | mean: {scores.mean().item():.3f} | min: {scores.min().item():.3f}')
        best_alignment = aligned_points.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]

    return best_alignment, best_transform, best_score
