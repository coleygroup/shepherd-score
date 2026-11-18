"""
Atom-identity (categorical) Gaussian overlap scoring with PyTorch.

Scoring channel for the ``vol_atomtype`` alignment mode: a Gaussian volume overlap partitioned by
a categorical per-atom label (the atomic number), so only atoms of the same element contribute to
the cross overlap. It is the pharmacophore color construction (per-type masked overlap, summed
over types, then a self-normalised Tanimoto or Tversky) over an arbitrary integer label. The
primitive is :func:`shepherd_score.score.gaussian_overlap.VAB_2nd_order`, so the width convention
matches the ``vol`` channel (``alpha=0.81``). Labels are compared by equality only.
"""
from typing import Literal, Union

import numpy as np
import torch

from shepherd_score.score.gaussian_overlap import VAB_2nd_order

_SIM_TYPE = Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit']


def _sigma_for(similarity: str) -> Union[float, None]:
    """Tversky ``sigma`` (weight on the reference self-overlap) for a similarity name, or ``None``
    for symmetric Tanimoto; the same mapping as :func:`pharmacophore_scoring.get_overlap_pharm`."""
    s = similarity.lower()
    if s == 'tanimoto':
        return None
    if s == 'tversky':
        return 0.95
    if s == 'tversky_ref':
        return 1.0
    if s == 'tversky_fit':
        return 0.05
    raise ValueError('Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')


def get_overlap_atomtype(labels_1: torch.Tensor,
                         labels_2: torch.Tensor,
                         centers_1: torch.Tensor,
                         centers_2: torch.Tensor,
                         alpha: float = 0.81,
                         similarity: _SIM_TYPE = 'tanimoto',
                         ) -> torch.Tensor:
    """
    Compute the atom-identity (categorical) Gaussian overlap score.

    Only atoms sharing the same label (e.g. atomic number) contribute to the cross overlap. The
    per-label cross and self overlaps are summed over labels, then reduced to a Tanimoto or
    Tversky similarity::

        Tanimoto = Σ_t VAB_t / (Σ_t VAA_t + Σ_t VBB_t - Σ_t VAB_t)
        Tversky  = Σ_t VAB_t / (sigma·Σ_t VAA_t + (1-sigma)·Σ_t VBB_t)   (clamped to 1)

    A label present in only one molecule contributes to that molecule's self-overlap (the
    denominator) but not to the cross overlap, as in the pharmacophore color convention. Labels
    are used for equality masking only. Batched ``(B,N,3)`` centers with 1-D labels are supported
    (the multi-start optimiser repeats the poses, not the labels).

    Parameters
    ----------
    labels_1 : torch.Tensor (N,)
        Integer-coded categorical label per reference atom (e.g. atomic number).
    labels_2 : torch.Tensor (M,)
        Integer-coded categorical label per fit atom.
    centers_1 : torch.Tensor (N,3) or (B,N,3)
        Reference atom coordinates.
    centers_2 : torch.Tensor (M,3) or (B,M,3)
        Fit atom coordinates (already SE(3)-transformed by the caller).
    alpha : float, optional
        Gaussian width for the overlap. Default 0.81 (volumetric, heavy atoms).
    similarity : str, optional
        'tanimoto' (default, symmetric) or a Tversky variant ('tversky', 'tversky_ref',
        'tversky_fit').

    Returns
    -------
    torch.Tensor
        Score(s): scalar for single-instance input, shape ``(B,)`` for batched centers.
    """
    if isinstance(labels_1, np.ndarray):
        labels_1 = torch.as_tensor(labels_1)
    if isinstance(labels_2, np.ndarray):
        labels_2 = torch.as_tensor(labels_2)

    batched = centers_1.dim() == 3
    # Running totals; batched centers keep a (B,) vector because VAB_2nd_order then returns (B,).
    VAB = torch.zeros(centers_1.shape[0], device=centers_1.device) if batched else torch.zeros((), device=centers_1.device)
    VAA = torch.zeros_like(VAB)
    VBB = torch.zeros_like(VAB)

    # Union of labels present in either molecule; each contributes to the relevant self-overlap sum.
    unique_labels = torch.unique(torch.cat((labels_1.reshape(-1), labels_2.reshape(-1))))
    for lbl in unique_labels:
        idx1 = torch.where(labels_1 == lbl)[0]
        idx2 = torch.where(labels_2 == lbl)[0]
        has1 = idx1.numel() > 0
        has2 = idx2.numel() > 0
        if has1:
            c1 = centers_1[..., idx1, :]
            VAA = VAA + VAB_2nd_order(c1, c1, alpha)
        if has2:
            c2 = centers_2[..., idx2, :]
            VBB = VBB + VAB_2nd_order(c2, c2, alpha)
        if has1 and has2:
            VAB = VAB + VAB_2nd_order(centers_1[..., idx1, :], centers_2[..., idx2, :], alpha)

    sigma = _sigma_for(similarity)
    if sigma is None:
        return VAB / (VAA + VBB - VAB)
    return torch.clamp_max(VAB / (sigma * VAA + (1 - sigma) * VBB), max=1.0)
