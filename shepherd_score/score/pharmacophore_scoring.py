"""
Pharmacophore scoring with PyTorch.
"""
from typing import Union, Callable, Literal, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from shepherd_score.score.gaussian_overlap import VAB_2nd_order, VAB_2nd_order_mask, VAB_2nd_order_cosine
from shepherd_score.score.gaussian_overlap import VAB_2nd_order_mask_batch, VAB_2nd_order_cosine_mask_batch
from shepherd_score.score.constants import P_TYPES, P_ALPHAS
P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))


def _compute_ref_overlap(overlap_func: Callable,
                         anchors_1: torch.Tensor,
                         alpha: float,
                         vectors_1: Union[torch.Tensor, None] = None,
                         allow_antiparallel: bool = False,
                         mask_1: Union[torch.Tensor, None] = None,
                         ) -> torch.Tensor:
    """ Both single instance and batched """
    # Just anchor volume overlap
    if (vectors_1 is None) and (mask_1 is None):
        VAA = overlap_func(anchors_1, anchors_1, alpha)
    # Just anchor volume overlap with masking for batching
    elif (vectors_1 is None) and (mask_1 is not None):
        VAA = overlap_func(anchors_1, anchors_1, alpha, mask_1=mask_1, mask_2=mask_1)
    # Anchor and vector volume overlap with masking for batching
    elif mask_1 is not None:
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel, mask_1=mask_1, mask_2=mask_1)
    # Anchor and vector volume overlap for single instance
    else:
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel)
    return VAA


def _compute_fit_overlap(overlap_func: Callable,
                         anchors_2: torch.Tensor,
                         alpha: float,
                         vectors_2: Union[torch.Tensor, None]  = None,
                         allow_antiparallel: bool = False,
                         mask_2: Union[torch.Tensor, None] = None
                         ) -> torch.Tensor:
    """ Both single instance and batched """
    # Just anchor volume overlap
    if (vectors_2 is None) and (mask_2 is None):
        VBB = overlap_func(anchors_2, anchors_2, alpha)
    # Just anchor volume overlap with masking for batching
    elif (vectors_2 is None) and (mask_2 is not None):
        VBB = overlap_func(anchors_2, anchors_2, alpha, mask_1=mask_2, mask_2=mask_2)
    # Anchor and vector volume overlap with masking for batching
    elif mask_2 is not None:
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel, mask_1=mask_2, mask_2=mask_2)
    # Anchor and vector volume overlap for single instance
    else:
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel)
    return VBB


def _compute_all_overlaps(overlap_func: Callable,
                          anchors_1: torch.Tensor,
                          anchors_2: torch.Tensor,
                          alpha: float,
                          vectors_1: Union[torch.Tensor, None] = None,
                          vectors_2: Union[torch.Tensor, None]  = None,
                          allow_antiparallel: bool = False,
                          mask_1: Union[torch.Tensor, None] = None,
                          mask_2: Union[torch.Tensor, None] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Handles both single instance and batched (inefficiently) """
    # Just anchor volume overlap
    if (vectors_1 is None or vectors_2 is None) and (mask_1 is None or mask_2 is None):
        VAB = overlap_func(anchors_1, anchors_2, alpha)
        VAA = overlap_func(anchors_1, anchors_1, alpha)
        VBB = overlap_func(anchors_2, anchors_2, alpha)
    # Just anchor volume overlap with masking for batching
    elif (vectors_1 is None or vectors_2 is None) and (mask_1 is not None or mask_2 is not None):
        VAB = overlap_func(anchors_1, anchors_2, alpha, mask_1=mask_1, mask_2=mask_2)
        VAA = overlap_func(anchors_1, anchors_1, alpha, mask_1=mask_1, mask_2=mask_1)
        VBB = overlap_func(anchors_2, anchors_2, alpha, mask_1=mask_2, mask_2=mask_2)
    # Anchor and vector volume overlap with masking for batching
    elif mask_1 is not None or mask_2 is not None:
        VAB = overlap_func(anchors_1, anchors_2, vectors_1, vectors_2, alpha, allow_antiparallel, mask_1=mask_1, mask_2=mask_2)
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel, mask_1=mask_1, mask_2=mask_1)
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel, mask_1=mask_2, mask_2=mask_2)
    # Anchor and vector volume overlap for single instance
    else:
        VAB = overlap_func(anchors_1, anchors_2, vectors_1, vectors_2, alpha, allow_antiparallel)
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel)
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel)
    return VAB, VAA, VBB


def _compute_all_overlaps_batch(overlap_func: Callable,
                                cdist_21,
                                cdist_22,
                                cdist_11,
                                alpha: float,
                                vmm_21: Union[torch.Tensor, None] = None,
                                vmm_22: Union[torch.Tensor, None] = None,
                                vmm_11: Union[torch.Tensor, None] = None,
                                allow_antiparallel: bool = False,
                                mask_1: Union[torch.Tensor, None] = None,
                                mask_2: Union[torch.Tensor, None] = None
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Only handles batched """
    # Just anchor volume overlap with masking for batching
    if (vmm_21 is None or vmm_22 is None or vmm_11 is None) and (mask_1 is not None or mask_2 is not None):
        VAB = overlap_func(cdist_21, alpha, mask_1=mask_1, mask_2=mask_2)
        VAA = overlap_func(cdist_11, alpha, mask_1=mask_1, mask_2=mask_1)
        VBB = overlap_func(cdist_22, alpha, mask_1=mask_2, mask_2=mask_2)
    # Anchor and vector volume overlap with masking for batching
    elif mask_1 is not None or mask_2 is not None:
        VAB = overlap_func(cdist_21, vmm_21, alpha, allow_antiparallel, mask_1=mask_1, mask_2=mask_2)
        VAA = overlap_func(cdist_11, vmm_11, alpha, allow_antiparallel, mask_1=mask_1, mask_2=mask_1)
        VBB = overlap_func(cdist_22, vmm_22, alpha, allow_antiparallel, mask_1=mask_2, mask_2=mask_2)
    return VAB, VAA, VBB


def tanimoto_func(VAB: torch.Tensor,
                  VAA: torch.Tensor,
                  VBB: torch.Tensor
                  ) -> torch.Tensor:
    """
    Computes Tanimoto similarity.
    Similarity(Tanimoto) = Overlap{1,2} / (Overlap{1,1} + Overlap{2,2} - Overlap{1,2})
    """
    return VAB/(VAA + VBB - VAB)


def tversky_func(VAB: torch.Tensor,
                 VAA: torch.Tensor,
                 VBB: torch.Tensor,
                 sigma: float
                 ) -> torch.Tensor:
    """
    Computes Tversky similarity -> clamped to be max of 1.0.
    sigma: [0,1]

    Similarity(Tversky) = Overlap{1,2} / (sigma*Overlap{1,1} + (1-sigma)*Overlap{2,2})
    """
    return torch.clamp_max(VAB/(sigma*VAA + (1-sigma)*VBB), max=1.0)


def get_vector_volume_overlap_score(ptype_str: str,
                                    ptype_1: torch.Tensor,
                                    ptype_2: torch.Tensor,
                                    anchors_1: torch.Tensor,
                                    anchors_2: torch.Tensor,
                                    vectors_1: torch.Tensor,
                                    vectors_2: torch.Tensor,
                                    allow_antiparallel: bool
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute volumentric overlap score with cosine similarity of vectors. Handles batching.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # single instance
    mask_inds_1 = torch.where(ptype_1 == ptype_idx)[0]
    mask_inds_2 = torch.where(ptype_2 == ptype_idx)[0]
    if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
        VAB, VAA, VBB = 0., 0., 0.
    elif len(mask_inds_1) == 0:
        VAB, VAA = 0., 0.
        VBB = _compute_fit_overlap(overlap_func=VAB_2nd_order_cosine,
                                    anchors_2=anchors_2[mask_inds_2],
                                    vectors_2=vectors_2[mask_inds_2],
                                    alpha = P_ALPHAS[ptype_str],
                                    allow_antiparallel = allow_antiparallel)
    elif len(mask_inds_2) == 0:
        VAB, VBB = 0., 0.
        VAA = _compute_ref_overlap(overlap_func=VAB_2nd_order_cosine,
                                    anchors_1=anchors_1[mask_inds_1],
                                    vectors_1=vectors_1[mask_inds_1],
                                    alpha = P_ALPHAS[ptype_str],
                                    allow_antiparallel = allow_antiparallel)
    else:
        VAB, VAA, VBB = _compute_all_overlaps(overlap_func=VAB_2nd_order_cosine,
                                                anchors_1=anchors_1[mask_inds_1],
                                                anchors_2=anchors_2[mask_inds_2],
                                                vectors_1=vectors_1[mask_inds_1],
                                                vectors_2=vectors_2[mask_inds_2],
                                                alpha = P_ALPHAS[ptype_str],
                                                allow_antiparallel = allow_antiparallel)
    return VAB, VAA, VBB


def get_vector_volume_overlap_score_batch(ptype_str: str,
                                          ptype_1: torch.Tensor,
                                          ptype_2: torch.Tensor,
                                          cdist_21: torch.Tensor,
                                          cdist_22: torch.Tensor,
                                          cdist_11: torch.Tensor,
                                          vmm_21: torch.Tensor,
                                          vmm_22: torch.Tensor,
                                          vmm_11: torch.Tensor,
                                          allow_antiparallel: bool = False
                                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute volumentric overlap score with cosine similarity of vectors. Only batching.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # batched
    mask_1 = torch.where(ptype_1 == ptype_idx, 1, 0)
    mask_2 = torch.where(ptype_2 == ptype_idx, 1, 0)
    VAB, VAA, VBB = _compute_all_overlaps_batch(overlap_func=VAB_2nd_order_cosine_mask_batch,
                                                cdist_21=cdist_21,
                                                cdist_22=cdist_22,
                                                cdist_11=cdist_11,
                                                vmm_21=vmm_21,
                                                vmm_22=vmm_22,
                                                vmm_11=vmm_11,
                                                alpha = P_ALPHAS[ptype_str],
                                                allow_antiparallel = allow_antiparallel,
                                                mask_1 = mask_1,
                                                mask_2 = mask_2)
    return VAB, VAA, VBB


def get_volume_overlap_score(ptype_str: str,
                             ptype_1: torch.Tensor,
                             ptype_2: torch.Tensor,
                             anchors_1: torch.Tensor,
                             anchors_2: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes volume overlap score single instance.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # single instance
    mask_inds_1 = torch.where(ptype_1 == ptype_idx)[0]
    mask_inds_2 = torch.where(ptype_2 == ptype_idx)[0]
    if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
        VAB, VAA, VBB = 0., 0., 0.
    elif len(mask_inds_1) == 0:
        VAB, VAA = 0., 0.
        VBB = _compute_fit_overlap(overlap_func=VAB_2nd_order,
                                    anchors_2=anchors_2[mask_inds_2],
                                    alpha = P_ALPHAS[ptype_str])
    elif len(mask_inds_2) == 0:
        VAB, VBB = 0., 0.
        VAA = _compute_ref_overlap(overlap_func=VAB_2nd_order,
                                    anchors_1=anchors_1[mask_inds_1],
                                    alpha = P_ALPHAS[ptype_str])
    else:
        VAB, VAA, VBB = _compute_all_overlaps(overlap_func=VAB_2nd_order,
                                                anchors_1=anchors_1[mask_inds_1],
                                                anchors_2=anchors_2[mask_inds_2],
                                                alpha = P_ALPHAS[ptype_str])
    return VAB, VAA, VBB


def get_volume_overlap_score_batch(ptype_str: str,
                                   ptype_1: torch.Tensor,
                                   ptype_2: torch.Tensor,
                                   cdist_21: torch.Tensor,
                                   cdist_22: torch.Tensor,
                                   cdist_11: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute volumentric overlap score with cosine similarity of vectors. Only batching.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # batched
    mask_1 = torch.where(ptype_1 == ptype_idx, 1, 0)
    mask_2 = torch.where(ptype_2 == ptype_idx, 1, 0)
    VAB, VAA, VBB = _compute_all_overlaps_batch(overlap_func=VAB_2nd_order_mask_batch,
                                                cdist_21=cdist_21,
                                                cdist_22=cdist_22,
                                                cdist_11=cdist_11,
                                                alpha = P_ALPHAS[ptype_str],
                                                mask_1 = mask_1,
                                                mask_2 = mask_2)
    return VAB, VAA, VBB


def get_volume_overlap_score_extended_points(ptype_str: str,
                                             ptype_1: torch.Tensor,
                                             ptype_2: torch.Tensor,
                                             anchors_1: torch.Tensor,
                                             anchors_2: torch.Tensor,
                                             vectors_1: torch.Tensor,
                                             vectors_2: torch.Tensor,
                                             only_extended: bool = False
                                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Score both the anchor and extended point volume overlap instead of a vector similarity.
    """    
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # single instance
    if len(ptype_1.shape) == 1 and len(ptype_2.shape) == 1:
        VAB, VAA, VBB = 0., 0., 0.
        mask_inds_1 = torch.where(ptype_1 == ptype_idx)[0]
        mask_inds_2 = torch.where(ptype_2 == ptype_idx)[0]
        if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
            VAB, VAA, VBB = 0., 0., 0.
        elif len(mask_inds_1) == 0:
            VAB, VAA = 0., 0.
            if not only_extended:
                VBB = _compute_fit_overlap(overlap_func=VAB_2nd_order,
                                           anchors_2=anchors_2[mask_inds_2],
                                           alpha = P_ALPHAS[ptype_str])
            else:
                VBB = 0.
            VBB += _compute_fit_overlap(overlap_func=VAB_2nd_order,
                                        anchors_2=vectors_2[mask_inds_2]+anchors_2[mask_inds_2],
                                        alpha = P_ALPHAS[ptype_str])
        elif len(mask_inds_2) == 0:
            VAB, VBB = 0., 0.
            if not only_extended:
                VAA = _compute_ref_overlap(overlap_func=VAB_2nd_order,
                                           anchors_1=anchors_1[mask_inds_1],
                                           alpha = P_ALPHAS[ptype_str])
            else:
                VAA = 0.
            VAA += _compute_ref_overlap(overlap_func=VAB_2nd_order,
                                        anchors_1=vectors_1[mask_inds_1]+anchors_1[mask_inds_1],
                                        alpha = P_ALPHAS[ptype_str])
        else:
            if not only_extended:
                VAB, VAA, VBB = _compute_all_overlaps(overlap_func=VAB_2nd_order,
                                                      anchors_1=anchors_1[mask_inds_1],
                                                      anchors_2=anchors_2[mask_inds_2],
                                                      alpha = P_ALPHAS[ptype_str])
            VAB_extended, VAA_extended, VBB_extended = _compute_all_overlaps(
                overlap_func=VAB_2nd_order,
                anchors_1=vectors_1[mask_inds_1]+anchors_1[mask_inds_1],
                anchors_2=vectors_2[mask_inds_2]+anchors_2[mask_inds_2],
                alpha = P_ALPHAS[ptype_str]
                )
            VAB += VAB_extended
            VAA += VAA_extended
            VBB += VBB_extended

    # batched
    elif len(ptype_1.shape) == 2 and len(ptype_2.shape) == 2:
        mask_1 = torch.where(ptype_1 == ptype_idx, 1, 0)
        mask_2 = torch.where(ptype_2 == ptype_idx, 1, 0)
        VAB = torch.zeros(ptype_1.shape[0], device=anchors_1.device)
        VAA = torch.zeros(ptype_1.shape[0], device=anchors_1.device)
        VBB = torch.zeros(ptype_1.shape[0], device=anchors_1.device)
        if not only_extended:
            VAB, VAA, VBB = _compute_all_overlaps(overlap_func=VAB_2nd_order_mask,
                                                  anchors_1=anchors_1,
                                                  anchors_2=anchors_2,
                                                  alpha = P_ALPHAS[ptype_str],
                                                  mask_1 = mask_1,
                                                  mask_2 = mask_2
                                                  )
        VAB_extended, VAA_extended, VBB_extended = _compute_all_overlaps(
            overlap_func = VAB_2nd_order_mask,
            anchors_1 = vectors_1+anchors_1,
            anchors_2 = vectors_2+anchors_2,
            alpha = P_ALPHAS[ptype_str],
            mask_1 = mask_1,
            mask_2 = mask_2
            )
        VAB += VAB_extended
        VAA += VAA_extended
        VBB += VBB_extended
    # otherwise
    else:
        raise ValueError(f"Arguments `ptype_1` and `ptype_2` must either be 1D or 2D (batched). Instead these shapes were given: {ptype_1.shape}, {ptype_2.shape}")
    return VAB, VAA, VBB

_SIM_TYPE = Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit']

def get_overlap_pharm(ptype_1: torch.Tensor,
                      ptype_2: torch.Tensor,
                      anchors_1: torch.Tensor,
                      anchors_2: torch.Tensor,
                      vectors_1: torch.Tensor,
                      vectors_2: torch.Tensor,
                      similarity: _SIM_TYPE = 'tanimoto',
                      extended_points: bool = False,
                      only_extended: bool = False
                      ) -> torch.Tensor:
    """
    Computes pharmacophore score.
    Accepts batching, but only if they are the same two molecules (or have the same number of features)
     --> Specifically used for alignment.

    Arguments
    ---------
    ptype_1 : torch.Tensor (N,) or (B,N)
        Indices specifying the pharmacophore type based on order of P_TYPES
    ptype_2 : torch.Tensor (M,) or (B,M)
        Indices specifying the pharmacophore type based on order of P_TYPES
    anchors_1 : torch.Tensor (N,3) or (B,N,3)
        Coordinates for the anchor points of each pharmacophore of molecule 1
    anchors_2 : torch.Tensor (M,3) or (B,M,3)
        Coordinates for the anchor points of each pharmacophore of molecule 2
    vectors_1 : torch.Tensor (N,3) or (B,N,3)
        Relative unit vectors of each pharmacophore of molecule 1
    vectors_2 : torch.Tensor (M,3) or (B,M,3)
        Relative unit vectors of each pharmacophore of molecule 2
    similarity : str
        Specifies what similarity function to use.
        'tanimoto' -- symmetric scoring function
        'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
        'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
        'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
    extended_points : bool
        Whether to score HBA/HBD with gaussian overlaps of extended points.
    only_extended : bool
        When `extended_points` is True, decide whether to only score the extended points (ignore
         anchor overlaps)

    Returns
    -------
    torch.Tensor (1,) or (B,) : score(s)
    """
    if isinstance(ptype_1, np.ndarray):
        ptype_1 = torch.Tensor(ptype_1)
    if isinstance(anchors_1, np.ndarray):
        anchors_1 = torch.Tensor(anchors_1)
    if isinstance(vectors_1, np.ndarray):
        vectors_1 = torch.Tensor(vectors_1)
    if isinstance(ptype_2, np.ndarray):
        ptype_2 = torch.Tensor(ptype_2)
    if isinstance(anchors_2, np.ndarray):
        anchors_2 = torch.Tensor(anchors_2)
    if isinstance(vectors_2, np.ndarray):
        vectors_2 = torch.Tensor(vectors_2)

    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func, sigma=0.05)
    else:
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    # Determine if single instance or batched
    if len(ptype_1.shape) == 1 and len(ptype_2.shape) == 1:
        batched = False
        assert anchors_1.shape == (ptype_1.shape[0], 3) and vectors_1.shape == anchors_1.shape, \
            f"Shapes of `anchors_1`, `vectors_1`, and `ptype_1` don't match. Should be (N,3), (N,3), and (N,) but {anchors_1.shape}, {vectors_1.shape}, and {ptype_1.shape} were given."
        assert anchors_2.shape == (ptype_2.shape[0], 3) and vectors_2.shape == anchors_2.shape, \
            f"Shapes of `anchors_2`, `vectors_2`, and `ptype_2` don't match. Should be (N,3), (N,3), and (N,) but {anchors_2.shape}, {vectors_2.shape}, and {ptype_2.shape} were given."

    elif len(ptype_1.shape) == 2 and len(ptype_2.shape) == 2:
        batched = True
        assert anchors_1.shape == (ptype_1.shape + (3,)) and vectors_1.shape == anchors_1.shape, \
            f"Shapes of `anchors_1`, `vectors_1`, and `ptype_1` don't match. Should be (B,N,3), (B,N,3), and (B,N) but {anchors_1.shape}, {vectors_1.shape}, and {ptype_1.shape} were given."
        assert anchors_2.shape == (ptype_2.shape + (3,)) and vectors_2.shape == anchors_2.shape, \
            f"Shapes of `anchors_2`, `vectors_2`, and `ptype_2` don't match. Should be (B,N,3), (B,N,3), and (B,N) but {anchors_2.shape}, {vectors_2.shape}, and {ptype_2.shape} were given."
    else:
        raise ValueError(f"Arguments `ptype_1` and `ptype_2` must either be 1D or 2D (batched). Instead these shapes were given: {ptype_1.shape}, {ptype_2.shape}")

    # Pharmacophores present in the molecules
    ptype_key2ind = {}
    unique_ptypes = torch.concat((torch.unique(ptype_1), torch.unique(ptype_2))).unique()
    for i, ptype_ind in enumerate(unique_ptypes.type(torch.int16)):
        ptype_key2ind[P_TYPES_LWRCASE[ptype_ind.item()]] = i

    # Initialize scores
    if batched:
        overlap = torch.zeros(ptype_1.shape[0]).to(anchors_1.device)
        ref_overlap = torch.zeros(ptype_1.shape[0]).to(anchors_1.device)
        fit_overlap = torch.zeros(ptype_1.shape[0]).to(anchors_1.device)

        vectors_1 = F.normalize(vectors_1, p=2, dim=2)
        vectors_2 = F.normalize(vectors_2, p=2, dim=2)
        cdist_21 = (torch.cdist(anchors_1, anchors_2)**2.0).permute(0,2,1)
        cdist_22 = (torch.cdist(anchors_2, anchors_2)**2.0)
        cdist_11 = (torch.cdist(anchors_1, anchors_1)**2.0)
        vmm_21 = torch.matmul(vectors_1, vectors_2.permute(0,2,1)).permute(0,2,1)
        vmm_22 = torch.matmul(vectors_2, vectors_2.permute(0,2,1))
        vmm_11 = torch.matmul(vectors_1, vectors_1.permute(0,2,1))
    else:
        overlap, ref_overlap, fit_overlap = 0., 0., 0.

    ## Score pharmacophores
    # Hydrophobe
    if 'hydrophobe' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_volume_overlap_score_batch(ptype_str='hydrophobe',
                                                           ptype_1=ptype_1,
                                                           ptype_2=ptype_2,
                                                           cdist_21 = cdist_21,
                                                           cdist_22 = cdist_22,
                                                           cdist_11 = cdist_11)
        else:
            VAB, VAA, VBB = get_volume_overlap_score(ptype_str='hydrophobe',
                                                     ptype_1=ptype_1,
                                                     ptype_2=ptype_2,
                                                     anchors_1=anchors_1,
                                                     anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Zn
    if 'znbinder' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_volume_overlap_score_batch(ptype_str='znbinder',
                                                           ptype_1=ptype_1,
                                                           ptype_2=ptype_2,
                                                           cdist_21 = cdist_21,
                                                           cdist_22 = cdist_22,
                                                           cdist_11 = cdist_11)
        else:
            VAB, VAA, VBB = get_volume_overlap_score(ptype_str='znbinder',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Anion
    if 'anion' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_volume_overlap_score_batch(ptype_str='anion',
                                                           ptype_1=ptype_1,
                                                           ptype_2=ptype_2,
                                                           cdist_21 = cdist_21,
                                                           cdist_22 = cdist_22,
                                                           cdist_11 = cdist_11)
        else:
            VAB, VAA, VBB = get_volume_overlap_score(ptype_str='anion',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Cation
    if 'cation' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_volume_overlap_score_batch(ptype_str='cation',
                                                           ptype_1=ptype_1,
                                                           ptype_2=ptype_2,
                                                           cdist_21 = cdist_21,
                                                           cdist_22 = cdist_22,
                                                           cdist_11 = cdist_11)
        else:
            VAB, VAA, VBB = get_volume_overlap_score(ptype_str='cation',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Aromatic
    if 'aromatic' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_vector_volume_overlap_score_batch(ptype_str='aromatic',
                                                                  ptype_1=ptype_1,
                                                                  ptype_2=ptype_2,
                                                                  cdist_21 = cdist_21,
                                                                  cdist_22 = cdist_22,
                                                                  cdist_11 = cdist_11,
                                                                  vmm_21 = vmm_21,
                                                                  vmm_22 = vmm_22,
                                                                  vmm_11 = vmm_11,
                                                                  allow_antiparallel=True)
        else:
            VAB, VAA, VBB = get_vector_volume_overlap_score(ptype_str='aromatic',
                                                            ptype_1=ptype_1,
                                                            ptype_2=ptype_2,
                                                            anchors_1=anchors_1,
                                                            anchors_2=anchors_2,
                                                            vectors_1=vectors_1,
                                                            vectors_2=vectors_2,
                                                            allow_antiparallel=True)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Acceptor
    if 'acceptor' in ptype_key2ind:
        if extended_points:
            VAB, VAA, VBB = get_volume_overlap_score_extended_points(ptype_str='acceptor',
                                                                     ptype_1=ptype_1,
                                                                     ptype_2=ptype_2,
                                                                     anchors_1=anchors_1,
                                                                     anchors_2=anchors_2,
                                                                     vectors_1=vectors_1,
                                                                     vectors_2=vectors_2,
                                                                     only_extended=only_extended)
        else:
            if batched:
                VAB, VAA, VBB = get_vector_volume_overlap_score_batch(ptype_str='acceptor',
                                                                      ptype_1=ptype_1,
                                                                      ptype_2=ptype_2,
                                                                      cdist_21 = cdist_21,
                                                                      cdist_22 = cdist_22,
                                                                      cdist_11 = cdist_11,
                                                                      vmm_21 = vmm_21,
                                                                      vmm_22 = vmm_22,
                                                                      vmm_11 = vmm_11,
                                                                      allow_antiparallel=False)
            else:
                VAB, VAA, VBB = get_vector_volume_overlap_score(ptype_str='acceptor',
                                                                ptype_1=ptype_1,
                                                                ptype_2=ptype_2,
                                                                anchors_1=anchors_1,
                                                                anchors_2=anchors_2,
                                                                vectors_1=vectors_1,
                                                                vectors_2=vectors_2,
                                                                allow_antiparallel=False)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Donors
    if 'donor' in ptype_key2ind:
        if extended_points:
            VAB, VAA, VBB = get_volume_overlap_score_extended_points(ptype_str='donor',
                                                                     ptype_1=ptype_1,
                                                                     ptype_2=ptype_2,
                                                                     anchors_1=anchors_1,
                                                                     anchors_2=anchors_2,
                                                                     vectors_1=vectors_1,
                                                                     vectors_2=vectors_2,
                                                                     only_extended=only_extended)
        else:
            if batched:
                VAB, VAA, VBB = get_vector_volume_overlap_score_batch(ptype_str='donor',
                                                                      ptype_1=ptype_1,
                                                                      ptype_2=ptype_2,
                                                                      cdist_21 = cdist_21,
                                                                      cdist_22 = cdist_22,
                                                                      cdist_11 = cdist_11,
                                                                      vmm_21 = vmm_21,
                                                                      vmm_22 = vmm_22,
                                                                      vmm_11 = vmm_11,
                                                                      allow_antiparallel=False)
            else:
                VAB, VAA, VBB = get_vector_volume_overlap_score(ptype_str='donor',
                                                                ptype_1=ptype_1,
                                                                ptype_2=ptype_2,
                                                                anchors_1=anchors_1,
                                                                anchors_2=anchors_2,
                                                                vectors_1=vectors_1,
                                                                vectors_2=vectors_2,
                                                                allow_antiparallel=False)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB
    
    # Halogen
    if 'halogen' in ptype_key2ind:
        if batched:
            VAB, VAA, VBB = get_vector_volume_overlap_score_batch(ptype_str='halogen',
                                                                  ptype_1=ptype_1,
                                                                  ptype_2=ptype_2,
                                                                  cdist_21 = cdist_21,
                                                                  cdist_22 = cdist_22,
                                                                  cdist_11 = cdist_11,
                                                                  vmm_21 = vmm_21,
                                                                  vmm_22 = vmm_22,
                                                                  vmm_11 = vmm_11,
                                                                  allow_antiparallel=False)
        else:
            VAB, VAA, VBB = get_vector_volume_overlap_score(ptype_str='halogen',
                                                            ptype_1=ptype_1,
                                                            ptype_2=ptype_2,
                                                            anchors_1=anchors_1,
                                                            anchors_2=anchors_2,
                                                            vectors_1=vectors_1,
                                                            vectors_2=vectors_2,
                                                            allow_antiparallel=False)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    scores = similarity_func(VAB=overlap, VAA=ref_overlap, VBB=fit_overlap)
    return scores


def get_pharm_combo_score(centers_1: torch.Tensor,
                          centers_2: torch.Tensor,
                          ptype_1: torch.Tensor,
                          ptype_2: torch.Tensor,
                          anchors_1: torch.Tensor,
                          anchors_2: torch.Tensor,
                          vectors_1: torch.Tensor,
                          vectors_2: torch.Tensor,
                          alpha: float = 0.81,
                          similarity: str = 'tanimoto',
                          extended_points: bool = False,
                          only_extended: bool = False
                          ) -> torch.Tensor:
    """ Compute a combined shape and pharmacophore score. """
    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func, sigma=0.05)
    else:
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    # Pharmacophore scoring
    pharm_score = get_overlap_pharm(ptype_1=ptype_1,
                                    ptype_2=ptype_2,
                                    anchors_1=anchors_1,
                                    anchors_2=anchors_2,
                                    vectors_1=vectors_1,
                                    vectors_2=vectors_2,
                                    similarity=similarity,
                                    extended_points=extended_points,
                                    only_extended=only_extended)
    
    # Shape scoring
    VAB = VAB_2nd_order(centers_1=centers_1,
                        centers_2=centers_2,
                        alpha=alpha)
    VAA = VAB_2nd_order(centers_1=centers_1,
                        centers_2=centers_1,
                        alpha=alpha)
    VBB = VAB_2nd_order(centers_1=centers_2,
                        centers_2=centers_2,
                        alpha=alpha)
    shape_score = similarity_func(VAB=VAB,
                                  VAA=VAA,
                                  VBB=VBB)

    score = (pharm_score + shape_score)/2
    return score