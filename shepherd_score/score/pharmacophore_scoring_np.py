"""
Pharmacophore scoring with NumPy.
"""

from typing import Union, Callable, Literal, Tuple
from functools import partial

import numpy as np

from shepherd_score.score.gaussian_overlap_np import VAB_2nd_order_np, VAB_2nd_order_cosine_np
from shepherd_score.score.constants import P_TYPES, P_ALPHAS
P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))


def _compute_ref_overlap_np(overlap_func: Callable,
                            anchors_1: np.ndarray,
                            alpha: float,
                            vectors_1: Union[np.ndarray, None] = None,
                            allow_antiparallel: bool = False
                            ) -> np.ndarray:
    """ Single instance only. """
    # Just anchor volume overlap
    if (vectors_1 is None):
        VAA = overlap_func(anchors_1, anchors_1, alpha)
    # Anchor and vector volume overlap for single instance
    else:
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel)
    return VAA


def _compute_fit_overlap_np(overlap_func: Callable,
                            anchors_2: np.ndarray,
                            alpha: float,
                            vectors_2: Union[np.ndarray, None]  = None,
                            allow_antiparallel: bool = False,
                            ) -> np.ndarray:
    """ Single instance only. """
    # Just anchor volume overlap
    if (vectors_2 is None):
        VBB = overlap_func(anchors_2, anchors_2, alpha)
    # Anchor and vector volume overlap for single instance
    else:
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel)
    return VBB


def _compute_all_overlaps_np(overlap_func: Callable,
                             anchors_1: np.ndarray,
                             anchors_2: np.ndarray,
                             alpha: float,
                             vectors_1: Union[np.ndarray, None] = None,
                             vectors_2: Union[np.ndarray, None] = None,
                             allow_antiparallel: bool = False,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Single instance only. """
    # Just anchor volume overlap
    if (vectors_1 is None or vectors_2 is None):
        VAB = overlap_func(anchors_1, anchors_2, alpha)
        VAA = overlap_func(anchors_1, anchors_1, alpha)
        VBB = overlap_func(anchors_2, anchors_2, alpha)
    # Anchor and vector volume overlap for single instance
    else:
        VAB = overlap_func(anchors_1, anchors_2, vectors_1, vectors_2, alpha, allow_antiparallel)
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel)
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel)
    return VAB, VAA, VBB


def tanimoto_func_np(VAB: np.ndarray,
                     VAA: np.ndarray,
                     VBB: np.ndarray
                     ) -> np.ndarray:
    """
    Computes Tanimoto similarity.
    Similarity(Tanimoto) = Overlap{1,2} / (Overlap{1,1} + Overlap{2,2} - Overlap{1,2})
    """
    return VAB/(VAA + VBB - VAB)


def tversky_func_np(VAB: np.ndarray,
                    VAA: np.ndarray,
                    VBB: np.ndarray,
                    sigma: float
                    ) -> np.ndarray:
    """
    Computes Tversky similarity -> clamped to be max of 1.0.
    sigma: [0,1]

    Similarity(Tversky) = Overlap{1,2} / (sigma*Overlap{1,1} + (1-sigma)*Overlap{2,2})
    """
    return np.clip(VAB/(sigma*VAA + (1-sigma)*VBB), a_min=None, a_max=1.0)


def get_vector_volume_overlap_score_np(ptype_str: str,
                                       ptype_1: np.ndarray,
                                       ptype_2: np.ndarray,
                                       anchors_1: np.ndarray,
                                       anchors_2: np.ndarray,
                                       vectors_1: np.ndarray,
                                       vectors_2: np.ndarray,
                                       allow_antiparallel: bool
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute volumentric overlap score with cosine similarity of vectors.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # single instance
    mask_inds_1 = np.where(ptype_1 == ptype_idx)[0]
    mask_inds_2 = np.where(ptype_2 == ptype_idx)[0]
    if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
        VAB, VAA, VBB = 0., 0., 0.
    elif len(mask_inds_1) == 0:
        VAB, VAA = 0., 0.
        VBB = _compute_fit_overlap_np(overlap_func=VAB_2nd_order_cosine_np,
                                      anchors_2=anchors_2[mask_inds_2],
                                      vectors_2=vectors_2[mask_inds_2],
                                      alpha = P_ALPHAS[ptype_str],
                                      allow_antiparallel = allow_antiparallel)
    elif len(mask_inds_2) == 0:
        VAB, VBB = 0., 0.
        VAA = _compute_ref_overlap_np(overlap_func=VAB_2nd_order_cosine_np,
                                      anchors_1=anchors_1[mask_inds_1],
                                      vectors_1=vectors_1[mask_inds_1],
                                      alpha = P_ALPHAS[ptype_str],
                                      allow_antiparallel = allow_antiparallel)
    else:
        VAB, VAA, VBB = _compute_all_overlaps_np(overlap_func=VAB_2nd_order_cosine_np,
                                                 anchors_1=anchors_1[mask_inds_1],
                                                 anchors_2=anchors_2[mask_inds_2],
                                                 vectors_1=vectors_1[mask_inds_1],
                                                 vectors_2=vectors_2[mask_inds_2],
                                                 alpha = P_ALPHAS[ptype_str],
                                                 allow_antiparallel = allow_antiparallel)
    return VAB, VAA, VBB


def get_volume_overlap_score_np(ptype_str: str,
                                ptype_1: np.ndarray,
                                ptype_2: np.ndarray,
                                anchors_1: np.ndarray,
                                anchors_2: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes volume overlap score single instance.
    """
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    # single instance
    mask_inds_1 = np.where(ptype_1 == ptype_idx)[0]
    mask_inds_2 = np.where(ptype_2 == ptype_idx)[0]
    if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
        VAB, VAA, VBB = 0., 0., 0.
    elif len(mask_inds_1) == 0:
        VAB, VAA = 0., 0.
        VBB = _compute_fit_overlap_np(overlap_func=VAB_2nd_order_np,
                                      anchors_2=anchors_2[mask_inds_2],
                                      alpha = P_ALPHAS[ptype_str])
    elif len(mask_inds_2) == 0:
        VAB, VBB = 0., 0.
        VAA = _compute_ref_overlap_np(overlap_func=VAB_2nd_order_np,
                                      anchors_1=anchors_1[mask_inds_1],
                                      alpha = P_ALPHAS[ptype_str])
    else:
        VAB, VAA, VBB = _compute_all_overlaps_np(overlap_func=VAB_2nd_order_np,
                                                 anchors_1=anchors_1[mask_inds_1],
                                                 anchors_2=anchors_2[mask_inds_2],
                                                 alpha = P_ALPHAS[ptype_str])
    return VAB, VAA, VBB


def get_volume_overlap_score_extended_points_np(ptype_str: str,
                                                ptype_1: np.ndarray,
                                                ptype_2: np.ndarray,
                                                anchors_1: np.ndarray,
                                                anchors_2: np.ndarray,
                                                vectors_1: np.ndarray,
                                                vectors_2: np.ndarray,
                                                only_extended: bool = False
                                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score both the anchor and extended point volume overlap instead of a vector similarity.
    """    
    ptype_str = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str)
    VAB, VAA, VBB = 0., 0., 0.
    mask_inds_1 = np.where(ptype_1 == ptype_idx)[0]
    mask_inds_2 = np.where(ptype_2 == ptype_idx)[0]
    if len(mask_inds_1) == 0 and len(mask_inds_2) == 0:
        VAB, VAA, VBB = 0., 0., 0.
    elif len(mask_inds_1) == 0:
        VAB, VAA = 0., 0.
        if not only_extended:
            VBB = _compute_fit_overlap_np(overlap_func=VAB_2nd_order_np,
                                            anchors_2=anchors_2[mask_inds_2],
                                            alpha = P_ALPHAS[ptype_str])
        else:
            VBB = 0.
        VBB += _compute_fit_overlap_np(overlap_func=VAB_2nd_order_np,
                                        anchors_2=vectors_2[mask_inds_2]+anchors_2[mask_inds_2],
                                        alpha = P_ALPHAS[ptype_str])
    elif len(mask_inds_2) == 0:
        VAB, VBB = 0., 0.
        if not only_extended:
            VAA = _compute_ref_overlap_np(overlap_func=VAB_2nd_order_np,
                                            anchors_1=anchors_1[mask_inds_1],
                                            alpha = P_ALPHAS[ptype_str])
        else:
            VAA = 0.
        VAA += _compute_ref_overlap_np(overlap_func=VAB_2nd_order_np,
                                        anchors_1=vectors_1[mask_inds_1]+anchors_1[mask_inds_1],
                                        alpha = P_ALPHAS[ptype_str])
    else:
        if not only_extended:
            VAB, VAA, VBB = _compute_all_overlaps_np(
                overlap_func=VAB_2nd_order_np,
                anchors_1=anchors_1[mask_inds_1],
                anchors_2=anchors_2[mask_inds_2],
                alpha = P_ALPHAS[ptype_str]
            )
        VAB_extended, VAA_extended, VBB_extended = _compute_all_overlaps_np(
            overlap_func=VAB_2nd_order_np,
            anchors_1=vectors_1[mask_inds_1]+anchors_1[mask_inds_1],
            anchors_2=vectors_2[mask_inds_2]+anchors_2[mask_inds_2],
            alpha = P_ALPHAS[ptype_str]
            )
        VAB += VAB_extended
        VAA += VAA_extended
        VBB += VBB_extended
    return VAB, VAA, VBB


_SIM_TYPE = Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit']

def get_overlap_pharm_np(ptype_1: np.ndarray,
                         ptype_2: np.ndarray,
                         anchors_1: np.ndarray,
                         anchors_2: np.ndarray,
                         vectors_1: np.ndarray,
                         vectors_2: np.ndarray,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False
                         ) -> np.ndarray:
    """
    NumPy implementation to compute pharmacophore score.
    Single instance only

    Arguments
    ---------
    ptype_1 : np.ndarray (N,)
        Indices specifying the pharmacophore type based on order of P_TYPES
    ptype_2 : np.ndarray (M,)
        Indices specifying the pharmacophore type based on order of P_TYPES
    anchors_1 : np.ndarray (N,3)
        Coordinates for the anchor points of each pharmacophore of molecule 1
    anchors_2 : np.ndarray (M,3)
        Coordinates for the anchor points of each pharmacophore of molecule 2
    vectors_1 : np.ndarray (N,3)
        Relative unit vectors of each pharmacophore of molecule 1
    vectors_2 : np.ndarray (M,3)
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
    np.ndarray (1,)
    """

    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func_np
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func_np, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func_np, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func_np, sigma=0.05)
    else:
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    # Determine if single instance or batched
    if len(ptype_1.shape) == 1 and len(ptype_2.shape) == 1:
        assert anchors_1.shape == (ptype_1.shape[0], 3) and vectors_1.shape == anchors_1.shape, \
            f"Shapes of `anchors_1`, `vectors_1`, and `ptype_1` don't match. Should be (N,3), (N,3), and (N,) but {anchors_1.shape}, {vectors_1.shape}, and {ptype_1.shape} were given."
        assert anchors_2.shape == (ptype_2.shape[0], 3) and vectors_2.shape == anchors_2.shape, \
            f"Shapes of `anchors_2`, `vectors_2`, and `ptype_2` don't match. Should be (N,3), (N,3), and (N,) but {anchors_2.shape}, {vectors_2.shape}, and {ptype_2.shape} were given."
    else:
        raise ValueError(f"Arguments `ptype_1` and `ptype_2` must either be 1D (single instances). Instead these shapes were given: {ptype_1.shape}, {ptype_2.shape}")

    # Pharmacophores present in the molecules
    ptype_key2ind = {}
    unique_ptypes = tuple(set(ptype_1).union(set(ptype_2)))
    for i, ptype_ind in enumerate(unique_ptypes):
        ptype_key2ind[P_TYPES_LWRCASE[ptype_ind]] = i

    # Initialize scores
    overlap, ref_overlap, fit_overlap = 0., 0., 0.

    ## Score pharmacophores
    # Hydrophobe
    if 'hydrophobe' in ptype_key2ind:
        VAB, VAA, VBB = get_volume_overlap_score_np(ptype_str='hydrophobe',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Zn
    if 'znbinder' in ptype_key2ind:
        VAB, VAA, VBB = get_volume_overlap_score_np(ptype_str='znbinder',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Anion
    if 'anion' in ptype_key2ind:
        VAB, VAA, VBB = get_volume_overlap_score_np(ptype_str='anion',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Cation
    if 'cation' in ptype_key2ind:
        VAB, VAA, VBB = get_volume_overlap_score_np(ptype_str='cation',
                                                    ptype_1=ptype_1,
                                                    ptype_2=ptype_2,
                                                    anchors_1=anchors_1,
                                                    anchors_2=anchors_2)
        overlap += VAB
        ref_overlap += VAA
        fit_overlap += VBB

    # Aromatic
    if 'aromatic' in ptype_key2ind:
        VAB, VAA, VBB = get_vector_volume_overlap_score_np(ptype_str='aromatic',
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
            VAB, VAA, VBB = get_volume_overlap_score_extended_points_np(ptype_str='acceptor',
                                                                        ptype_1=ptype_1,
                                                                        ptype_2=ptype_2,
                                                                        anchors_1=anchors_1,
                                                                        anchors_2=anchors_2,
                                                                        vectors_1=vectors_1,
                                                                        vectors_2=vectors_2,
                                                                        only_extended=only_extended)
        else:
            VAB, VAA, VBB = get_vector_volume_overlap_score_np(ptype_str='acceptor',
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
            VAB, VAA, VBB = get_volume_overlap_score_extended_points_np(ptype_str='donor',
                                                                        ptype_1=ptype_1,
                                                                        ptype_2=ptype_2,
                                                                        anchors_1=anchors_1,
                                                                        anchors_2=anchors_2,
                                                                        vectors_1=vectors_1,
                                                                        vectors_2=vectors_2,
                                                                        only_extended=only_extended)
        else:
            VAB, VAA, VBB = get_vector_volume_overlap_score_np(ptype_str='donor',
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
        VAB, VAA, VBB = get_vector_volume_overlap_score_np(ptype_str='halogen',
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


def get_pharm_combo_score(centers_1: np.ndarray,
                          centers_2: np.ndarray,
                          ptype_1: np.ndarray,
                          ptype_2: np.ndarray,
                          anchors_1: np.ndarray,
                          anchors_2: np.ndarray,
                          vectors_1: np.ndarray,
                          vectors_2: np.ndarray,
                          alpha: float = 0.81,
                          similarity: str = 'tanimoto',
                          extended_points: bool = False,
                          only_extended: bool = False
                          ) -> np.ndarray:
    """ Compute a combined shape and pharmacophore score. """
    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func_np
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func_np, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func_np, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func_np, sigma=0.05)
    else:
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    # Pharmacophore scoring
    pharm_score = get_overlap_pharm_np(ptype_1=ptype_1,
                                       ptype_2=ptype_2,
                                       anchors_1=anchors_1,
                                       anchors_2=anchors_2,
                                       vectors_1=vectors_1,
                                       vectors_2=vectors_2,
                                       similarity=similarity,
                                       extended_points=extended_points,
                                       only_extended=only_extended)
    # Shape scoring
    VAB = VAB_2nd_order_np(centers_1=centers_1,
                           centers_2=centers_2,
                           alpha=alpha)
    VAA = VAB_2nd_order_np(centers_1=centers_1,
                           centers_2=centers_1,
                           alpha=alpha)
    VBB = VAB_2nd_order_np(centers_1=centers_2,
                           centers_2=centers_2,
                           alpha=alpha)
    shape_score = similarity_func(VAB=VAB,
                                  VAA=VAA,
                                  VBB=VBB)

    score = (pharm_score + shape_score)/2
    return score
