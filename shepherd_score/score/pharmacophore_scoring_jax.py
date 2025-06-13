"""
Pharmacophore scoring with JAX.

This module may not be as fast as the NumPy version.
TODO: Try to optimize this module.
"""
from typing import Union, Callable, Literal, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, Array

from shepherd_score.score.gaussian_overlap_jax import VAB_2nd_order_jax, VAB_2nd_order_cosine_jax
from shepherd_score.score.gaussian_overlap_jax import VAB_2nd_order_jax_mask, VAB_2nd_order_cosine_jax_mask
from shepherd_score.score.constants import P_TYPES, P_ALPHAS
P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))


@partial(jit, static_argnames=['overlap_func', 'allow_antiparallel'])
def _compute_ref_overlap_jax(overlap_func: Callable,
                             anchors_1: Array,
                             alpha: float,
                             vectors_1: Union[Array, None] = None,
                             allow_antiparallel: bool = False
                             ) -> Array:
    """ Single instance only. """
    # Just anchor volume overlap
    if (vectors_1 is None):
        VAA = overlap_func(anchors_1, anchors_1, alpha)
    # Anchor and vector volume overlap for single instance
    else:
        VAA = overlap_func(anchors_1, anchors_1, vectors_1, vectors_1, alpha, allow_antiparallel)
    return VAA


@partial(jit, static_argnames=['overlap_func', 'allow_antiparallel'])
def _compute_fit_overlap_jax(overlap_func: Callable,
                             anchors_2: Array,
                             alpha: float,
                             vectors_2: Union[Array, None] = None,
                             allow_antiparallel: bool = False,
                             ) -> Array:
    """ Single instance only. """
    # Just anchor volume overlap
    if (vectors_2 is None):
        VBB = overlap_func(anchors_2, anchors_2, alpha)
    # Anchor and vector volume overlap for single instance
    else:
        VBB = overlap_func(anchors_2, anchors_2, vectors_2, vectors_2, alpha, allow_antiparallel)
    return VBB




@jit
def tanimoto_func_jax(VAB: Array,
                      VAA: Array,
                      VBB: Array
                      ) -> Array:
    """
    Computes Tanimoto similarity.
    Similarity(Tanimoto) = Overlap{1,2} / (Overlap{1,1} + Overlap{2,2} - Overlap{1,2})
    """
    return VAB / (VAA + VBB - VAB)


@jit
def tversky_func_jax(VAB: Array,
                     VAA: Array,
                     VBB: Array,
                     sigma: float
                     ) -> Array:
    """
    Computes Tversky similarity -> clamped to be max of 1.0.
    sigma: [0,1]

    Similarity(Tversky) = Overlap{1,2} / (sigma*Overlap{1,1} + (1-sigma)*Overlap{2,2})
    """
    return jnp.clip(VAB / (sigma * VAA + (1 - sigma) * VBB), min=None, max=1.0) 


def get_vector_volume_overlap_score_jax(ptype_str: str,
                                        ptype_1: Array,
                                        ptype_2: Array,
                                        anchors_1: Array,
                                        anchors_2: Array,
                                        vectors_1: Array,
                                        vectors_2: Array,
                                        allow_antiparallel: bool
                                        ) -> Tuple[Array, Array, Array]:
    """
    Compute volumentric overlap score with cosine similarity of vectors. JAX version.
    """
    ptype_str_lwr = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str_lwr)
    
    mask_1 = ptype_1 == ptype_idx
    mask_2 = ptype_2 == ptype_idx

    VAB = VAB_2nd_order_cosine_jax_mask(anchors_1, anchors_2, vectors_1, vectors_2, mask_1, mask_2, P_ALPHAS[ptype_str_lwr], allow_antiparallel)
    VAA = VAB_2nd_order_cosine_jax_mask(anchors_1, anchors_1, vectors_1, vectors_1, mask_1, mask_1, P_ALPHAS[ptype_str_lwr], allow_antiparallel)
    VBB = VAB_2nd_order_cosine_jax_mask(anchors_2, anchors_2, vectors_2, vectors_2, mask_2, mask_2, P_ALPHAS[ptype_str_lwr], allow_antiparallel)

    return VAB, VAA, VBB


def get_volume_overlap_score_jax(ptype_str: str,
                                 ptype_1: Array,
                                 ptype_2: Array,
                                 anchors_1: Array,
                                 anchors_2: Array
                                 ) -> Tuple[Array, Array, Array]:
    """
    Computes volume overlap score single instance. JAX version.
    """
    ptype_str_lwr = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str_lwr)
    
    mask_1 = ptype_1 == ptype_idx
    mask_2 = ptype_2 == ptype_idx

    VAB = VAB_2nd_order_jax_mask(anchors_1, anchors_2, mask_1, mask_2, P_ALPHAS[ptype_str_lwr])
    VAA = VAB_2nd_order_jax_mask(anchors_1, anchors_1, mask_1, mask_1, P_ALPHAS[ptype_str_lwr])
    VBB = VAB_2nd_order_jax_mask(anchors_2, anchors_2, mask_2, mask_2, P_ALPHAS[ptype_str_lwr])

    return VAB, VAA, VBB


def get_volume_overlap_score_extended_points_jax(ptype_str: str,
                                                 ptype_1: Array,
                                                 ptype_2: Array,
                                                 anchors_1: Array,
                                                 anchors_2: Array,
                                                 vectors_1: Array,
                                                 vectors_2: Array,
                                                 only_extended: bool = False
                                                 ) -> Tuple[Array, Array, Array]:
    """
    Score both the anchor and extended point volume overlap instead of a vector similarity. JAX version.
    """
    ptype_str_lwr = ptype_str.lower()
    ptype_idx = P_TYPES_LWRCASE.index(ptype_str_lwr)
    alpha = P_ALPHAS[ptype_str_lwr]

    mask_1 = ptype_1 == ptype_idx
    mask_2 = ptype_2 == ptype_idx
    
    VAB, VAA, VBB = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

    # Score extended points
    VAB_extended = VAB_2nd_order_jax_mask(anchors_1 + vectors_1, anchors_2 + vectors_2, mask_1, mask_2, alpha)
    VAA_extended = VAB_2nd_order_jax_mask(anchors_1 + vectors_1, anchors_1 + vectors_1, mask_1, mask_1, alpha)
    VBB_extended = VAB_2nd_order_jax_mask(anchors_2 + vectors_2, anchors_2 + vectors_2, mask_2, mask_2, alpha)

    VAB += VAB_extended
    VAA += VAA_extended
    VBB += VBB_extended
    
    # Score anchors if not only_extended
    if not only_extended:
        VAB_anchor = VAB_2nd_order_jax_mask(anchors_1, anchors_2, mask_1, mask_2, alpha)
        VAA_anchor = VAB_2nd_order_jax_mask(anchors_1, anchors_1, mask_1, mask_1, alpha)
        VBB_anchor = VAB_2nd_order_jax_mask(anchors_2, anchors_2, mask_2, mask_2, alpha)
        
        VAB += VAB_anchor
        VAA += VAA_anchor
        VBB += VBB_anchor

    return VAB, VAA, VBB


_SIM_TYPE = Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit']

@partial(jit, static_argnames=['similarity', 'extended_points', 'only_extended'])
def get_overlap_pharm_jax(ptype_1: Array,
                          ptype_2: Array,
                          anchors_1: Array,
                          anchors_2: Array,
                          vectors_1: Array,
                          vectors_2: Array,
                          similarity: _SIM_TYPE = 'tanimoto',
                          extended_points: bool = False,
                          only_extended: bool = False
                          ) -> Array:
    """
    JAX implementation to compute pharmacophore score.
    Single instance only.

    Arguments
    ---------
    ptype_1 : Array (N,)
        Indices specifying the pharmacophore type based on order of P_TYPES
    ptype_2 : Array (M,)
        Indices specifying the pharmacophore type based on order of P_TYPES
    anchors_1 : Array (N,3)
        Coordinates for the anchor points of each pharmacophore of molecule 1
    anchors_2 : Array (M,3)
        Coordinates for the anchor points of each pharmacophore of molecule 2
    vectors_1 : Array (N,3)
        Relative unit vectors of each pharmacophore of molecule 1
    vectors_2 : Array (M,3)
        Relative unit vectors of each pharmacophore of molecule 2
    similarity : str
        Specifies what similarity function to use.
    extended_points : bool
        Whether to score HBA/HBD with gaussian overlaps of extended points.
    only_extended : bool
        When `extended_points` is True, decide whether to only score the extended points (ignore anchor overlaps)

    Returns
    -------
    Array (1,)
    """

    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func_jax
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func_jax, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func_jax, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func_jax, sigma=0.05)
    else:
        # JAX jit doesn't like dynamic error raising based on string values well.
        # Consider validating this outside or ensuring `similarity` is from a known set.
        # For now, assume valid input due to _SIM_TYPE or raise error outside jit scope.
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    overlap, ref_overlap, fit_overlap = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

    # Hydrophobe
    VAB_h, VAA_h, VBB_h = get_volume_overlap_score_jax(ptype_str='hydrophobe',
                                                       ptype_1=ptype_1, ptype_2=ptype_2,
                                                       anchors_1=anchors_1, anchors_2=anchors_2)
    overlap += VAB_h; ref_overlap += VAA_h; fit_overlap += VBB_h

    # ZnBinder
    VAB_z, VAA_z, VBB_z = get_volume_overlap_score_jax(ptype_str='znbinder',
                                                       ptype_1=ptype_1, ptype_2=ptype_2,
                                                       anchors_1=anchors_1, anchors_2=anchors_2)
    overlap += VAB_z; ref_overlap += VAA_z; fit_overlap += VBB_z
    
    # Anion
    VAB_an, VAA_an, VBB_an = get_volume_overlap_score_jax(ptype_str='anion',
                                                          ptype_1=ptype_1, ptype_2=ptype_2,
                                                          anchors_1=anchors_1, anchors_2=anchors_2)
    overlap += VAB_an; ref_overlap += VAA_an; fit_overlap += VBB_an

    # Cation
    VAB_cat, VAA_cat, VBB_cat = get_volume_overlap_score_jax(ptype_str='cation',
                                                             ptype_1=ptype_1, ptype_2=ptype_2,
                                                             anchors_1=anchors_1, anchors_2=anchors_2)
    overlap += VAB_cat; ref_overlap += VAA_cat; fit_overlap += VBB_cat

    # Aromatic
    VAB_ar, VAA_ar, VBB_ar = get_vector_volume_overlap_score_jax(ptype_str='aromatic',
                                                                 ptype_1=ptype_1, ptype_2=ptype_2,
                                                                 anchors_1=anchors_1, anchors_2=anchors_2,
                                                                 vectors_1=vectors_1, vectors_2=vectors_2,
                                                                 allow_antiparallel=True)
    overlap += VAB_ar; ref_overlap += VAA_ar; fit_overlap += VBB_ar
    
    # Acceptor
    if extended_points:
        VAB_acc, VAA_acc, VBB_acc = get_volume_overlap_score_extended_points_jax(
                                        ptype_str='acceptor', ptype_1=ptype_1, ptype_2=ptype_2,
                                        anchors_1=anchors_1, anchors_2=anchors_2,
                                        vectors_1=vectors_1, vectors_2=vectors_2,
                                        only_extended=only_extended)
    else:
        VAB_acc, VAA_acc, VBB_acc = get_vector_volume_overlap_score_jax(
                                        ptype_str='acceptor', ptype_1=ptype_1, ptype_2=ptype_2,
                                        anchors_1=anchors_1, anchors_2=anchors_2,
                                        vectors_1=vectors_1, vectors_2=vectors_2,
                                        allow_antiparallel=False)
    overlap += VAB_acc; ref_overlap += VAA_acc; fit_overlap += VBB_acc

    # Donor
    if extended_points:
        VAB_don, VAA_don, VBB_don = get_volume_overlap_score_extended_points_jax(
                                        ptype_str='donor', ptype_1=ptype_1, ptype_2=ptype_2,
                                        anchors_1=anchors_1, anchors_2=anchors_2,
                                        vectors_1=vectors_1, vectors_2=vectors_2,
                                        only_extended=only_extended)
    else:
        VAB_don, VAA_don, VBB_don = get_vector_volume_overlap_score_jax(
                                        ptype_str='donor', ptype_1=ptype_1, ptype_2=ptype_2,
                                        anchors_1=anchors_1, anchors_2=anchors_2,
                                        vectors_1=vectors_1, vectors_2=vectors_2,
                                        allow_antiparallel=False)
    overlap += VAB_don; ref_overlap += VAA_don; fit_overlap += VBB_don

    # Halogen
    VAB_hal, VAA_hal, VBB_hal = get_vector_volume_overlap_score_jax(
                                    ptype_str='halogen', ptype_1=ptype_1, ptype_2=ptype_2,
                                    anchors_1=anchors_1, anchors_2=anchors_2,
                                    vectors_1=vectors_1, vectors_2=vectors_2,
                                    allow_antiparallel=False)
    overlap += VAB_hal; ref_overlap += VAA_hal; fit_overlap += VBB_hal
    
    scores = similarity_func(VAB=overlap, VAA=ref_overlap, VBB=fit_overlap)
    return scores


@partial(jit, static_argnames=['similarity', 'extended_points', 'only_extended'])
def get_pharm_combo_score_jax(centers_1: Array,
                              centers_2: Array,
                              ptype_1: Array,
                              ptype_2: Array,
                              anchors_1: Array,
                              anchors_2: Array,
                              vectors_1: Array,
                              vectors_2: Array,
                              alpha: float = 0.81,
                              similarity: str = 'tanimoto',
                              extended_points: bool = False,
                              only_extended: bool = False
                              ) -> Array:
    """ Compute a combined shape and pharmacophore score. JAX version. """
    # Similarity scoring
    if similarity.lower() == 'tanimoto':
        similarity_func = tanimoto_func_jax
    elif similarity.lower() == 'tversky':
        similarity_func = partial(tversky_func_jax, sigma=0.95)
    elif similarity.lower() == 'tversky_ref':
        similarity_func = partial(tversky_func_jax, sigma=1.)
    elif similarity.lower() == 'tversky_fit':
        similarity_func = partial(tversky_func_jax, sigma=0.05)
    else:
        raise ValueError(f'Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')

    # Pharmacophore scoring
    pharm_score = get_overlap_pharm_jax(ptype_1=ptype_1,
                                        ptype_2=ptype_2,
                                        anchors_1=anchors_1,
                                        anchors_2=anchors_2,
                                        vectors_1=vectors_1,
                                        vectors_2=vectors_2,
                                        similarity=similarity,
                                        extended_points=extended_points,
                                        only_extended=only_extended)
    
    # Shape scoring
    VAB_shape = VAB_2nd_order_jax(centers_1=centers_1,
                                  centers_2=centers_2,
                                  alpha=alpha)
    VAA_shape = VAB_2nd_order_jax(centers_1=centers_1,
                                  centers_2=centers_1,
                                  alpha=alpha)
    VBB_shape = VAB_2nd_order_jax(centers_1=centers_2,
                                  centers_2=centers_2,
                                  alpha=alpha)
    shape_score = similarity_func(VAB=VAB_shape,
                                  VAA=VAA_shape,
                                  VBB=VBB_shape)

    score = (pharm_score + shape_score) / 2.0
    return score 