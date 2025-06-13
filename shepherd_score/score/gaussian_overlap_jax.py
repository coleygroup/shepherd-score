""" 
Gaussian volume overlap scoring functions -- Shape-only (i.e., not color)
JAX VERSION (~ 6x faster than numpy)

Batched and non-batched functionalities

Reference math:
https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K
https://doi.org/10.1021/j100011a016
"""
from jax import jit, Array
import jax.numpy as jnp
import jax


###################################################################################################
####### JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX JAX #######
###################################################################################################

@jit
def jax_cdist(X_1: Array,
              X_2: Array
              ) -> Array:
    """
    Jax implementation pairwise euclidian distances.
    
    Parameters
    ----------
    X_1 : Array (N, P)
    X_2 : Array (M, P)
    
    Returns
    -------
    Array (N, M)
        Distance matrix between X_1 and X_2.
    """
    distances = jnp.linalg.norm((X_1[:, None, :] - X_2[None, :, :]), axis=-1)
    return distances

@jit
def jax_sq_cdist(X_1: Array,
                 X_2: Array
                 ) -> Array:
    """
    Jax implementation pairwise SQUARED euclidian distances.
    
    Parameters
    ----------
    X_1 : Array (N, P)
    X_2 : Array (M, P)
    
    Returns
    -------
    Array (N, M)
        Distance matrix between X_1 and X_2, squared.
    """
    distances = jnp.sum(jnp.square((X_1[:, None, :] - X_2[None, :, :])), axis=-1)
    return distances


def VAB_2nd_order_jax(centers_1: Array, centers_2: Array, alpha: float) -> Array:
    """ 2nd order volume overlap of AB """
    R2 = jax_sq_cdist(centers_1, centers_2)

    VAB_2nd_order = jnp.sum(jnp.pi**(1.5) * jnp.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))
    return VAB_2nd_order


def shape_tanimoto_jax(centers_1: Array, centers_2: Array, alpha: float) -> Array:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order_jax(centers_1, centers_1, alpha)
    VBB = VAB_2nd_order_jax(centers_2, centers_2, alpha)
    VAB = VAB_2nd_order_jax(centers_1, centers_2, alpha)
    return VAB / (VAA + VBB - VAB)

@jit
def get_overlap_jax(centers_1: Array,
                    centers_2: Array,
                    alpha: float = 0.81
                    ) -> Array:
    """ Compute ROCS Gaussian volume overlap using jitted jax function. """
    tanimoto = shape_tanimoto_jax(centers_1, centers_2, alpha)
    return tanimoto


@jit
def _mask_prod_jax(mask_1: Array, mask_2: Array):
    return mask_1[:, None] * mask_2[None, :]


def VAB_2nd_order_jax_mask(centers_1: Array, centers_2: Array,
                            mask_1: Array, mask_2: Array,
                            alpha: float) -> Array:
    """ 2nd order volume overlap of AB """
    R2 = jax_sq_cdist(centers_1, centers_2)
    M2 = _mask_prod_jax(mask_1, mask_2)

    VAB_2nd_order = jnp.sum(M2 * jnp.pi**(1.5) * jnp.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))
    return VAB_2nd_order


def shape_tanimoto_jax_mask(centers_1: Array, centers_2: Array,
                            mask_1: Array, mask_2: Array,
                            alpha: float) -> Array:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order_jax_mask(centers_1, centers_1, mask_1, mask_1, alpha)
    VBB = VAB_2nd_order_jax_mask(centers_2, centers_2, mask_2, mask_2, alpha)
    VAB = VAB_2nd_order_jax_mask(centers_1, centers_2, mask_1, mask_2, alpha)
    return VAB / (VAA + VBB - VAB)

@jit
def get_overlap_jax_mask(centers_1: Array,
                         centers_2: Array,
                         mask_1: Array,
                         mask_2: Array,
                         alpha: float = 0.81
                         ) -> Array:
    """ Compute ROCS Gaussian volume overlap using jitted jax function. """
    tanimoto = shape_tanimoto_jax_mask(centers_1, centers_2, mask_1, mask_2, alpha)
    return tanimoto


def _VAB_2nd_order_cosine_jax(centers_1: Array,
                              centers_2: Array,
                              vectors_1: Array,
                              vectors_2: Array,
                              alpha: float,
                              allow_antiparallel: bool,
                              ) -> Array:
    """
    2nd order volume overlap of AB weighted by cosine similarity (JAX version) - implementation part.
    """
    R2 = jax_sq_cdist(centers_1, centers_2)  # (N1, N2)
    term_common = (jnp.pi**1.5) / ((2 * alpha)**1.5)

    # Normalize vectors
    vec1_norm = vectors_1 / jnp.linalg.norm(vectors_1, axis=-1, keepdims=True)
    vec2_norm = vectors_2 / jnp.linalg.norm(vectors_2, axis=-1, keepdims=True)

    # Cosine similarity: (N1, N2)
    V2_sim = jnp.dot(vec1_norm, vec2_norm.T)

    V2_sim = jax.lax.cond(
        allow_antiparallel,
        lambda x: jnp.abs(x),    # True branch
        lambda x: jnp.clip(x, 0., 1.),  # False branch
        V2_sim
    )
    V2_weighted = (V2_sim + 2.) / 3.

    VAB_second_order = jnp.sum(term_common *
                               V2_weighted *  # REMOVED .T : V2_weighted is (N1,N2), R2 is (N1,N2)
                               jnp.exp(-(alpha / 2) * R2))
    return VAB_second_order

VAB_2nd_order_cosine_jax = jit(_VAB_2nd_order_cosine_jax, static_argnames=["allow_antiparallel"])


def _VAB_2nd_order_cosine_jax_mask(centers_1: Array,
                                   centers_2: Array,
                                   vectors_1: Array,
                                   vectors_2: Array,
                                   mask_1: Array,
                                   mask_2: Array,
                                   alpha: float,
                                   allow_antiparallel: bool,
                                   ) -> Array:
    """
    2nd order volume overlap of AB weighted by cosine similarity (JAX version) - implementation part.
    """
    R2 = jax_sq_cdist(centers_1, centers_2)  # (N1, N2)
    M2 = _mask_prod_jax(mask_1, mask_2)
    term_common = (jnp.pi**1.5) / ((2 * alpha)**1.5)

    # Normalize vectors
    vec1_norm = vectors_1 / jnp.linalg.norm(vectors_1, axis=-1, keepdims=True)
    vec2_norm = vectors_2 / jnp.linalg.norm(vectors_2, axis=-1, keepdims=True)

    # Cosine similarity: (N1, N2)
    V2_sim = jnp.dot(vec1_norm, vec2_norm.T)

    V2_sim = jax.lax.cond(
        allow_antiparallel,
        lambda x: jnp.abs(x),    # True branch
        lambda x: jnp.clip(x, 0., 1.),  # False branch
        V2_sim
    )
    V2_weighted = (V2_sim + 2.) / 3.

    VAB_second_order = jnp.sum(term_common *
                                 M2 *
                                 V2_weighted *
                                 jnp.exp(-(alpha / 2) * R2))
    return VAB_second_order

VAB_2nd_order_cosine_jax_mask = jit(_VAB_2nd_order_cosine_jax_mask, static_argnames=["allow_antiparallel"])
