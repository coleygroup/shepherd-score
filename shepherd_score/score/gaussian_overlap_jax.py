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
