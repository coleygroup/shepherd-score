""" 
Gaussian volume overlap scoring functions -- Shape-only (i.e., not color)
NUMPY VERSION

Single instance functionality only.

Reference math:
https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K
https://doi.org/10.1021/j100011a016
"""
import numpy as np
from scipy.spatial import distance

###################################################################################################
####### NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY NUMPY #######
###################################################################################################

def VAB_2nd_order_np(centers_1, centers_2, alpha) -> np.ndarray:
    """ 2nd order volume overlap of AB """
    R2 = (distance.cdist(centers_1, centers_2)**2.0).T

    VAB_2nd_order = np.sum(np.pi**(1.5) * np.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))
    return VAB_2nd_order


def shape_tanimoto_np(centers_1, centers_2, alpha) -> np.ndarray:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order_np(centers_1, centers_1, alpha)
    VBB = VAB_2nd_order_np(centers_2, centers_2, alpha)
    VAB = VAB_2nd_order_np(centers_1, centers_2, alpha)
    return VAB / (VAA + VBB - VAB)


def get_overlap_np(centers_1:np.ndarray,
                   centers_2:np.ndarray,
                   alpha:float = 0.81
                   ) -> np.ndarray:
    """ NumPy implementation of shape similarity via gaussian overlaps (single instance) """
    tanimoto = shape_tanimoto_np(centers_1, centers_2, alpha)
    return tanimoto


def VAB_2nd_order_cosine_np(centers_1: np.ndarray,
                            centers_2: np.ndarray,
                            vectors_1: np.ndarray,
                            vectors_2: np.ndarray,
                            alpha: float,
                            allow_antiparallel: bool,
                            ) -> np.ndarray:
    """
    2nd order volume overlap of AB weighted by cosine similarity.
    NumPy implementation with single instance functionality.
    """
    if len(centers_1.shape) == 2:
        # Normalize vectors for cosine similarity
        norm_v1 = np.linalg.norm(vectors_1, axis=1, keepdims=True)
        norm_v2 = np.linalg.norm(vectors_2, axis=1, keepdims=True)

        # Avoid division by zero if a vector is all zeros
        vec1_norm = np.divide(vectors_1, norm_v1, out=np.zeros_like(vectors_1), where=norm_v1!=0)
        vec2_norm = np.divide(vectors_2, norm_v2, out=np.zeros_like(vectors_2), where=norm_v2!=0)

        # cosine similarity
        V2 = np.matmul(vec1_norm, vec2_norm.T).T # Now uses normalized vectors
        if allow_antiparallel:
            V2 = np.abs(V2)
        else:
            V2 = np.clip(V2, 0., 1.)

        V2 = (V2 + 2.)/3. # Following PheSA's suggestion for weighting

        R2 = (distance.cdist(centers_1, centers_2)**2.0).T

        VAB_second_order = np.sum(np.pi**(1.5) * V2 * np.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))

    return VAB_second_order
