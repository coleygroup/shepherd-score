"""
Alignment implementation in Jax.
"""
from typing import Union, List, Tuple
import numpy as np

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, Array
import optax

import torch

from shepherd_score.score.gaussian_overlap_jax import get_overlap_jax
from shepherd_score.score.electrostatic_scoring_jax import get_overlap_esp_jax, esp_combo_score_jax
from shepherd_score.score.pharmacophore_scoring_jax import get_overlap_pharm_jax, _SIM_TYPE
from shepherd_score.alignment_utils.pca_jax import quaternions_for_principal_component_alignment_jax, rotation_axis_jax, vmap_angle_between_vecs_jax, vmap_quaternion_from_axis_angle_jax
from shepherd_score.alignment_utils.se3_jax import get_SE3_transform_jax, apply_SE3_transform_jax
from shepherd_score.alignment import _initialize_se3_params, _initialize_se3_params_with_translations

vmap_get_overlap_jax = vmap(get_overlap_jax, (None, 0, None))
vmap_get_overlap_esp_jax = vmap(get_overlap_esp_jax, (None, 0, None, None, None, None))
vmap_esp_combo_score = vmap(esp_combo_score_jax, (None, 0,
                                                  None, 0,
                                                  None, 0,
                                                  None, None,
                                                  None, None,
                                                  None, None,
                                                  None, None, None, None))
vmap_apply_SE3_transform_jax = vmap(apply_SE3_transform_jax, (None, 0))
vmap_get_SE3_transform_jax = vmap(get_SE3_transform_jax, 0)


def apply_SO3_transform_jax(vectors: Array, se3_matrix: Array) -> Array:
    """
    Apply SO(3) transformation (rotation) to a set of vectors.
    """
    rotation_matrix = se3_matrix[..., :3, :3]
    return jnp.matmul(vectors, rotation_matrix.transpose())


vmap_apply_SO3_transform_jax = vmap(apply_SO3_transform_jax, (None, 0))


def _get_points_fibonacci_jax(num_samples: int) -> Array:
    """
    Generate points on unit sphere using fibonacci approach. Jax implementation.
    Adapted from Morfeus:
     https://github.com/digital-chemistry-laboratory/morfeus/blob/main/morfeus/geometry.py
    
    Parameters
    ----------
    num_samples : int
        Number of points to sample from the surface of a sphere
    
    Returns
    -------
    Array (num_samples,3)
        Coordinates of the sampled points.
    """
    offset = 2.0 / num_samples
    increment = jnp.pi * (3.0 - jnp.sqrt(5.0))

    i = jnp.arange(num_samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = jnp.sqrt(1 - jnp.square(y))
    phi = jnp.mod((i + 1), num_samples) * increment
    x = jnp.cos(phi) * r
    z = jnp.sin(phi) * r

    points = jnp.column_stack((x, y, z))
    return points


def _objective_ROCS_overlay_jax(se3_params: Array,
                                ref_points: Array,
                                fit_points: Array,
                                alpha: float
                                ) -> Array:
    """
    Objective function to optimize ROCS overlay.
    Jax implementation.
    
    Parameters
    ----------
    se3_params : Array (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    alpha : float
        Gaussian width parameter used in scoring function.
    
    Returns
    -------
    loss : Array (1,)
        1 - Tanimoto score
    """        
    se3_matrix = get_SE3_transform_jax(se3_params)
    fit_points = apply_SE3_transform_jax(fit_points, se3_matrix)
    score = get_overlap_jax(ref_points, fit_points, alpha)
    return score

batched_obj_ROCS_overlay_helper = vmap(_objective_ROCS_overlay_jax, (0, None, None, None))

def objective_ROCS_overlay_jax(se3_params: Array,
                               ref_points: Array,
                               fit_points: Array,
                               alpha: float
                               ) -> Array:
    """
    Objective function to optimize ROCS overlay.
    Jax implementation.
    
    Parameters
    ----------
    se3_params : Array (batch, 7)
        Parameters for SE(3) transformation. Expects batch.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : Array (N,3)
        Reference points. (NOT batched since it assumes the same reference points).
    fit_points : Array (batch, M,3)
        Expects batch.
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
        If you want to optimize to the same fit_points, with a batch of different
        se3_params, try use jnp.tile(fit_points, (batch, 1, 1)).
    alpha : float
        Gaussian width parameter used in scoring function.
    
    Returns
    -------
    loss : Array (1,)
        1 - Tanimoto score
    """
    scores = batched_obj_ROCS_overlay_helper(se3_params,
                                             ref_points,
                                             fit_points,
                                             alpha)
    return 1 - scores.mean()
    

def _quats_from_fibo_jax(num_samples: int):
    """
    Computes the quaternions corresponding to the a uniform distribution (deterministic) of
    rotations. Does this by finding out the quaternions necessary to rotate a unit vector
    to points sampled on a sphere from the golden spiral method or Fibonacci sphere surface
    sampling.
    Jax implementation.

    Parameters
    ----------
    num_samples : int
        Number of rotations to generate.

    Returns
    -------
    quaternions : Array (num_samples, 4)
        quaternions corresponding to each rotation.
    """
    fibo = _get_points_fibonacci_jax(num_samples)
    unit_v = jnp.tile(jnp.array([1., 0., 0.]),
                      (num_samples, 1))
    
    # quaternions = __quats_from_fibo_jax(unit_v, fibo)
    angles = vmap_angle_between_vecs_jax(unit_v, fibo)
    axes = rotation_axis_jax(unit_v, fibo)
    quaternions = vmap_quaternion_from_axis_angle_jax(axes, angles)
    return quaternions


def _get_45_fibo_jax() -> Array:
    """ Precomputed values for se3_params_from_fibo(45).
    Returns
    -------
    Array (45,4)
        Corresponding quaternions for se3_params_from_fibo(45).
    """
    return jnp.array([[ 0.6501596 ,  0.        , -0.10890594, -0.7519521 ],
                      [ 0.71811795,  0.        ,  0.24900949, -0.64984685],
                      [ 0.79960614,  0.        , -0.22734107, -0.5558292 ],
                      [ 0.48607868,  0.        ,  0.09597147, -0.8686294 ],
                      [ 0.8678287 ,  0.        ,  0.18554172, -0.46092048],
                      [ 0.6441806 ,  0.        , -0.49103084, -0.58644706],
                      [ 0.58135426,  0.        ,  0.53663224, -0.61159873],
                      [ 0.9219894 ,  0.        , -0.13865991, -0.36153716],
                      [ 0.37174237,  0.        , -0.4017539 , -0.8368999 ],
                      [ 0.82034767,  0.        ,  0.4505742 , -0.3521542 ],
                      [ 0.7915699 ,  0.        , -0.5098301 , -0.3368833 ],
                      [ 0.35016882,  0.        ,  0.62455714, -0.69807595],
                      [ 0.9682232 ,  0.        ,  0.0993299 , -0.22951545],
                      [ 0.48625368,  0.        , -0.7709624 , -0.41130796],
                      [ 0.6632823 ,  0.        ,  0.69872594, -0.26802734],
                      [ 0.92916685,  0.        , -0.3295777 , -0.16741402],
                      [ 0.13607754,  0.        , -0.1463197 , -0.97983336],
                      [ 0.9195395 ,  0.        ,  0.37396038, -0.12083343],
                      [ 0.6908489 ,  0.        , -0.71145827, -0.12866619],
                      [ 0.427207  ,  0.        ,  0.89058506, -0.15605238],
                      [ 0.9967814 ,  0.        , -0.06662399, -0.04458794],
                      [ 0.2999607 ,  0.        , -0.95107055, -0.07408379],
                      [ 0.78085893, -0.        ,  0.6247074 ,  0.        ],
                      [ 0.8650692 ,  0.        , -0.5009943 ,  0.02568838],
                      [ 0.15980992, -0.        ,  0.9471624 ,  0.2781082 ],
                      [ 0.9745988 , -0.        ,  0.21325576,  0.06840423],
                      [ 0.5568162 ,  0.        , -0.8151512 ,  0.15963776],
                      [ 0.57879627, -0.        ,  0.79255456,  0.19196929],
                      [ 0.962584  ,  0.        , -0.23290652,  0.13851605],
                      [ 0.20126757,  0.        , -0.60178804,  0.7728793 ],
                      [ 0.86761075, -0.        ,  0.45306247,  0.20490502],
                      [ 0.7600118 ,  0.        , -0.5942492 ,  0.2631538 ],
                      [ 0.3819389 , -0.        ,  0.71805334,  0.5818266 ],
                      [ 0.96679044, -0.        ,  0.03724971,  0.2528412 ],
                      [ 0.46128264,  0.        , -0.67306805,  0.57809824],
                      [ 0.7085131 , -0.        ,  0.575984  ,  0.40773967],
                      [ 0.8734927 ,  0.        , -0.33189464,  0.3561691 ],
                      [ 0.35904366, -0.        ,  0.09578869,  0.92839223],
                      [ 0.8831887 , -0.        ,  0.24063599,  0.40258166],
                      [ 0.6643608 ,  0.        , -0.48505744,  0.56863344],
                      [ 0.58328235, -0.        ,  0.43531075,  0.6857742 ],
                      [ 0.8708025 ,  0.        , -0.08129112,  0.4848656 ],
                      [ 0.5442492 ,  0.        , -0.19216032,  0.81661934],
                      [ 0.74993277, -0.        ,  0.2244347 ,  0.622278  ],
                      [ 0.77770305,  0.        ,  0.        ,  0.6286318 ]])


def _initialize_se3_params_jax(ref_points: Array,
                               fit_points: Array,
                               num_repeats: int = 50
                               ) -> Array:
    """
    Initialize SE(3) parameter guesses. Jax implementation.
    SLOWER THAN TORCH.
    
    First four values are the quaternion and the last three
    are the translation. 
    All initial translations are to align fit_points COM with ref_points' COM.

    The first set corresponds to no rotation.
    The next four (if applicable) correspond to principal component alignment with ref_points.
    All other transformations are rotations generated from Fibonacci sampling of points on a
    sphere.

    Parameters
    ----------
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    
    Returns
    -------
    se3_params : Array (num_repeats, 7)
        Initial guesses for the SE(3) transformation parameters.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    ref_points_com = ref_points.mean(0)
    fit_points_com = fit_points.mean(0)

    # Always do all principal components if num_repeats is greater than 1
    if num_repeats < 5:
        num_repeats = 5

    # First guess keeps the original orientation but aligns the COMs
    # Switch to just local optimization, no COM alignment
    se3_params = jnp.zeros((num_repeats, 7))
    se3_params = se3_params.at[0, :4].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    # se3_params = se3_params.at[0, 4:].set(-fit_points_com + ref_points_com)

    # Align the principal components for the next 4
    pca_quats = quaternions_for_principal_component_alignment_jax(ref_points, fit_points)
    se3_params = se3_params.at[1:5, :4].set(jnp.array(pca_quats)) # rotation component for centered points
    SE3_rotation = vmap_get_SE3_transform_jax(se3_params.at[1:5].get()) # only rotation
    # Rotate translation to COM in original coordinates
    T = vmap_apply_SE3_transform_jax(fit_points_com, SE3_rotation).squeeze()
    # Apply translation to center COMs by taking into account implicit translation done in PCA
    se3_params = se3_params.at[1:5, 4:].set(- T + ref_points_com)
    # Do random rotations 
    if num_repeats > 5:
        if num_repeats == 50:
            # Precomputed se3_params from fibonacci sampling of 45
            se3_params = se3_params.at[5:, :4].set(_get_45_fibo_jax())
        else:
            se3_params = se3_params.at[5:, :4].set(_quats_from_fibo_jax(num_repeats - 5))
        # Adjust translation to COM with the corresponding rotations
        SE3_rotation = vmap_get_SE3_transform_jax(se3_params.at[5:].get()) # only rotation
        T = vmap_apply_SE3_transform_jax(fit_points_com, SE3_rotation).squeeze()
        # Apply translation to center COMs by taking into account implicit translation done with rotations
        se3_params = se3_params.at[5:, 4:].set(- T + ref_points_com)

    return se3_params

# TRIED TO REPLACE PYTORCH VERSION BUT NO REAL SPEEDUP
# def _quats_from_fibo_np(num_samples: int):
#     """
#     Computes the quaternions corresponding to the a uniform distribution (deterministic) of
#     rotations. Does this by finding out the quaternions necessary to rotate a unit vector
#     to points sampled on a sphere from the golden spiral method or Fibonacci sphere surface
#     sampling.

#     Parameters
#     ----------
#     num_samples : int
#         Number of rotations to generate.

#     Returns
#     -------
#     quaternions : torch.Tensor (num_samples, 4)
#         quaternions corresponding to each rotation.
#     """
#     fibo = _get_points_fibonacci(num_samples)
#     unit_v = np.array([1., 0., 0.])
    
#     quaternions = np.zeros((num_samples, 4))
#     for i in range(num_samples):
#         angles = angle_between_vecs_np(unit_v, fibo[i])
#         axes = rotation_axis_np(unit_v, fibo[i])
#         quaternions[i] = quaternion_from_axis_angle_np(axes, angles)
#     return quaternions

# def _initialize_se3_params_with_translations_np(ref_points: np.ndarray,
#                                                 fit_points: np.ndarray,
#                                                 trans_centers: np.ndarray,
#                                                 num_repeats_per_trans: int = 10
#                                                 ) -> np.ndarray:
#     """
#     Slower than Torch so use Torch version. Scales linearlly with num_repeats_per_trans.
#     """
#     # Initial guess for SE(3) parameters (quaternion followed by translation)
#     ref_points_com = ref_points.mean(0)
#     fit_points_com = fit_points.mean(0)

#     num_repeats = num_repeats_per_trans * trans_centers.shape[0] + 5

#     # First guess keeps the original orientation but aligns the COMs
#     se3_params = np.zeros((num_repeats, 7))
#     se3_params[0, :4] = np.array([1.0, 0.0, 0.0, 0.0]) 
#     se3_params[0, 4:] = -fit_points_com + ref_points_com

#     pca_quats = quaternions_for_principal_component_alignment_np(ref_points, fit_points)
#     se3_params[1:5, :4] = pca_quats # rotation component for centered points
#     fit_points_com = fit_points_com.reshape(1,-1)
#     for i in range(1,5):
#         SE3_rotation = get_SE3_transform_np(se3_params[i]) # only rotation
#         # Rotate translation to COM in original coordinates
#         T = apply_SE3_transform_np(fit_points_com, SE3_rotation).squeeze()
#         # Apply translation to center COMs by taking into account implicit translation done in PCA
#         se3_params[i, 4:] = - T + ref_points_com

#     # Do random rotations
#     quats = _quats_from_fibo_np(num_repeats_per_trans)

#     quats = quats / np.linalg.norm(_quats_from_fibo_np(10), 2, 1, keepdims=True)
#     se3_params[5:, :4] = np.tile(quats, (trans_centers.shape[0], 1))
#     # Construct SE(3) transformation matrix for rotations
#     SE3_rotation = np.eye(4)
#     T = np.zeros((num_repeats_per_trans, 3))
#     for i in range(num_repeats_per_trans):
#         SE3_rotation[:3, :3] = quaternions_to_rotation_matrix_np(quats[i])

#         # Adjust translation to COM with the corresponding rotations
#         T[i] = apply_SE3_transform_np(fit_points_com, SE3_rotation)
#     T = np.tile(T, (trans_centers.shape[0], 1))
#     # translation to atoms
#     trans_centers_rep = np.repeat(trans_centers, num_repeats_per_trans, 0)
#     # Apply translation to center COMs by taking into account implicit translation done with rotations
#     se3_params[5:, 4:] = - T + trans_centers_rep
#     return se3_params


jit_val_grad_obj_ROCS = jit(value_and_grad(objective_ROCS_overlay_jax))

def optimize_ROCS_overlay_jax(ref_points: Array,
                              fit_points: Array,
                              alpha: float,
                              num_repeats: int = 50,
                              trans_centers: Union[Array, np.ndarray, None] = None,
                              lr: float = 0.1,
                              max_num_steps: int = 200,
                              verbose: bool = False
                              ) -> Tuple[Array]:
    """
    Optimize alignment of fit_points with respect to ref_points using SE(3) transformations and
    maximizing gaussian overlap score.

    If num_repeats is 1, the initial guess for alignment is an identity rotation and aligned COMs.
    If num_repeats is 5 or greater, four initial guesses are aligned using principal components.
    
    Parameters
    ----------
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    alpha : float
        Gaussian width parameter used in scoring function.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    trans_centers : array (P, 3) (default=None)
        Locations to translate fit_points' center of mass as an initial guesses for optimization.
        At each translation center, 10 rotations are also sampled. So the number of initializations
        scales as (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
        aligned COM's.
        If None, then num_repeats rotations are done with aligned COM's.
    lr : float (default=0.1)
        Learning rate or step-size for optimization
    max_num_steps : int (default=200)
        Maximum number of steps to optimize over.
    verbose : bool (False)
        Print statements about initial and final similarity scores. Further, it will print scores
        during optimization at very 100 steps.

    Returns
    -------
    tuple
        aligned_points : Array (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        SE3_transform : Array (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : Array (1,)
            Tanimoto shape similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    # FASTER USING TORCH
    # se3_params = _initialize_se3_params_jax(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=torch.Tensor(np.array(ref_points)),
                                            fit_points=torch.Tensor(np.array(fit_points)),
                                            num_repeats=num_repeats).detach()
        if num_repeats == 1:
            se3_params = se3_params.unsqueeze(0)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=torch.Tensor(np.array(ref_points)),
            fit_points=torch.Tensor(np.array(fit_points)),
            trans_centers=torch.Tensor(np.array(trans_centers)),
            num_repeats_per_trans=10).detach()
    
    if len(se3_params.shape) == 1:
        se3_params.unsqueeze(0)
    se3_params = jnp.array(se3_params)

    # Create optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(se3_params)

    # Optimization loop
    if verbose:
        print(f'Initial shape similarity score: {get_overlap_jax(ref_points, fit_points, alpha):.3}')
    last_loss = 1
    counter = 0
    for step in range(max_num_steps):
        # Forward pass: compute objective function and gradients
        loss, grads = jit_val_grad_obj_ROCS(se3_params, ref_points, fit_points, alpha)
        updates, opt_state = optimizer.update(grads, opt_state, se3_params)
        se3_params = optax.apply_updates(se3_params, updates)
        
        # early stopping
        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    SE3_transform = vmap_get_SE3_transform_jax(se3_params)
    aligned_points = vmap_apply_SE3_transform_jax(fit_points, SE3_transform)
    scores = vmap_get_overlap_jax(ref_points,
                                  aligned_points,
                                  alpha)
    if num_repeats == 1:
        if verbose:
            print(f'Optimized shape similarity score: {scores:.3}')
        best_alignment = aligned_points
        best_transform = SE3_transform
        best_score = scores
    else:
        if verbose:
            print(f'Optimized shape similarity score -- max: {scores.max():3} | mean: {scores.mean():.3} | min: {scores.min():3}')
        best_idx = jnp.argmax(scores)
        best_alignment = aligned_points.at[best_idx].get()
        best_transform = SE3_transform.at[best_idx].get()
        best_score = scores.at[best_idx].get()
    return best_alignment, best_transform, best_score


def _objective_ROCS_esp_overlay_jax(se3_params: Array,
                                    ref_points: Array,
                                    fit_points: Array,
                                    ref_charges: Array,
                                    fit_charges: Array,
                                    alpha: float,
                                    lam: float
                                    ) -> Array:
    """
    Objective function to optimize ROCS esp overlay.
    Jax implementation.
    
    Parameters
    ----------
    se3_params : Array (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    alpha : float
        Gaussian width parameter used in scoring function.
    
    Returns
    -------
    loss : Array (1,)
        1 - Tanimoto score
    """        
    se3_matrix = get_SE3_transform_jax(se3_params)
    fit_points = apply_SE3_transform_jax(fit_points, se3_matrix)
    score = get_overlap_esp_jax(ref_points, fit_points, ref_charges, fit_charges, alpha, lam)
    return score

batched_obj_ROCS_esp_overlay_helper = vmap(_objective_ROCS_esp_overlay_jax, (0, None, None, None, None, None, None))


def objective_ROCS_esp_overlay_jax(se3_params: Array,
                                   ref_points: Array,
                                   fit_points: Array,
                                   ref_charges: Array,
                                   fit_charges: Array,
                                   alpha: float,
                                   lam: float
                                   ) -> Array:
    """
    Objective function to optimize ROCS esp overlay. 
    
    Parameters
    ----------
    se3_params : Array (batch, 7)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    ref_charges : Array (N,)
        Electric potential at the corresponding ref_points coordinates.
    fit_charges : Array (M,)
        Electric potential at the corresponding fit_points coordinates
    alpha : float
        Gaussian width parameter used in scoring function.
    lam : float
        Scaling term for charges used in the exponential kernel of the ESP scoring function.
    
    Returns
    -------
    loss : Array (1,)
        1 - mean(ESP Tanimoto score).
    """        
    scores = batched_obj_ROCS_esp_overlay_helper(se3_params, ref_points, fit_points, ref_charges, fit_charges, alpha, lam)
    return 1-scores.mean()


jit_val_grad_obj_ROCS_esp = jit(value_and_grad(objective_ROCS_esp_overlay_jax))


def optimize_ROCS_esp_overlay_jax(ref_points: Array,
                                  fit_points: Array,
                                  ref_charges: Array,
                                  fit_charges: Array,
                                  alpha: float,
                                  lam: float,
                                  num_repeats: int = 50,
                                  trans_centers: Union[Array, np.ndarray, None] = None,
                                  lr: float = 0.1,
                                  max_num_steps: int = 200,
                                  verbose: bool = False) -> Tuple[Array]:
    """
    Optimize alignment of fit_points with respect to ref_points using SE(3) transformations and
    maximizing electrostatic-weighted gaussian overlap score.
    
    Parameters
    ----------
    ref_points : Array (N,3)
        Reference points.
    fit_points : Array (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    ref_charges : Array (batch, N) or (N,)
        Electric potential at the corresponding ref_points coordinates.
    fit_charges : Array (batch, N) or (N,)
        Electric potential at the corresponding fit_points coordinates
    alpha : float
        Gaussian width parameter used in scoring function.
    lam : float
        Scaling term for charges used in the exponential kernel of the ESP scoring function.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    trans_centers : array (P, 3) (default=None)
        Locations to translate fit_points' center of mass as an initial guesses for optimization.
        At each translation center, 10 rotations are also sampled. So the number of initializations
        scales as (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
        aligned COM's.
        If None, then num_repeats rotations are done with aligned COM's.
    lr : float (default=0.1)
        Learning rate or step-size for optimization
    max_num_steps : int (default=200)
        Maximum number of steps to optimize over.
    verbose : bool (False)
        Print statements about initial and final similarity scores. Further, it will print scores
        during optimization at very 100 steps.

    Returns
    -------
    tuple
        aligned_points : Array (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        SE3_transform : Array (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : Array (1,)
            Tanimoto shape+ESP similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    # FASTER USING TORCH
    # se3_params = _initialize_se3_params_jax(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=torch.Tensor(np.array(ref_points)),
                                            fit_points=torch.Tensor(np.array(fit_points)),
                                            num_repeats=num_repeats).detach()
        if num_repeats == 1:
            se3_params = se3_params.unsqueeze(0)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=torch.Tensor(np.array(ref_points)),
            fit_points=torch.Tensor(np.array(fit_points)),
            trans_centers=torch.Tensor(np.array(trans_centers)),
            num_repeats_per_trans=10).detach()
    
    if len(se3_params.shape) == 1:
        se3_params.unsqueeze(0)
    se3_params = jnp.array(se3_params)

    # Create optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(se3_params)

    # Optimization loop
    if verbose:
        print(f'Initial shape similarity score: {get_overlap_esp_jax(ref_points, fit_points, ref_charges, fit_charges, alpha, lam):.3}')
    last_loss = 1
    counter = 0
    for step in range(max_num_steps):
        loss, grads = jit_val_grad_obj_ROCS_esp(se3_params, ref_points, fit_points, ref_charges, fit_charges, alpha, lam)
        updates, opt_state = optimizer.update(grads, opt_state, se3_params)
        se3_params = optax.apply_updates(se3_params, updates)
        
        # early stopping
        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    SE3_transform = vmap_get_SE3_transform_jax(se3_params)
    aligned_points = vmap_apply_SE3_transform_jax(fit_points, SE3_transform)
    scores = vmap_get_overlap_esp_jax(ref_points,
                                      aligned_points,
                                      ref_charges,
                                      fit_charges,
                                      alpha,
                                      lam)
    if num_repeats == 1:
        if verbose:
            print(f'Optimized shape+ESP similarity score: {scores:.3}')
        best_alignment = aligned_points
        best_transform = SE3_transform
        best_score = scores
    else:
        if verbose:
            print(f'Optimized shape+ESP similarity score -- max: {scores.max():3} | mean: {scores.mean():.3} | min: {scores.min():3}')
        best_idx = jnp.argmax(scores)
        best_alignment = aligned_points.at[best_idx].get()
        best_transform = SE3_transform.at[best_idx].get()
        best_score = scores.at[best_idx].get()
    return best_alignment, best_transform, best_score


def _objective_esp_combo_score_overlay_jax(se3_params,
                                           ref_centers_w_H,
                                           fit_centers_w_H,
                                           ref_centers,
                                           fit_centers,
                                           ref_points,
                                           fit_points,
                                           ref_partial_charges,
                                           fit_partial_charges,
                                           ref_surf_esp,
                                           fit_surf_esp,
                                           ref_radii,
                                           fit_radii,
                                           alpha,
                                           lam,
                                           probe_radii=1.0,
                                           esp_weight=0.5) -> Array:
    """
    Helper function to apply se3_param transformations to all fit related coordinates.
    Compute the score for that transformation.
    """

    se3_matrix = get_SE3_transform_jax(se3_params)
    fit_centers_w_H = apply_SE3_transform_jax(fit_centers_w_H, se3_matrix)
    fit_centers = apply_SE3_transform_jax(fit_centers, se3_matrix)
    fit_points = apply_SE3_transform_jax(fit_points, se3_matrix)

    score = esp_combo_score_jax(ref_centers_w_H,
                                fit_centers_w_H,
                                ref_centers,
                                fit_centers,
                                ref_points,
                                fit_points,
                                ref_partial_charges,
                                fit_partial_charges,
                                ref_surf_esp,
                                fit_surf_esp,
                                ref_radii,
                                fit_radii,
                                alpha,
                                lam,
                                probe_radii,
                                esp_weight)
    return score


batched_obj_esp_combo_score_helper = vmap(_objective_esp_combo_score_overlay_jax, (0,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None,
                                                                                   None, None))


def objective_esp_combo_score_overlay_jax(se3_params,
                                          ref_centers_w_H,
                                          fit_centers_w_H,
                                          ref_centers,
                                          fit_centers,
                                          ref_points,
                                          fit_points,
                                          ref_partial_charges,
                                          fit_partial_charges,
                                          ref_surf_esp,
                                          fit_surf_esp,
                                          ref_radii,
                                          fit_radii,
                                          alpha,
                                          lam,
                                          probe_radii=1.0,
                                          esp_weight=0.5) -> Array:
    """
    Computes the esp combo score in batch, takes the mean and convert to a loss.
    """
    scores = batched_obj_esp_combo_score_helper(se3_params,
                                                ref_centers_w_H,
                                                fit_centers_w_H,
                                                ref_centers,
                                                fit_centers,
                                                ref_points,
                                                fit_points,
                                                ref_partial_charges,
                                                fit_partial_charges,
                                                ref_surf_esp,
                                                fit_surf_esp,
                                                ref_radii,
                                                fit_radii,
                                                alpha,
                                                lam,
                                                probe_radii,
                                                esp_weight)
    return 1-scores.mean()

jit_val_grad_obj_esp_combo_score_overlay = jit(value_and_grad(objective_esp_combo_score_overlay_jax))

def convert_to_jnp_array(arr):
    if not isinstance(arr, Array):
        arr = jnp.array(arr)
    return arr


def optimize_esp_combo_score_overlay_jax(ref_centers_w_H: Union[Array, np.ndarray],
                                         fit_centers_w_H: Union[Array, np.ndarray],
                                         ref_centers: Union[Array, np.ndarray],
                                         fit_centers: Union[Array, np.ndarray],
                                         ref_points: Union[Array, np.ndarray],
                                         fit_points: Union[Array, np.ndarray],
                                         ref_partial_charges: Union[Array, np.ndarray, List],
                                         fit_partial_charges: Union[Array, np.ndarray, List],
                                         ref_surf_esp: Union[Array, np.ndarray],
                                         fit_surf_esp: Union[Array, np.ndarray],
                                         ref_radii: Union[Array, np.ndarray, List],
                                         fit_radii: Union[Array, np.ndarray, List],
                                         alpha: float,
                                         lam: float,
                                         probe_radius: float = 1.0,
                                         esp_weight: float = 0.5,
                                         num_repeats: int = 50,
                                         trans_centers: Union[Array, np.ndarray, None] = None,
                                         lr: float = 0.1,
                                         max_num_steps: int = 200,
                                         verbose: bool = False) -> Tuple[Array]:
    """
    Optimize alignment of fit_points with respect to ref_points using SE(3) transformations and
    maximizing ShaEP score.
    
    Parameters
    ----------
    ref_centers_w_H : Array (N + n_H, 3)
        Coordinates of atom centers INCLUDING hydrogens of reference molecule.
        Used for computing electrostatic potential.
        Same for fit_centers_w_H except (M + m_H, 3).

    ref_centers : Array (N, 3) or (n_surf, 3)
        Coordinates of points for reference molecule used to compute shape similarity.
        Use atom centers for volumentric similarity. Use surface centers for surface similarity.
        Same for fit_centers except (M, 3) or (m_surf, 3).
    
    ref_points : Array (n_surf, 3)
        Coordinates of surface points for referencemolecule.
        Same for fit_points except (m_surf, 3).
    
    ref_partial_charges : Array (N + n_H,)
        Partial charges corresponding to the atoms in ref_centers_w_H.
        Same for fit_partial_charges except (M + m_H,).
    
    ref_surf_esp : Array (n_surf,)
        The electrostatic potential calculated at each surface point (ref_points).
        Same for fit_surf_esp except (m_surf,)
    
    ref_radii : Array (N + n_H,)
        vdW radii corresponding to the atoms in centers_w_H_1 (angstroms)
        Same for fit_radii except (M + m_H,)

    alpha : float
        Gaussian width parameter used in shape similarity scoring function.

    lam : float (default = 0.001)
        Electrostatic potential weighting parameter (smaller = higher weight).
        0.001 was chosen as default based empirical observations of the distribution of scores
        generated by _esp_comparison before summation.

    probe_radius : float (default = 1.0)
        Surface points found within vdW radii + probe radius will be masked out. Surface generation
        uses a probe radius of 1.2 (radius of hydrogen) so we use a slightly lower radius for be
        more tolerant.
    
    esp_weight : float (default = 0.5)
        Weight to be placed on electrostatic similarity with respect to shape similarity.
        0 = only shape similarity
        1 = only electrostatic similarity

    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.

    trans_centers : array (P, 3) (default=None)
        Locations to translate fit_points' center of mass as an initial guesses for optimization.
        At each translation center, 10 rotations are also sampled. So the number of initializations
        scales as (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
        aligned COM's.
        If None, then num_repeats rotations are done with aligned COM's.

    lr : float (default=0.1)
        Learning rate or step-size for optimization

    max_num_steps : int (default=200)
        Maximum number of steps to optimize over.

    verbose : bool (False)
        Print statements about initial and final similarity scores. Further, it will print scores
        during optimization at very 100 steps.

    Returns
    -------
    tuple
        aligned_points : Array (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        SE3_transform : Array (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : Array (1,)
            ShaEP similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    # FASTER USING TORCH
    # se3_params = _initialize_se3_params_jax(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=torch.Tensor(np.array(ref_points)),
                                            fit_points=torch.Tensor(np.array(fit_points)),
                                            num_repeats=num_repeats).detach()
        if num_repeats == 1:
            se3_params = se3_params.unsqueeze(0)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=torch.Tensor(np.array(ref_points)),
            fit_points=torch.Tensor(np.array(fit_points)),
            trans_centers=torch.Tensor(np.array(trans_centers)),
            num_repeats_per_trans=10).detach()
    
    if len(se3_params.shape) == 1:
        se3_params.unsqueeze(0)
    se3_params = jnp.array(se3_params)
    
    ref_centers_w_H = convert_to_jnp_array(ref_centers_w_H)
    fit_centers_w_H = convert_to_jnp_array(fit_centers_w_H)
    ref_centers = convert_to_jnp_array(ref_centers)
    fit_centers = convert_to_jnp_array(fit_centers)
    ref_points = convert_to_jnp_array(ref_points)
    fit_points = convert_to_jnp_array(fit_points)
    ref_partial_charges = convert_to_jnp_array(ref_partial_charges)
    fit_partial_charges = convert_to_jnp_array(fit_partial_charges)
    ref_surf_esp = convert_to_jnp_array(ref_surf_esp)
    fit_surf_esp = convert_to_jnp_array(fit_surf_esp)
    ref_radii = convert_to_jnp_array(ref_radii)
    fit_radii = convert_to_jnp_array(fit_radii)

    # Create optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(se3_params)

    # Optimization loop
    if verbose:
        init_score = esp_combo_score_jax(ref_centers_w_H,
                                         fit_centers_w_H,
                                         ref_centers,
                                         fit_centers,
                                         ref_points,
                                         fit_points,
                                         ref_partial_charges,
                                         fit_partial_charges,
                                         ref_surf_esp,
                                         fit_surf_esp,
                                         ref_radii,
                                         fit_radii,
                                         alpha,
                                         lam,
                                         probe_radius,
                                         esp_weight)
        print(f'Initial ShaEP-inspired similarity score: {init_score:.3}')
    last_loss = 1
    counter = 0
    for step in range(max_num_steps):
        loss, grads = jit_val_grad_obj_esp_combo_score_overlay(se3_params,
                                                               ref_centers_w_H,
                                                               fit_centers_w_H,
                                                               ref_centers,
                                                               fit_centers,
                                                               ref_points,
                                                               fit_points,
                                                               ref_partial_charges,
                                                               fit_partial_charges,
                                                               ref_surf_esp,
                                                               fit_surf_esp,
                                                               ref_radii,
                                                               fit_radii,
                                                               alpha,
                                                               lam,
                                                               probe_radius,
                                                               esp_weight)
        updates, opt_state = optimizer.update(grads, opt_state, se3_params)
        se3_params = optax.apply_updates(se3_params, updates)
        
        # early stopping
        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    SE3_transform = vmap_get_SE3_transform_jax(se3_params)
    aligned_points = vmap_apply_SE3_transform_jax(fit_points, SE3_transform)
    aligned_centers_w_H = vmap_apply_SE3_transform_jax(fit_centers_w_H, SE3_transform)
    aligned_centers = vmap_apply_SE3_transform_jax(fit_centers, SE3_transform)
    scores = vmap_esp_combo_score(ref_centers_w_H,
                                  aligned_centers_w_H,
                                  ref_centers,
                                  aligned_centers,
                                  ref_points,
                                  aligned_points,
                                  ref_partial_charges,
                                  fit_partial_charges,
                                  ref_surf_esp,
                                  fit_surf_esp,
                                  ref_radii,
                                  fit_radii,
                                  alpha,
                                  lam,
                                  probe_radius,
                                  esp_weight)
    if num_repeats == 1:
        if verbose:
            print(f'Optimized ShaEP inspired similarity score: {scores:.3}')
        best_alignment = aligned_points
        best_transform = SE3_transform
        best_score = scores
    else:
        if verbose:
            print(f'Optimized ShaEP inspired similarity score -- max: {scores.max():3} | mean: {scores.mean():.3} | min: {scores.min():3}')
        best_idx = jnp.argmax(scores)
        best_alignment = aligned_points.at[best_idx].get()
        best_transform = SE3_transform.at[best_idx].get()
        best_score = scores.at[best_idx].get()
    return best_alignment, best_transform, best_score


def _objective_pharm_overlay_jax(se3_params: Array,
                                 ref_pharms: Array,
                                 fit_pharms: Array,
                                 ref_anchors: Array,
                                 fit_anchors: Array,
                                 ref_vectors: Array,
                                 fit_vectors: Array,
                                 similarity: _SIM_TYPE = 'tanimoto',
                                 extended_points: bool = False,
                                 only_extended: bool = False
                                 ) -> Array:
    """
    Objective function to optimize pharmacophore overlay for a single instance.
    """
    se3_matrix = get_SE3_transform_jax(se3_params)
    fit_anchors_transformed = apply_SE3_transform_jax(fit_anchors, se3_matrix)
    fit_vectors_transformed = apply_SO3_transform_jax(fit_vectors, se3_matrix)

    score = get_overlap_pharm_jax(ptype_1=ref_pharms,
                                  ptype_2=fit_pharms,
                                  anchors_1=ref_anchors,
                                  anchors_2=fit_anchors_transformed,
                                  vectors_1=ref_vectors,
                                  vectors_2=fit_vectors_transformed,
                                  similarity=similarity,
                                  extended_points=extended_points,
                                  only_extended=only_extended)
    return score

batched_obj_pharm_overlay_helper = vmap(_objective_pharm_overlay_jax, (0, None, None, None, None, None, None, None, None, None))

def objective_pharm_overlay_jax(se3_params: Array,
                                ref_pharms: Array,
                                fit_pharms: Array,
                                ref_anchors: Array,
                                fit_anchors: Array,
                                ref_vectors: Array,
                                fit_vectors: Array,
                                similarity: _SIM_TYPE = 'tanimoto',
                                extended_points: bool = False,
                                only_extended: bool = False
                                ) -> Array:
    """
    Objective function to optimize pharmacophore overlay. Batched.
    """
    scores = batched_obj_pharm_overlay_helper(se3_params,
                                              ref_pharms,
                                              fit_pharms,
                                              ref_anchors,
                                              fit_anchors,
                                              ref_vectors,
                                              fit_vectors,
                                              similarity,
                                              extended_points,
                                              only_extended)
    return 1 - scores.mean()


jit_val_grad_obj_pharm_overlay = jit(value_and_grad(objective_pharm_overlay_jax), static_argnames=('similarity', 'extended_points', 'only_extended'))


def optimize_pharm_overlay_jax(ref_pharms: Array,
                               fit_pharms: Array,
                               ref_anchors: Array,
                               fit_anchors: Array,
                               ref_vectors: Array,
                               fit_vectors: Array,
                               similarity: _SIM_TYPE = 'tanimoto',
                               extended_points: bool = False,
                               only_extended: bool = False,
                               num_repeats: int = 50,
                               trans_centers: Union[Array, np.ndarray, None] = None,
                               lr: float = 0.1,
                               max_num_steps: int = 200,
                               verbose: bool = False
                               ) -> Tuple[Array, Array, Array, Array]:
    """
    Optimize alignment of fit_anchors with respect to ref_anchors using SE(3) transformations and
    maximizing pharmacophore overlap score. JAX implementation.
    """
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=torch.Tensor(np.array(ref_anchors)),
                                            fit_points=torch.Tensor(np.array(fit_anchors)),
                                            num_repeats=num_repeats).detach()
        if num_repeats == 1:
            se3_params = se3_params.unsqueeze(0)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=torch.Tensor(np.array(ref_anchors)),
            fit_points=torch.Tensor(np.array(fit_anchors)),
            trans_centers=torch.Tensor(np.array(trans_centers)),
            num_repeats_per_trans=10).detach()

    if len(se3_params.shape) == 1:
        se3_params.unsqueeze(0)
    se3_params = jnp.array(se3_params)
    current_num_repeats = se3_params.shape[0]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(se3_params)

    if verbose:
        init_score = get_overlap_pharm_jax(ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors, similarity, extended_points, only_extended)
        print(f'Initial pharmacophore similarity score: {init_score:.3}')
    
    last_loss = 1
    counter = 0

    for step in range(max_num_steps):
        loss, grads = jit_val_grad_obj_pharm_overlay(se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vectors, fit_vectors, similarity, extended_points, only_extended)
        updates, opt_state = optimizer.update(grads, opt_state, se3_params)
        se3_params = optax.apply_updates(se3_params, updates)

        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break
    
    SE3_transform = vmap_get_SE3_transform_jax(se3_params)
    aligned_anchors = vmap_apply_SE3_transform_jax(fit_anchors, SE3_transform)
    aligned_vectors = vmap_apply_SO3_transform_jax(fit_vectors, SE3_transform)

    scores = vmap(get_overlap_pharm_jax, (None, None, None, 0, None, 0, None, None, None))(
        ref_pharms, fit_pharms, ref_anchors, aligned_anchors, ref_vectors, aligned_vectors, similarity, extended_points, only_extended
    )

    if current_num_repeats == 1:
        if verbose:
            print(f'Optimized pharmacophore similarity score: {scores.squeeze():.3}')
        best_alignment = aligned_anchors.squeeze()
        best_aligned_vectors = aligned_vectors.squeeze()
        best_transform = SE3_transform.squeeze()
        best_score = scores.squeeze()
    else:
        if verbose:
            print(f'Optimized pharmacophore similarity score -- max: {scores.max():.3} | mean: {scores.mean():.3} | min: {scores.min():.3}')
        best_idx = jnp.argmax(scores)
        best_alignment = aligned_anchors[best_idx]
        best_aligned_vectors = aligned_vectors[best_idx]
        best_transform = SE3_transform[best_idx]
        best_score = scores[best_idx]
        
    return best_alignment, best_aligned_vectors, best_transform, best_score
