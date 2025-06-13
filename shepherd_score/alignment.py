""" 
Alignment algorithms using Torch-based scoring functions.
Torch based functions can perform on batches as well as single instances.
"""
from copy import deepcopy
from typing import Union, List, Tuple
import torch
import torch.nn.functional as F
from torch import optim
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolAlign

from shepherd_score.generate_point_cloud import _get_points_fibonacci
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring import get_overlap_esp, esp_combo_score
from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm, _SIM_TYPE
from shepherd_score.alignment_utils.se3 import get_SE3_transform, apply_SE3_transform, quaternions_to_rotation_matrix, apply_SO3_transform
from shepherd_score.alignment_utils.pca_np import quaternions_for_principal_component_alignment_np
from shepherd_score.alignment_utils.pca import angle_between_vecs, rotation_axis, quaternion_from_axis_angle


def objective_ROCS_overlay(se3_params: torch.Tensor,
                           ref_points: torch.Tensor,
                           fit_points: torch.Tensor,
                           alpha: float
                          ) -> torch.Tensor:
    """
    Objective function to optimize ROCS overlay. Supports batched and non-batched inputs.
    If the inputs are batched, the loss is the average across the batch.
    
    Parameters
    ----------
    se3_params : torch.Tensor (batch, 7) or (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : torch.Tensor (batch, N, 3) or (N,3)
        Reference points. If you want to optimize to the same ref_points, with a batch of different
        se3_params, try use torch.Tensor.repeat((batch, 1, 1)).
    fit_points : torch.Tensor (batch, M, 3) or (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
        If you want to optimize to the same fit_points, with a batch of different
        se3_params, try use torch.Tensor.repeat((batch, 1, 1)).
    alpha : float
        Gaussian width parameter used in scoring function.
    
    Returns
    -------
    loss : torch.Tensor (1,)
        1 - average(Tanimoto score).
    """
    if len(fit_points.shape) - 1 != len(se3_params.shape):
        err_mssg = f'Instead these shapes were given: fit_points {fit_points.shape} and se3_params {se3_params.shape}'
        if len(fit_points.shape) == 2: # expect single instance
            raise ValueError(f'Since "fit_points" is a single point cloud, there should only be one set of "se3_params" for each batch. {err_mssg}')
        elif len(fit_points.shape) == 3: # expect batch
            raise ValueError(f'Since "fit_points" is batched, there should be a row of "se3_params" for each batch. {err_mssg}')
    
    se3_matrix = get_SE3_transform(se3_params)
    fit_points = apply_SE3_transform(fit_points, se3_matrix)
    score = get_overlap(ref_points, fit_points, alpha)
    
    # Single instance
    if len(se3_params.shape) == 1:
        return 1-score # maximize overlap
    # Batch
    elif len(se3_params.shape) == 2:
        return 1-score.mean()


def _quats_from_fibo(num_samples: int):
    """
    Computes the quaternions corresponding to the a uniform distribution (deterministic) of
    rotations. Does this by finding out the quaternions necessary to rotate a unit vector
    to points sampled on a sphere from the golden spiral method or Fibonacci sphere surface
    sampling.

    Parameters
    ----------
    num_samples : int
        Number of rotations to generate.

    Returns
    -------
    quaternions : torch.Tensor (num_samples, 4)
        quaternions corresponding to each rotation.
    """
    fibo = torch.Tensor(_get_points_fibonacci(num_samples))
    unit_v = torch.Tensor([1., 0., 0.]).repeat((num_samples, 1))
    
    angles = angle_between_vecs(unit_v, fibo)
    axes = rotation_axis(unit_v, fibo)
    quaternions = quaternion_from_axis_angle(axes, angles)
    return quaternions


def _get_45_fibo() -> torch.Tensor:
    """ Precomputed values for se3_params_from_fibo(45).
    Returns
    -------
    torch.Tensor (45,4)
        Corresponding quaternions for se3_params_from_fibo(45).
    """
    return torch.Tensor([[ 0.6501596 ,  0.        , -0.10890594, -0.7519521 ],
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


def _initialize_se3_params(ref_points: torch.Tensor,
                           fit_points: torch.Tensor,
                           num_repeats: int = 50
                           ) -> torch.Tensor:
    """
    Initialize SE(3) parameter guesses. First four values are the quaternion and the last three
    are the translation.
    All initial translations are to align fit_points COM with ref_points' COM.

    The first set corresponds to no rotation.
    The next four (if applicable) correspond to principal component alignment with ref_points.
    All other transformations are rotations generated from Fibonacci sampling of points on a
    sphere.

    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
        Reference points.
    fit_points : torch.Tensor (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    
    Returns
    -------
    se3_params : torch.Tensor (num_repeats, 7)
        Initial guesses for the SE(3) transformation parameters.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    ref_points_com = ref_points.mean(0)
    fit_points_com = fit_points.mean(0)

    # Always do all principal components if num_repeats is greater than 1
    if num_repeats > 1 and num_repeats < 5:
        num_repeats = 5

    # Center the masses together as an initial guess
    # Switch to just local optimization, no COM alignment
    if num_repeats == 1:
        se3_params = torch.zeros(7, device=ref_points.device)
        se3_params[:4] = torch.tensor([1.0, 0.0, 0.0, 0.0]) 
        # se3_params[4:] = -fit_points_com + ref_points_com
    else:
        # First guess keeps the original orientation but aligns the COMs
        se3_params = torch.zeros((num_repeats, 7), device=ref_points.device)
        se3_params[0, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0]) 
        se3_params[0, 4:] = -fit_points_com + ref_points_com

        # Align the principal components for the next 4
        if num_repeats >= 5:
            pca_quats = quaternions_for_principal_component_alignment_np(ref_points.cpu().numpy(), fit_points.cpu().numpy())
            se3_params[1:5, :4] = torch.from_numpy(pca_quats) # rotation component for centered points
            SE3_rotation = get_SE3_transform(se3_params[1:5]) # only rotation
            # Rotate translation to COM in original coordinates
            T = apply_SE3_transform(fit_points_com.repeat(4,1).unsqueeze(1), SE3_rotation).squeeze()
            # Apply translation to center COMs by taking into account implicit translation done in PCA
            se3_params[1:5, 4:] = - T + ref_points_com
        # Do random rotations 
        if num_repeats > 5:
            if num_repeats == 50:
                # Precomputed se3_params from fibonacci sampling of 45
                se3_params[5:, :4] = _get_45_fibo().to(ref_points.device)
            else:
                se3_params[5:, :4] = _quats_from_fibo(num_repeats - 5).to(ref_points.device)
            # Adjust translation to COM with the corresponding rotations
            SE3_rotation = get_SE3_transform(se3_params[5:]) # only rotation
            T = apply_SE3_transform(fit_points_com.repeat(se3_params[5:].shape[0],1).unsqueeze(1),
                                    SE3_rotation).squeeze()
            # Apply translation to center COMs by taking into account implicit translation done with rotations
            se3_params[5:, 4:] = - T + ref_points_com

    # make these parameters trainable
    se3_params.requires_grad = True
    return se3_params


def _initialize_se3_params_with_translations(ref_points: torch.Tensor,
                                             fit_points: torch.Tensor,
                                             trans_centers: torch.Tensor,
                                             num_repeats_per_trans: int = 10
                                             ) -> torch.Tensor:
    """
    Initialize SE(3) parameter guesses. First four values are the quaternion and the last three
    are the translation.
    All initial translations are to align fit_points COM with ref_points' COM.

    The first set corresponds to no rotation.
    The next four (if applicable) correspond to principal component alignment with ref_points.
    All other transformations are rotations generated from Fibonacci sampling of points on a
    sphere.

    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
        Reference points.
    fit_points : torch.Tensor (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    
    Returns
    -------
    se3_params : torch.Tensor (num_repeats, 7)
        Initial guesses for the SE(3) transformation parameters.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    ref_points_com = ref_points.mean(0)
    fit_points_com = fit_points.mean(0)

    num_repeats = num_repeats_per_trans * trans_centers.shape[0] + 5

    # First guess keeps the original orientation but aligns the COMs
    se3_params = torch.zeros((num_repeats, 7), device=ref_points.device)
    se3_params[0, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0]) 
    se3_params[0, 4:] = -fit_points_com + ref_points_com

    pca_quats = quaternions_for_principal_component_alignment_np(ref_points.cpu().numpy(), fit_points.cpu().numpy())
    se3_params[1:5, :4] = torch.from_numpy(pca_quats) # rotation component for centered points
    SE3_rotation = get_SE3_transform(se3_params[1:5]) # only rotation
    # Rotate translation to COM in original coordinates
    T = apply_SE3_transform(fit_points_com.repeat(4,1).unsqueeze(1), SE3_rotation).squeeze()
    # Apply translation to center COMs by taking into account implicit translation done in PCA
    se3_params[1:5, 4:] = - T + ref_points_com

    # Do random rotations
    if num_repeats_per_trans == 45:
        # Precomputed se3_params from fibonacci sampling of 45
        quats = _get_45_fibo().to(ref_points.device)
    else:
        quats = _quats_from_fibo(num_repeats_per_trans).to(ref_points.device)

    quats = F.normalize(quats, p=2, dim=1)
    se3_params[5:, :4] = quats.repeat(trans_centers.shape[0], 1)
    rotation_matrices = quaternions_to_rotation_matrix(quats)
    # Construct SE(3) transformation matrix
    SE3_rotation = torch.eye(4, device=se3_params.device).repeat((quats.shape[0],1,1))
    SE3_rotation[:, :3, :3] = rotation_matrices

    # Adjust translation to COM with the corresponding rotations
    T = apply_SE3_transform(fit_points_com.repeat(num_repeats_per_trans, 1).unsqueeze(1),
                            SE3_rotation).squeeze().repeat(trans_centers.shape[0], 1)
    # translation to atoms
    trans_centers_rep = torch.repeat_interleave(trans_centers, num_repeats_per_trans, 0).to(device=se3_params.device)
    # Apply translation to center COMs by taking into account implicit translation done with rotations
    se3_params[5:, 4:] = - T + trans_centers_rep

    # make these parameters trainable
    se3_params.requires_grad = True
    return se3_params


def optimize_ROCS_overlay(ref_points: torch.Tensor,
                          fit_points: torch.Tensor,
                          alpha: float,
                          num_repeats: int = 50,
                          trans_centers: Union[torch.Tensor, None] = None,
                          lr: float = 0.1,
                          max_num_steps: int = 200,
                          verbose: bool = False
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize alignment of fit_points with respect to ref_points using SE(3) transformations and
    maximizing gaussian overlap score.

    If num_repeats is 1, the initial guess for alignment is an identity rotation and aligned COMs.
    If num_repeats is 5 or greater, four initial guesses are aligned using principal components.
    
    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
        Reference points.
    fit_points : torch.Tensor (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    alpha : float
        Gaussian width parameter used in scoring function.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    trans_centers : torch.Tensor (P, 3) (default=None)
        Locations to translate fit_points' center of mass as an initial guesses for optimization.
        At each translation center, 10 rotations are also sampled. So the number of initializations
        scales as (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
        aligned COM's.
        If None, then num_repeats rotations are done with aligned COM's.
    lr : float (default=0.1)
        Learning rate or step-size for optimization
    max_num_steps : int (default=200)10
        Maximum number of steps to optimize over.
    verbose : bool (False)
        Print initial and final similarity scores with scores every 100 steps.

    Returns
    -------
    tuple
        aligned_points : torch.Tensor (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        SE3_transform : torch.Tensor (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : torch.Tensor (1,)
            Tanimoto shape similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_points,
            fit_points=fit_points,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    
    num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1
    # Create optimizer
    optimizer = optim.Adam([se3_params], lr=lr)

    # Optimization loop
    if verbose:
        print(f'Initial shape similarity score: {get_overlap(ref_points, fit_points, alpha):.3}')
    last_loss = 1
    counter = 0
    # ref_points will be broadcast by the objective/scoring function
    if num_repeats == 1:
        fit_points_to_transform = fit_points 
    else: 
        fit_points_to_transform = fit_points.repeat((num_repeats,1,1))

    for step in range(max_num_steps):
        # Forward pass: compute objective function and gradients
        loss = objective_ROCS_overlay(se3_params=se3_params,
                                      ref_points=ref_points,
                                      fit_points=fit_points_to_transform,
                                      alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1-loss.item()}")
        
        # early stopping
        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    optimized_se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(optimized_se3_params)
    aligned_points = apply_SE3_transform(fit_points_to_transform, SE3_transform)
    scores = get_overlap(centers_1=ref_points,
                         centers_2=aligned_points,
                         alpha=alpha)
    if num_repeats == 1:
        if verbose:
            print(f'Optimized shape similarity score: {scores:.3}')
        best_alignment = aligned_points.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        if verbose:
            print(f'Optimized shape similarity score -- max: {scores.max():3} | mean: {scores.mean():.3} | min: {scores.min():3}')
        best_idx = torch.argmax(scores.detach().cpu())
        best_alignment = aligned_points.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]
    return best_alignment, best_transform, best_score


def objective_ROCS_esp_overlay(se3_params: torch.Tensor,
                               ref_points: torch.Tensor,
                               fit_points: torch.Tensor,
                               ref_charges: torch.Tensor,
                               fit_charges: torch.Tensor,
                               alpha: float,
                               lam: float
                               ) -> torch.Tensor:
    """
    Objective function to optimize ROCS overlay. Supports batched and non-batched inputs.
    If the inputs are batched, the loss is the average across the batch.
    
    Parameters
    ----------
    se3_params : torch.Tensor (batch, 7) or (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_points : torch.Tensor (batch, N, 3) or (N,3)
        Reference points.
    fit_points : torch.Tensor (batch, M, 3) or (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    ref_charges : torch.Tensor (batch, N) or (N,)
        Electric potential at the corresponding ref_points coordinates.
    fit_charges : torch.Tensor (batch, M) or (M,)
        Electric potential at the corresponding fit_points coordinates
    alpha : float
        Gaussian width parameter used in scoring function.
    lam : float
        Scaling term for charges used in the exponential kernel of the ESP scoring function.
    
    Returns
    -------
    loss : torch.Tensor (1,)
        1 - mean(ESP Tanimoto score).
    """
    if len(fit_points.shape) - 1 != len(se3_params.shape):
        err_mssg = f'Instead these shapes were given: fit_points {fit_points.shape} and se3_params {se3_params.shape}'
        if len(fit_points.shape) == 2: # expect single instance
            raise ValueError(f'Since "fit_points" is a single point cloud, there should only be one set of "se3_params". {err_mssg}')
        elif len(fit_points.shape) == 3: # expect batch
            raise ValueError(f'Since "fit_points" is batched, there should be a row of "se3_params" for each batch. {err_mssg}')
    
    # Validate correspondence of points and charges dimensions.
    if len(fit_points.shape) -1 != len(fit_charges.shape) and not (fit_points.shape[:-1] == fit_charges.shape): # Check for (B,M,3) vs (B,M) or (M,3) vs (M,)
        raise ValueError(f'fit_charges should correspond to fit_points point-wise. Instead these shapes were given: fit_points {fit_points.shape} and fit_charges {fit_charges.shape}')
    if len(ref_points.shape) - 1 != len(ref_charges.shape) and not (ref_points.shape[:-1] == ref_charges.shape): # Check for (B,N,3) vs (B,N) or (N,3) vs (N,)
        raise ValueError(f'ref_charges should correspond to ref_points point-wise. Instead these shapes were given: ref_points {ref_points.shape} and ref_charges {ref_charges.shape}')
        
    se3_matrix = get_SE3_transform(se3_params)
    transformed_fit_points = apply_SE3_transform(fit_points, se3_matrix)
    score = get_overlap_esp(centers_1=ref_points,
                            centers_2=transformed_fit_points,
                            charges_1=ref_charges,
                            charges_2=fit_charges,
                            alpha=alpha,
                            lam=lam)
    
    # Single instance
    if len(se3_params.shape) == 1:
        return 1-score # maximize overlap
    # Batch
    elif len(se3_params.shape) == 2:
        return 1-score.mean()


def optimize_ROCS_esp_overlay(ref_points: torch.Tensor,
                              fit_points: torch.Tensor,
                              ref_charges: torch.Tensor,
                              fit_charges: torch.Tensor,
                              alpha: float,
                              lam: float,
                              num_repeats: int = 50,
                              trans_centers: Union[torch.Tensor, None] = None,
                              lr: float = 0.1,
                              max_num_steps: int = 200,
                              verbose: bool = False
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize alignment of fit_points with respect to ref_points using SE(3) transformations and
    maximizing electrostatic-weighted gaussian overlap score.
    
    Parameters
    ----------
    ref_points : torch.Tensor (N,3)
        Reference points.
    fit_points : torch.Tensor (M,3)
        Set of points to apply SE(3) transformations to maximize shape similarity with ref_points.
    ref_charges : torch.Tensor (batch, N) or (N,)
        Electric potential at the corresponding ref_points coordinates.
    fit_charges : torch.Tensor (batch, N) or (N,)
        Electric potential at the corresponding fit_points coordinates
    alpha : float
        Gaussian width parameter used in scoring function.
    lam : float
        Scaling term for charges used in the exponential kernel of the ESP scoring function.
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    trans_centers : torch.Tensor (P, 3) (default=None)
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
        Print initial and final similarity scores with scores every 100 steps.

    Returns
    -------
    tuple
        aligned_points : torch.Tensor (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        SE3_transform : torch.Tensor (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : torch.Tensor (1,)
            Tanimoto shape similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_points,
            fit_points=fit_points,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    
    current_num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    # Create optimizer
    optimizer = optim.Adam([se3_params], lr=lr)

    # Optimization loop
    if verbose:
        print(f'Initial ESP similarity score: {get_overlap_esp(ref_points, fit_points, ref_charges, fit_charges, alpha, lam):.3}')
    
    last_loss = torch.tensor(float('inf'), device=ref_points.device) # Initialize with a high value
    counter = 0

    # Prepare fit_points and fit_charges for transformation loop
    if current_num_repeats == 1:
        fit_points_to_transform = fit_points
        fit_charges_for_objective = fit_charges
    else:
        fit_points_to_transform = fit_points.repeat((current_num_repeats,1,1))
        # Ensure fit_charges is correctly shaped for repeat: (M,) -> (B,M) or (B,M) -> (B,M)
        if fit_charges.dim() == 1: # (M,)
             fit_charges_for_objective = fit_charges.repeat((current_num_repeats,1))
        elif fit_charges.dim() == 2 and fit_charges.shape[0] == 1 : # (1,M)
             fit_charges_for_objective = fit_charges.repeat((current_num_repeats,1))
        elif fit_charges.dim() == 2 and fit_charges.shape[0] == fit_points.shape[0] and fit_points.dim()==3 and current_num_repeats == fit_charges.shape[0]: # Already batched correctly for fit_points
             fit_charges_for_objective = fit_charges # This case should ideally not occur if initial fit_charges is single mol
        else: # Default/expected: initial fit_charges is for single molecule, to be repeated
             fit_charges_for_objective = fit_charges.repeat((current_num_repeats,1))


    for step in range(max_num_steps):
        loss = objective_ROCS_esp_overlay(se3_params=se3_params,
                                          ref_points=ref_points,
                                          fit_points=fit_points_to_transform,
                                          ref_charges=ref_charges,
                                          fit_charges=fit_charges_for_objective,
                                          alpha=alpha,
                                          lam=lam)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1-loss.item():.3f}")
        
        # early stopping
        if torch.abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    optimized_se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(optimized_se3_params)
    aligned_points = apply_SE3_transform(fit_points_to_transform, SE3_transform)
    
    scores = get_overlap_esp(centers_1=ref_points,  
                             charges_1=ref_charges, 
                             centers_2=aligned_points,
                             charges_2=fit_charges_for_objective,
                             alpha=alpha,
                             lam=lam)
    
    if current_num_repeats == 1:
        if verbose:
            print(f'Optimized ESP similarity score: {scores.item():.3}')
        best_alignment = aligned_points.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        best_idx = torch.argmax(scores.detach().cpu())
        if verbose:
            print(f'Optimized ESP similarity score -- max: {scores[best_idx].item():.3} | mean: {scores.mean().item():.3} | min: {scores.min().item():.3}')
        best_alignment = aligned_points.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]
        
    return best_alignment, best_transform, best_score


def objective_esp_combo_score_overlay(se3_params: torch.Tensor,
                                      ref_centers_w_H: torch.Tensor,
                                      fit_centers_w_H: torch.Tensor,
                                      ref_centers: torch.Tensor,
                                      fit_centers: torch.Tensor,
                                      ref_points: torch.Tensor,
                                      fit_points: torch.Tensor,
                                      ref_partial_charges: torch.Tensor,
                                      fit_partial_charges: torch.Tensor,
                                      ref_surf_esp: torch.Tensor,
                                      fit_surf_esp: torch.Tensor,
                                      ref_radii: torch.Tensor,
                                      fit_radii: torch.Tensor,
                                      alpha: float,
                                      lam: float,
                                      probe_radius: float,
                                      esp_weight: float
                                      ) -> torch.Tensor:
    """
    Objective for ESP combo score. Handles broadcasting for ref_* inputs.
    fit_* inputs are expected to be repeated if se3_params is batched.
    """
    # Validate that fit_* inputs that are transformed have a consistent batch dimension with se3_params
    if len(fit_points.shape) - 1 != len(se3_params.shape): # Using fit_points as representative
        err_mssg = f'Shape mismatch: fit_points {fit_points.shape}, se3_params {se3_params.shape}'
        if len(fit_points.shape) == 2:
             raise ValueError(f'Single fit_points expects single se3_params. {err_mssg}')
        elif len(fit_points.shape) == 3:
             raise ValueError(f'Batched fit_points expects batched se3_params. {err_mssg}')

    se3_matrix = get_SE3_transform(se3_params)
    transformed_fit_centers_w_H = apply_SE3_transform(fit_centers_w_H, se3_matrix)
    transformed_fit_centers = apply_SE3_transform(fit_centers, se3_matrix)
    transformed_fit_points = apply_SE3_transform(fit_points, se3_matrix)
    
    score = esp_combo_score(centers_w_H_1=ref_centers_w_H,
                            centers_w_H_2=transformed_fit_centers_w_H,
                            centers_1=ref_centers,
                            centers_2=transformed_fit_centers,
                            points_1=ref_points,
                            points_2=transformed_fit_points,
                            partial_charges_1=ref_partial_charges,
                            partial_charges_2=fit_partial_charges,
                            point_charges_1=ref_surf_esp,
                            point_charges_2=fit_surf_esp,
                            radii_1=ref_radii,
                            radii_2=fit_radii,
                            alpha=alpha,
                            lam=lam,
                            probe_radius=probe_radius,
                            esp_weight=esp_weight
                            )

    if len(se3_params.shape) == 1:
        return 1-score
    elif len(se3_params.shape) == 2:
        return 1-score.mean()


def optimize_esp_combo_score_overlay(ref_centers_w_H: torch.Tensor,
                                     fit_centers_w_H: torch.Tensor,
                                     ref_centers: torch.Tensor,
                                     fit_centers: torch.Tensor,
                                     ref_points: torch.Tensor,
                                     fit_points: torch.Tensor,
                                     ref_partial_charges: torch.Tensor,
                                     fit_partial_charges: torch.Tensor,
                                     ref_surf_esp: torch.Tensor,
                                     fit_surf_esp: torch.Tensor,
                                     ref_radii: torch.Tensor,
                                     fit_radii: torch.Tensor,
                                     alpha: float,
                                     lam: float,
                                     probe_radius: float = 1.0,
                                     esp_weight: float = 0.5,
                                     num_repeats: int = 50,
                                     trans_centers: Union[torch.Tensor, None] = None,
                                     lr: float = 0.1,
                                     max_num_steps: int = 200,
                                     verbose: bool = False
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize alignment using ESP combo score.
    """
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_points,
            fit_points=fit_points,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    
    current_num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    optimizer = optim.Adam([se3_params], lr=lr)

    if verbose:
        init_score = esp_combo_score(
            centers_w_H_1=ref_centers_w_H,
            centers_w_H_2=fit_centers_w_H,
            centers_1=ref_centers,
            centers_2=fit_centers,
            points_1=ref_points,
            points_2=fit_points,
            partial_charges_1=ref_partial_charges,
            partial_charges_2=fit_partial_charges,
            point_charges_1=ref_surf_esp,
            point_charges_2=fit_surf_esp,
            radii_1=ref_radii,
            radii_2=fit_radii,
            alpha=alpha,
            lam=lam,
            probe_radius=probe_radius,
            esp_weight=esp_weight
        )
        print(f'Initial ESP-combo score: {init_score.item():.3}')
    
    last_loss = torch.tensor(float('inf'), device=ref_points.device)
    counter = 0

    # Prepare fit_* tensors for the optimization loop
    if current_num_repeats == 1:
        fit_centers_w_H_obj = fit_centers_w_H
        fit_centers_obj = fit_centers
        fit_points_obj = fit_points
        fit_partial_charges_obj = fit_partial_charges
        fit_surf_esp_obj = fit_surf_esp
        fit_radii_obj = fit_radii
    else:
        fit_centers_w_H_obj = fit_centers_w_H.repeat((current_num_repeats, 1, 1))
        fit_centers_obj = fit_centers.repeat((current_num_repeats, 1, 1))
        fit_points_obj = fit_points.repeat((current_num_repeats, 1, 1))
        # For 1D tensors like charges/radii, ensure correct repeat
        fit_partial_charges_obj = fit_partial_charges.repeat((current_num_repeats, 1) if fit_partial_charges.dim() > 0 else (current_num_repeats,))
        fit_surf_esp_obj = fit_surf_esp.repeat((current_num_repeats, 1) if fit_surf_esp.dim() > 0 else (current_num_repeats,))
        fit_radii_obj = fit_radii.repeat((current_num_repeats, 1) if fit_radii.dim() > 0 else (current_num_repeats,))

    for step in range(max_num_steps):
        loss = objective_esp_combo_score_overlay(
            se3_params=se3_params,
            ref_centers_w_H=ref_centers_w_H,
            fit_centers_w_H=fit_centers_w_H_obj ,
            ref_centers=ref_centers,
            fit_centers=fit_centers_obj,
            ref_points=ref_points,
            fit_points=fit_points_obj,
            ref_partial_charges=ref_partial_charges,
            fit_partial_charges=fit_partial_charges_obj,
            ref_surf_esp=ref_surf_esp,
            fit_surf_esp=fit_surf_esp_obj,
            ref_radii=ref_radii,
            fit_radii=fit_radii_obj,
            alpha=alpha,
            lam=lam,
            probe_radius=probe_radius,
            esp_weight=esp_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1-loss.item():.3f}")
        
        if torch.abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    optimized_se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(optimized_se3_params)

    # Transform the correct fit inputs for final scoring
    aligned_fit_centers_w_H = apply_SE3_transform(fit_centers_w_H_obj, SE3_transform)
    aligned_fit_centers = apply_SE3_transform(fit_centers_obj, SE3_transform)
    aligned_fit_points = apply_SE3_transform(fit_points_obj, SE3_transform)
    
    scores = esp_combo_score(
        centers_w_H_1=ref_centers_w_H,
        centers_w_H_2=aligned_fit_centers_w_H,
        centers_1=ref_centers,
        centers_2=aligned_fit_centers,
        points_1=ref_points,
        points_2=aligned_fit_points,
        partial_charges_1=ref_partial_charges,
        partial_charges_2=fit_partial_charges_obj,
        point_charges_1=ref_surf_esp,
        point_charges_2=fit_surf_esp_obj,
        radii_1=ref_radii,
        radii_2=fit_radii_obj,
        alpha=alpha,
        lam=lam,
        probe_radius=probe_radius,
        esp_weight=esp_weight
    )

    if current_num_repeats == 1:
        if verbose:
            print(f'Optimized ESP-combo score: {scores.item():.3}')
        best_alignment = aligned_fit_points.cpu() # Consistent: return aligned surface points
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        best_idx = torch.argmax(scores.detach().cpu())
        if verbose:
            print(f'Optimized ESP-combo score -- max: {scores[best_idx].item():.3} | mean: {scores.mean().item():.3} | min: {scores.min().item():.3}')
        best_alignment = aligned_fit_points.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]
        
    return best_alignment, best_transform, best_score


def crippen_align(ref_rdmol: Chem.rdchem.Mol,
                  fit_rdmol: Chem.rdchem.Mol
                  ) -> Chem.rdchem.Mol:
    """
    Align fit_rdmol with respect to ref_rdmol with rdkit's Crippen Alignment algorithm.
    
    Parameters
    ----------
    ref_rdmol : rdkit.Chem.rdchem.Mol
        Reference molecule that fit_rdmol is aligned to.
    fit_rdmol : rdkit.Chem.rdchem.Mol
        Fit molecule that will be aligned to the reference.

    Returns
    -------
    aligned_fit_rdmol : rdkit.Chem.rdchem.Mol
        Fit molecule with new aligned coordinates.
    """
    ref_rdmol2 = deepcopy(Chem.RemoveHs(ref_rdmol))
    fit_rdmol2 = deepcopy(Chem.RemoveHs(fit_rdmol))
    prbCrippen = rdMolDescriptors._CalcCrippenContribs(fit_rdmol2)
    refCrippen = rdMolDescriptors._CalcCrippenContribs(ref_rdmol)
    alignment = rdMolAlign.GetCrippenO3A(fit_rdmol2, ref_rdmol2, prbCrippen, refCrippen, 0, 0)
    alignment.Align()
    return fit_rdmol2


def objective_pharm_overlay(se3_params: torch.Tensor,
                            ref_pharms: torch.Tensor,
                            fit_pharms: torch.Tensor,
                            ref_anchors: torch.Tensor,
                            fit_anchors: torch.Tensor,
                            ref_vectors: torch.Tensor,
                            fit_vectors: torch.Tensor,
                            similarity: _SIM_TYPE = 'tanimoto',
                            extended_points: bool = False,
                            only_extended: bool = False
                            ) -> torch.Tensor:
    """
    Objective function to optimize ROCS overlay. Supports batched and non-batched inputs.
    If the inputs are batched, the loss is the average across the batch.
    
    Parameters
    ----------
    se3_params : torch.Tensor (batch, 7) or (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    ref_anchors : torch.Tensor (batch, N, 3) or (N,3)
        Reference anchors. If you want to optimize to the same ref_anchors, with a batch of different
        se3_params, try use torch.Tensor.repeat((batch, 1, 1)).
    fit_anchors : torch.Tensor (batch, M, 3) or (M,3)
        Set of anchors to apply SE(3) transformations to maximize shape similarity with ref_anchors.
        If you want to optimize to the same fit_anchors, with a batch of different
        se3_params, try use torch.Tensor.repeat((batch, 1, 1)).
    ref_charges : torch.Tensor (batch, N) or (N,)
        Electric potential at the corresponding ref_anchors coordinates.
    fit_charges : torch.Tensor (batch, N) or (N,)
        Electric potential at the corresponding fit_anchors coordinates
    alpha : float
        Gaussian width parameter used in scoring function.
    lam : float
        Scaling term for charges used in the exponential kernel of the ESP scoring function.
    
    Returns
    -------
    loss : torch.Tensor (1,)
        1 - mean(ESP Tanimoto score).
    """
    if len(fit_anchors.shape) - 1 != len(se3_params.shape):
        err_mssg = f'Instead these shapes were given: fit_anchors {fit_anchors.shape} and se3_params {se3_params.shape}'
        if len(fit_anchors.shape) == 2: # expect single instance
            raise ValueError(f'Since "fit_anchors" is a single point cloud, there should only be one set of "se3_params" for each batch. {err_mssg}')

        elif len(fit_anchors.shape) == 3: # expect batch
            raise ValueError(f'Since "fit_anchors" is batched, there should be a row of "se3_params" for each batch. {err_mssg}')
        
    se3_matrix = get_SE3_transform(se3_params)
    fit_anchors = apply_SE3_transform(fit_anchors, se3_matrix)
    fit_vectors = apply_SO3_transform(fit_vectors, se3_matrix)
    score = get_overlap_pharm(ptype_1=ref_pharms,
                            ptype_2=fit_pharms,
                            anchors_1=ref_anchors,
                            anchors_2=fit_anchors,
                            vectors_1=ref_vectors,
                            vectors_2=fit_vectors,
                            similarity=similarity,
                            extended_points=extended_points,
                            only_extended=only_extended)
    # Single instance
    if len(se3_params.shape) == 1:
        return 1-score # maximize overlap
    # Batch
    elif len(se3_params.shape) == 2:
        return 1-score.mean()


def optimize_pharm_overlay(ref_pharms: torch.Tensor,
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
    Optimize alignment of fit_anchors with respect to ref_anchors using SE(3) transformations and
    maximizing electrostatic-weighted gaussian overlap score.
    
    Parameters
    ----------
    ref_pharms : torch.Tensor (N,) Indices reflecting pharmacophore type of reference molecule
    fit_pharms : torch.Tensor (N,) Indices reflecting pharmacophore type of fit molecule
    ref_anchors : torch.Tensor (N,3) Reference pharmacophore positions (anchors).
    fit_anchors : torch.Tensor (M,3) Set of anchors to align pharmacophores to ref.
    ref_vectors : torch.Tensor (batch, N, 3) or (N,3) Relative unit vectors to the anchor anchors.
    fit_vectors : torch.Tensor (batch, N, 3) or (N,3) Relative unit vectors to the anchor anchors.
    similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
        Specifies what similarity function to use.
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
    extended_points : bool of whether to score HBA/HBD with gaussian overlaps of extended points.
    only_extended : bool for when `extended_points` is True, decide whether to only score the
                    extended points (ignore anchor overlaps)
    num_repeats : int (default=50)
        Number of different random initializations of SE(3) transformation parameters.
    trans_centers : torch.Tensor (P, 3) (default=None)
        Locations to translate fit_points' center of mass as an initial guesses for optimization.
        At each translation center, 10 rotations are also sampled. So the number of initializations
        scales as (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
        aligned COM's.
        If None, then num_repeats rotations are done with aligned COM's.
    lr : float (default=0.1) Learning rate or step-size for optimization
    max_num_steps : int (default=200) Maximum number of steps to optimize over.
    verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

    Returns
    -------
    tuple
        aligned_points : torch.Tensor (M,3)
            The transformed point cloud for fit_points using the optimized SE(3) transformation for
            alignment with ref_points.
        aligned_vectors : torch.Tensor (M,3)
            The transformed vectors for fit_vectors using the optimized SO(3) transformation for
            aligment with ref_points.
        SE3_transform : torch.Tensor (4,4)
            Optimized SE(3) transformation matrix used to obtain aligned_points from fit_points.
        score : torch.Tensor (1,)
            Tanimoto shape similarity score for the optimal transformation.
    """
    # Initial guess for SE(3) parameters (quaternion followed by translation)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_points=ref_anchors, fit_points=fit_anchors, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(
            ref_points=ref_anchors,
            fit_points=fit_anchors,
            trans_centers=trans_centers,
            num_repeats_per_trans=10)
    num_repeats = len(se3_params) if len(se3_params.shape) == 2 else 1

    # Create optimizer
    optimizer = optim.Adam([se3_params], lr=lr)

    # Optimization loop
    if verbose:
        init_score = get_overlap_pharm(
            ref_pharms,
            fit_pharms,
            ref_anchors,
            fit_anchors,
            ref_vectors,
            fit_vectors,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended
        )
        print(f'Initial pharmacophore similarity score: {init_score:.3}')
    last_loss = 1
    counter = 0
    ref_pharms_rep = ref_pharms.repeat((num_repeats,1)).squeeze(0)
    fit_pharms_rep = fit_pharms.repeat((num_repeats,1)).squeeze(0)
    ref_anchors_rep = ref_anchors.repeat((num_repeats,1,1)).squeeze(0)
    fit_anchors_rep = fit_anchors.repeat((num_repeats,1,1)).squeeze(0)
    ref_vectors_rep = ref_vectors.repeat((num_repeats,1,1)).squeeze(0)
    fit_vectors_rep = fit_vectors.repeat((num_repeats,1,1)).squeeze(0)

    for step in range(max_num_steps):
        # Forward pass: compute objective function and gradients
        loss = objective_pharm_overlay(
            se3_params=se3_params,
            ref_pharms=ref_pharms_rep,
            fit_pharms=fit_pharms_rep,
            ref_anchors=ref_anchors_rep,
            fit_anchors=fit_anchors_rep,
            ref_vectors=ref_vectors_rep,
            fit_vectors=fit_vectors_rep,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if verbose and step % 100 == 0:
            print(f"Step {step}, Score: {1-loss.item()}")
        
        # early stopping
        if abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    # Extract optimized SE(3) parameters
    optimized_se3_params = se3_params.detach()
    SE3_transform = get_SE3_transform(optimized_se3_params)
    aligned_anchors = apply_SE3_transform(fit_anchors_rep, SE3_transform)
    aligned_vectors = apply_SO3_transform(fit_vectors_rep, SE3_transform)
    scores = get_overlap_pharm(
        ptype_1=ref_pharms_rep,
        ptype_2=fit_pharms_rep,
        anchors_1=ref_anchors_rep,
        anchors_2=aligned_anchors,
        vectors_1=ref_vectors_rep,
        vectors_2=aligned_vectors,
        similarity=similarity,
        extended_points=extended_points,
        only_extended=only_extended
    )
    if num_repeats == 1:
        if verbose:
            print(f'Optimized pharmacophore similarity score: {scores:.3}')
        best_alignment = aligned_anchors.cpu()
        best_aligned_vectors = aligned_vectors.cpu()
        best_transform = SE3_transform.cpu()
        best_score = scores.cpu()
    else:
        if verbose:
            print(f'Optimized pharmacophore similarity score -- max: {scores.max():3} | mean: {scores.mean():.3} | min: {scores.min():3}')
        best_idx = torch.argmax(scores.detach().cpu())
        best_alignment = aligned_anchors.cpu()[best_idx]
        best_aligned_vectors = aligned_vectors.cpu()[best_idx]
        best_transform = SE3_transform.cpu()[best_idx]
        best_score = scores.cpu()[best_idx]
    return best_alignment, best_aligned_vectors, best_transform, best_score
