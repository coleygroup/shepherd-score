""" 
Gaussian volume overlap scoring functions -- Shape-only (i.e., not color)

Batched and non-batched functionalities

Reference math:
https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K
https://doi.org/10.1021/j100011a016
"""
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F


def VAB_2nd_order(centers_1: torch.Tensor,
                  centers_2: torch.Tensor,
                  alpha: float
                  ) -> torch.Tensor:
    """
    2nd order volume overlap of AB.
    Torch implementation supporting single instances, matched batches, and
    broadcasting scenarios.
    The function relies on torch.cdist for calculating squared distances,
    which handles necessary broadcasting for batch dimensions efficiently.

    R2_cdist will have a shape like (Batch, N_c1, N_c2) or (N_c1, N_c2)
    depending on the input shapes. torch.cdist handles the broadcasting
    of batch dimensions. For example:
    - c1=(N,3), c2=(M,3)       -> cdist_out=(N,M)
    - c1=(B,N,3), c2=(B,M,3)   -> cdist_out=(B,N,M)
    - c1=(N,3), c2=(B,M,3)     -> cdist_out=(B,N,M) (c1 broadcasted)
    - c1=(1,N,3), c2=(B,M,3)   -> cdist_out=(B,N,M) (c1 broadcasted)
    """
    R2_cdist = torch.cdist(centers_1, centers_2)**2.0

    # Batched case
    if len(R2_cdist.shape) == 3:
        R2 = R2_cdist.permute(0, 2, 1)  # (B, N_c2, N_c1)
        
        VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                               ((2 * alpha)**1.5) *
                                               torch.exp(-(alpha / 2) * R2),
                                               dim=2),
                                     dim=1)  # Resulting shape: (B,)

    # Single instance
    elif len(R2_cdist.shape) == 2:
        # No transpose needed since sum
        VAB_second_order = torch.sum((np.pi**1.5) /
                                     ((2 * alpha)**1.5) *
                                     torch.exp(-(alpha / 2) * R2_cdist))
    else:
        raise ValueError(
            f"Unexpected shape from torch.cdist: {R2_cdist.shape}. "
            f"Input shapes were: centers_1={centers_1.shape}, centers_2={centers_2.shape}"
        )

    return VAB_second_order


def shape_tanimoto(centers_1: torch.Tensor,
                   centers_2: torch.Tensor,
                   alpha: float
                   ) -> torch.Tensor:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order(centers_1, centers_1, alpha)
    VBB = VAB_2nd_order(centers_2, centers_2, alpha)
    VAB = VAB_2nd_order(centers_1, centers_2, alpha)
    return VAB / (VAA + VBB - VAB)


def get_overlap(centers_1: Union[torch.Tensor, np.ndarray],
                centers_2: Union[torch.Tensor, np.ndarray],
                alpha:float = 0.81
                ) -> torch.Tensor:
    """
    Volumetric shape similarity with tunable "alpha" Gaussian width parameter.
    Handles single instances, matched batches, and broadcasting scenarios
    (e.g., centers_1=(N,3) or (1,N,3) and centers_2=(B,M,3)). PyTorch implementation.

    Parameters
    ----------
    centers_1 : Union[torch.Tensor, np.ndarray] (batch, N, 3) or (N, 3)
        Coordinates of each point of the first point cloud.
        Can be (N,3) for a single instance, (B,N,3) for a batch, or (1,N,3)
        for a single instance to be broadcast against a batch in centers_2.
    centers_2 : Union[torch.Tensor, np.ndarray] (batch, M, 3) or (M, 3)
        Coordinates of each point of the second point cloud.
        Can be (M,3) for a single instance, (B,M,3) for a batch, or (1,M,3)
        for a single instance to be broadcast against a batch in centers_1.
    alpha : float (default=0.81)
        Gaussian width parameter. Lower value corresponds to wider Gaussian (longer tail).
    
    Returns
    -------
    torch.Tensor : (batch,) or scalar
        The Tanimoto similarity score. Returns a scalar if both inputs are single instances,
        or a 1D tensor of shape (batch,) if at least one input is batched.
    """
    if isinstance(centers_1, np.ndarray):
        centers_1 = torch.Tensor(centers_1)
    if isinstance(centers_2, np.ndarray):
        centers_2 = torch.Tensor(centers_2)
    tanimoto = shape_tanimoto(centers_1, centers_2, alpha)
    return tanimoto


def VAB_2nd_order_mask(centers_1: torch.Tensor,
                       centers_2: torch.Tensor,
                       alpha: float,
                       mask_1: torch.Tensor,
                       mask_2: torch.Tensor
                       ) -> torch.Tensor:
    """
    2nd order volume overlap of AB with masking.
    Torch implementation supporting single instances, matched batches, and broadcasting.
    Masks are applied to the interaction terms.

    Parameters
    ----------
    centers_1 : torch.Tensor (N,3) or (B,N,3) or (1,N,3)
        Coordinates for the first set of points.
    centers_2 : torch.Tensor (M,3) or (B,M,3) or (1,M,3)
        Coordinates for the second set of points.
    alpha : float
        Gaussian width parameter.
    mask_1 : torch.Tensor (N,) or (B,N) or (1,N)
        Mask for centers_1. Boolean or float (0/1).
    mask_2 : torch.Tensor (M,) or (B,M) or (1,M)
        Mask for centers_2. Boolean or float (0/1).

    Returns
    -------
    torch.Tensor
        Scalar or (B,) tensor of overlap scores.
    """
    R2_cdist = torch.cdist(centers_1, centers_2)**2.0  # Shape: (B, N1, N2) or (N1, N2)

    m1_final = mask_1.float()
    m2_final = mask_2.float()

    # Batched case
    if R2_cdist.dim() == 3:
        R2 = R2_cdist.permute(0, 2, 1) # Shape: (B, N2, N1)

        # Ensure masks are at least 2D for broadcasting with batch dimension
        if m1_final.dim() == 1:
            m1_final = m1_final.unsqueeze(0) # (1, N1)
        if m2_final.dim() == 1:
            m2_final = m2_final.unsqueeze(0) # (1, N2)
        
        # m1_final_unsqueezed: (B or 1, N1, 1), m2_final_unsqueezed: (B or 1, 1, N2)
        m1_final_unsqueezed = m1_final.unsqueeze(2)
        m2_final_unsqueezed = m2_final.unsqueeze(1)
        
        # mask_mat will broadcast to (B, N1, N2) then permuted to (B, N2, N1)
        mask_mat = (m1_final_unsqueezed * m2_final_unsqueezed).permute(0, 2, 1)

        VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                               ((2 * alpha)**1.5) *
                                               mask_mat *
                                               torch.exp(-(alpha / 2) * R2),
                                               dim=2),
                                     dim=1)
    # Single instance
    elif R2_cdist.dim() == 2: # R2_cdist is (N1, N2)
        # m1_final should be (N1,), m2_final should be (N2,)
        # mask_mat should be (N1, N2) for direct multiplication with R2_cdist
        mask_mat = m1_final.unsqueeze(1) * m2_final.unsqueeze(0) # (N1,1) * (1,N2) -> (N1,N2)
        
        VAB_second_order = torch.sum((np.pi**1.5) /
                                     ((2 * alpha)**1.5) *
                                     mask_mat *
                                     torch.exp(-(alpha / 2) *
                                     R2_cdist))
    else:
        raise ValueError(
            f"Unexpected shape from torch.cdist: {R2_cdist.shape}. "
            f"Input shapes were: centers_1={centers_1.shape}, centers_2={centers_2.shape}"
        )
    return VAB_second_order


def VAB_2nd_order_mask_batch(cdist_21: torch.Tensor,
                             alpha: float,
                             mask_1: torch.Tensor,
                             mask_2: torch.Tensor
                             ) -> torch.Tensor:
    """
    2nd order volume overlap of AB (batched) with masking, using precomputed cdist.
    Assumes inputs `cdist_21`, `mask_1`, `mask_2` are already batched and broadcast-compatible.

    Parameters
    ----------
    cdist_21 : torch.Tensor (B,M,N)
        Precomputed squared Euclidean distances: (torch.cdist(centers_2, centers_1)**2.0).
        Note the order: cdist(c2, c1) gives (B, M, N) which is R_21^2.
        If cdist(c1,c2).permute(0,2,1) was used, it's also (B,M,N).
    alpha : float
        Gaussian width parameter.
    mask_1 : torch.Tensor (B,N) or (1,N)
        Mask for the first set of points (corresponding to N in cdist_21).
    mask_2 : torch.Tensor (B,M) or (1,M)
        Mask for the second set of points (corresponding to M in cdist_21).
    
    Returns
    -------
    torch.Tensor : (B,)
        Batched Tanimoto similarity scores.
    """
    # mask_1: (B,N) -> (B,1,N) after unsqueeze
    # mask_2: (B,M) -> (B,M,1) after unsqueeze
    # mask_mat: (B,M,1) * (B,1,N) -> (B,M,N) due to broadcasting rules
    mask_mat = (mask_2.unsqueeze(2) * mask_1.unsqueeze(1))

    VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                            ((2 * alpha)**1.5) *
                                            mask_mat *
                                            torch.exp(-(alpha / 2) * cdist_21),
                                            dim = 2),
                                dim = 1)
    return VAB_second_order


def VAB_2nd_order_cosine(centers_1: torch.Tensor,
                         centers_2: torch.Tensor,
                         vectors_1: torch.Tensor,
                         vectors_2: torch.Tensor,
                         alpha: float,
                         allow_antiparallel: bool,
                         ) -> torch.Tensor:
    """
    2nd order volume overlap of AB weighted by cosine similarity.
    Torch implementation supporting single instances, matched batches, and broadcasting.

    Parameters
    ----------
    centers_1 : torch.Tensor (N,3) or (B,N,3) or (1,N,3)
    centers_2 : torch.Tensor (M,3) or (B,M,3) or (1,M,3)
    vectors_1 : torch.Tensor (N,3) or (B,N,3) or (1,N,3)
    vectors_2 : torch.Tensor (M,3) or (B,M,3) or (1,M,3)
    alpha : float
    allow_antiparallel : bool

    Returns
    -------
    torch.Tensor
        Scalar or (B,) tensor of overlap scores.
    """
    R2_cdist = torch.cdist(centers_1, centers_2)**2.0  # (B, N1, N2) or (N1, N2)

    # Normalize vectors
    # For batched: (B, N, 3) -> (B, N, 3)
    # For single: (N, 3) -> (N, 3)
    norm_dim = vectors_1.dim() - 1
    vec1_norm = F.normalize(vectors_1, p=2, dim=norm_dim)
    vec2_norm = F.normalize(vectors_2, p=2, dim=norm_dim)

    # Batched case
    if R2_cdist.dim() == 3:
        R2_permuted = R2_cdist.permute(0, 2, 1)  # (B, N2, N1)
        
        # Cosine similarity V2: (B, N1, N2)
        # vec1_norm: (B or 1, N1, 3), vec2_norm: (B or 1, N2, 3)
        # Need vec2_norm.permute(0,2,1) for matmul -> (B or 1, 3, N2)
        # Result of matmul: (B, N1, N2)
        V2_sim_N1_N2 = torch.matmul(vec1_norm, vec2_norm.permute(0,2,1) if vec2_norm.dim() == 3 else vec2_norm.T)
        V2_sim_permuted = V2_sim_N1_N2.permute(0, 2, 1) # (B, N2, N1) for multiplication with R2_permuted

        if allow_antiparallel:
            V2_sim_permuted = torch.abs(V2_sim_permuted)
        else:
            V2_sim_permuted = torch.clamp(V2_sim_permuted, 0., 1.) # wrong direction should be 0 rather than negative
        V2_weighted_permuted = (V2_sim_permuted + 2.) / 3. # Following PheSA's suggestion for weighting

        VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                               ((2 * alpha)**1.5) *
                                               V2_weighted_permuted *
                                               torch.exp(-(alpha / 2) * R2_permuted), # R2 is (B, N2, N1)
                                               dim=2),
                                     dim=1)
    # Single instance
    elif R2_cdist.dim() == 2: # R2_cdist is (N1, N2)
        # Cosine similarity V2_sim: (N1, N2)
        # vec1_norm: (N1,3), vec2_norm: (N2,3)
        V2_sim_N1_N2 = torch.matmul(vec1_norm, vec2_norm.T) # (N1,N2)
        # No transpose needed for V2_sim_N1_N2 if R2_cdist is used directly

        if allow_antiparallel:
            V2_sim_N1_N2 = torch.abs(V2_sim_N1_N2)
        else:
            V2_sim_N1_N2 = torch.clamp(V2_sim_N1_N2, 0., 1.)
        V2_weighted_N1_N2 = (V2_sim_N1_N2 + 2.) / 3. # Following PheSA's suggestion for weighting

        VAB_second_order = torch.sum((np.pi**1.5) \
                                     / ((2 * alpha)**1.5) \
                                     * V2_weighted_N1_N2 \
                                     * torch.exp(-(alpha / 2) *
                                     R2_cdist)) # Use R2_cdist directly
    else:
        raise ValueError(
            f"Unexpected shape from torch.cdist: {R2_cdist.shape}. "
            f"Input shapes were: centers_1={centers_1.shape}, centers_2={centers_2.shape}"
        )
    return VAB_second_order


def VAB_2nd_order_cosine_mask(centers_1: torch.Tensor,
                              centers_2: torch.Tensor,
                              vectors_1: torch.Tensor,
                              vectors_2: torch.Tensor,
                              alpha: float,
                              allow_antiparallel: bool,
                              mask_1: torch.Tensor,
                              mask_2: torch.Tensor
                              ) -> torch.Tensor:
    """
    2nd order volume overlap of AB weighted by cosine similarity, with masking.
    Torch implementation supporting single instances, matched batches, and broadcasting.

    Parameters
    ----------
    centers_1 : torch.Tensor (N,3) or (B,N,3) or (1,N,3)
    centers_2 : torch.Tensor (M,3) or (B,M,3) or (1,M,3)
    vectors_1 : torch.Tensor (N,3) or (B,N,3) or (1,N,3)
    vectors_2 : torch.Tensor (M,3) or (B,M,3) or (1,M,3)
    alpha : float
    allow_antiparallel : bool
    mask_1 : torch.Tensor (N,) or (B,N) or (1,N)
    mask_2 : torch.Tensor (M,) or (B,M) or (1,M)

    Returns
    -------
    torch.Tensor
        Scalar or (B,) tensor of overlap scores.
    """
    R2_cdist = torch.cdist(centers_1, centers_2)**2.0  # (B, N1, N2) or (N1, N2)

    # Normalize vectors
    norm_dim_v1 = vectors_1.dim() - 1
    norm_dim_v2 = vectors_2.dim() - 1
    vec1_norm = F.normalize(vectors_1, p=2, dim=norm_dim_v1)
    vec2_norm = F.normalize(vectors_2, p=2, dim=norm_dim_v2)
    
    m1_final = mask_1.float()
    m2_final = mask_2.float()

    # Batched case
    if R2_cdist.dim() == 3:
        R2_permuted = R2_cdist.permute(0, 2, 1)  # (B, N2, N1)

        # Cosine similarity V2_sim: (B, N1, N2) -> permute to (B, N2, N1)
        V2_sim = torch.matmul(vec1_norm, vec2_norm.permute(0,2,1) if vec2_norm.dim() == 3 else vec2_norm.T)
        V2_sim_permuted = V2_sim.permute(0, 2, 1)

        if allow_antiparallel:
            V2_sim_permuted = torch.abs(V2_sim_permuted)
        else:
            V2_sim_permuted = torch.clamp(V2_sim_permuted, 0., 1.) # wrong direction should be 0 rather than negative
        V2_weighted_permuted = (V2_sim_permuted + 2.) / 3. # Following PheSA's suggestion for weighting

        # Prepare masks
        if m1_final.dim() == 1: m1_final = m1_final.unsqueeze(0)
        if m2_final.dim() == 1: m2_final = m2_final.unsqueeze(0)
        mask_mat_permuted = (m1_final.unsqueeze(2) * m2_final.unsqueeze(1)).permute(0, 2, 1) # (B, N2, N1)

        VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                               ((2 * alpha)**1.5) *
                                               mask_mat_permuted *
                                               V2_weighted_permuted *
                                               torch.exp(-(alpha / 2) * R2_permuted), # R2 is (B, N2, N1)
                                               dim=2),
                                     dim=1)
    # Single instance
    elif R2_cdist.dim() == 2: # R2_cdist is (N1, N2)

        # Cosine similarity V2_sim: (N1, N2)
        V2_sim_N1_N2 = torch.matmul(vec1_norm, vec2_norm.T) # (N1,N2)
        # No transpose needed for V2_sim if R2_cdist is used directly

        if allow_antiparallel:
            V2_sim_N1_N2 = torch.abs(V2_sim_N1_N2)
        else:
            V2_sim_N1_N2 = torch.clamp(V2_sim_N1_N2, 0., 1.) # wrong direction should be 0 rather than negative
        V2_weighted_N1_N2 = (V2_sim_N1_N2 + 2.) / 3. # Following PheSA's suggestion for weighting

        # Mask mat should be (N1,N2)
        mask_mat_N1_N2 = m1_final.unsqueeze(1) * m2_final.unsqueeze(0) # (N1,1)*(1,N2) -> (N1,N2)

        VAB_second_order = torch.sum((np.pi**1.5) /
                                     ((2 * alpha)**1.5) *
                                     mask_mat_N1_N2 *
                                     V2_weighted_N1_N2 *
                                     torch.exp(-(alpha / 2) *R2_cdist)
                                     )
    else:
        raise ValueError(
            f"Unexpected shape from torch.cdist: {R2_cdist.shape}. "
            f"Input shapes were: centers_1={centers_1.shape}, centers_2={centers_2.shape}"
        )
    return VAB_second_order


def VAB_2nd_order_cosine_mask_batch(cdist_21: torch.Tensor,
                                    vmm_21: torch.Tensor,
                                    alpha: float,
                                    allow_antiparallel: bool,
                                    mask_1: torch.Tensor,
                                    mask_2: torch.Tensor
                                    ) -> torch.Tensor:
    """
    2nd order volume overlap of AB (batched) weighted by cosine similarity, 
    with masking, using precomputed cdist and vector dot products (vmm).
    Assumes inputs `cdist_21`, `vmm_21`, `mask_1`, `mask_2` are already batched and broadcast-compatible.

    Parameters
    ----------
    cdist_21 : torch.Tensor (B,M,N)
        Precomputed squared Euclidean distances, e.g., (torch.cdist(centers_2, centers_1)**2.0).
    vmm_21 : torch.Tensor (B,M,N)
        Precomputed dot products of normalized vectors, e.g., torch.matmul(vectors_2, vectors_1.permute(0,2,1)).
        This corresponds to cosine similarities if vectors were normalized.
    alpha : float
        Gaussian width parameter.
    allow_antiparallel : bool
        If true, absolute cosine similarity is used.
    mask_1 : torch.Tensor (B,N) or (1,N)
        Mask for the first set of points/vectors (N dimension).
    mask_2 : torch.Tensor (B,M) or (1,M)
        Mask for the second set of points/vectors (M dimension).

    Returns
    -------
    torch.Tensor : (B,)
        Batched Tanimoto similarity scores.
    """
    # Cosine similarity weighting
    vmm_21_processed = vmm_21
    if allow_antiparallel:
        vmm_21_processed = torch.abs(vmm_21_processed)
    else:
        vmm_21_processed = torch.clamp(vmm_21_processed, 0., 1.) # wrong direction should be 0 rather than negative
    vmm_21_weighted = (vmm_21_processed + 2.) / 3. # Following PheSA's suggestion for weighting

    # Prepare mask_mat (B,M,N)
    mask_mat = (mask_2.unsqueeze(2) * mask_1.unsqueeze(1))

    VAB_second_order = torch.sum(torch.sum((np.pi**1.5) /
                                            ((2 * alpha)**1.5) *
                                            mask_mat *
                                            vmm_21_weighted *
                                            torch.exp(-(alpha / 2) * cdist_21),
                                            dim = 2),
                                dim = 1)
    return VAB_second_order


##################################################################
##################### Older implementations ######################
##################################################################

def VAB_2nd_order_batched(centers_1: torch.Tensor,
                          centers_2: torch.Tensor,
                          alphas_1: torch.Tensor,
                          alphas_2: torch.Tensor,
                          prefactors_1: torch.Tensor,
                          prefactors_2: torch.Tensor
                          ) -> torch.Tensor:
    """ 
    Calculate the 2nd order volume overlap of AB -- batched functionality
    
    Parameters
    ----------
        centers_1 : (torch.Tensor) (batch_size, num_atoms_1, 3)
            Coordinates of atoms in molecule 1

        centers_2 : (torch.Tensor) (batch_size, num_atoms_2, 3)
            Coordinates of atoms in molecule 2

        alphas_1 : (torch.Tensor) (batch_size, num_atoms_1)
            Alpha values for atoms in molecule 1

        alphas_2 : (torch.Tensor) (batch_size, num_atoms_2)
            Alpha values for atoms in molecule 2

        prefactors_1 : (torch.Tensor) (batch_size, num_atoms_1)
            Prefactor values for atoms in molecule 1

        prefactors_2 : (torch.Tensor) (batch_size, num_atoms_2)
            Prefactor values for atoms in molecule 2
    
    Returns
    -------
    torch.Tensor (batch_size,)
        Representing the 2nd order volume overlap of AB for each batch
    """
    R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)

    prefactor1_prod_prefactor2 = (prefactors_1.unsqueeze(1) * prefactors_2.unsqueeze(2))

    alpha1_prod_alpha2 = (alphas_1.unsqueeze(1) * alphas_2.unsqueeze(2))
    alpha1_sum_alpha2 = (alphas_1.unsqueeze(1) + alphas_2.unsqueeze(2))
    VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                        prefactor1_prod_prefactor2 *
                                        torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2) /
                                        (alpha1_sum_alpha2**(1.5)),
                                        dim = 2),
                                dim = 1)
    return VAB_second_order


def shape_tanimoto_batched(centers_1: torch.Tensor,
                           centers_2: torch.Tensor,
                           alphas_1: torch.Tensor,
                           alphas_2: torch.Tensor,
                           prefactors_1: torch.Tensor,
                           prefactors_2: torch.Tensor
                           ) -> torch.Tensor:
    """
    Calculate the Tanimoto shape similarity between two batches of molecules.
    """
    VAA = VAB_2nd_order_batched(centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1)
    VBB = VAB_2nd_order_batched(centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2)
    VAB = VAB_2nd_order_batched(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return VAB / (VAA + VBB - VAB)


def get_overlap_batch(centers_1:torch.Tensor,
                      centers_2:torch.Tensor,
                      prefactor:float = 0.8,
                      alpha:float = 0.81) -> torch.Tensor:
    """ Computes the gaussian overlap for a batch of centers. """
    # initialize prefactor and alpha matrices
    prefactors_1 = (torch.ones(centers_1.shape[0]) * prefactor).unsqueeze(0)
    prefactors_2 = (torch.ones(centers_2.shape[0]) * prefactor).unsqueeze(0)
    alphas_1 = (torch.ones(prefactors_1.shape[0]) * alpha).unsqueeze(0)
    alphas_2 = (torch.ones(prefactors_2.shape[0]) * alpha).unsqueeze(0)

    tanimoto_score = shape_tanimoto_batched(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return tanimoto_score


def VAB_2nd_order_full(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2) -> torch.Tensor:
    """ 2nd order volume overlap of AB """
    R2 = (torch.cdist(centers_1, centers_2, compute_mode='use_mm_for_euclid_dist')**2.0).T
    prefactor1_prod_prefactor2 = prefactors_1 * prefactors_2.unsqueeze(1)
    alpha1_prod_alpha2 = alphas_1 * alphas_2.unsqueeze(1)
    alpha1_sum_alpha2 = alphas_1 + alphas_2.unsqueeze(1)

    VAB_second_order = torch.sum(np.pi**(1.5) * prefactor1_prod_prefactor2 * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2) / (alpha1_sum_alpha2**(1.5)))
    return VAB_second_order


def shape_tanimoto_full(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2) -> torch.Tensor:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order_full(centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1)
    VBB = VAB_2nd_order_full(centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2)
    VAB = VAB_2nd_order_full(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return VAB / (VAA + VBB - VAB)


def get_overlap_full(centers_1:torch.Tensor,
                     centers_2:torch.Tensor,
                     prefactor:float = 0.8,
                     alpha:float = 0.81
                     ) -> torch.Tensor:
    """ Computes the gaussian overlap for a batch of centers with custom prefactor and alpha values."""
    prefactors_1 = torch.ones(centers_1.shape[0]) * prefactor
    prefactors_2 = torch.ones(centers_2.shape[0]) * prefactor
    alphas_1 = torch.ones(prefactors_1.shape[0]) * alpha
    alphas_2 = torch.ones(prefactors_2.shape[0]) * alpha

    tanimoto = shape_tanimoto_full(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return tanimoto
