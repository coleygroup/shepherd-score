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
    Torch implemenation with batched and single instance functionality.
    """
    # Single instance
    if len(centers_1.shape) == 2:
        R2 = (torch.cdist(centers_1, centers_2, compute_mode='use_mm_for_euclid_dist')**2.0).T

        VAB_second_order = torch.sum(np.pi**(1.5) * torch.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))

    # Batched
    elif len(centers_1.shape) == 3:
        R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
    
        VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                            torch.exp(-(alpha / 2) * R2) /
                                            ((2*alpha)**(1.5)),
                                            dim = 2),
                                    dim = 1)
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
    Volumentric shape similarity with tunable "alpha" Gaussian width parameter.
    Handles both batched and single instances. PyTorch implementation.

    Parameters
    ----------
    centers_1 : Union[torch.Tensor, np.ndarray] (batch, N, 3) or (N, 3)
        Coordinates of each point of the first point cloud.
    centers_2 : Union[torch.Tensor, np.ndarray] (batch, N, 3) or (N, 3)
        Coordinates of each point of the second point cloud.
    alpha : float (default=0.81)
        Gaussian width parameter. Lower value corresponds to wider Gaussian (longer tail).
    
    Returns
    -------
    torch.Tensor : (batch, 1) or (1,)
    """
    if isinstance(centers_1, np.ndarray):
        centers_1 = torch.Tensor(centers_1)
    if isinstance(centers_2, np.ndarray):
        centers_2 = torch.Tensor(centers_2)
    # initialize prefactor and alpha matrices
    tanimoto = shape_tanimoto(centers_1, centers_2, alpha)
    return tanimoto


def VAB_2nd_order_mask(centers_1: torch.Tensor,
                       centers_2: torch.Tensor,
                       alpha: float,
                       mask_1: torch.Tensor,
                       mask_2: torch.Tensor
                       ) -> torch.Tensor:
    """
    2nd order volume overlap of AB.
    Torch implemenation with batched and single instance functionality using masking.

    centers_1 : torch.Tensor (N,3) or (B,N,3)
    centers_2 : torch.Tensor (M,3) or (B,M,3)
    mask_1 : torch.Tensor (N,) or (B,N)
    mask_2 : torch.Tensor (M,) or (B,M)
    """
    # Single instance
    if len(centers_1.shape) == 2:
        masked_c1 = centers_1[mask_1.bool()]
        masked_c2 = centers_2[mask_2.bool()]
        R2 = (torch.cdist(masked_c1, masked_c2, compute_mode='use_mm_for_euclid_dist')**2.0).T

        VAB_second_order = torch.sum(np.pi**(1.5) * torch.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))

    # Batched
    elif len(centers_1.shape) == 3:
        # Fast assuming low number of centers and high number of instances
        R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
        mask_mat = (mask_1[:, :,None] * mask_2[:, None,:]).permute(0,2,1)

        VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                               mask_mat * 
                                               torch.exp(-(alpha / 2) * R2) /
                                               ((2*alpha)**(1.5)),
                                               dim = 2),
                                    dim = 1)
    return VAB_second_order


def VAB_2nd_order_mask_batch(cdist_21: torch.Tensor,
                             alpha: float,
                             mask_1: torch.Tensor,
                             mask_2: torch.Tensor
                             ) -> torch.Tensor:
    """
    2nd order volume overlap of AB.
    Torch implemenation with only batched functionality using masking and precomputed cdist

    cdist_21 : torch.Tensor (B,M,N)
        Precomputed (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
    alpha : float
    mask_1 : torch.Tensor (B,N)
    mask_2 : torch.Tensor (B,M)
    """
    # Fast assuming low number of centers and high number of instances
    mask_mat = (mask_1[:, :,None] * mask_2[:, None,:]).permute(0,2,1)

    VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                            mask_mat * 
                                            torch.exp(-(alpha / 2) * cdist_21) /
                                            ((2*alpha)**(1.5)),
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
    Torch implemenation with batched and single instance functionality.
    """

    # Single instance
    if len(centers_1.shape) == 2:
        # cosine similarity
        vectors_1 = F.normalize(vectors_1, p=2, dim=1)
        vectors_2 = F.normalize(vectors_2, p=2, dim=1)
        V2 = torch.matmul(vectors_1, vectors_2.T).T
        if allow_antiparallel:
            # 
            V2 = torch.abs(V2)
        else:
            V2 = torch.clamp(V2, 0., 1.)

        V2 = (V2 + 2.)/3. # Following PheSA's suggestion for weighting

        R2 = (torch.cdist(centers_1, centers_2, compute_mode='use_mm_for_euclid_dist')**2.0).T

        VAB_second_order = torch.sum(np.pi**(1.5) * V2 * torch.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))

    # Batched
    elif len(centers_1.shape) == 3:
        # Cosine similarity
        vectors_1 = F.normalize(vectors_1, p=2, dim=2)
        vectors_2 = F.normalize(vectors_2, p=2, dim=2)
        V2 = torch.matmul(vectors_1, vectors_2.permute(0,2,1)).permute(0,2,1)
        if allow_antiparallel:
            V2 = torch.abs(V2)
        else:
            V2 = torch.clamp(V2, 0., 1.) # wrong direction should be 0 rather than negative

        V2 = (V2 + 2.)/3. # Following PheSA's suggestion for weighting

        R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
    
        VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                               V2 * 
                                               torch.exp(-(alpha / 2) * R2) /
                                               ((2*alpha)**(1.5)),
                                               dim = 2),
                                     dim = 1)
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
    2nd order volume overlap of AB weighted by cosine similarity.
    Torch implemenation with batched and single instance functionality.
    """

    # Single instance
    if len(centers_1.shape) == 2:
        # cosine similarity
        masked_centers_1 = centers_1[mask_1.bool()]
        masked_centers_2 = centers_2[mask_2.bool()]
        masked_vectors_1 = vectors_1[mask_1.bool()]
        masked_vectors_2 = vectors_2[mask_2.bool()]
        masked_vectors_1 = F.normalize(masked_vectors_1, p=2, dim=1)
        masked_vectors_2 = F.normalize(masked_vectors_2, p=2, dim=1)
        V2 = torch.matmul(masked_vectors_1, masked_vectors_2.T).T
        if allow_antiparallel:
            V2 = torch.abs(V2)
        else:
            V2 = torch.clamp(V2, 0., 1.)

        V2 = (V2 + 2.)/3. # Following PheSA's suggestion for weighting

        R2 = (torch.cdist(masked_centers_1, masked_centers_2, compute_mode='use_mm_for_euclid_dist')**2.0).T

        VAB_second_order = torch.sum(np.pi**(1.5) * V2 * torch.exp(-(alpha / 2) * R2) / ((2*alpha)**(1.5)))

    # Batched
    elif len(centers_1.shape) == 3:
        # Cosine similarity
        vectors_1 = F.normalize(vectors_1, p=2, dim=2)
        vectors_2 = F.normalize(vectors_2, p=2, dim=2)
        V2 = torch.matmul(vectors_1, vectors_2.permute(0,2,1)).permute(0,2,1)
        if allow_antiparallel:
            V2 = torch.abs(V2)
        else:
            V2 = torch.clamp(V2, 0., 1.) # wrong direction should be 0 rather than negative

        V2 = (V2 + 2.)/3. # Following PheSA's suggestion for weighting

        R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
        mask_mat = (mask_1[:, :, None] * mask_2[:, None, :]).permute(0,2,1)
    
        VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                               mask_mat *
                                               V2 * 
                                               torch.exp(-(alpha / 2) * R2) /
                                               ((2*alpha)**(1.5)),
                                               dim = 2),
                                     dim = 1)
    return VAB_second_order


def VAB_2nd_order_cosine_mask_batch(cdist_21: torch.Tensor,
                                    vmm_21: torch.Tensor,
                                    alpha: float,
                                    allow_antiparallel: bool,
                                    mask_1: torch.Tensor,
                                    mask_2: torch.Tensor
                                    ) -> torch.Tensor:
    """
    2nd order volume overlap of AB weighted by cosine similarity.
    Torch implemenation for only batched inputs where cdist and vmm were precomputed.

    cdist_21 : torch.Tensor (B,M,N)
        Precomputed (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
    vmm_21 : torch.Tensor (B,M,N)
        Precomputed torch.matmul(vectors_1, vectors_2.permute(0,2,1)).permute(0,2,1).
        Assume that vectors_1, vectors_2 were normalized and thus vmm = cosine similarity.
    """
    # Cosine similarity
    if allow_antiparallel:
        vmm_21 = torch.abs(vmm_21)
    else:
        vmm_21 = torch.clamp(vmm_21, 0., 1.) # wrong direction should be 0 rather than negative

    vmm_21 = (vmm_21 + 2.)/3. # Following PheSA's suggestion for weighting

    mask_mat = (mask_1[:, :, None] * mask_2[:, None, :]).permute(0,2,1)

    VAB_second_order = torch.sum(torch.sum(np.pi**(1.5) *
                                            mask_mat *
                                            vmm_21 * 
                                            torch.exp(-(alpha / 2) * cdist_21) /
                                            ((2*alpha)**(1.5)),
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

    # initialize prefactor and alpha matrices
    prefactors_1 = torch.ones(centers_1.shape[0]) * prefactor
    prefactors_2 = torch.ones(centers_2.shape[0]) * prefactor
    alphas_1 = torch.ones(prefactors_1.shape[0]) * alpha
    alphas_2 = torch.ones(prefactors_2.shape[0]) * alpha

    tanimoto = shape_tanimoto_full(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return tanimoto
