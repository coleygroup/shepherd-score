""" 
Gaussian volume overlap scoring functions combined with continuous electrostatics
PYTORCH VERSION.

Batched and non-batched functionalities

Reference math:
https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K
https://doi.org/10.1021/j100011a016
"""
import numpy as np
import torch
from shepherd_score.score.constants import COULOMB_SCALING, LAM_SCALING
from shepherd_score.score.gaussian_overlap import get_overlap


def VAB_2nd_order_esp(centers_1: torch.Tensor,
                      centers_2: torch.Tensor,
                      charges_1: torch.Tensor,
                      charges_2: torch.Tensor,
                      alpha: float,
                      lam: float
                      ) -> torch.Tensor:
    """ Torch implementation. Handles batching"""
    # Single instance
    if len(centers_1.shape) == 2:
        R2 = (torch.cdist(centers_1, centers_2, compute_mode='use_mm_for_euclid_dist')**2.0).T
        C2 = (torch.cdist(charges_1, charges_2, compute_mode='use_mm_for_euclid_dist')**2.0).T

        VAB_2nd_order = torch.sum(np.pi**(1.5) \
                                  * torch.exp(-(alpha / 2) * R2) \
                                  / ((2*alpha)**(1.5))\
                                  * torch.exp(-C2/lam))

    # Batched
    elif len(centers_1.shape) == 3:
        R2 = (torch.cdist(centers_1, centers_2)**2.0).permute(0,2,1)
        C2 = (torch.cdist(charges_1, charges_2)**2.0).permute(0,2,1)
    
        VAB_2nd_order = torch.sum(torch.sum(np.pi**(1.5) \
                                            * torch.exp(-(alpha / 2) * R2) \
                                            / ((2*alpha)**(1.5))\
                                            * torch.exp(-C2/lam),
                                            dim = 2),
                                    dim = 1)
    return VAB_2nd_order

def shape_tanimoto_esp(centers_1: torch.Tensor,
                       centers_2: torch.Tensor,
                       charges_1: torch.Tensor,
                       charges_2: torch.Tensor,
                       alpha: float,
                       lam: float
                       ) -> torch.Tensor:
    """ Compute Tanimoto shape similarity """
    VAA = VAB_2nd_order_esp(centers_1, centers_1, charges_1, charges_1, alpha, lam)
    VBB = VAB_2nd_order_esp(centers_2, centers_2, charges_2, charges_2, alpha, lam)
    VAB = VAB_2nd_order_esp(centers_1, centers_2, charges_1, charges_2, alpha, lam)
    return VAB / (VAA + VBB - VAB)


def get_overlap_esp(centers_1: torch.Tensor,
                    centers_2: torch.Tensor,
                    charges_1: torch.Tensor,
                    charges_2: torch.Tensor,
                    alpha: float = 0.81,
                    lam: float = 0.3*LAM_SCALING
                    ) -> torch.Tensor:
    """
    Torch implementation.
    Compute electrostatic similarity which weights Gaussian volume overlap by electrostatics. 
    The Tanimoto score is used.

    Typically `lam=0.3*LAM_SCALING` is used for surface point clouds and `lam=0.1` for partial charge
    weighted volumetric overlap.
    
    Parameters
    ----------
    centers_1 : torch.Tensor (batch, N, 3) or (N, 3)
        Coordinates for the sets of points representing molecule 1.
    centers_2 : torch.Tensor (batch, N, 3) or (N, 3)
        Coordinates for the sets of points representing molecule 2.
    charges_1 : torch.Tensor (batch, N) or (N,)
        Electrostatic energy for the sets of points representing molecule 1.
    charges_2 : torch.Tensor (batch, N) or (N,)
        Electrostatic energy for the sets of points representing molecule 2.
    alpha : float
        Parameter controlling the width of the Gaussians.
    lam : float
        Parameter controlling the influence of electrostatics.
    
    Returns
    -------
    tanimoto_esp : torch.Tensor (batch, 1) or (1,)
        Tanimoto similarities of electrostatics.
    """
    if isinstance(centers_1, np.ndarray):
        centers_1 = torch.Tensor(centers_1)
    if isinstance(centers_2, np.ndarray):
        centers_2 = torch.Tensor(centers_2)
    if isinstance(charges_1, np.ndarray):
        charges_1 = torch.Tensor(charges_1)
    if isinstance(charges_2, np.ndarray):
        charges_2 = torch.Tensor(charges_2)
    # initialize prefactor and alpha matrices
    if len(charges_1.shape) == 1:
        charges_1 = charges_1.reshape((-1,1))
    elif len(charges_1.shape) == 2:
        charges_1 = charges_1.unsqueeze(2)
    if len(charges_2.shape) == 1:
        charges_2 = charges_2.reshape((-1,1))
    elif len(charges_2.shape) == 2:
        charges_2 = charges_2.unsqueeze(2)
    

    tanimoto_esp = shape_tanimoto_esp(centers_1, centers_2,
                                      charges_1, charges_2,
                                      alpha,
                                      lam)
    return tanimoto_esp


def _esp_comparison(points_1: torch.Tensor,
                    centers_w_H_2: torch.Tensor, # EXPECTS HYDROGENS INCLUDED
                    partial_charges_2: torch.Tensor,
                    points_charges_1: torch.Tensor,
                    radii_2: torch.Tensor,
                    probe_radius: float = 1.0,
                    lam: float = 0.001
                    ) -> torch.Tensor:
    """
    Helper function for computing the electrostatic potential (ESP) component of ShaEP score.
    It computes the difference in ESP at surface/observer points of molecule 1 for the ESP values
    generated by molecule 1 and molecule 2. It masks out observer points if they are in
    molecule 2's volume defined by vdW+probe_radius.
    Expects single instance or batched. This will ONLY check the shape of points_1 to deterimine
    if it is batched or not so errors in the shape of the other tensors may or may not be caught.

    Parameters
    ----------
    points_1 : torch.Tensor (N_surf, 3) or (batch, N_surf, 3)
        Surface points of molecule 1 for which ESP's will be computed and compared.

    centers_w_H_2 : torch.Tensor (M + m_H, 3) or (batch, M + m_H, 3)
        Coordinates for atoms (including hydrogens) of molecule 2. Used in calculation of ESP at
        points_1 and masking out those within molecule 2's volume.
    
    partial_charges_2 : torch.Tensor (M + m_H,) or (batch, M + m_H,)
        Partial charges corresponding to centers_w_H_2. Used to calculate ESP.

    points_charges_1 : torch.Tensor (N_surf,) or (batch, N_surf,)
        Precalculated ESP's of molecule 1 corresponding to points_1.
    
    radii_2 : torch.Tensor (M + m_H,) or (batch, M + m_H,)
        Radii of each atom corresponding to centers_w_H_2. Used for masking operation.
    
    probe_radius : float (default = 1.0)
        Probe radius (default is 1 angstrom). Surfaces assumed to be generated with vdW radius and
        a probe radius of 1.2 angstroms (vdW radius of hydrogen). 1.0 used rather than 1.2 as a
        tolerance.
    
    lam : float (default = 0.001)
        Electrostatic potential weighting parameter (smaller = higher weight).
        0.001 was chosen as default based empirical observations of the distribution of scores
        generated before the summation in this function.

    Returns
    -------
    torch.Tensor (1,) or (batch, 1)
        Point to point ESP comparison. Scores range: [0, N_surf]. Score decreases for differences
        in ESP or due to masking of poorly aligned surface points.
    """
    lam = LAM_SCALING * lam
    # return
    distances = torch.cdist(points_1, centers_w_H_2)
    # single instance
    if len(points_1.shape) == 2:
        # mask out molecule 1 surface points that are within molecule 2
        mask = torch.where(torch.all(distances >= radii_2 + probe_radius, axis=1), 1., 0.)
        # Calculate the potentials
        esp_at_surf_1 = torch.matmul(partial_charges_2, 1 / distances.T) * COULOMB_SCALING

        esp = torch.sum(mask * torch.exp(-torch.square(points_charges_1 - esp_at_surf_1)/lam))

    # batched
    elif len(points_1.shape) == 3:
        # mask out molecule 1 surface points that are within molecule 2
        mask = torch.where(torch.all(distances >= radii_2.unsqueeze(1) + probe_radius, axis=2), 1., 0.)
        # Calculate the potentials
        esp_at_surf_1 = torch.matmul(partial_charges_2.unsqueeze(1), 1 / distances.permute(0,2,1)) * COULOMB_SCALING

        esp = torch.sum(mask * torch.exp(-torch.square(points_charges_1.unsqueeze(1) - esp_at_surf_1)/lam).squeeze(), axis=1)

    return esp


def esp_combo_score(centers_w_H_1: torch.Tensor,
                    centers_w_H_2: torch.Tensor,
                    centers_1: torch.Tensor,
                    centers_2: torch.Tensor,
                    points_1: torch.Tensor,
                    points_2: torch.Tensor,
                    partial_charges_1: torch.Tensor,
                    partial_charges_2: torch.Tensor,
                    point_charges_1: torch.Tensor,
                    point_charges_2: torch.Tensor,
                    radii_1: torch.Tensor,
                    radii_2: torch.Tensor,
                    alpha: float,
                    lam: float=0.001,
                    probe_radius: float=1.0,
                    esp_weight: float=0.5
                    ) -> torch.Tensor:
    """
    Computes a similarity score defined by ShaEP. It is a balanced score between electrostatics
    and shape similarity.
    Single instance or batch accepted (in the 0th dimension).
    This will ONLY check the shape of points_1 to deterimine if it is batched or not so errors
    in the shape of the other tensors may or may not be caught.

    Parameters
    ----------
    centers_w_H_1 : torch.Tensor (N + n_H, 3) | (batch, N + n_H, 3)
        Coordinates of atom centers INCLUDING hydrogens of molecule 1.
        Used for computing electrostatic potential.
        Same for centers_w_H_2 except (M + m_H, 3).

    centers_1 : torch.Tensor (N, 3) or (n_surf, 3) | (batch, N, 3) or (batch, n_surf, 3)
        Coordinates of points for molecule 1 used to compute shape similarity.
        Use atom centers for volumentric similarity. Use surface centers for surface similarity.
        Same for centers except (M, 3) or (m_surf, 3).
    
    points_1 : torch.Tensor (n_surf, 3) | (batch, n_surf, 3)
        Coordinates of surface points for molecule 1.
        Same for points_2 except (m_surf, 3).
    
    partial_charges_1 : torch.Tensor (N + n_H,) | (batch, N + n_H,)
        Partial charges corresponding to the atoms in centers_w_H_1.
        Same for partial_charges_2 except (M + m_H,).
    
    point_charges_1 : torch.Tensor (n_surf,) | (batch, n_surf,)
        The electrostatic potential calculated at each surface point (points_1).
        Same for point_charges_1 except (m_surf,)
    
    radii_1 : torch.Tensor (N + n_H,) | (batch, N + n_H,)
        vdW radii corresponding to the atoms in centers_w_H_1 (angstroms)
        Same for radii_2 except (M + m_H,)
    
    alpha : float
        Gaussian width parameter used for shape similarity.
    
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
    
    Returns
    -------
    torch.Tensor (1,) or (batch, 1)
        Similarity score (range: [0, 1]). Higher is more similar.
    """

    # Calculate the difference in ESP at the surface of molecule 1
    #   Expects hydrogens for the centers
    if len(points_1.shape) == len(points_2.shape):
        esp_1 = _esp_comparison(points_1, centers_w_H_2, partial_charges_2, point_charges_1, radii_2, probe_radius, lam)
        esp_2 = _esp_comparison(points_2, centers_w_H_1, partial_charges_1, point_charges_2, radii_1, probe_radius, lam)
    else:
        raise ValueError(f"Inputs points_1 and points_2 should have the same dimensions but got {len(points_1.shape)} and {len(points_2.shape)}.")

    if len(points_1.shape) == 2:
        electrostatic_sim = (esp_1 + esp_2) / (len(points_1) + len(points_2))
    elif len(points_1.shape) == 3:
        electrostatic_sim = (esp_1 + esp_2) / (points_1.shape[1] + points_2.shape[1])
    else:
        raise ValueError(f"Recieved points_1 with shape {points_1.shape} when it should be (n_surf, 3) or (batch, n_surf, 3).")
    volumetric_sim = get_overlap(centers_1, centers_2, alpha)

    score = esp_weight*electrostatic_sim + (1-esp_weight)*volumetric_sim
    return score
