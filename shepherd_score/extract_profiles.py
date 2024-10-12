"""
Convenience module to hold functions used to extract interaction profiles.
"""
import numpy as np
import open3d
from rdkit import Chem

from shepherd_score.score.constants import COULOMB_SCALING
from shepherd_score.generate_point_cloud import get_molecular_surface, get_atomic_vdw_radii
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores, get_pharmacophores_dict

def get_electrostatic_potential(mol: Chem.Mol,
                                partial_charges: np.ndarray,
                                surf_pos: np.ndarray
                                ) -> np.ndarray:
    """
    Get the electrostatic potential (ESP) at each surface point.

    Arguments
    ---------
    mol : rdkit.Chem.Mol object with a conformer
    partial_charges : np.ndarray (N,) atom-mapped to the `mol` object
    surf_pos : np.ndarray (M,3) sampled surface positions

    Returns
    -------
    np.ndarray (M,) electrostatic potential at each surface point
    """
    centers = mol.GetConformer().GetPositions()
    distances = np.linalg.norm(surf_pos[:, np.newaxis] - centers, axis=2)
    # Calculate the potentials
    E_pot = np.dot(partial_charges, 1 / distances.T) * COULOMB_SCALING
    # Ensure that invalid distances (where distance is 0) are handled
    E_pot[np.isinf(E_pot)] = 0    
    return E_pot.astype(np.float32)