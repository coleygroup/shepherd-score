"""
Molecule class to hold molecule geometries and extract interaction profiles.
MoleculePair class facilitates alignment with interaction profiles.
"""
from typing import Union, List, Optional, Tuple
from copy import deepcopy
import sys

import numpy as np
import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point3D
import torch

from shepherd_score.score.constants import COULOMB_SCALING, LAM_SCALING, ALPHA

from shepherd_score.generate_point_cloud import get_atom_coords, get_atomic_vdw_radii, get_molecular_surface, get_molecular_surface_const_density
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE, get_overlap_pharm
from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_esp_overlay, optimize_esp_combo_score_overlay
from shepherd_score.alignment import optimize_pharm_overlay
from shepherd_score.alignment_utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np


def update_mol_coordinates(mol: Chem.Mol, coordinates: Union[List, np.ndarray]) -> Chem.Mol:
    """
    Updates the coordinates of a 3D RDKit mol object with a new set of coordinates
    
    Args:
        mol -- RDKit mol object with 3D coordinates to be replaced
        coordinates -- list/array of new [x,y,z] coordinates
    
    Returns:
        mol_new -- deep-copied RDKit mol object with updated 3D coordinates
    """
    mol_new = deepcopy(mol)
    conf = mol_new.GetConformer()
    for i in range(mol_new.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return mol_new


class Molecule:
    """
    Molecule contains ways to hold/generate molecule geometries
    """
    def __init__(self,
                 mol: Chem.rdchem.Mol,
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 probe_radius: Optional[float] = None,
                 surface_points: Optional[np.ndarray] = None,
                 partial_charges : Optional[np.ndarray] = None,
                 electrostatics: Optional[np.ndarray] = None,
                 pharm_multi_vector: Optional[bool] = None,
                 pharm_types: Optional[np.ndarray] = None,
                 pharm_ancs: Optional[np.ndarray] = None,
                 pharm_vecs: Optional[np.ndarray] = None
                 ):
        """
        Molecule constructor to extract interaction profiles.
        
        If `partial_charges` are not provided, they will be generated using MMFF94 which may
        result in subpar performance.

        Arguments
        ----------
        mol : rdkit.Chem.rdchem.Mol
        num_surf_points : Optional[int] Number of surface points to sample.
            If None, the surface point cloud is not generated. More efficient if only doing volumentric.
        density : Optional[np.ndarray]
            Density of points to sample on molecular surface.
            If None, the surface point cloud is not generated. More efficient if only doing volumentric.
            If both num_surf_points and density are not None, num_surf_points supercedes density.
        surface_points : Optional[np.ndarray] (M,3) Surface points if they were previously generated.
        probe_radius : Optional[float] the radius of a probe atom to act as a "solvent accessible surface".
            Default is 1.2 if `None` is passed.
        partial_charges : Optional[np.ndarray] (N,) Partial charges for each atom.
            If `None` is passed and ESP surface is generated, it will default to MMFF94 partial charges.
        electrostatics : Optional[np.ndarray] (M,) Electrostatic potential if they were previously generated.
        pharm_multi_vector : Optional[bool] If None, don't generate pharmacophores, else generate
            pharmacophores with/without (true/false) multi-vectors.
        pharm_types : Optional[np.ndarray] (P,) Types of pharmacophores.
        pharm_ancs : Optional[np.ndarray] (P,3) Anchor positions of pharmacophores.
        pharm_vecs : Optional[np.ndarray] (P,3) Unit vectors relative to anchor positions of pharmacophores.
        """
        self.mol = mol
        self.atom_pos = Chem.RemoveHs(mol).GetConformer().GetPositions()
        if surface_points is None:
            self.num_surf_points = num_surf_points
        else:
            self.num_surf_points = len(surface_points)
        self.density = density

        if isinstance(partial_charges, list):
            partial_charges = np.array(partial_charges)

        if isinstance(partial_charges, np.ndarray):
            self.partial_charges = partial_charges
        else:
            self.partial_charges = self.get_partial_charges()
        self.radii = get_atomic_vdw_radii(mol)

        if surface_points is None:
            self.probe_radius = probe_radius if probe_radius is not None else 1.2
            if isinstance(num_surf_points, int):
                self.surf_pos = self.get_pc()
            elif isinstance(density, float):
                self.surf_pos = self.get_pc(use_density=True)
            else: # if None then don't generate a point cloud
                self.surf_pos = None
                self.surf_esp = None
        else:
            self.surf_pos = surface_points
            self.probe_radius = probe_radius if probe_radius is not None else 1.2
        
        if self.surf_pos is not None and self.partial_charges is not None:
            if not isinstance(electrostatics, np.ndarray):
                self.surf_esp = self.get_electrostatic_potential()
            else:
                self.surf_esp = electrostatics

        # Indices for atoms that aren't hydrogens
        self._nonH_atoms_idx = np.array([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1])

        self.pharm_multi_vector = pharm_multi_vector
        if isinstance(pharm_types, np.ndarray) and isinstance(pharm_ancs, np.ndarray) and isinstance(pharm_vecs, np.ndarray):
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = pharm_types, pharm_ancs, pharm_vecs
        else:
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = None, None, None
            if self.pharm_multi_vector is not None:
                self.get_pharmacophore(
                    multi_vector=self.pharm_multi_vector,
                    exclude=[],
                    check_access=False,
                    scale=1.
                )


    def get_partial_charges(self) -> np.ndarray:
        """
        Get the partial charges on each atom using MMFF.
        """
        molec_props = Chem.AllChem.MMFFGetMoleculeProperties(self.mol)
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(self.mol.GetAtoms())])
        return charges.astype(np.float32)


    def get_pc(self, use_density=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the point cloud positions.
        """
        self.mol, centers = get_atom_coords(self.mol, MMFF_optimize=False)
        if use_density:
            positions = get_molecular_surface_const_density(centers,
                                                            self.radii,
                                                            self.density,
                                                            probe_radius=self.probe_radius,
                                                            num_samples_per_atom=25)
        else:
            positions = get_molecular_surface(centers,
                                              self.radii,
                                              num_points=self.num_surf_points,
                                              probe_radius=self.probe_radius,
                                              num_samples_per_atom = 25)
        return positions.astype(np.float32)


    def get_electrostatic_potential(self) -> np.ndarray:
        """
        Get the electrostatic potential at each surface point.
        """
        centers = self.mol.GetConformer().GetPositions()
        distances = np.linalg.norm(self.surf_pos[:, np.newaxis] - centers, axis=2)
        # Calculate the potentials
        E_pot = np.dot(self.partial_charges, 1 / distances.T) * COULOMB_SCALING
        # Ensure that invalid distances (where distance is 0) are handled
        E_pot[np.isinf(E_pot)] = 0    
        return E_pot.astype(np.float32)


    def center_to(self, xyz_means: np.ndarray) -> None:
        """
        If you want to center the molecule with respect to a certain coordinate frame.
        """
        self.atom_pos -= xyz_means
        trans = np.eye(4)
        trans[:3,3] = -xyz_means
        Chem.rdMolTransforms.TransformConformer(self.mol.GetConformer(), trans)
        if self.surf_pos is not None:
            self.surf_pos -= xyz_means
        if self.pharm_ancs is not None:
            self.pharm_ancs -= xyz_means


    def get_pharmacophore(self,
                          multi_vector: bool = True,
                          exclude: List[int] = [],
                          check_access: bool = False,
                          scale: float = 1):
        """ Get the pharmacophores of the molecule. """
        self.pharm_types, self.pharm_ancs, self.pharm_vecs = get_pharmacophores(
            self.mol,
            multi_vector=multi_vector,
            exclude=exclude,
            check_access=check_access,
            scale=scale
        )


class MoleculePair:
    """ Pair of Molecule objects to facilitate alignment. """

    def __init__(self,
                 ref_mol: Union[Chem.rdchem.Mol, Molecule],
                 fit_mol: Union[Chem.rdchem.Mol, Molecule],
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 do_center: bool = False,
                 device = -1):
        """
        A pair of molecules. A refence molecule and a fit molecule that can be aligned to the fit.
        There are a number of alignments that can be done:
        - Volumetric (with and without hydrogens)
        - Volumetric with partial charge weighting (with and without hydrogens)
        - Surface
        - Surface with electrostatic potential weighting
        - ShaEP scoring (esp-combo)
        - Pharmacophores (with various settings for using extended points rather than vectors)

        Similarly, you can score with surface, Surf+ESP, and pharmacophores

        Parameters
        ----------
        ref_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Reference molecule.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        fit_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Molecule to fit to the reference.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        num_surf_points : Optional[int] (default = None)
            Number of surface points to sample if rdkit Mol objects are given.
            MUST provide a value for surface or ESP alignment.
        density : Optional[float] (default = None)
            Density of points to sample if rdkit Mol objects are given.
            An integer intput for num_surf_points supercedes the density call.
        do_center : bool (default = False)
            THIS IS CRUCIAL
            Whether to initially align molecule centers together. For global optimizations, set to
            True. For scoring of current alignment or local alignment set to False.
        device : pytorch Device (default = -1)
            Device to use if you want to align with PyTorch downstream.
            Default places alignment computation on CPU.
        """
        # Generate surfaces if not a Molecule object
        if not isinstance(ref_mol, Chem.rdchem.Mol):
            self.ref_molec = ref_mol
        else:
            self.ref_molec = Molecule(ref_mol, num_surf_points=num_surf_points, density=density)
        if not isinstance(fit_mol, Chem.rdchem.Mol):
            self.fit_molec = fit_mol
        else:
            self.fit_molec = Molecule(fit_mol, num_surf_points=num_surf_points, density=density)

        self.num_surf_points = num_surf_points
        self.density = density
        if density is not None and num_surf_points is None:
            self.num_surf_points = True
        if not isinstance(device, torch.device):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Center to origin
        if do_center:
            self.ref_molec.center_to(self.ref_molec.atom_pos.mean(0))
            self.fit_molec.center_to(self.fit_molec.atom_pos.mean(0))

        self.transform_vol = np.eye(4)
        self.sim_aligned_vol = None
        
        self.transform_vol_noH = np.eye(4)
        self.sim_aligned_vol_noH = None

        self.transform_surf = np.eye(4)
        self.sim_aligned_surf = None

        self.transform_esp = np.eye(4)
        self.sim_aligned_esp = None

        self.transform_vol_esp = np.eye(4)
        self.sim_aligned_vol_esp = None

        self.transform_vol_esp_noH = np.eye(4)
        self.sim_aligned_vol_esp_noH = None

        self.transform_esp_combo = np.eye(4)
        self.sim_aligned_esp_combo = None

        self.transform_pharm = np.eye(4)
        self.sim_aligned_pharm = None


    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using volumetric similarity.

        Optimally aligned score found in `self.sim_aligned_vol` and the optimal SE(3)
        transformation is at `self.transform_vol`. If `no_H` is True, append '_noH' to them.

        Arguments
        ---------
        no_H : bool (default = True) to not include hydrogens in volumetric similarity.
        num_repeats : int (default=50)
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3) coordinates of transformed atoms
        """
        if no_H:
            ref_atom_pos = self.ref_molec.atom_pos
            fit_atom_pos = self.fit_molec.atom_pos
        else:
            ref_atom_pos = self.ref_molec.mol.GetConformer().GetPositions()
            # ref_atom_pos -= ref_atom_pos.mean(0)
            fit_atom_pos = self.fit_molec.mol.GetConformer().GetPositions()
            # fit_atom_pos -= fit_atom_pos.mean(0)
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_ROCS_overlay_jax, optimize_ROCS_esp_overlay_jax, optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay_jax(
                ref_points=jnp.array(ref_atom_pos),
                fit_points=jnp.array(fit_atom_pos),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_fit_points = np.array(aligned_fit_points)
        else:
            # PyTorch
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay(
                ref_points=torch.from_numpy(ref_atom_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(fit_atom_pos).to(torch.float32).to(self.device),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            se3_transform = se3_transform.numpy()
            score = score.numpy()
            aligned_fit_points = aligned_fit_points.numpy()
        if no_H:
            self.transform_vol_noH = se3_transform
            self.sim_aligned_vol_noH = score
        else:
            self.transform_vol = se3_transform
            self.sim_aligned_vol = score
        return aligned_fit_points


    def align_with_vol_esp(self,
                           lam: float,
                           no_H: bool = True,
                           num_repeats: int = 50,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = 200,
                           use_jax: bool = False,
                           verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using volume similarity weighted by partial charge
        Toggle with_H parameter for scoring with or without hydrogens.

        Typically `lam=0.1` is used.
        Optimally aligned score found in `self.sim_aligned_vol_esp` and the optimal SE(3)
        transformation is at `self.transform_vol_esp`. If `no_H` is True, append '_noH' to them.

        Arguments
        ---------
        lam : float partial charge weighting parameter
        no_H : bool (default = True) to not include hydrogens in volumetric similarity.
        num_repeats : int (default=50)
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3) coordinates of transformed atoms
        """
        if no_H:
            ref_mol_partial_charges = self.ref_molec.partial_charges[self.ref_molec._nonH_atoms_idx]
            fit_mol_partial_charges = self.fit_molec.partial_charges[self.fit_molec._nonH_atoms_idx]
            ref_mol_pos = self.ref_molec.atom_pos
            fit_mol_pos = self.fit_molec.atom_pos
        else:
            ref_mol_partial_charges = self.ref_molec.partial_charges
            fit_mol_partial_charges = self.fit_molec.partial_charges
            ref_mol_pos = self.ref_molec.mol.GetConformer().GetPositions()
            # ref_mol_pos -= ref_mol_pos.mean(0) # move COM to origin
            fit_mol_pos = self.fit_molec.mol.GetConformer().GetPositions()
            # fit_mol_pos -= fit_mol_pos.mean(0)

        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_ROCS_overlay_jax, optimize_ROCS_esp_overlay_jax, optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay_jax(
                ref_points=jnp.array(ref_mol_pos),
                fit_points=jnp.array(fit_mol_pos),
                ref_charges=jnp.array(ref_mol_partial_charges),
                fit_charges=jnp.array(fit_mol_partial_charges),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_fit_points = np.array(aligned_fit_points)
            
        else: # Use Torch implementation
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay(
                ref_points=torch.from_numpy(ref_mol_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(fit_mol_pos).to(torch.float32).to(self.device),
                ref_charges=torch.from_numpy(ref_mol_partial_charges).to(torch.float32).to(self.device),
                fit_charges=torch.from_numpy(fit_mol_partial_charges).to(torch.float32).to(self.device),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            se3_transform = se3_transform.numpy()
            score = score.numpy()
            aligned_fit_points = aligned_fit_points.numpy()
        
        if no_H:
            self.transform_vol_esp_noH = se3_transform
            self.sim_aligned_vol_esp_noH = score
        else:
            self.transform_vol_esp = se3_transform
            self.sim_aligned_vol_esp = score
        return aligned_fit_points
    
    
    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = 50,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = 200,
                        use_jax: bool = False,
                        verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using surface similarity.

        Optimally aligned score found in `self.sim_aligned_surf` and the optimal SE(3)
        transformation is at `self.transform_surf`.

        Arguments
        ---------
        alpha : float Gaussian width parameter for overlap
        num_repeats : int (default=50)
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3) coordinates of transformed atoms
        """
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_ROCS_overlay_jax, optimize_ROCS_esp_overlay_jax, optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay_jax(
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_surf = np.array(se3_transform)
            self.sim_aligned_surf = np.array(score)
            return np.array(aligned_fit_points)
        else:
            # Torch
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay(
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_surf = se3_transform.numpy()
            self.sim_aligned_surf = score.numpy()
            return aligned_fit_points.numpy()


    def align_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using ESP+surface similarity.
        `lam` is scaled by (1e4/(4*55.263*np.pi))**2 for correct units.

        Typically, `lam=0.3` is used and is scaled internally.

        Optimally aligned score found in `self.sim_aligned_esp` and the optimal SE(3)
        transformation is at `self.transform_esp`.

        Arguments
        ---------
        alpha : float Gaussian width parameter for overlap
        lam : float (default = 0.3) Weighting factor for ESP scoring. Scaled internally.
        num_repeats : int (default=50)
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3) coordinates of transformed atoms
        """
        lam_scaled = LAM_SCALING * lam
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_ROCS_overlay_jax, optimize_ROCS_esp_overlay_jax, optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay_jax(
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                ref_charges=jnp.array(self.ref_molec.surf_esp),
                fit_charges=jnp.array(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_esp = np.array(se3_transform)
            self.sim_aligned_esp = np.array(score)
            return np.array(aligned_fit_points)
        else: # Use Torch implementation
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay(
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                ref_charges=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                fit_charges=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_esp = se3_transform.numpy()
            self.sim_aligned_esp = score.numpy()
            return aligned_fit_points.numpy()
    

    def align_with_esp_combo(self,
                             alpha: float,
                             lam: float = 0.001,
                             probe_radius: float = 1.0,
                             esp_weight: float = 0.5,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             use_jax: bool = False,
                             verbose: bool = False):
        """
        Align using ShaEP similarity score.
        If alpha is 0.81, then it automatically uses volumetric shape similarity.
        Otherwise, it uses surface shape similarity.

        Optimally aligned score found in `self.sim_aligned_esp_combo` and the optimal SE(3)
        transformation is at `self.transform_esp_combo`.

        Arguments
        ---------
        alpha : float Gaussian width parameter for overlap
        lam : float ESP weighting parameter
        probe_radius : float (default = 0.1) Surface points found within vdW radii + probe radius
            will be masked out. Surface generation uses a probe radius of 1.2 by default (radius of
            hydrogen) so we use a slightly lower radius for be more tolerant.
        esp_weight : float (default = 0.5) How much to weight shape vs esp_combo similarity ([0,1])
        num_repeats : int (default=50)
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3) coordinates of transformed atoms
        """
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_ROCS_overlay_jax, optimize_ROCS_esp_overlay_jax, optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_esp_combo_score_overlay_jax(
                ref_centers_w_H=jnp.array(self.ref_molec.mol.GetConformer().GetPositions()),
                fit_centers_w_H=jnp.array(self.fit_molec.mol.GetConformer().GetPositions()),
                ref_centers=jnp.array(self.ref_molec.atom_pos) if alpha == 0.81 else jnp.array(self.ref_molec.surf_pos),
                fit_centers=jnp.array(self.fit_molec.atom_pos) if alpha == 0.81 else jnp.array(self.fit_molec.surf_pos),
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                ref_partial_charges=jnp.array(self.ref_molec.partial_charges),
                fit_partial_charges=jnp.array(self.fit_molec.partial_charges), 
                ref_surf_esp=jnp.array(self.ref_molec.surf_esp),
                fit_surf_esp=jnp.array(self.fit_molec.surf_esp),
                ref_radii=jnp.array(self.ref_molec.radii),
                fit_radii=jnp.array(self.fit_molec.radii),
                alpha=alpha, 
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_esp_combo = np.array(se3_transform)
            self.sim_aligned_esp_combo = np.array(score)
            return np.array(aligned_fit_points)
        else:
            if alpha == 0.81:
                ref_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.atom_pos).to(torch.float32).to(self.device)
            else:
                ref_centers = torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device)

            aligned_fit_points, se3_transform, score = optimize_esp_combo_score_overlay(
                ref_centers_w_H=torch.from_numpy(self.ref_molec.mol.GetConformer().GetPositions()).to(torch.float32).to(self.device),
                fit_centers_w_H=torch.from_numpy(self.fit_molec.mol.GetConformer().GetPositions()).to(torch.float32).to(self.device),
                ref_centers=ref_centers,
                fit_centers=fit_centers,
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                ref_partial_charges=torch.from_numpy(self.ref_molec.partial_charges).to(torch.float32).to(self.device),
                fit_partial_charges=torch.from_numpy(self.fit_molec.partial_charges).to(torch.float32).to(self.device), 
                ref_surf_esp=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                fit_surf_esp=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
                ref_radii=torch.from_numpy(self.ref_molec.radii).to(torch.float32).to(self.device),
                fit_radii=torch.from_numpy(self.fit_molec.radii).to(torch.float32).to(self.device),
                alpha=alpha, 
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_esp_combo = se3_transform.numpy()
            self.sim_aligned_esp_combo = score.numpy()
            return aligned_fit_points.numpy()


    def align_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = 50,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = 200,
                         use_jax: bool = False,
                         verbose: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align fit_molec to ref_molec using pharmacophore similarity.

        Optimally aligned score found in `self.sim_aligned_pharm` and the optimal SE(3)
        transformation is at `self.transform_pharm`.

        Arguments
        ---------
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
            Number of different random initializations of SO(3) transformation parameters.
        trans_init : bool (default = False)
            Apply translation initializiation for alignment. `fit_molec`'s COM is translated to
            each `ref_molecs`'s pharmacophore, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's.
            If None, then num_repeats rotations are done with aligned COM's.
        lr : float (default=0.1) Learning rate or step-size for optimization
        max_num_steps : int (default=200) Maximum number of steps to optimize over.
        use_jax : bool (default = False) toggle to use Jax instead of PyTorch
        verbose : bool (False) Print initial and final similarity scores with scores every 100 steps.

        Returns
        -------
        Tuple
            aligned_fit_anchors : np.ndarray (P, 3) aligned coordinates of pharmacophores positions
            aligned_fit_vectors : np.ndarray (P, 3) aligned coordinates of pharmacophore vectors
        """
        if use_jax:
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from .alignment_jax import optimize_pharm_overlay_jax
            
            aligned_fit_anchors, aligned_fit_vectors, se3_transform, score = optimize_pharm_overlay_jax(
                ref_pharms=jnp.array(self.ref_molec.pharm_types),
                fit_pharms=jnp.array(self.fit_molec.pharm_types),
                ref_anchors=jnp.array(self.ref_molec.pharm_ancs),
                fit_anchors=jnp.array(self.fit_molec.pharm_ancs),
                ref_vectors=jnp.array(self.ref_molec.pharm_vecs),
                fit_vectors=jnp.array(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                num_repeats=num_repeats,
                trans_centers=self.ref_molec.pharm_ancs if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_pharm = np.array(se3_transform)
            self.sim_aligned_pharm = np.array(score)
            return np.array(aligned_fit_anchors), np.array(aligned_fit_vectors)
        # PyTorch
        aligned_fit_anchors, aligned_fit_vectors, se3_transform, score = optimize_pharm_overlay(
            ref_pharms=torch.from_numpy(self.ref_molec.pharm_types).to(torch.float32).to(self.device),
            fit_pharms=torch.from_numpy(self.fit_molec.pharm_types).to(torch.float32).to(self.device),
            ref_anchors=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device),
            fit_anchors=torch.from_numpy(self.fit_molec.pharm_ancs).to(torch.float32).to(self.device),
            ref_vectors=torch.from_numpy(self.ref_molec.pharm_vecs).to(torch.float32).to(self.device),
            fit_vectors=torch.from_numpy(self.fit_molec.pharm_vecs).to(torch.float32).to(self.device),
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=num_repeats,
            trans_centers=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device) if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose
        )

        self.transform_pharm = se3_transform.numpy()
        self.sim_aligned_pharm = score.numpy()
        return aligned_fit_anchors.numpy(), aligned_fit_vectors.numpy()
    

    def score_with_surf(self,
                        alpha: float,
                        use: str = 'np'
                        ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using surface similarity given current alignment.
        By default it uses the numpy implementation.

        Arguments
        ---------
        alpha : float Gaussian width parameter for overlap
        use : str (default = 'np') define what implementation to use
            For numpy use: 'np' or 'numpy'
            For jax use: 'jax' or 'jnp'
            For torch use: 'torch' or 'pytorch'

        Returns
        -------
        score : np.ndarray (1,) similarity score
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use == 'jax' or use == 'jnp': # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from shepherd_score.score.gaussian_overlap_jax import get_overlap_jax
            score = get_overlap_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                alpha=alpha,
            )
            return np.array(score)
        elif use == 'torch' or use == 'pytorch':
            # Torch
            score = get_overlap(
                centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                centers_2=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                alpha=alpha,
            )
            return score.cpu().numpy()
        elif use == 'np' or use == 'numpy':
            score = get_overlap_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                alpha=alpha,
            )
            return score

    
    def score_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       use: str = 'np'
                       ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using ESP+surface similarity given current alignment.
        `lam` is scaled by (1e4/(4*55.263*np.pi))**2 for correct units.

        Typically `lam` = 0.3 is used and is scaled internally.
        By default it uses the numpy implementation.

        Arguments
        ---------
        alpha : float Gaussian width parameter for overlap
        lam : float Weighting factor for ESP scoring
        use : str (default = 'np') define what implementation to use
            For numpy use: 'np' or 'numpy'
            For jax use: 'jax' or 'jnp'
            For torch use: 'torch' or 'pytorch'
        
        Returns
        -------
        score : np.ndarray (1,) similarity score
        """
        lam_scaled = LAM_SCALING * lam
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use in ('jax', 'jnp'): # Use Jax implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
            from shepherd_score.score.electrostatic_scoring_jax import get_overlap_esp_jax
            score = get_overlap_esp_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                charges_1=jnp.array(self.ref_molec.surf_esp),
                charges_2=jnp.array(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
            )
            return np.array(score)
        elif use in ('torch', 'pytorch'): # Use Torch implementation
            score = get_overlap_esp(
                centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                centers_2=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                charges_1=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                charges_2=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
                alpha=alpha,
                lam=lam_scaled,
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=self.fit_molec.surf_esp,
                alpha=alpha,
                lam=lam_scaled,
            )
            return score


    def score_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         use: str = 'np'
                         ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using pharmacophore similarity given current alignment.
        By default it uses the numpy implementation.

        Arguments
        ---------
        similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
        Specifies what similarity function to use.
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
        extended_points : bool of whether to score HBA/HBD with gaussian overlaps of extended points.
        only_extended : bool for when `extended_points` is True, decide whether to only score the
                        extended points (ignore anchor overlaps)
        use : str (default = 'np') define what implementation to use
            For numpy use: 'np' or 'numpy'
            For jax use: 'jax' or 'jnp'
            For torch use: 'torch' or 'pytorch'
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if use in ('jax', 'jnp'):
            raise NotImplementedError(
                f'Jax version of alignment has not been implemented yet. Use NumPy or PyTorch version by setting `use`.'
            )
        elif use in ('torch', 'pytorch'):
            # PyTorch
            score = get_overlap_pharm(
                ptype_1=torch.from_numpy(self.ref_molec.pharm_types).to(torch.float32).to(self.device),
                ptype_2=torch.from_numpy(self.fit_molec.pharm_types).to(torch.float32).to(self.device),
                anchors_1=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device),
                anchors_2=torch.from_numpy(self.fit_molec.pharm_ancs).to(torch.float32).to(self.device),
                vectors_1=torch.from_numpy(self.ref_molec.pharm_vecs).to(torch.float32).to(self.device),
                vectors_2=torch.from_numpy(self.fit_molec.pharm_vecs).to(torch.float32).to(self.device),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_pharm_np(
                ptype_1=self.ref_molec.pharm_types,
                ptype_2=self.fit_molec.pharm_types,
                anchors_1=self.ref_molec.pharm_ancs,
                anchors_2=self.fit_molec.pharm_ancs,
                vectors_1=self.ref_molec.pharm_vecs,
                vectors_2=self.fit_molec.pharm_vecs,
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score
    

    def get_transformed_mol_and_feats(self,
                                      se3_transform: np.ndarray
                                      ) -> Tuple:
        """
        Get an RDKit mol object and applicable features with a transformation applied.

        Arguments
        ---------
        se3_transform : np.ndarray (4,4) SE(3) transformation matrix.

        Returns
        -------
        Tuple
            transformed_mol : rdkit.Chem.Mol with transformed coordinates
            transformed_surf_pos : np.ndarray with transformed surface or None if N/A
            transformed_pharm_ancs : np.ndarray with transformed pharm positions or None if N/A
            transformed_pharm_vecs : np.ndarray with transformed pharm vectors or None if N/A
        """
        # Transform mol
        transformed_mol = update_mol_coordinates(
            mol=self.fit_molec.mol,
            coordinates=apply_SE3_transform_np(
                points=self.fit_molec.mol.GetConformer().GetPositions(),
                SE3_transform=se3_transform
            )
        )

        # Transform surface points
        transformed_surf_pos = None
        if self.fit_molec.surf_pos is not None:
            transformed_surf_pos = apply_SE3_transform_np(
                points=self.fit_molec.surf_pos,
                SE3_transform=se3_transform
            )

        # Transform pharmacophore features
        transformed_pharm_ancs = None
        transformed_pharm_vecs = None
        if self.fit_molec.pharm_ancs is not None and self.fit_molec.pharm_vecs is not None:
            transformed_pharm_ancs = apply_SE3_transform_np(
                points=self.fit_molec.pharm_ancs,
                SE3_transform=se3_transform
            )
            transformed_pharm_vecs = apply_SO3_transform_np(
                points=self.fit_molec.pharm_vecs,
                SE3_transform=se3_transform
            )
        return transformed_mol, transformed_surf_pos, transformed_pharm_ancs, transformed_pharm_vecs


    def get_transformed_molecule(self,
                                 se3_transform: np.ndarray
                                 ) -> Molecule:
        """
        Get Molecule object transformation applied to all applicable features for the fit molecule.

        Arguments
        ---------
        se3_transform : np.ndarray (4,4) SE(3) transformation matrix.

        Returns
        -------
        Molecule object with transformed features.
        """
        (transformed_mol,
        transformed_surf_pos,
        transformed_pharm_ancs,
        transformed_pharm_vecs) = self.get_transformed_mol_and_feats(se3_transform=se3_transform)

        transformed_fit_molec = Molecule(mol=transformed_mol,
                                         probe_radius=self.fit_molec.probe_radius,
                                         surface_points=transformed_surf_pos,
                                         partial_charges=self.fit_molec.partial_charges,
                                         electrostatics=self.fit_molec.surf_esp,
                                         pharm_multi_vector=self.fit_molec.pharm_multi_vector,
                                         pharm_types=self.fit_molec.pharm_types,
                                         pharm_ancs=transformed_pharm_ancs,
                                         pharm_vecs=transformed_pharm_vecs
                                         )
        return transformed_fit_molec
