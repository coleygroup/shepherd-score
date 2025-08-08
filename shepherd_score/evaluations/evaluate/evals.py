"""
Evaluation pipeline classes for generated molecules.
"""

import sys
import os
from typing import Union, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import itertools
from importlib.metadata import distributions

import numpy as np
import pandas as pd
from rdkit import Chem

if any(d.metadata["Name"] == 'rdkit' for d in distributions()):
    from rdkit.Contrib.SA_Score import sascorer
else:
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
    from SA_Score import sascorer

from rdkit.Chem import QED, Crippen, Lipinski, rdFingerprintGenerator
from rdkit.Chem.rdMolAlign import GetBestRMS, AlignMol
from rdkit.DataStructs import TanimotoSimilarity

from shepherd_score.evaluations.utils.convert_data import extract_mol_from_xyz_block, get_mol_from_atom_pos 

from shepherd_score.score.constants import ALPHA, LAM_SCALING

from shepherd_score.conformer_generation import optimize_conformer_with_xtb_from_xyz_block, single_point_xtb_from_xyz

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np

RNG = np.random.default_rng()
morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=True)

TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])


class ConfEval:
    """ Generated conformer evaluation pipeline """

    def __init__(self,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 solvent: Optional[str] = None,
                 num_processes: int = 1):
        """
        Base class for evaluation of a single generated conformer.

        Checks validity with RDKit pipeline and xTB single point calculation and optimization.
        Calculates 2D graph properties for valid molecules.

        Automatically aligns relaxed structure to the original structure via rdkit RMS.

        Arguments
        ---------
        atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot
            encoding.
        positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.
        solvent : str solvent type for xtb relaxation
        num_processes : int (default = 1) number of processors to use for xtb relaxation and RDKit
            RMSD alignment.
        """
        self.xyz_block = None
        self.mol = None
        self.smiles = None
        self.molblock = None
        self.energy = None
        self.partial_charges = None

        self.solvent = solvent
        self.charge = 0

        self.xyz_block_post_opt = None
        self.mol_post_opt = None
        self.smiles_post_opt = None
        self.molblock_post_opt = None
        self.energy_post_opt = None
        self.partial_charges_post_opt = None

        self.is_valid = False
        self.is_valid_post_opt = False
        self.is_graph_consistent = False

        # 2D graph features
        self.SA_score = None
        self.QED = None
        self.logP = None
        self.fsp3 = None
        self.morgan_fp = None

        self.SA_score_post_opt = None
        self.QED_post_opt = None
        self.logP_post_opt = None
        self.fsp3_post_opt = None
        self.morgan_fp_post_opt = None

        # Consistency in 3D
        self.strain_energy = None
        self.rmsd = None

        # 1. Converts coords + atom_ids -> xyz block
        # 2. Get mol from xyz block
        self.mol, self.charge, self.xyz_block = get_mol_from_atom_pos(atoms=atoms, positions=positions)

        # 3. Get xtb energy and charges of initial conformation
        try:
            self.energy, self.partial_charges = single_point_xtb_from_xyz(xyz_block=self.xyz_block,
                                                                          solvent=self.solvent,
                                                                          charge=self.charge,
                                                                          num_cores=num_processes,
                                                                          temp_dir=TMPDIR)
            self.partial_charges = np.array(self.partial_charges)
        except Exception as e:
            pass
        self.is_valid = self.mol is not None and self.partial_charges is not None
        if self.is_valid:
            self.smiles = Chem.MolToSmiles(Chem.RemoveHs(self.mol))
            self.molblock = Chem.MolToMolBlock(self.mol)

        # 4. Relax structure with xtb
        try:
            xtb_out = optimize_conformer_with_xtb_from_xyz_block(self.xyz_block,
                                                                solvent=self.solvent,
                                                                num_cores=num_processes,
                                                                charge=self.charge,
                                                                temp_dir=TMPDIR)
            self.xyz_block_post_opt, self.energy_post_opt, self.partial_charges_post_opt = xtb_out
            self.partial_charges_post_opt = np.array(self.partial_charges_post_opt)

            # 5. Check if relaxed_structure is valid
            self.mol_post_opt = extract_mol_from_xyz_block(xyz_block=self.xyz_block_post_opt,
                                                           charge=self.charge)
        except Exception as e:
            pass

        self.is_valid_post_opt = self.mol_post_opt is not None and self.partial_charges_post_opt is not None

        # 6. Check if 2D molecular graphs are consistent
        if self.is_valid and self.is_valid_post_opt:
            self.is_graph_consistent = Chem.MolToSmiles(self.mol) == Chem.MolToSmiles(self.mol_post_opt)
            # Align post-opt mol with RMSD
            mol_atom_ids = list(range(self.mol.GetNumAtoms()))
            mol_post_opt_atom_ids = list(range(self.mol_post_opt.GetNumAtoms()))
            AlignMol(prbMol=self.mol_post_opt, refMol=self.mol, atomMap=[i for i in zip(mol_post_opt_atom_ids, mol_atom_ids)])

        if self.is_valid_post_opt:
            self.smiles_post_opt = Chem.MolToSmiles(Chem.RemoveHs(self.mol_post_opt))
            self.molblock_post_opt = Chem.MolToMolBlock(self.mol_post_opt)

        # 7. Calculate strain energy with relaxed structure
        if self.energy is not None and self.energy_post_opt is not None:
            self.strain_energy = self.energy - self.energy_post_opt

        # 8. Calculate RMSD from relaxed structure
        if self.is_graph_consistent:
            mol_copy = deepcopy(Chem.RemoveHs(self.mol))
            mol_post_opt_copy = deepcopy(Chem.RemoveHs(self.mol_post_opt))
            self.rmsd = GetBestRMS(prbMol=mol_copy, refMol=mol_post_opt_copy, numThreads=num_processes)

        # 9. 2D graph properties
        if self.is_valid:
            self.SA_score = sascorer.calculateScore(Chem.RemoveHs(self.mol))
            self.QED = QED.qed(self.mol)
            self.logP = Crippen.MolLogP(self.mol)
            self.fsp3 = Lipinski.FractionCSP3(self.mol)
            self.morgan_fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.mol))
        
        # 10. 2D graph properties post optimization
        if self.is_valid_post_opt:
            self.SA_score_post_opt = sascorer.calculateScore(Chem.RemoveHs(self.mol_post_opt))
            self.QED_post_opt = QED.qed(self.mol_post_opt)
            self.logP_post_opt = Crippen.MolLogP(self.mol_post_opt)
            self.fsp3_post_opt = Lipinski.FractionCSP3(self.mol_post_opt)
            self.morgan_fp_post_opt = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.mol_post_opt))


    def to_pandas(self):
        """
        Convert the stored attributes to a pd.Series (for global attributes).

        Arguments
        ---------
        self

        Returns
        -------
        pd.Series : holds attributes in an easy to visualize way
        """
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            global_attrs[key] = value

        series_global = pd.Series(global_attrs)

        return series_global


class ConsistencyEval(ConfEval):
    """
    Evaluation of the consistency between jointly generated molecules' features.
    Consistency in terms of similarity scores.
    """
    def __init__(self,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 surf_points: Optional[np.ndarray] = None,
                 surf_esp: Optional[np.ndarray] = None,
                 pharm_feats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 probe_radius: float = 1.2,
                 num_processes: int = 1):
        """
        Consistency evaluation class for jointly generated molecule and features
        using 3D similarity scoring functions. Inherits from ConfEval so that it
        can first run a conformer evaluation on the generated molecule.

        Must supply `atoms` and `positions` AND at least one of the features necessary for
        similarity scoring.

        IMPORTANT ASSUMPTIONS
        1. Gaussian width parameter (alpha) for surface similarity was fitted to a probe radius of
            1.2 A.
        2. ESP weighting parameter (lam) for electrostatic similarity is set to 0.3 which was
            tested for assumption 1.

        Arguments
        ---------
        atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M)
            one-hot encoding.
        positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.
        surf_points : Optional[np.ndarray] (M,3) generated surface point cloud.
        surf_esp : Optional[np.ndarray] (M,) generated electrostatic potential on surface.
        pharm_feats : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] where the arrays are as follows:
            pharm_types : Optional[np.ndarray] (P,) type of pharmacophore defined by
                shepherd_score.score.utils.constants.P_TYPES
            pharm_ancs : Optional[np.ndarray] (P,3) anchor positions of pharmacophore
            pharm_vecs : Optional[np.ndarray] (P,3) unit vector of each pharmacophore relative to anchor
        pharm_multi_vector : Optional[bool] Use multiple vectors to represent Aro/HBA/HBD or single
        solvent : str solvent type for xtb relaxation
        probe_radius : Optional[float] radius of probe atom used to generate solvent accessible
            surface. (default = 1.2) which is the VdW radius of a hydrogen
        num_processes : int number of processors to use for xtb relaxation
        """
        if not (isinstance(atoms, np.ndarray) or isinstance(positions, np.ndarray)):
            raise ValueError(f"Must provide `atoms` and `positions` as np.ndarrays. Instead {type(atoms)} and {type(positions)} were given.")

        super().__init__(atoms=atoms, positions=positions, solvent=solvent, num_processes=num_processes)

        self.molec = None
        self.probe_radius = probe_radius
        self.molec_regen = None
        self.molec_post_opt = None

        self.sim_surf_consistent = None
        self.sim_esp_consistent = None
        self.sim_pharm_consistent = None

        self.sim_surf_consistent_relax = None
        self.sim_esp_consistent_relax = None
        self.sim_pharm_consistent_relax = None

        self.sim_surf_consistent_relax_optimal = None
        self.sim_esp_consistent_relax_optimal = None
        self.sim_pharm_consistent_relax_optimal = None
        
        if pharm_feats is not None:
            pharm_types, pharm_ancs, pharm_vecs = pharm_feats
            num_pharms = len(pharm_types)

            if pharm_ancs.shape != (num_pharms, 3) or pharm_vecs.shape != (num_pharms, 3):
                raise ValueError(
                    f'Provided pharmacophore features do not match dimensions: pharm_types {pharm_types.shape}, pharm_ancs {pharm_ancs.shape}, pharm_vecs {pharm_vecs.shape}'
                )
        else:
            pharm_types, pharm_ancs, pharm_vecs = None, None, None

        has_pharm_features = (isinstance(pharm_types, np.ndarray)
                              and isinstance(pharm_ancs, np.ndarray)
                              and isinstance(pharm_vecs, np.ndarray))
        if has_pharm_features and pharm_multi_vector is None:
            print('WARNING: Generated pharmacophore features provided, but `pharm_multi_vector` is None.')
            print('         Pharmacophore similarity not computed.')
        if not isinstance(surf_points, np.ndarray) and not isinstance(surf_esp, np.ndarray) and not has_pharm_features:
            raise ValueError(f'Must provide at least one of the generated representations: surface, electrostatics, or pharmacophores.')
        
        # Scoring parameters
        self.num_surf_points = len(surf_points) if surf_points is not None else None
        # Assumes no radius scaling with probe_radius=1.2
        self.alpha = ALPHA(self.num_surf_points) if self.num_surf_points is not None else None
        self.lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT
        self.lam_scaled = self.lam * LAM_SCALING # -> ONLY TO BE USED FOR get_overlap_esp*

        # Self-consistency of features for generated molecule
        if self.is_valid:
            # Generate a Molecule object with generated features
            self.molec = Molecule(
                self.mol,
                partial_charges=np.array(self.partial_charges),
                surface_points=surf_points,
                electrostatics=surf_esp,
                pharm_types=pharm_types,
                pharm_ancs=pharm_ancs,
                pharm_vecs=pharm_vecs,
                probe_radius=self.probe_radius
            )
            # Generate a Molecule object with regenerated features
            self.molec_regen = Molecule(
                self.mol,
                num_surf_points = self.num_surf_points,
                probe_radius=self.probe_radius,
                partial_charges = np.array(self.partial_charges),
                pharm_multi_vector = pharm_multi_vector if has_pharm_features else None
            )

            if self.molec.surf_pos is not None:
                self.sim_surf_consistent = get_overlap_np(
                    self.molec.surf_pos,
                    self.molec_regen.surf_pos,
                    alpha=self.alpha
                )

            if self.molec.surf_esp is not None and self.molec.surf_pos is not None:
                self.sim_esp_consistent = get_overlap_esp_np(
                    self.molec.surf_pos, self.molec_regen.surf_pos,
                    self.molec.surf_esp, self.molec_regen.surf_esp,
                    alpha=self.alpha,
                    lam=self.lam_scaled
                )

            if has_pharm_features and pharm_multi_vector is not None:
                self.sim_pharm_consistent = get_overlap_pharm_np(
                    self.molec.pharm_types,
                    self.molec_regen.pharm_types,
                    self.molec.pharm_ancs,
                    self.molec_regen.pharm_ancs,
                    self.molec.pharm_vecs,
                    self.molec_regen.pharm_vecs,
                    similarity='tanimoto',
                    extended_points=False,
                    only_extended=False
                )

        # Consistency between generated molecule and relaxed structure and features
        if self.is_valid and self.is_valid_post_opt: 
            # Generate a Molecule object of relaxed structure
            self.molec_post_opt = Molecule(
                self.mol_post_opt,
                num_surf_points = self.num_surf_points,
                probe_radius=self.probe_radius,
                partial_charges = np.array(self.partial_charges_post_opt),
                pharm_multi_vector = pharm_multi_vector if has_pharm_features else None
            )

            # Score only since we already align w.r.t. RMS of the generated atomic point cloud
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_consistent_relax = get_overlap_np(
                    self.molec.surf_pos,
                    self.molec_post_opt.surf_pos,
                    alpha=self.alpha
                )
            if self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:            
                self.sim_esp_consistent_relax = get_overlap_esp_np(
                    self.molec.surf_pos, self.molec_post_opt.surf_pos,
                    self.molec.surf_esp, self.molec_post_opt.surf_esp,
                    alpha=self.alpha,
                    lam=self.lam_scaled
                )
            if isinstance(pharm_multi_vector, bool) and self.molec_post_opt.pharm_ancs is not None and self.molec.pharm_ancs is not None:
                self.sim_pharm_consistent_relax = get_overlap_pharm_np(
                    self.molec.pharm_types,
                    self.molec_post_opt.pharm_types,
                    self.molec.pharm_ancs,
                    self.molec_post_opt.pharm_ancs,
                    self.molec.pharm_vecs,
                    self.molec_post_opt.pharm_vecs,
                    similarity='tanimoto',
                    extended_points=False,
                    only_extended=False
                )

            # Alignment with scoring functions
            mp_ref_and_relaxed = MoleculePair(self.molec,
                                              self.molec_post_opt,
                                              num_surf_points=self.num_surf_points,
                                              do_center=False)
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_consistent_relax_optimal = self._align_with_surface(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:            
                self.sim_esp_consistent_relax_optimal = self._align_with_esp(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if isinstance(pharm_multi_vector, bool) and self.molec_post_opt.pharm_ancs is not None and self.molec.pharm_ancs is not None:
                self.sim_pharm_consistent_relax_optimal = self._align_with_pharm(mp_ref_and_relaxed=mp_ref_and_relaxed)


    def _align_with_surface(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with surface.

        Returns
        -------
        float : Surface similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_surf(
            self.alpha,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )
        surf_similarity = mp_ref_and_relaxed.sim_aligned_surf
        return float(surf_similarity)
    

    def _align_with_esp(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with ESP

        Returns
        -------
        float : ESP similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_esp(
            self.alpha,
            lam=self.lam,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        ) 
        esp_similarity = mp_ref_and_relaxed.sim_aligned_esp
        return float(esp_similarity)


    def _align_with_pharm(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with pharmacophores

        Returns
        -------
        float : Pharmacophore similarity score of optimally aligned molecule.
        """
        aligned_fit_anchors, aligned_vectors = mp_ref_and_relaxed.align_with_pharm(
            similarity='tanimoto',
            extended_points=False,
            only_extended=False,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )
        pharm_similarity = mp_ref_and_relaxed.sim_aligned_pharm
        return float(pharm_similarity)


class ConditionalEval(ConfEval):
    """ Evaluation of conditionally generated molecules' quality and similarity. """

    def __init__(self,
                 ref_molec: Molecule,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 condition: str,
                 num_surf_points: int = 400,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 num_processes: int = 1):
        """
        Evaluation pipeline for conditionally-generated molecules.
        Inherits from ConfEval so that it can first run a conformer evaluation on the generated
        molecule.

        IMPORTANT ASSUMPTIONS
        1. Gaussian width parameter (alpha) for surface similarity assumes a probe radius of 1.2A.
        2. ESP weighting parameter (lam) for electrostatic similarity is set to 0.3 which was
           tested for assumption 1.

        Arguments
        ---------
        ref_molec : Molecule object of reference/target molecule. Must contain the representation
            that was used for conditioning
        atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot encoding.
        positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.
        condition : str for which the molecule was conditioned on out of ('surface', 'esp', 'pharm',
            'all'). Used for alignment. Choose 'esp' or 'all' if you want to compute ESP-aligned
            scores for other profiles.
        num_surf_points : int (default = 400) Number of surface points to sample for similiarity scoring.
        pharm_multi_vector : Optional[bool] Use multiple vectors to represent Aro/HBA/HBD or single
        solvent : str solvent type for xtb relaxation
        num_processes : int number of processors to use for xtb relaxation
        """
        condition = condition.lower()
        self.condition = None
        if 'surf' in condition or 'shape' in condition:
            self.condition = 'surface'
        elif 'esp' in condition or 'electrostatic' in condition:
            self.condition = 'esp'
        elif 'pharm' in condition:
            self.condition = 'pharm'
        elif condition == 'all':
            self.condition = 'all'
        else:
            raise ValueError(f'`condition` must contain one of the following: "surf", "esp", "pharm", or "all". Instead, {condition} was given.')

        super().__init__(atoms=atoms,
                         positions=positions,
                         solvent=solvent,
                         num_processes=num_processes)

        self.sim_surf_target = None
        self.sim_esp_target = None
        self.sim_pharm_target = None

        self.sim_surf_target_relax = None
        self.sim_esp_target_relax = None
        self.sim_pharm_target_relax = None

        self.sim_surf_target_relax_optimal = None
        self.sim_esp_target_relax_optimal = None
        self.sim_pharm_target_relax_optimal = None

        self.sim_surf_target_relax_esp_aligned = None
        self.sim_pharm_target_relax_esp_aligned = None
        
        # Scoring parameters
        self.num_surf_points = num_surf_points
        self.alpha = ALPHA(self.num_surf_points) # Fitted to probe_radius=1.2
        self.lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT
        self.lam_scaled = self.lam * LAM_SCALING # -> ONLY TO BE USED FOR get_overlap_esp*

        self.ref_molec = ref_molec # Reference Molecule object
        self.molec_regen = None
        self.molec_post_opt = None

        if self.is_valid:
            # Generate a Molecule object with regenerated features
            self.molec_regen = Molecule(
                self.mol,
                num_surf_points = self.num_surf_points,
                partial_charges = np.array(self.partial_charges),
                pharm_multi_vector = pharm_multi_vector,
                probe_radius=self.ref_molec.probe_radius
            )

            if self.molec_regen.surf_pos is not None:
                self.sim_surf_target = get_overlap_np(
                    self.molec_regen.surf_pos,
                    self.ref_molec.surf_pos,
                    alpha=self.alpha
                )

            if self.molec_regen.surf_esp is not None and self.molec_regen.surf_pos is not None:
                self.sim_esp_target = get_overlap_esp_np(
                    self.molec_regen.surf_pos, self.ref_molec.surf_pos,
                    self.molec_regen.surf_esp, self.ref_molec.surf_esp,
                    alpha=self.alpha,
                    lam=self.lam_scaled
                )

            if pharm_multi_vector is not None and self.ref_molec.pharm_ancs is not None:
                self.sim_pharm_target = get_overlap_pharm_np(
                    self.molec_regen.pharm_types,
                    self.ref_molec.pharm_types,
                    self.molec_regen.pharm_ancs,
                    self.ref_molec.pharm_ancs,
                    self.molec_regen.pharm_vecs,
                    self.ref_molec.pharm_vecs,
                    similarity='tanimoto',
                    extended_points=False,
                    only_extended=False
                )

        # Similarity between relaxed structure and target molecule -> first align
        if self.is_valid_post_opt:
            # Generate a Molecule object of relaxed structure
            self.molec_post_opt = Molecule(
                self.mol_post_opt,
                num_surf_points = self.num_surf_points,
                partial_charges = np.array(self.partial_charges_post_opt),
                pharm_multi_vector = pharm_multi_vector,
                probe_radius=self.ref_molec.probe_radius
            )

            # Score based on RMS alignment
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_target_relax = get_overlap_np(
                    self.molec_post_opt.surf_pos,
                    self.ref_molec.surf_pos,
                    alpha=self.alpha
                )

            if self.molec_post_opt.surf_esp is not None and self.molec_post_opt.surf_pos is not None:
                self.sim_esp_target_relax = get_overlap_esp_np(
                    self.molec_post_opt.surf_pos, self.ref_molec.surf_pos,
                    self.molec_post_opt.surf_esp, self.ref_molec.surf_esp,
                    alpha=self.alpha,
                    lam=self.lam_scaled
                )

            if pharm_multi_vector is not None and self.ref_molec.pharm_ancs is not None:
                self.sim_pharm_target_relax = get_overlap_pharm_np(
                    self.molec_post_opt.pharm_types,
                    self.ref_molec.pharm_types,
                    self.molec_post_opt.pharm_ancs,
                    self.ref_molec.pharm_ancs,
                    self.molec_post_opt.pharm_vecs,
                    self.ref_molec.pharm_vecs,
                    similarity='tanimoto',
                    extended_points=False,
                    only_extended=False
                )

            # Align and score w.r.t. specified condition
            mp_ref_and_relaxed = MoleculePair(self.ref_molec,
                                              self.molec_post_opt,
                                              num_surf_points=self.num_surf_points,
                                              do_center=False)
            if (self.condition == 'surface' or self.condition == 'all') and self.molec_post_opt.surf_pos is not None:
                self.sim_surf_target_relax_optimal = self._align_with_surface(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if (self.condition == 'esp' or self.condition == 'all') and self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:
                self.sim_esp_target_relax_optimal = self._align_with_esp(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if (self.condition == 'pharm' or self.condition == 'all') and isinstance(pharm_multi_vector, bool):
                self.sim_pharm_target_relax_optimal = self._align_with_pharm(mp_ref_and_relaxed=mp_ref_and_relaxed)

            # Compute ESP-aligned surf and pharmacophore similarity scores
            if mp_ref_and_relaxed.transform_esp is not None and self.condition in ('esp', 'all'):
                molec_post_opt_esp_aligned = mp_ref_and_relaxed.get_transformed_molecule(mp_ref_and_relaxed.transform_esp)
                esp_aligned_molec_pair = MoleculePair(ref_mol=self.ref_molec,
                                                      fit_mol=molec_post_opt_esp_aligned,
                                                      num_surf_points=self.ref_molec.num_surf_points,
                                                      do_center=False)
                self.sim_surf_target_relax_esp_aligned = esp_aligned_molec_pair.score_with_surf(
                    alpha=self.alpha, use='np'
                )
                if isinstance(pharm_multi_vector, bool):
                    self.sim_pharm_target_relax_esp_aligned = esp_aligned_molec_pair.score_with_pharm(
                        similarity='tanimoto',
                        extended_points=False,
                        only_extended=False,
                        use='np'
                    )


    def _align_with_surface(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with surface.

        Returns
        -------
        float : Surface similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_surf(
            self.alpha,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )

        surf_similarity = mp_ref_and_relaxed.sim_aligned_surf
        return float(surf_similarity)
    

    def _align_with_esp(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with ESP

        Returns
        -------
        float : ESP similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_esp(
            self.alpha,
            lam=self.lam,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        ) 
        esp_similarity = mp_ref_and_relaxed.sim_aligned_esp
        return float(esp_similarity)


    def _align_with_pharm(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with pharmacophores

        Returns
        -------
        float : Pharmacophore similarity score of optimally aligned molecule.
        """
        aligned_fit_anchors, aligned_vectors = mp_ref_and_relaxed.align_with_pharm(
            similarity='tanimoto',
            extended_points=False,
            only_extended=False,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )
        pharm_similarity = mp_ref_and_relaxed.sim_aligned_pharm
        return float(pharm_similarity)
