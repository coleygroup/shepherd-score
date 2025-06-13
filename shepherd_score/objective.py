"""
Module contains `Objective` class used for computing similarity scores between reference and fit
molecules.
"""

from typing import Union, Optional, List, Tuple
import os
from tqdm import tqdm

import numpy as np
from rdkit import Chem
import open3d # imported because sometimes order matters
import torch

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.alignment_utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np
from shepherd_score.conformer_generation import embed_conformer_from_smiles, optimize_conformer_with_xtb
from shepherd_score.conformer_generation import generate_opt_conformers, generate_opt_conformers_xtb, charges_from_single_point_conformer_with_xtb
from shepherd_score.score.constants import ALPHA

TMPDIR = '.'
if 'TMPDIR' in os.environ:
    TMPDIR = os.environ['TMPDIR']


class GeneralObjective:
    """
    Objective class that computes similarity scores of fit molecules given a reference molecule.
    """
    def __init__(self,
                 ref_mol: Union[str, Chem.Mol],
                 rep: str = 'esp',
                 xtb_opt: bool = False,
                 num_points: Optional[int] = None,
                 use_vol: bool = False,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 num_processes: int = 1):
        """
        Generalized objective class that computes similarity scores of fit molecules given a
        reference.
        Constructor -- sets up the reference molecule and relevant representations.

        Arguments
        ---------
        ref_mol : str or Chem.Mol representing reference molecule.
        rep : str for representation chosen from ('shape', 'esp', 'pharm')
        xtb_opt : bool to optimize conformers with xtb or just use EKTG embedding and MMFF opt.
        num_points : Optional[int] if `rep` is 'shape' or 'esp' this is necessary to generate the
            surface point cloud.
        use_vol : bool (default = False) toggle to True if you want to use volumetric scoring.
            Ignores `num_points` if True.
        pharm_multi_vector : Optional[bool] whether to represent pharmacophores with mulitple
            vectors (if applicable) or a single, averaged vector.
        solvent : Optional[str] solvent to use if optimizing with xtb. Default (None) is gas phase.
        num_processes : int (default = 1) Number of proccesses to use during xtb optimization
        """
        self.ref_partial_charges = None
        self.representation = rep.lower()
        self.xtb_opt = xtb_opt
        self.solvent = solvent
        self.use_vol = use_vol
        self.ref_molec = None

        self.buffer = {} # stores past SMILES and their similarity scores

        self.num_points = num_points
        if self.use_vol:
            self.alpha = 0.81
            self.num_points = None
        else:
            self.alpha = ALPHA(self.num_points) if self.num_points is not None else None
        
        self.lam = 0.1 if self.use_vol else 0.3

        self.pharm_multi_vector = pharm_multi_vector

        self.num_processes = num_processes

        self.sim_score_distr_with_resample = np.array([1.])
        self.sim_score_upper_bound = 1.
        
        if self.representation == 'shape':
            pass
        elif self.representation in ('electrostatics', 'esp'):
            self.representation = 'esp'
        elif self.representation in ('pharmacophore', 'pharmacophores', 'pharm'):
            self.representation = 'pharm'
        else:
            raise ValueError(f'Please enter a valid key for `rep`. "{rep}" was given')

        if self.representation in ('shape', 'esp') and self.num_points is None and not self.use_vol:
            raise ValueError(f'Either `use_vol` must be True or `num_points` must be supplied for surface point cloud.')
        if self.representation == 'pharm' and self.pharm_multi_vector is None:
            raise ValueError(f'`pharm_multi_vector` must be supplied for surface point cloud. {pharm_multi_vector} was given.')

        if isinstance(ref_mol, str):
            ref_mol = embed_conformer_from_smiles(ref_mol, attempts=50, MMFF_optimize=True)
            if self.xtb_opt:
                ref_mol, _, self.ref_partial_charges = optimize_conformer_with_xtb(
                    ref_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    temp_dir=TMPDIR
                )
            else:
                self.ref_partial_charges = np.array(charges_from_single_point_conformer_with_xtb(
                    ref_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    temp_dir=TMPDIR
                ))
        elif isinstance(ref_mol, Chem.Mol):
            try:
                ref_mol.GetConformer()
                has_conformer = True
            except:
                has_conformer = False

            if not has_conformer:
                ref_mol = embed_conformer_from_smiles(ref_mol, attempts=50, MMFF_optimize=True)
                if self.xtb_opt:
                    ref_mol, _, self.ref_partial_charges = optimize_conformer_with_xtb(
                        ref_mol,
                        solvent=self.solvent,
                        num_cores=self.num_processes,
                        temp_dir=TMPDIR
                    )
            if self.ref_partial_charges is None:
                self.ref_partial_charges = np.array(charges_from_single_point_conformer_with_xtb(
                    ref_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    temp_dir=TMPDIR
                ))
        else:
            raise ValueError(f'`ref_molec` must be str or Chem.Mol object. Instead {type(ref_mol)} was given.')

        # Get Molecule objects
        if self.representation in ('shape', 'esp'):
            self.ref_molec = Molecule(mol=ref_mol,
                                      num_surf_points=self.num_points,
                                      partial_charges=self.ref_partial_charges)
        
        elif self.representation == 'pharm':
            self.ref_molec = Molecule(mol=ref_mol,
                                      pharm_multi_vector=self.pharm_multi_vector)
        
        if self.representation in ('shape', 'esp') and not self.use_vol:
            self.sim_score_distr_with_resample = self.resampling_surf_scores()
            self.sim_score_upper_bound = max(self.sim_score_distr_with_resample)

        self.buffer[Chem.MolToSmiles(self.ref_molec.mol)] = self.sim_score_upper_bound


    def resampling_surf_scores(self) -> np.ndarray:
        """
        Capture distribution of surface similarity or surface ESP scores caused by resampling
        surface.

        Returns
        -------
        score_distr : np.ndarray (if not relevant)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.num_points >=150 and torch.cuda.is_available():
            use_torch = True
        else:
            use_torch = False

        num_repeats = 5

        score_distr = np.empty(num_repeats)
        if self.representation == 'shape':
            for i in range(num_repeats):
                molec = Molecule(mol=self.ref_molec.mol,
                                 num_surf_points=self.ref_molec.num_surf_points,
                                 probe_radius=self.ref_molec.probe_radius,
                                 partial_charges=self.ref_molec.partial_charges)
                if use_torch:
                    score_distr[i] = get_overlap(
                        centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(device),
                        centers_2=torch.from_numpy(molec.surf_pos).to(torch.float32).to(device),
                        alpha=self.alpha
                    ).cpu().numpy()
                else:
                    score_distr[i] = get_overlap_np(
                        centers_1=self.ref_molec.surf_pos,
                        centers_2=molec.surf_pos,
                        alpha=self.alpha
                    )

        elif self.representation == 'esp':
            for i in range(num_repeats):
                molec = Molecule(mol=self.ref_molec.mol,
                                 num_surf_points=self.ref_molec.num_surf_points,
                                 probe_radius=self.ref_molec.probe_radius,
                                 partial_charges=self.ref_molec.partial_charges)
                if use_torch:
                    score_distr[i] = get_overlap_esp(
                        centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(device),
                        centers_2=torch.from_numpy(molec.surf_pos).to(torch.float32).to(device),
                        charges_1=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(device),
                        charges_2=torch.from_numpy(molec.surf_esp).to(torch.float32).to(device),
                        alpha=self.alpha,
                        lam=self.lam
                    ).cpu().numpy()
                else:
                    score_distr[i] = get_overlap_esp_np(
                        centers_1=self.ref_molec.surf_pos, 
                        centers_2=molec.surf_pos,
                        charges_1=self.ref_molec.surf_esp,
                        charges_2=molec.surf_esp,
                        alpha=self.alpha,
                        lam=self.lam
                    )
        else:
            score_distr = np.array([1.]*num_repeats)

        return score_distr


    def _score(self,
               fit_mol: Chem.Mol,
               fit_partial_charges: Optional[np.ndarray] = None,
               trans_init: bool = False,
               use_jax: bool = False) -> float:
        """
        Align and score a fit molecule to the reference molecule.

        Arguments
        ---------
        fit_mol : Chem.Mol fit molecule to compare to reference.
        fit_partial_charges : np.ndarray of partial charges for each atom
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False

        Returns
        -------
        float : similarity score
        """
        if fit_partial_charges is not None:
            fit_partial_charges = np.array(fit_partial_charges)
        if self.representation in ('shape', 'electrostatics', 'esp'):
            fit_molec = Molecule(mol=fit_mol,
                                 num_surf_points=self.num_points,
                                 partial_charges=fit_partial_charges)

        elif self.representation in ('pharmacophore', 'pharmacophores', 'pharm', 'p'):
            fit_molec = Molecule(mol=fit_mol,
                                 pharm_multi_vector=self.pharm_multi_vector)

        molec_pair = MoleculePair(self.ref_molec, fit_molec, num_surf_points=self.num_points)

        if self.representation == 'shape':
            if self.use_vol:
                molec_pair.align_with_vol(no_H=True, trans_init=trans_init, use_jax=use_jax)
                score = molec_pair.sim_aligned_vol_noH
            else:
                molec_pair.align_with_surf(alpha=self.alpha, trans_init=trans_init, use_jax=use_jax)
                score = molec_pair.sim_aligned_surf

        elif self.representation == 'esp':
            if self.use_vol:
                molec_pair.align_with_vol_esp(lam=self.lam, no_H=True, trans_init=trans_init, use_jax=use_jax)
                score = molec_pair.sim_aligned_vol_esp_noH
            else:
                molec_pair.align_with_esp(self.alpha,
                                        lam=self.lam,
                                        trans_init=trans_init,
                                        use_jax=use_jax)
                score = molec_pair.sim_aligned_esp
        
        elif self.representation == 'pharm':
            molec_pair.align_with_pharm(similarity='tanimoto',
                                        extended_points=False,
                                        only_extended=False,
                                        trans_init=trans_init,
                                        use_jax=False)
            score = molec_pair.sim_aligned_pharm

        return score


    def score(self,
              smiles: str,
              num_conformers: int = 1,
              trans_init: bool = False,
              use_jax: bool = False) -> float:
        """
        Align and score a fit molecule to the reference molecule.

        Arguments
        ---------
        smiles : str SMILES string of molecule to compare to reference.
        num_conformers : int (default = 1) Max number of conformers to generate and score.
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False

        Returns
        -------
        float : similarity score (or max score if num_conformers > 1)
        """
        fit_partial_charges = None
        if num_conformers == 1:
            fit_mol = embed_conformer_from_smiles(smiles, attempts=50, MMFF_optimize=True)

            if self.xtb_opt:
                fit_mol, _, fit_partial_charges = optimize_conformer_with_xtb(
                    fit_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    temp_dir=TMPDIR
                )
            if fit_partial_charges is None:
                fit_partial_charges = charges_from_single_point_conformer_with_xtb(
                    fit_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    temp_dir=TMPDIR
                )
            score = self._score(fit_mol,
                                trans_init=trans_init,
                                use_jax=use_jax)
            return score

        if self.xtb_opt:
            fit_mols, fit_partial_charges = generate_opt_conformers_xtb(smiles, MMFF_optimize=True, verbose=False)
        else:
            fit_mols = generate_opt_conformers(smiles, MMFF_optimize=True, verbose=False)
            fit_partial_charges = []
            for m in fit_mols:
                fit_partial_charges.append(charges_from_single_point_conformer_with_xtb(
                    conformer=m, 
                    solvent=self.solvent, 
                    num_cores=self.num_processes, 
                    temp_dir=TMPDIR
                ))

        scores = []
        for i, fit_mol in enumerate(fit_mols):
            scores.append(self._score(fit_mol,
                                      fit_partial_charges=fit_partial_charges[i],
                                      trans_init=trans_init,
                                      use_jax=use_jax))

        return max(scores)


    def score_multiple(self,
                       smiles: List[str],
                       num_conformers: int = 1,
                       trans_init: bool = False,
                       use_jax: bool = False) -> List[float]:
        """
        Aligns and scores multiple fit SMILES.

        Arguments
        ---------
        smiles : List[str] of SMILES strings of molecules to compare to reference.
        num_conformers : int (default = 1) Max number of conformers to generate and score.
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False

        Returns
        -------
        List[float] : similarity scores. Returns a value of -1 if Objective.score fails for any reason.
        """
        scores = []
        for smi in smiles:
            smi = Chem.CanonSmiles(smi)
            if smi in self.buffer:
                # skip if we've already computed it
                scores.append(self.buffer[smi])
            else:
                try:
                    scores.append(self.score(smi,
                                             num_conformers=num_conformers,
                                             trans_init=trans_init,
                                             use_jax=use_jax))
                except:
                    scores.append(-1.)

            self.buffer[smi] = scores[-1] # store {smiles : score}

        return scores
    


class Objective:
    """
    Objective class that computes similarity scores of fit molecules given a reference molecule.
    """
    def __init__(self,
                 ref_molblock: str,
                 ref_partial_charges: np.array,
                 num_points: int = 400,
                 pharm_multi_vector: Optional[bool] = False,
                 probe_radius: float = 0.6,
                 xtb_opt: bool = True,
                 solvent: Optional[str] = 'water',
                 num_processes: int = 4):
        """
        Objective class that aligns fit molecules given a reference using ESP 3D similarity,
        and scoring with ESP+Pharmacophore combined 3D similarity.
        Constructor -- sets up the reference molecule and relevant representations.

        Arguments
        ---------
        ref_molblock : str molblock of the reference molecule.
        ref_partial_charges : np.array partial charges of reference molecule
        num_points : Optional[int] if `rep` is 'shape' or 'esp' this is necessary to generate the
            surface point cloud.
        pharm_multi_vector : Optional[bool] (default=False) whether to represent pharmacophores with
        mulitple vectors (if applicable) or a single, averaged vector. `None` does not generate
            pharmacophores.
        xtb_opt : bool to optimize conformers with xtb or just use EKTG embedding and MMFF opt.
        solvent : Optional[str] (default='water') solvent to use if optimizing with xtb.
            `None` is gas phase.
        num_processes : int (default = 1) Number of proccesses to use during xtb optimization
        """
        self.ref_partial_charges = None
        self.xtb_opt = xtb_opt
        self.solvent = solvent
        self.ref_molec = None
        self.probe_radius = probe_radius

        self.buffer = {} # stores past SMILES and their similarity scores

        self.num_points = num_points
        self.alpha = ALPHA(self.num_points)
        
        self.lam = 0.3

        self.pharm_multi_vector = pharm_multi_vector

        self.num_processes = num_processes

        self.sim_score_distr_with_resample = np.array([1.])
        self.sim_score_upper_bound = 1.

        ref_mol = Chem.MolFromMolBlock(ref_molblock, removeHs=False)
        self.ref_partial_charges = np.array(ref_partial_charges)

        # Get Molecule objects -- generate surf, esp, and pharms
        self.ref_molec = Molecule(mol=ref_mol,
                                  probe_radius=self.probe_radius,
                                  num_surf_points=self.num_points,
                                  partial_charges=self.ref_partial_charges,
                                  pharm_multi_vector=self.pharm_multi_vector)

        self.sim_score_distr_with_resample = self.resampling_surf_scores()
        self.sim_score_upper_bound = max(self.sim_score_distr_with_resample)

        self.buffer[Chem.MolToSmiles(self.ref_molec.mol)] = self.sim_score_upper_bound


    def resampling_surf_scores(self) -> np.ndarray:
        """
        Capture distribution of ESP surface + pharmacophore scores caused by resampling
        surface.

        Returns
        -------
        score_distr : np.ndarray (if not relevant)
        """
        num_repeats = 50

        esp_score_distr = np.empty(num_repeats)
        for i in range(num_repeats):
            molec = Molecule(mol=self.ref_molec.mol,
                             num_surf_points=self.ref_molec.num_surf_points,
                             probe_radius=self.ref_molec.probe_radius,
                             partial_charges=self.ref_molec.partial_charges)
            esp_score_distr[i] = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos, 
                centers_2=molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=molec.surf_esp,
                alpha=self.alpha,
                lam=self.lam
            )
        
        pharm_score = get_overlap_pharm_np(self.ref_molec.pharm_types, self.ref_molec.pharm_types,
                                         self.ref_molec.pharm_ancs, self.ref_molec.pharm_ancs,
                                         self.ref_molec.pharm_vecs, self.ref_molec.pharm_vecs,
                                         'tanimoto', extended_points=False, only_extended=False)

        return esp_score_distr + pharm_score


    def _score(self,
               fit_mol: Chem.Mol,
               fit_partial_charges: np.ndarray,
               trans_init: bool = False,
               use_jax: bool = False) -> float:
        """
        Align and score a fit molecule to the reference molecule.

        Arguments
        ---------
        fit_mol : Chem.Mol fit molecule to compare to reference.
        fit_partial_charges : np.ndarray of partial charges for each atom
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False for alignment

        Returns
        -------
        float : similarity score
        """
        fit_molec = Molecule(mol=fit_mol,
                             num_surf_points=self.num_points,
                             partial_charges=np.array(fit_partial_charges),
                             pharm_multi_vector=self.pharm_multi_vector,
                             probe_radius=self.ref_molec.probe_radius)

        molec_pair = MoleculePair(self.ref_molec, fit_molec, num_surf_points=self.num_points)

        molec_pair.align_with_esp(self.alpha,
                                  lam=self.lam,
                                  trans_init=trans_init,
                                  use_jax=use_jax)
        esp_score = molec_pair.sim_aligned_esp
        se3_transform_esp = molec_pair.transform_esp

        transformed_pharm_ancs = apply_SE3_transform_np(fit_molec.pharm_ancs, se3_transform_esp)
        transformed_pharm_vecs = apply_SO3_transform_np(fit_molec.pharm_vecs, se3_transform_esp)

        pharm_score = get_overlap_pharm_np(self.ref_molec.pharm_types, fit_molec.pharm_types,
                                         self.ref_molec.pharm_ancs, transformed_pharm_ancs,
                                         self.ref_molec.pharm_vecs, transformed_pharm_vecs,
                                         'tanimoto', extended_points=False, only_extended=False)
        return esp_score, pharm_score


    def score(self,
              smiles: str,
              num_conformers: int = 5,
              trans_init: bool = False,
              use_jax: bool = False) -> Tuple[float, float]:
        """
        Align and score a fit molecule to the reference molecule.

        Arguments
        ---------
        smiles : str SMILES string of molecule to compare to reference.
        num_conformers : int (default = 1) Max number of conformers to generate and score.
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False for alignment

        Returns
        -------
        Tuple
            float : ESP similarity score (max=1.)
            float : Pharm similarity score (max=1.)
        """
        fit_partial_charges = None
        if num_conformers == 1:
            fit_mol = embed_conformer_from_smiles(smiles, attempts=50, MMFF_optimize=True)
            charge = Chem.GetFormalCharge(fit_mol)

            if self.xtb_opt:
                fit_mol, _, fit_partial_charges = optimize_conformer_with_xtb(
                    fit_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    charge=charge,
                    temp_dir=TMPDIR
                )
            if fit_partial_charges is None:
                fit_partial_charges = charges_from_single_point_conformer_with_xtb(
                    fit_mol,
                    solvent=self.solvent,
                    num_cores=self.num_processes,
                    charge=charge,
                    temp_dir=TMPDIR
                )
            esp_score, pharm_score = self._score(
                fit_mol,
                fit_partial_charges=fit_partial_charges,
                trans_init=trans_init,
                use_jax=use_jax
            )
            return esp_score, pharm_score

        else:
            # if num_conformers is more than 1
            if self.xtb_opt:
                charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))
                fit_mols, _, fit_partial_charges = generate_opt_conformers_xtb(
                    smiles, charge=charge, MMFF_optimize=True, verbose=False, num_confs=num_conformers, temp_dir=TMPDIR
                )
            else:
                fit_mols = generate_opt_conformers(smiles, MMFF_optimize=True, verbose=False, num_confs=num_conformers)
                fit_partial_charges = []
                for m in fit_mols:
                    fit_partial_charges.append(charges_from_single_point_conformer_with_xtb(
                        conformer=m, 
                        solvent=self.solvent, 
                        num_cores=self.num_processes,
                        charge=charge,
                        temp_dir=TMPDIR
                    ))

            scores = []
            for i, fit_mol in enumerate(fit_mols):
                scores.append(self._score(fit_mol,
                                        fit_partial_charges=fit_partial_charges[i],
                                        trans_init=trans_init,
                                        use_jax=use_jax))
            total_scores = [sum(s) for s in scores]
            ind_max_score = np.argmax(np.array(total_scores))
            return scores[ind_max_score]


    def score_multiple(self,
                       smiles: List[str],
                       num_conformers: int = 5,
                       trans_init: bool = False,
                       use_jax: bool = False,
                       verbose=False) -> List[float]:
        """
        Aligns and scores multiple fit SMILES.

        Arguments
        ---------
        smiles : List[str] of SMILES strings of molecules to compare to reference.
        num_conformers : int (default = 1) Max number of conformers to generate and score.
        trans_init : bool (default = False) Whether to initially translate fit COM to each ref atom
            during alignment.
        use_jax : bool (default = False) uses torch if False for alignment

        Returns
        -------
        List[float] : similarity scores. Returns a value of -1 if Objective.score fails for any reason.
        """
        scores = []
        if verbose:
            pbar = tqdm(smiles, total=len(smiles))
        else:
            pbar = smiles
        for smi in pbar:
            try:
                # Canonicalize smiles
                smi = Chem.CanonSmiles(smi)
            except:
                # if not a valid smiles skip
                scores.append(-1.)
                self.buffer[smi] = {'esp': None,
                                    'pharm': None}
                continue

            if smi in self.buffer:
                # skip if we've already computed it
                scores.append(self.buffer[smi]['esp'] + self.buffer[smi]['pharm'])
            else:
                try:
                    esp_score, pharm_score = self.score(
                        smi,
                        num_conformers=num_conformers,
                        trans_init=trans_init,
                        use_jax=use_jax
                    )
                    scores.append(esp_score + pharm_score)
                except:
                    scores.append(-1.)
                    esp_score = None
                    pharm_score = None

            self.buffer[smi] = {'esp': esp_score,
                                'pharm': pharm_score}

        return scores
