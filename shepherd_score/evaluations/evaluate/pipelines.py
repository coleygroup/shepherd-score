"""
Evaluation pipeline classes for generated molecules.
"""

import sys
import os
import logging
import traceback
from typing import Union, List, Tuple, Optional, Dict, Any, Literal
from pathlib import Path
from tqdm import tqdm
import multiprocessing
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

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np

from shepherd_score.conformer_generation import set_thread_limits

from shepherd_score.evaluations.evaluate.evals import ConfEval, ConsistencyEval, ConditionalEval
from shepherd_score.evaluations.evaluate._pipeline_eval_single import _eval_unconditional_single, _create_failed_result
from shepherd_score.evaluations.evaluate._pipeline_eval_single import _eval_conditional_single, _create_conditional_failed_result
from shepherd_score.evaluations.evaluate._pipeline_eval_single import _eval_consistency_single, _create_consistency_failed_result
from shepherd_score.evaluations.evaluate._pipeline_eval_single import _compute_consistency_upper_bounds

RNG = np.random.default_rng()
morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=True)

TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])

# Configure logging for worker processes
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _unpack_eval_unconditional_single(args):
    return _eval_unconditional_single(*args)

def _unpack_eval_conditional_single(args):
    return _eval_conditional_single(*args)

def _unpack_eval_consistency_single(args):
    return _eval_consistency_single(*args)


class UnconditionalEvalPipeline:
    """ Unconditional evaluation pipeline """

    def __init__(self,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 solvent: Optional[str] = None):
        """
        Evaluation pipeline for a list of unconditionally generated molecules.

        Arguments
        ---------
        generated_mols : List containing tuple of np.ndarrays holding: atomic numbers (N,)
            and corresponding positions (N,3)
        solvent : Optional[str] implicit solvent model to use for xtb relaxation
        """
        self.generated_mols = generated_mols
        self.smiles = [None] * len(generated_mols)
        self.smiles_post_opt = [None] * len(generated_mols)
        self.molblocks = [None] * len(generated_mols)
        self.molblocks_post_opt = [None] * len(generated_mols)
        self.num_generated_mols = len(generated_mols)

        self.solvent = solvent

        self.num_valid = 0
        self.num_valid_post_opt = 0
        self.num_consistent_graph = 0

        # Individual properties
        self.strain_energies = np.empty(self.num_generated_mols)
        self.rmsds = np.empty(self.num_generated_mols)
        self.SA_scores = np.empty(self.num_generated_mols)
        self.logPs = np.empty(self.num_generated_mols)
        self.QEDs = np.empty(self.num_generated_mols)
        self.fsp3s = np.empty(self.num_generated_mols)
        self.morgan_fps = [None] * self.num_generated_mols

        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = [None] * self.num_generated_mols

        # Overall metrics
        self.frac_valid = None
        self.frac_valid_post_opt = None
        self.frac_consistent = None
        self.frac_unique = None
        self.frac_unique_post_opt = None
        self.avg_graph_diversity = None
        self.graph_similarity_matrix = None


    def evaluate(self,
                 num_processes: int = 1,
                 num_workers: int = 1,
                 verbose: bool = False,
                 *,
                 mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                 ):
        """
        Run the evaluation pipeline.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        num_workers : int number of parallel worker processes. Constraint: num_workers*num_processes <= available CPUs
            Only recommended if `generated_mols` is > 100 due to start-up overhead of new processes.
            If num_workers > 1, multiprocessing is used, and not much is gained by setting
            num_processes > 1 in this case.
        verbose : bool for whether to print tqdm progress bar
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.
            'spawn' is recommended for most cases.

        Returns
        -------
        None : Just updates the class attributes
        """
        available_cpus = multiprocessing.cpu_count() or 1
        if num_workers < 1:
            num_workers = 1
        max_workers_allowed = max(1, available_cpus // max(1, num_processes))
        if num_workers > max_workers_allowed:
            num_workers = max_workers_allowed

        if num_workers > 1:
            # Don't need to use multiprocessing context since Open3D not used and xtb handled internally
            multiprocessing.set_start_method(mp_context, force=True)
            inputs = [(i, atoms, positions, self.solvent, 1)
                      for i, (atoms, positions) in enumerate(self.generated_mols)]
            with multiprocessing.Pool(num_workers) as pool:
                if verbose:
                    pbar = tqdm(total=self.num_generated_mols, desc='Unconditional Eval')

                results_iter = pool.imap_unordered(_unpack_eval_unconditional_single, inputs, chunksize=1)

                pending_results = {i: None for i in range(self.num_generated_mols)}
                completed = 0

                for res in results_iter:
                    idx = res['i']
                    pending_results[idx] = res
                    self._process_single_result(res, idx)
                    completed += 1
                    if verbose:
                        pbar.update(1)

                if verbose:
                    pbar.close()

                # Handle any missing results
                for i in range(self.num_generated_mols):
                    if pending_results[i] is None:
                        logger.warning(f"Missing result for molecule {i}, creating failed result")
                        failed_res = _create_failed_result(i, "Worker result missing")
                        self._process_single_result(failed_res, i)
        else:
            # Single process evaluation
            if verbose:
                pbar = tqdm(enumerate(self.generated_mols),
                            desc='Unconditional Eval',
                            total=self.num_generated_mols)
            else:
                pbar = enumerate(self.generated_mols)

            for i, gen_mol in pbar:
                atoms, positions = gen_mol

                res = _eval_unconditional_single(
                    i, atoms, positions, self.solvent, num_processes
                )
                self._process_single_result(res, i)

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity()


    def _process_single_result(self, res: Dict[str, Any], i: int):
        """Helper method to process a single evaluation result."""
        if res['is_valid']:
            self.num_valid += 1
        self.smiles[i] = res['smiles']
        self.molblocks[i] = res['molblock']

        if res['is_valid_post_opt']:
            self.num_valid_post_opt += 1
        self.smiles_post_opt[i] = res['smiles_post_opt']
        self.molblocks_post_opt[i] = res['molblock_post_opt']

        if res['molblock'] is not None:
            try:
                mol_pre = Chem.MolFromMolBlock(res['molblock'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_pre))
                self.morgan_fps[i] = fp
            except Exception:
                self.morgan_fps[i] = None

        if res['molblock_post_opt'] is not None:
            try:
                mol_post_opt = Chem.MolFromMolBlock(res['molblock_post_opt'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_post_opt))
                self.morgan_fps_post_opt[i] = fp
            except Exception:
                self.morgan_fps_post_opt[i] = None

        self.num_consistent_graph += 1 if res['is_graph_consistent'] else 0

        self.strain_energies[i] = res['strain_energy']
        self.rmsds[i] = res['rmsd']
        self.SA_scores[i] = res['SA_score']
        self.QEDs[i] = res['QED']
        self.logPs[i] = res['logP']
        self.fsp3s[i] = res['fsp3']

        self.SA_scores_post_opt[i] = res['SA_score_post_opt']
        self.QEDs_post_opt[i] = res['QED_post_opt']
        self.logPs_post_opt[i] = res['logP_post_opt']
        self.fsp3s_post_opt[i] = res['fsp3_post_opt']


    def get_attr(self, obj, attr: str):
        """ Gets an attribute of `obj` via the string name. If it is None, then return np.nan """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
    
    def get_frac_valid(self):
        """ Fraction of generated molecules that were valid. """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """ Fraction of generated molecules that were valid after relaxation. """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac

    def get_diversity(self) -> Tuple[float, np.ndarray]:
        """
        Get average molecular graph diversity (average dissimilarity) as defined by GenBench3D (arXiv:2407.04424)
        and the tanimioto similarity matrix of fingerprints.

        Returns
        -------
        tuple
            avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
            similarity_matrix : np.ndarray (N,N) similarity matrix
        """
        if self.num_consistent_graph == 0:
            return None, None
        similarity_matrix = np.zeros((self.num_consistent_graph, self.num_consistent_graph))
        running_avg_diversity_sum = 0
        for i, fp1 in enumerate(self.morgan_fps):
            for j, fp2 in enumerate(self.morgan_fps):
                if i == j:
                    similarity_matrix[i,j] = 1
                if i > j: # symmetric
                    similarity_matrix[i,j] = similarity_matrix[j,i]
                else:
                    similarity_matrix[i,j] = TanimotoSimilarity(fp1, fp2)
                    running_avg_diversity_sum += (1 - similarity_matrix[i,j])
        # from GenBench3D: arXiv:2407.04424
        avg_diversity = running_avg_diversity_sum / ((self.num_consistent_graph - 1)*self.num_consistent_graph / 2)
        return avg_diversity, similarity_matrix


    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('smiles', 'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt'):
                continue
            elif key == 'graph_similarity_matrix' or key == 'graph_similarity_matrix_post_opt':
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise


class ConditionalEvalPipeline:
    """ Evaluation pipeline for conditionally generate molecules. """

    def __init__(self,
                 ref_molec: Molecule,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 condition: str,
                 num_surf_points: int = 400,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 ):
        """
        Initialize attributes for conditional evaluation pipeline.

        Arguments
        ---------
        ref_molec : Molecule obj of reference/target molecule that was used for conditioning.
            Must contain the 3D representation that was used for conditioning (i.e., shape, ESP, or
            pharmacophores).
        generated_mols : List containing tuple of np.ndarrays holding: atomic numbers (N,)
            and corresponding positions (N,3)
        condition : str for which the molecule was conditioned on out of ('surface', 'esp',
            'pharm', 'all'). Used for alignment.
        num_surf_points : int (default = 4) Number of surface points to sample for similarity scoring.
            Must match the number of surface points in ref_molec.
        pharm_multi_vector : Optional[bool] Use multiple vectors to represent Aro/HBA/HBD or single.
            Choose whatever was used during joint generation and the settings for ref_molec should
            match.
        solvent : str solvent type for xtb relaxation
        """
        self.generated_mols = generated_mols
        self.num_generated_mols = len(self.generated_mols)
        self.solvent = solvent        

        self.pharm_multi_vector = pharm_multi_vector
        self.condition = condition
        self.num_surf_points = num_surf_points
        self.lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT
        self.lam_scaled = self.lam * LAM_SCALING # -> ONLY TO BE USED FOR get_overlap_esp*

        self.ref_molec = ref_molec
        if self.ref_molec.num_surf_points != self.num_surf_points:
            raise ValueError(
                f'The number of surface points in the reference molecule ({self.ref_molec.num_surf_points}) does not match `num_surf_points` ({self.num_surf_points}).'
            )
        self.ref_molblock = Chem.MolToMolBlock(ref_molec.mol)
        self.ref_mol_SA_score = sascorer.calculateScore(Chem.RemoveHs(self.ref_molec.mol))
        self.ref_mol_QED = QED.qed(self.ref_molec.mol)
        self.ref_mol_logP = Crippen.MolLogP(self.ref_molec.mol)
        self.ref_mol_fsp3 = Lipinski.FractionCSP3(self.ref_molec.mol)
        self.ref_mol_morgan_fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.ref_molec.mol))
        resampling_scores = self.resampling_surf_scores()
        self.ref_surf_resampling_scores = resampling_scores[0]
        self.ref_surf_esp_resampling_scores = resampling_scores[1]
        self.sims_surf_upper_bound = max(self.ref_surf_resampling_scores)
        self.sims_esp_upper_bound = max(self.ref_surf_esp_resampling_scores)

        self.smiles = [None] * self.num_generated_mols
        self.smiles_post_opt = [None] * self.num_generated_mols
        self.molblocks = [None] * self.num_generated_mols
        self.molblocks_post_opt = [None] * self.num_generated_mols
        self.num_valid = 0
        self.num_valid_post_opt = 0
        self.num_consistent_graph = 0

        # Individual properties
        self.strain_energies = np.empty(self.num_generated_mols)
        self.rmsds = np.empty(self.num_generated_mols)
        self.SA_scores = np.empty(self.num_generated_mols)
        self.logPs = np.empty(self.num_generated_mols)
        self.QEDs = np.empty(self.num_generated_mols)
        self.fsp3s = np.empty(self.num_generated_mols)
        self.morgan_fps = [None] * self.num_generated_mols

        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = [None] * self.num_generated_mols

        # Overall metrics
        self.frac_valid = None
        self.frac_valid_post_opt = None
        self.frac_consistent = None
        self.frac_unique = None
        self.frac_unique_post_opt = None
        self.avg_graph_diversity = None
        
        # 3D similarity scores
        self.sims_surf_target = np.empty(self.num_generated_mols)
        self.sims_esp_target = np.empty(self.num_generated_mols)
        self.sims_pharm_target = np.empty(self.num_generated_mols)

        self.sims_surf_target_relax = np.empty(self.num_generated_mols)
        self.sims_esp_target_relax = np.empty(self.num_generated_mols)
        self.sims_pharm_target_relax = np.empty(self.num_generated_mols)

        self.sims_surf_target_relax_optimal = np.empty(self.num_generated_mols)
        self.sims_esp_target_relax_optimal = np.empty(self.num_generated_mols)
        self.sims_pharm_target_relax_optimal = np.empty(self.num_generated_mols)

        self.sims_surf_target_relax_esp_aligned = np.empty(self.num_generated_mols)
        self.sims_pharm_target_relax_esp_aligned = np.empty(self.num_generated_mols)

        # 2D similarities
        self.graph_similarities = np.empty(self.num_generated_mols)
        self.graph_similarities_post_opt = np.empty(self.num_generated_mols)


    def evaluate(self,
                 num_processes: int = 1,
                 num_workers: int = 1,
                 verbose: bool=False,
                 *,
                 mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                 ):
        """ 
        Run conditional evaluation on every generated molecule and store collective values.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        num_workers : int number of workers to use for multiprocessing
            If num_workers > 1, multiprocessing is used, and not much is gained by setting
            num_processes > 1.
            There is an associated overhead of starting up new processes and doing score evaluations.
        verbose : bool for whether to display tqdm
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.
            'spawn' is recommended for most cases.

        Returns
        -------
        None : Just updates the class attributes
        """
        available_cpus = multiprocessing.cpu_count() or 1
        if num_workers < 1:
            num_workers = 1
        max_workers_allowed = max(1, available_cpus // max(1, num_processes))
        if num_workers > max_workers_allowed:
            num_workers = max_workers_allowed
        
        if num_workers > 1:
            multiprocessing.set_start_method(mp_context, force=True)
            with set_thread_limits(num_processes):
                inputs = [(i, self.ref_molec, self.condition, self.num_surf_points, self.pharm_multi_vector, atoms, positions, self.solvent, 1)
                        for i, (atoms, positions) in enumerate(self.generated_mols)]
                with multiprocessing.Pool(num_workers) as pool:
                    if verbose:
                        pbar = tqdm(total=self.num_generated_mols, desc='Conditional Eval')

                    # Use imap_unordered for better progress tracking
                    results_iter = pool.imap_unordered(_unpack_eval_conditional_single, inputs, chunksize=1)

                    # Create a mapping to track original indices since imap_unordered doesn't preserve order
                    pending_results = {i: None for i in range(self.num_generated_mols)}

                    for res in results_iter:
                        idx = res['i']
                        pending_results[idx] = res
                        self._process_single_result(res, idx)
                        if verbose:
                            pbar.update(1)

                    if verbose:
                        pbar.close()

                    # Handle any missing results
                    for i in range(self.num_generated_mols):
                        if pending_results[i] is None:
                            logger.warning(f"Missing result for molecule {i}, creating failed result")
                            failed_res = _create_conditional_failed_result(i, "Worker result missing")
                            self._process_single_result(failed_res, i)
        else:
            # Single process evaluation
            if verbose:
                pbar = tqdm(enumerate(self.generated_mols),
                            desc='Conditional Eval',
                            total=self.num_generated_mols)
            else:
                pbar = enumerate(self.generated_mols)

            for i, gen_mol in pbar:
                atoms, positions = gen_mol

                res = _eval_conditional_single(
                    i, self.ref_molec, self.condition, self.num_surf_points,
                    self.pharm_multi_vector, atoms, positions, self.solvent, num_processes
                )
                self._process_single_result(res, i)

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity = self.get_diversity()


    def _process_single_result(self, res: Dict[str, Any], i: int):
        """Helper method to process a single evaluation result."""
        if res['is_valid']:
            self.num_valid += 1
        self.smiles[i] = res['smiles']
        self.molblocks[i] = res['molblock']

        if res['is_valid_post_opt']:
            self.num_valid_post_opt += 1
        self.smiles_post_opt[i] = res['smiles_post_opt']
        self.molblocks_post_opt[i] = res['molblock_post_opt']

        if res['molblock'] is not None:
            try:
                mol_pre = Chem.MolFromMolBlock(res['molblock'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_pre))
                self.morgan_fps[i] = fp
                if self.ref_mol_morgan_fp is not None:
                    self.graph_similarities[i] = TanimotoSimilarity(fp, self.ref_mol_morgan_fp)
                else:
                    self.graph_similarities[i] = np.nan
            except Exception:
                self.morgan_fps[i] = None
                self.graph_similarities[i] = np.nan
        else:
            self.morgan_fps[i] = None
            self.graph_similarities[i] = np.nan

        if res['molblock_post_opt'] is not None:
            try:
                mol_post_opt = Chem.MolFromMolBlock(res['molblock_post_opt'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_post_opt))
                self.morgan_fps_post_opt[i] = fp
                if self.ref_mol_morgan_fp is not None:
                    self.graph_similarities_post_opt[i] = TanimotoSimilarity(fp, self.ref_mol_morgan_fp)
                else:
                    self.graph_similarities_post_opt[i] = np.nan
            except Exception:
                self.morgan_fps_post_opt[i] = None
                self.graph_similarities_post_opt[i] = np.nan
        else:
            self.morgan_fps_post_opt[i] = None
            self.graph_similarities_post_opt[i] = np.nan

        self.num_consistent_graph += 1 if res['is_graph_consistent'] else 0

        self.strain_energies[i] = res['strain_energy']
        self.rmsds[i] = res['rmsd']
        self.SA_scores[i] = res['SA_score']
        self.QEDs[i] = res['QED']
        self.logPs[i] = res['logP']
        self.fsp3s[i] = res['fsp3']

        self.SA_scores_post_opt[i] = res['SA_score_post_opt']
        self.QEDs_post_opt[i] = res['QED_post_opt']
        self.logPs_post_opt[i] = res['logP_post_opt']
        self.fsp3s_post_opt[i] = res['fsp3_post_opt']

        self.sims_surf_target[i] = res['sim_surf_target']
        self.sims_esp_target[i] = res['sim_esp_target']
        self.sims_pharm_target[i] = res['sim_pharm_target']
        self.sims_surf_target_relax[i] = res['sim_surf_target_relax']
        self.sims_esp_target_relax[i] = res['sim_esp_target_relax']
        self.sims_pharm_target_relax[i] = res['sim_pharm_target_relax']

        self.sims_surf_target_relax_optimal[i] = res['sim_surf_target_relax_optimal']
        self.sims_esp_target_relax_optimal[i] = res['sim_esp_target_relax_optimal']
        self.sims_pharm_target_relax_optimal[i] = res['sim_pharm_target_relax_optimal']

        self.sims_surf_target_relax_esp_aligned[i] = res['sim_surf_target_relax_esp_aligned']
        self.sims_pharm_target_relax_esp_aligned[i] = res['sim_pharm_target_relax_esp_aligned']


    def resampling_surf_scores(self) -> Union[np.ndarray, None]:
        """
        Capture distribution of surface similarity and surface ESP scores caused by resampling
        surface.

        Returns
        -------
        Tuple
            surf_scores : np.ndarray or None (if not relevant)
            esp_scores : np.ndarray or None (if not relevant)
        """
        surf_scores = np.empty(50)
        esp_scores = np.empty(50)
        for i in range(50):
            molec = Molecule(mol=self.ref_molec.mol,
                             num_surf_points=self.num_surf_points,
                             probe_radius=self.ref_molec.probe_radius,
                             partial_charges=np.array(self.ref_molec.partial_charges))
            surf_scores[i] = get_overlap_np(
                self.ref_molec.surf_pos,
                molec.surf_pos,
                alpha=ALPHA(molec.num_surf_points)
            )
            esp_scores[i] = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos, 
                centers_2=molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=molec.surf_esp,
                alpha=ALPHA(molec.num_surf_points),
                lam=self.lam_scaled
            )
            
        return surf_scores, esp_scores
            

    def get_attr(self, obj, attr: str):
        """ Gets an attribute of `obj` via the string name. If it is None, then return np.nan """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
        
    def get_frac_valid(self):
        """ Fraction of generated molecules that were valid. """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """ Fraction of generated molecules that were valid after relaxation. """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac


    def get_diversity(self) -> float:
        """
        Get average molecular graph diversity (average dissimilarity) w.r.t. target.

        Returns
        -------
        avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
        """
        avg_diversity = np.nanmean(1 - self.graph_similarities)
        return avg_diversity
    
    
    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('smiles', 'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt', 'ref_molec'):
                continue
            elif key in ('ref_surf_resampling_scores', 'ref_surf_esp_resampling_scores'):
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise


def resample_surf_scores(ref_molec: Molecule,
                         num_samples: int = 20,
                         eval_surf: bool = True,
                         eval_esp: bool = True,
                         lam_scaled: float = 0.3 * LAM_SCALING
                         ) -> Tuple[Union[np.ndarray, None]]:
    """
    Helper function to get a baseline of resampling the surface and scoring.
    """
    surf_scores = np.empty(num_samples)
    esp_scores = np.empty(num_samples)
    if eval_surf is None or ref_molec.num_surf_points is None:
        return None, None
    if eval_esp is None:
        esp_scores = None
    for i in range(num_samples):
        molec = Molecule(mol=ref_molec.mol,
                         num_surf_points=ref_molec.num_surf_points,
                         probe_radius=ref_molec.probe_radius,
                         partial_charges=np.array(ref_molec.partial_charges))
        surf_scores[i] = get_overlap_np(ref_molec.surf_pos,
                                        molec.surf_pos,
                                        alpha=ALPHA(molec.num_surf_points))
        if eval_esp:
            esp_scores[i] = get_overlap_esp_np(centers_1=ref_molec.surf_pos, 
                                               centers_2=molec.surf_pos,
                                               charges_1=ref_molec.surf_esp,
                                               charges_2=molec.surf_esp,
                                               alpha=ALPHA(molec.num_surf_points),
                                               lam=lam_scaled)
    return surf_scores, esp_scores


class ConsistencyEvalPipeline:
    """ Evaluation pipeline for unconditionally generated molecules with consistency check. """

    def __init__(self,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 generated_surf_points: Optional[List[np.ndarray]] = None,
                 generated_surf_esp: Optional[List[np.ndarray]] = None,
                 generated_pharm_feats: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
                 probe_radius: float = 1.2,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 random_molblock_charges: Optional[List[Tuple]] = None
                 ):
        """
        Initialize attributes for consistency evaluation pipeline.

        Arguments
        ---------
        generated_mols : List containing tuple of np.ndarrays holding: atomic numbers (N,)
            and corresponding positions (N,3)

        Jointly generated features subject for evaluation.

        generated_surf_points : Optional List[np.ndarray (M, 3)] List containing all surface
            point clouds.
        generated_surf_esp : Optional List[np.ndarray (M,)] List containing corresponding ESP
            values of the generated_surf_points.
        generated_pharm_feats : Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] containing:
            generated_pharm_types : np.ndarray (P,) containing pharmacophore types as ints.
            generated_pharm_ancs : np.ndarray (P, 3) containing pharm anchor coordinates.
            generated_pharm_vecs : np.ndarray (P, 3) containing pharm vectors relative unit vecs.
        probe_radius : float (default = 1.2) Probe radius used for solvent accessible surface
        pharm_multi_vector : Optional[bool] Use multiple vectors to represent Aro/HBA/HBD or single
            if `generated_pharm_feats` is used.
            Choose whatever was used during joint generation and the settings for ref_molec should
            match.
        solvent : str solvent type for xtb relaxation
        random_molblock_charges : Optional[List[Tuple]] Contains molblock_charges to randomly
            select from, and align with (re-)generated sample.
        """
        self.generated_mols = generated_mols
        self.num_generated_mols = len(self.generated_mols)
        self.solvent = solvent
        self.probe_radius = probe_radius
        self.random_molblock_charges = random_molblock_charges
        if self.random_molblock_charges is not None:
            self.num_random_molblock_charges = len(self.random_molblock_charges)
        else:
            self.num_random_molblock_charges = None

        # Check that the lengths are the same
        if generated_surf_points is not None:
            assert self.num_generated_mols == len(generated_surf_points)
        self.generated_surf_points = generated_surf_points
        if generated_surf_esp is not None:
            assert self.num_generated_mols == len(generated_surf_esp)
        self.generated_surf_esp = generated_surf_esp
        if self.generated_surf_esp is not None and self.generated_surf_points is None:
            raise ValueError(f'`generated_surf_pos` must also be provided if `generated_surf_esp` is given.')

        if generated_pharm_feats is not None: # unpack
            self.generated_pharm_feats = generated_pharm_feats
        else:
            self.generated_pharm_feats = None

        self.pharm_multi_vector = pharm_multi_vector

        self.smiles = [None] * self.num_generated_mols
        self.smiles_post_opt = [None] * self.num_generated_mols
        self.molblocks = [None] * self.num_generated_mols
        self.molblocks_post_opt = [None] * self.num_generated_mols
        self.num_valid = 0
        self.num_valid_post_opt = 0
        self.num_consistent_graph = 0

        # Individual properties
        self.strain_energies = np.empty(self.num_generated_mols)
        self.rmsds = np.empty(self.num_generated_mols)
        self.SA_scores = np.empty(self.num_generated_mols)
        self.logPs = np.empty(self.num_generated_mols)
        self.QEDs = np.empty(self.num_generated_mols)
        self.fsp3s = np.empty(self.num_generated_mols)
        self.morgan_fps = [None] * self.num_generated_mols

        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = [None] * self.num_generated_mols

        # Overall metrics
        self.frac_valid = None
        self.frac_valid_post_opt = None
        self.frac_consistent = None
        self.frac_unique = None
        self.frac_unique_post_opt = None
        self.avg_graph_diversity = None
        self.graph_similarity_matrix = None
        self.avg_graph_diversity_post_opt = None
        self.graph_similarity_matrix_post_opt = None
        
        # 3D similarity scores
        self.sims_surf_consistent = np.empty(self.num_generated_mols)
        self.sims_esp_consistent = np.empty(self.num_generated_mols)
        self.sims_pharm_consistent = np.empty(self.num_generated_mols)

        self.sims_surf_upper_bound = np.empty(self.num_generated_mols)
        self.sims_esp_upper_bound = np.empty(self.num_generated_mols)

        self.sims_surf_lower_bound = np.empty(self.num_generated_mols)
        self.sims_esp_lower_bound = np.empty(self.num_generated_mols)
        self.sims_pharm_lower_bound = np.empty(self.num_generated_mols)

        self.sims_surf_consistent_relax = np.empty(self.num_generated_mols)
        self.sims_esp_consistent_relax = np.empty(self.num_generated_mols)
        self.sims_pharm_consistent_relax = np.empty(self.num_generated_mols)

        self.sims_surf_consistent_relax_optimal = np.empty(self.num_generated_mols)
        self.sims_esp_consistent_relax_optimal = np.empty(self.num_generated_mols)
        self.sims_pharm_consistent_relax_optimal = np.empty(self.num_generated_mols)


    def evaluate(self,
                 num_processes: int = 1,
                 num_workers: int = 1,
                 verbose: bool = False,
                 *,
                 mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                 ):
        """
        Run consistency evaluation on every generated molecule and store collective values.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        num_workers : int number of workers to use for multiprocessing
            If num_workers > 1, multiprocessing is used, and not much is gained by setting
            num_processes > 1 in this case.
            There is an associated overhead of starting up new processes and doing score evaluations.
        verbose : bool for whether to display tqdm
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.
            'spawn' is recommended for most cases.

        Returns
        -------
        None : Just updates the class attributes
        """
        available_cpus = multiprocessing.cpu_count() or 1
        if num_workers < 1:
            num_workers = 1
        max_workers_allowed = max(1, available_cpus // max(1, num_processes))
        if num_workers > max_workers_allowed:
            num_workers = max_workers_allowed

        if num_workers > 1:
            multiprocessing.set_start_method(mp_context, force=True)
            with set_thread_limits(num_processes):
                inputs = [(i, atoms, positions,
                        self.generated_surf_points[i] if self.generated_surf_points is not None else None,
                        self.generated_surf_esp[i] if self.generated_surf_esp is not None else None,
                        self.generated_pharm_feats[i] if self.generated_pharm_feats is not None else None,
                        self.pharm_multi_vector, self.probe_radius, self.solvent, 1,
                        self.random_molblock_charges, i)  # Use i as random seed for reproducibility
                        for i, (atoms, positions) in enumerate(self.generated_mols)]
                with multiprocessing.Pool(num_workers) as pool:
                    if verbose:
                        pbar = tqdm(total=self.num_generated_mols, desc='Consistency Eval')

                    results_iter = pool.imap_unordered(_unpack_eval_consistency_single, inputs, chunksize=1)

                    pending_results = {i: None for i in range(self.num_generated_mols)}

                    for res in results_iter:
                        idx = res['i']
                        pending_results[idx] = res
                        self._process_single_result(res, idx)
                        if verbose:
                            pbar.update(1)

                    if verbose:
                        pbar.close()

                    # Handle any missing results
                    for i in range(self.num_generated_mols):
                        if pending_results[i] is None:
                            logger.warning(f"Missing result for molecule {i}, creating failed result")
                            failed_res = _create_consistency_failed_result(i, "Worker result missing")
                            self._process_single_result(failed_res, i)
        else:
            # Single process evaluation
            if verbose:
                pbar = tqdm(enumerate(self.generated_mols),
                            desc='Consistency Eval',
                            total=self.num_generated_mols)
            else:
                pbar = enumerate(self.generated_mols)

            for i, gen_mol in pbar:
                atoms, positions = gen_mol
                surf_points = self.generated_surf_points[i] if self.generated_surf_points is not None else None
                surf_esp = self.generated_surf_esp[i] if self.generated_surf_esp is not None else None
                pharm_feats = self.generated_pharm_feats[i] if self.generated_pharm_feats is not None else None

                res = _eval_consistency_single(
                    i, atoms, positions, surf_points, surf_esp, pharm_feats,
                    self.pharm_multi_vector, self.probe_radius, self.solvent, num_processes,
                    self.random_molblock_charges, i  # Use i as random seed for reproducibility
                )
                self._process_single_result(res, i)

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity(post_opt=False)
        self.avg_graph_diversity_post_opt, self.graph_similarity_matrix_post_opt = self.get_diversity(post_opt=True)


    def _process_single_result(self, res: Dict[str, Any], i: int):
        """Helper method to process a single evaluation result."""
        if res['is_valid']:
            self.num_valid += 1
        self.smiles[i] = res['smiles']
        self.molblocks[i] = res['molblock']

        if res['is_valid_post_opt']:
            self.num_valid_post_opt += 1
        self.smiles_post_opt[i] = res['smiles_post_opt']
        self.molblocks_post_opt[i] = res['molblock_post_opt']

        if res['molblock'] is not None:
            try:
                mol_pre = Chem.MolFromMolBlock(res['molblock'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_pre))
                self.morgan_fps[i] = fp
            except Exception:
                self.morgan_fps[i] = None

        if res['molblock_post_opt'] is not None:
            try:
                mol_post_opt = Chem.MolFromMolBlock(res['molblock_post_opt'], removeHs=False)
                fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(mol_post_opt))
                self.morgan_fps_post_opt[i] = fp
            except Exception:
                self.morgan_fps_post_opt[i] = None

        self.num_consistent_graph += 1 if res['is_graph_consistent'] else 0

        self.strain_energies[i] = res['strain_energy']
        self.rmsds[i] = res['rmsd']
        self.SA_scores[i] = res['SA_score']
        self.QEDs[i] = res['QED']
        self.logPs[i] = res['logP']
        self.fsp3s[i] = res['fsp3']

        self.SA_scores_post_opt[i] = res['SA_score_post_opt']
        self.QEDs_post_opt[i] = res['QED_post_opt']
        self.logPs_post_opt[i] = res['logP_post_opt']
        self.fsp3s_post_opt[i] = res['fsp3_post_opt']

        # Consistency 3D similarity attributes
        self.sims_surf_consistent[i] = res['sim_surf_consistent']
        self.sims_esp_consistent[i] = res['sim_esp_consistent']
        self.sims_pharm_consistent[i] = res['sim_pharm_consistent']

        self.sims_surf_consistent_relax[i] = res['sim_surf_consistent_relax']
        self.sims_esp_consistent_relax[i] = res['sim_esp_consistent_relax']
        self.sims_pharm_consistent_relax[i] = res['sim_pharm_consistent_relax']

        self.sims_surf_consistent_relax_optimal[i] = res['sim_surf_consistent_relax_optimal']
        self.sims_esp_consistent_relax_optimal[i] = res['sim_esp_consistent_relax_optimal']
        self.sims_pharm_consistent_relax_optimal[i] = res['sim_pharm_consistent_relax_optimal']

        # Lower bound similarities
        self.sims_surf_lower_bound[i] = res['sim_surf_lower_bound']
        self.sims_esp_lower_bound[i] = res['sim_esp_lower_bound']
        self.sims_pharm_lower_bound[i] = res['sim_pharm_lower_bound']

        # Upper bound similarities
        self.sims_surf_upper_bound[i] = res['sim_surf_upper_bound']
        self.sims_esp_upper_bound[i] = res['sim_esp_upper_bound']


    def resampling_surf_scores(self,
                               consis_eval: ConsistencyEval,
                               num_samples: int = 20) -> Tuple[Union[np.ndarray, None]]:
        """
        Capture distribution of surface similarity and surface ESP scores caused by resampling
        surface.
        
        Arguments
        ---------
        consis_eval : ConsistencyEval obj to check similarity scores caused by resampling
        num_samples : int (default = 20) number of times to resample surface and score

        Returns
        -------
        Tuple
            surf_scores : np.ndarray or None (if not relevant)
            esp_scores : np.ndarray or None (if not relevant)
        """
        ref_molec = consis_eval.molec
        surf_scores, esp_scores = resample_surf_scores(
            ref_molec=ref_molec,
            num_samples=num_samples,
            eval_surf=consis_eval.molec.surf_pos is not None,
            eval_esp=consis_eval.molec.surf_esp is not None,
            lam_scaled=consis_eval.lam_scaled
        )            
        return surf_scores, esp_scores

    
    @staticmethod
    def resampling_upper_bounds(consis_eval: ConsistencyEval,
                                num_samples: int = 5,
                                num_surf_points: Optional[int] = None
                                ) -> Tuple[Union[float, None]]:
        """
        Compute the expectation (upper bound) of similarity score caused by stochastic surface
        sampling by calculating the mean similarity between pairwise comparisons.

        Arguments
        ---------
        consis_eval : ConsistencyEval
        num_samples = 5

        Returns
        -------
        Tuple
            upper_bound_surf : float or None surface similarity upper bound
            upper_bound_esp : float or None ESP similarity upper bound
        """
        return _compute_consistency_upper_bounds(
            consis_eval, num_samples, num_surf_points
        )
            

    def get_attr(self, obj, attr: str):
        """ Gets an attribute of `obj` via the string name. If it is None, then return np.nan """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
        
    def get_frac_valid(self):
        """ Fraction of generated molecules that were valid. """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """ Fraction of generated molecules that were valid after relaxation. """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac


    def get_diversity(self, post_opt=False) -> Tuple[float, np.ndarray]:
        """
        Get average molecular graph diversity (average dissimilarity) as defined by GenBench3D (arXiv:2407.04424)
        and the tanimioto similarity matrix of fingerprints.

        Returns
        -------
        tuple
            avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
            similarity_matrix : np.ndarray (N,N) similarity matrix
        """
        if self.num_consistent_graph == 0:
            return None, None
        if post_opt:
            fps = self.morgan_fps_post_opt
        else:
            fps = self.morgan_fps
        similarity_matrix = np.zeros((self.num_consistent_graph, self.num_consistent_graph))
        running_avg_diversity_sum = 0
        for i, fp1 in enumerate(fps):
            for j, fp2 in enumerate(fps):
                if i == j:
                    similarity_matrix[i,j] = 1
                if i > j: # symmetric
                    similarity_matrix[i,j] = similarity_matrix[j,i]
                else:
                    similarity_matrix[i,j] = TanimotoSimilarity(fp1, fp2)
                    running_avg_diversity_sum += (1 - similarity_matrix[i,j])
        # from GenBench3D: arXiv:2407.04424
        avg_diversity = running_avg_diversity_sum / ((self.num_consistent_graph - 1)*self.num_consistent_graph / 2)
        return avg_diversity, similarity_matrix
    

    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('random_molblock_charges', 'num_random_molblock_charges', 'smiles',
                       'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt'):
                continue
            elif key == 'graph_similarity_matrix' or key == 'graph_similarity_matrix_post_opt':
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise
