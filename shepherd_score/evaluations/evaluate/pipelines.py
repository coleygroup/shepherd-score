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

from shepherd_score.score.constants import ALPHA, LAM_SCALING

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np

from shepherd_score.evaluations.evaluate.evals import ConfEval, ConsistencyEval, ConditionalEval

RNG = np.random.default_rng()
morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=True)

TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])

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
        self.smiles = []
        self.smiles_post_opt = []
        self.molblocks = []
        self.molblocks_post_opt = []
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
        self.morgan_fps = []

        self.strain_energies_post_opt = np.empty(self.num_generated_mols)
        self.rmsds_post_opt = np.empty(self.num_generated_mols)
        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = []

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
                 verbose: bool = False
                 ):
        """
        Run the evaluation pipeline.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        verbose : bool for whether to print tqdm progress bar
        """
        if verbose:
            pbar = tqdm(enumerate(self.generated_mols), desc='Unconditional Eval',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
        for i, gen_mol in pbar:
            atoms, positions = gen_mol
            conf_eval = ConfEval(atoms=atoms, positions=positions, solvent=self.solvent, num_processes=num_processes)

            if conf_eval.morgan_fp is not None:
                self.morgan_fps.append(conf_eval.morgan_fp)
            if conf_eval.is_valid:
                self.num_valid += 1
                self.smiles.append(conf_eval.smiles)
                self.molblocks.append(conf_eval.molblock)
            if conf_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1
                self.smiles_post_opt.append(conf_eval.smiles_post_opt)
                self.molblocks_post_opt.append(conf_eval.molblock_post_opt)

            self.num_consistent_graph += 1 if conf_eval.is_graph_consistent else 0

            self.strain_energies[i] = self.get_attr(conf_eval, 'strain_energy')
            self.rmsds[i] = self.get_attr(conf_eval, 'rmsd')
            self.SA_scores[i] = self.get_attr(conf_eval, 'SA_score')
            self.QEDs[i] = self.get_attr(conf_eval, 'QED')
            self.logPs[i] = self.get_attr(conf_eval, 'logP')
            self.fsp3s[i] = self.get_attr(conf_eval, 'fsp3')

            self.SA_scores_post_opt[i] = self.get_attr(conf_eval, 'SA_score_post_opt')
            self.QEDs_post_opt[i] = self.get_attr(conf_eval, 'QED_post_opt')
            self.logPs_post_opt[i] = self.get_attr(conf_eval, 'logP_post_opt')
            self.fsp3s_post_opt[i] = self.get_attr(conf_eval, 'fsp3_post_opt')

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity()


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

        self.smiles = []
        self.smiles_post_opt = []
        self.molblocks = []
        self.molblocks_post_opt = []
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
        self.morgan_fps = []

        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = []

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
                 verbose: bool=False):
        """ 
        Run conditional evaluation on every generated molecule and store collective values.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        verbose : bool for whether to display tqdm

        Returns
        -------
        None : Just updates the class attributes
        """
        if verbose:
            pbar = tqdm(enumerate(self.generated_mols),
                        desc='Conditional Eval',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
        for i, gen_mol in pbar:
            atoms, positions = gen_mol

            cond_eval = ConditionalEval(
                ref_molec=self.ref_molec,
                atoms=atoms,
                positions=positions,
                condition=self.condition,
                num_surf_points=self.num_surf_points,
                pharm_multi_vector=self.pharm_multi_vector,
                num_processes=num_processes,
                solvent=self.solvent
            )

            # Conformer attributes
            self.num_consistent_graph += 1 if cond_eval.is_graph_consistent else 0
            self.molblocks.append(cond_eval.molblock)
            self.molblocks_post_opt.append(cond_eval.molblock_post_opt)

            if cond_eval.is_valid:
                self.num_valid += 1
                self.smiles.append(cond_eval.smiles)
            else:
                self.smiles.append(None)
            if cond_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1
                self.smiles_post_opt.append(cond_eval.smiles_post_opt)
            else:
                self.smiles_post_opt.append(None)

            self.strain_energies[i] = self.get_attr(cond_eval, 'strain_energy')
            self.rmsds[i] = self.get_attr(cond_eval, 'rmsd')
            self.SA_scores[i] = self.get_attr(cond_eval, 'SA_score')
            self.QEDs[i] = self.get_attr(cond_eval, 'QED')
            self.logPs[i] = self.get_attr(cond_eval, 'logP')
            self.fsp3s[i] = self.get_attr(cond_eval, 'fsp3')
            if cond_eval.morgan_fp is not None:
                self.graph_similarities[i] = TanimotoSimilarity(cond_eval.morgan_fp, self.ref_mol_morgan_fp)
            else:
                self.graph_similarities[i] = np.nan
            if cond_eval.morgan_fp_post_opt is not None:
                self.graph_similarities_post_opt[i] = TanimotoSimilarity(cond_eval.morgan_fp_post_opt, self.ref_mol_morgan_fp)
            else:
                self.graph_similarities_post_opt[i] = np.nan
            
            self.SA_scores_post_opt[i] = self.get_attr(cond_eval, 'SA_score_post_opt')
            self.QEDs_post_opt[i] = self.get_attr(cond_eval, 'QED_post_opt')
            self.logPs_post_opt[i] = self.get_attr(cond_eval, 'logP_post_opt')
            self.fsp3s_post_opt[i] = self.get_attr(cond_eval, 'fsp3_post_opt')

            # Conditional attributes
            self.sims_surf_target[i] = self.get_attr(cond_eval, 'sim_surf_target')
            self.sims_esp_target[i] = self.get_attr(cond_eval, 'sim_esp_target')
            self.sims_pharm_target[i] = self.get_attr(cond_eval, 'sim_pharm_target')

            self.sims_surf_target_relax[i] = self.get_attr(cond_eval, 'sim_surf_target_relax')
            self.sims_esp_target_relax[i] = self.get_attr(cond_eval, 'sim_esp_target_relax')
            self.sims_pharm_target_relax[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax')      

            self.sims_surf_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_surf_target_relax_optimal')
            self.sims_esp_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_esp_target_relax_optimal')
            self.sims_pharm_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax_optimal')

            self.sims_surf_target_relax_esp_aligned[i] = self.get_attr(cond_eval, 'sim_surf_target_relax_esp_aligned')
            self.sims_pharm_target_relax_esp_aligned[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax_esp_aligned')

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity = self.get_diversity()

    
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

        self.smiles = []
        self.smiles_post_opt = []
        self.molblocks = []
        self.molblocks_post_opt = []
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
        self.morgan_fps = []

        self.SA_scores_post_opt = np.empty(self.num_generated_mols)
        self.logPs_post_opt = np.empty(self.num_generated_mols)
        self.QEDs_post_opt = np.empty(self.num_generated_mols)
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)
        self.morgan_fps_post_opt = []

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
                 verbose: bool=False):
        """ 
        Run consistency evaluation on every generated molecule and store collective values.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        verbose : bool for whether to display tqdm

        Returns
        -------
        None : Just updates the class attributes
        """

        if verbose:
            pbar = tqdm(enumerate(self.generated_mols), desc='Consistency Eval',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
        for i, gen_mol in pbar:
            atoms, positions = gen_mol
            surf_points = self.generated_surf_points[i] if self.generated_surf_points is not None else None
            surf_esp = self.generated_surf_esp[i] if self.generated_surf_esp is not None else None
            pharm_feats = self.generated_pharm_feats[i] if self.generated_pharm_feats is not None else None
            if self.num_random_molblock_charges is not None:
                rand_ind_for_lower_bound = RNG.choice(self.num_random_molblock_charges, 1)[0]
            else:
                rand_ind_for_lower_bound = 0

            consis_eval = ConsistencyEval(
                atoms=atoms,
                positions=positions,
                surf_points=surf_points,
                surf_esp=surf_esp,
                pharm_feats=pharm_feats,
                pharm_multi_vector=self.pharm_multi_vector,
                probe_radius=self.probe_radius,
                num_processes=num_processes,
                solvent=self.solvent
            )

            # Conformer attributes
            self.num_consistent_graph += 1 if consis_eval.is_graph_consistent else 0

            self.molblocks.append(consis_eval.molblock)
            self.molblocks_post_opt.append(consis_eval.molblock_post_opt)
            self.smiles.append(consis_eval.smiles)

            if consis_eval.is_valid:
                self.num_valid += 1

                # Compute similarity score lower bounds
                if self.num_random_molblock_charges is not None:
                    rand_molblock_charges = self.random_molblock_charges[rand_ind_for_lower_bound]
                    rand_molec = Molecule(
                        mol=Chem.MolFromMolBlock(rand_molblock_charges[0], removeHs=False),
                        num_surf_points=consis_eval.molec_regen.num_surf_points,
                        partial_charges=np.array(rand_molblock_charges[1]),
                        pharm_multi_vector=consis_eval.molec_regen.pharm_multi_vector
                    )

                    mp = MoleculePair(ref_mol=consis_eval.molec_regen,
                                      fit_mol=rand_molec,
                                      num_surf_points=consis_eval.molec_regen.num_surf_points)

                    # align and compare to molec_regen
                    if consis_eval.molec_regen.surf_pos is not None:
                        mp.align_with_surf(alpha=ALPHA(mp.num_surf_points),
                                           num_repeats=50,
                                           trans_init=False,
                                           use_jax=False)
                        self.sims_surf_lower_bound[i] = mp.sim_aligned_surf
                    else:
                        self.sims_surf_lower_bound[i] = np.nan
                    if consis_eval.molec_regen.surf_esp is not None:
                        mp.align_with_esp(alpha=ALPHA(mp.num_surf_points),
                                          lam=consis_eval.lam_scaled,
                                          num_repeats=50,
                                          trans_init=False,
                                          use_jax=False)
                        self.sims_esp_lower_bound[i] = mp.sim_aligned_esp
                    else:
                        self.sims_esp_lower_bound[i] = np.nan
                    if consis_eval.molec_regen.pharm_ancs is not None:
                        mp.align_with_pharm(num_repeats=50,
                                            trans_init=False,
                                            use_jax=False)
                        self.sims_pharm_lower_bound[i] = mp.sim_aligned_pharm
                    else:
                        self.sims_pharm_lower_bound[i] = np.nan
                else:
                    self.sims_surf_lower_bound[i] = np.nan
                    self.sims_esp_lower_bound[i] = np.nan
                    self.sims_pharm_lower_bound[i] = np.nan

            if consis_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1
                self.smiles_post_opt.append(consis_eval.smiles_post_opt)

            # only compute upper bound if consistent
            if consis_eval.is_valid and consis_eval.is_valid_post_opt:
                # Upper bound
                surf_scores, esp_scores = self.resampling_upper_bounds(
                    consis_eval=consis_eval,
                    num_samples=5
                )
                if surf_scores is not None:
                    self.sims_surf_upper_bound[i] = surf_scores
                else:
                    self.sims_surf_upper_bound[i] = np.nan

                if esp_scores is not None:
                    self.sims_esp_upper_bound[i] = esp_scores
                else:
                    self.sims_esp_upper_bound[i] = np.nan
            else:
                self.sims_esp_upper_bound[i] = np.nan
                self.sims_surf_upper_bound[i] = np.nan

            self.strain_energies[i] = self.get_attr(consis_eval, 'strain_energy')
            self.rmsds[i] = self.get_attr(consis_eval, 'rmsd')
            self.SA_scores[i] = self.get_attr(consis_eval, 'SA_score')
            self.QEDs[i] = self.get_attr(consis_eval, 'QED')
            self.logPs[i] = self.get_attr(consis_eval, 'logP')
            self.fsp3s[i] = self.get_attr(consis_eval, 'fsp3')

            self.SA_scores_post_opt[i] = self.get_attr(consis_eval, 'SA_score_post_opt')
            self.QEDs_post_opt[i] = self.get_attr(consis_eval, 'QED_post_opt')
            self.logPs_post_opt[i] = self.get_attr(consis_eval, 'logP_post_opt')
            self.fsp3s_post_opt[i] = self.get_attr(consis_eval, 'fsp3_post_opt')

            # Conditional attributes
            self.sims_surf_consistent[i] = self.get_attr(consis_eval, 'sim_surf_consistent')
            self.sims_esp_consistent[i] = self.get_attr(consis_eval, 'sim_esp_consistent')
            self.sims_pharm_consistent[i] = self.get_attr(consis_eval, 'sim_pharm_consistent')

            self.sims_surf_consistent_relax[i] = self.get_attr(consis_eval, 'sim_surf_consistent_relax')
            self.sims_esp_consistent_relax[i] = self.get_attr(consis_eval, 'sim_esp_consistent_relax')
            self.sims_pharm_consistent_relax[i] = self.get_attr(consis_eval, 'sim_pharm_consistent_relax')

            self.sims_surf_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_surf_consistent_relax_optimal')
            self.sims_esp_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_esp_consistent_relax_optimal')
            self.sims_pharm_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_pharm_consistent_relax_optimal')

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity(post_opt=False)
        self.avg_graph_diversity_post_opt, self.graph_similarity_matrix_post_opt = self.get_diversity(post_opt=True)


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
        eval_surf = consis_eval.molec_post_opt.surf_pos is not None
        eval_esp = consis_eval.molec_post_opt.surf_esp is not None and consis_eval.molec_post_opt.surf_pos is not None
        if eval_surf is False and eval_esp is False:
            return None, None
        
        if num_surf_points is None:
            num_surf_points = consis_eval.num_surf_points

        # extract multiple instances of the interaction profiles
        molecs_ls = []
        for _ in range(num_samples):
            molec_extract = Molecule(
                mol=consis_eval.mol_post_opt,
                num_surf_points=num_surf_points,
                probe_radius=consis_eval.probe_radius,
                partial_charges=consis_eval.partial_charges_post_opt,
            )
            molecs_ls.append(molec_extract)

        # Score all combinations
        all_surf_scores = []
        all_esp_scores = []
        inds_all_combos = list(itertools.combinations(list(range(len(molecs_ls))), 2))

        for inds in inds_all_combos:
            molec_1 = molecs_ls[inds[0]]
            molec_2 = molecs_ls[inds[1]]

            if eval_surf:
                # surface scoring
                score = get_overlap_np(
                    centers_1=molec_1.surf_pos,
                    centers_2=molec_2.surf_pos,
                    alpha=ALPHA(num_surf_points)
                )
                all_surf_scores.append(score)
            else:
                all_surf_scores = None

            if eval_esp:
                # ESP surface scoring
                # MAKE SURE TO SCALE LAMBDA
                score = get_overlap_esp_np(
                    centers_1=molec_1.surf_pos,
                    centers_2=molec_2.surf_pos,
                    charges_1=molec_1.surf_esp,
                    charges_2=molec_2.surf_esp,
                    alpha=ALPHA(num_surf_points),
                    lam = consis_eval.lam_scaled
                )
                all_esp_scores.append(score)
            else:
                all_esp_scores = None

        upper_bound_surf = None
        upper_bound_esp = None
        if all_surf_scores is not None:
            upper_bound_surf = np.nanmean(np.array(all_surf_scores))
        if all_esp_scores is not None:
            upper_bound_esp = np.nanmean(np.array(all_esp_scores))

        return float(upper_bound_surf), float(upper_bound_esp)
            

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
            fps = self.morgan_fps
        else:
            fps = self.morgan_fps_post_opt
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
