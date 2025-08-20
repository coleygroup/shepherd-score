"""
Evaluation pipeline classes for generated molecules.
"""

import os
import logging
import traceback
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path

import itertools

import numpy as np
from rdkit import Chem

from shepherd_score.score.constants import ALPHA, LAM_SCALING

from shepherd_score.container import Molecule, MoleculePair

from shepherd_score.evaluations.evaluate.evals import ConfEval, ConsistencyEval, ConditionalEval

TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])

# Configure logging for worker processes
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _create_failed_result(i: int, error_msg: str) -> Dict[str, Any]:
    """Create a result dict for failed evaluations with all required fields."""
    return {
        'i': i,
        'is_valid': False,
        'is_valid_post_opt': False,
        'is_graph_consistent': False,
        'smiles': None,
        'smiles_post_opt': None,
        'molblock': None,
        'molblock_post_opt': None,
        'strain_energy': np.nan,
        'rmsd': np.nan,
        'SA_score': np.nan,
        'QED': np.nan,
        'logP': np.nan,
        'fsp3': np.nan,
        'SA_score_post_opt': np.nan,
        'QED_post_opt': np.nan,
        'logP_post_opt': np.nan,
        'fsp3_post_opt': np.nan,
        'error': error_msg
    }


def _create_conditional_failed_result(i: int, error_msg: str) -> Dict[str, Any]:
    """Create a result dict for failed conditional evaluations with all required fields."""
    base_failed_results = _create_failed_result(i, error_msg)
    return {
        **base_failed_results,
        # Additional Conditional 3D similarity attributes
        'sim_surf_target': np.nan,
        'sim_esp_target': np.nan,
        'sim_pharm_target': np.nan,
        'sim_surf_target_relax': np.nan,
        'sim_esp_target_relax': np.nan,
        'sim_pharm_target_relax': np.nan,
        'sim_surf_target_relax_optimal': np.nan,
        'sim_esp_target_relax_optimal': np.nan,
        'sim_pharm_target_relax_optimal': np.nan,
        'sim_surf_target_relax_esp_aligned': np.nan,
        'sim_pharm_target_relax_esp_aligned': np.nan,
    }


def _create_consistency_failed_result(i: int, error_msg: str) -> Dict[str, Any]:
    """Create a result dict for failed consistency evaluations with all required fields."""
    base_failed_results = _create_failed_result(i, error_msg)
    return {
        **base_failed_results,
        # Additional Consistency 3D similarity attributes
        'sim_surf_consistent': np.nan,
        'sim_esp_consistent': np.nan,
        'sim_pharm_consistent': np.nan,
        'sim_surf_consistent_relax': np.nan,
        'sim_esp_consistent_relax': np.nan,
        'sim_pharm_consistent_relax': np.nan,
        'sim_surf_consistent_relax_optimal': np.nan,
        'sim_esp_consistent_relax_optimal': np.nan,
        'sim_pharm_consistent_relax_optimal': np.nan,
        # Lower bound similarities
        'sim_surf_lower_bound': np.nan,
        'sim_esp_lower_bound': np.nan,
        'sim_pharm_lower_bound': np.nan,
        # Upper bound similarities
        'sim_surf_upper_bound': np.nan,
        'sim_esp_upper_bound': np.nan,
    }


def _compute_consistency_lower_bounds(
    consis_eval: ConsistencyEval,
    random_molblock_charges: Optional[List[Tuple]] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute similarity score lower bounds for consistency evaluation.

    Returns
    -------
    Tuple[float, float, float]
        sim_surf_lower_bound, sim_esp_lower_bound, sim_pharm_lower_bound
    """
    if random_molblock_charges is None or len(random_molblock_charges) == 0:
        return np.nan, np.nan, np.nan

    try:
        # Set seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Select random molecule for lower bound calculation
        rand_ind = np.random.choice(len(random_molblock_charges))
        rand_molblock_charges = random_molblock_charges[rand_ind]

        # Create random molecule for comparison
        rand_molec = Molecule(
            mol=Chem.MolFromMolBlock(rand_molblock_charges[0], removeHs=False),
            num_surf_points=consis_eval.molec_regen.num_surf_points,
            partial_charges=np.array(rand_molblock_charges[1]),
            pharm_multi_vector=consis_eval.molec_regen.pharm_multi_vector
        )

        # Create molecule pair for alignment
        mp = MoleculePair(
            ref_mol=consis_eval.molec_regen,
            fit_mol=rand_molec,
            num_surf_points=consis_eval.molec_regen.num_surf_points
        )

        sim_surf_lower_bound = np.nan
        sim_esp_lower_bound = np.nan
        sim_pharm_lower_bound = np.nan

        # Align and compare to molec_regen
        if consis_eval.molec_regen.surf_pos is not None:
            mp.align_with_surf(alpha=ALPHA(mp.num_surf_points),
                              num_repeats=50,
                              trans_init=False,
                              use_jax=False)
            sim_surf_lower_bound = mp.sim_aligned_surf
        if consis_eval.molec_regen.surf_esp is not None:
            mp.align_with_esp(alpha=ALPHA(mp.num_surf_points),
                             lam=consis_eval.lam_scaled,
                             num_repeats=50,
                             trans_init=False,
                             use_jax=False)
            sim_esp_lower_bound = mp.sim_aligned_esp
        if consis_eval.molec_regen.pharm_ancs is not None:
            mp.align_with_pharm(num_repeats=50,
                               trans_init=False,
                               use_jax=False)
            sim_pharm_lower_bound = mp.sim_aligned_pharm

        return sim_surf_lower_bound, sim_esp_lower_bound, sim_pharm_lower_bound

    except Exception as e:
        logger.warning(f"Lower bound calculation failed: {e}")
        return np.nan, np.nan, np.nan


def _compute_consistency_upper_bounds(
    consis_eval: ConsistencyEval,
    num_samples: int = 5,
    num_surf_points: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute similarity score upper bounds for consistency evaluation.

    Returns
    -------
    Tuple[float, float]
        sim_surf_upper_bound, sim_esp_upper_bound
    """
    try:
        if num_surf_points is None:
            num_surf_points = consis_eval.num_surf_points

        eval_surf = consis_eval.molec_post_opt.surf_pos is not None
        eval_esp = consis_eval.molec_post_opt.surf_esp is not None and consis_eval.molec_post_opt.surf_pos is not None
        if eval_surf is False and eval_esp is False:
            return np.nan, np.nan

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
                from shepherd_score.score.gaussian_overlap_np import get_overlap_np
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
                from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
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

        upper_bound_surf = np.nan
        upper_bound_esp = np.nan
        if all_surf_scores is not None:
            upper_bound_surf = np.nanmean(np.array(all_surf_scores))
        if all_esp_scores is not None:
            upper_bound_esp = np.nanmean(np.array(all_esp_scores))

        return float(upper_bound_surf), float(upper_bound_esp)

    except Exception as e:
        logger.warning(f"Upper bound calculation failed: {e}")
        return np.nan, np.nan


def _eval_unconditional_single(i: int,
                               atoms: np.ndarray,
                               positions: np.ndarray,
                               solvent: Optional[str],
                               num_processes: int) -> Dict[str, Any]:
    """
    Evaluate a single molecule and preserve necessary attributes for the pipeline while avoiding
    pickling issues.
    """
    try:
        # Input validation
        if atoms is None or positions is None:
            raise ValueError("atoms and positions cannot be None")
        if len(atoms) != len(positions):
            raise ValueError("atoms and positions must have same length")
        if len(atoms) == 0:
            raise ValueError("Empty molecule")
        
        conf_eval = ConfEval(atoms=atoms, positions=positions, solvent=solvent, num_processes=num_processes)
        
        res = {
            'i': i,
            'is_valid': conf_eval.is_valid,
            'is_valid_post_opt': conf_eval.is_valid_post_opt,
            'is_graph_consistent': conf_eval.is_graph_consistent,
            'smiles': conf_eval.smiles if conf_eval.is_valid else None,
            'smiles_post_opt': conf_eval.smiles_post_opt if conf_eval.is_valid_post_opt else None,
            'molblock': conf_eval.molblock if conf_eval.is_valid else None,
            'molblock_post_opt': conf_eval.molblock_post_opt if conf_eval.is_valid_post_opt else None,
            'strain_energy': conf_eval.strain_energy if conf_eval.strain_energy is not None else np.nan,
            'rmsd': conf_eval.rmsd if conf_eval.rmsd is not None else np.nan,
            'SA_score': conf_eval.SA_score if conf_eval.SA_score is not None else np.nan,
            'QED': conf_eval.QED if conf_eval.QED is not None else np.nan,
            'logP': conf_eval.logP if conf_eval.logP is not None else np.nan,
            'fsp3': conf_eval.fsp3 if conf_eval.fsp3 is not None else np.nan,
            'SA_score_post_opt': conf_eval.SA_score_post_opt if conf_eval.SA_score_post_opt is not None else np.nan,
            'QED_post_opt': conf_eval.QED_post_opt if conf_eval.QED_post_opt is not None else np.nan,
            'logP_post_opt': conf_eval.logP_post_opt if conf_eval.logP_post_opt is not None else np.nan,
            'fsp3_post_opt': conf_eval.fsp3_post_opt if conf_eval.fsp3_post_opt is not None else np.nan,
            'error': None
        }
        return res
        
    except Exception as e:
        error_msg = f"Unconditional evaluation failed for molecule {i}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return _create_failed_result(i, error_msg)


def _eval_conditional_single(i: int,
                             ref_molec: Molecule,
                             condition: str,
                             num_surf_points: int,
                             pharm_multi_vector: Optional[bool],
                             atoms: np.ndarray,
                             positions: np.ndarray,
                             solvent: Optional[str],
                             num_processes: int) -> Dict[str, Any]:
    """
    Evaluate a single molecule and preserve necessary attributes for the pipeline while avoiding
    pickling issues.
    """
    try:
        # Input validation
        if atoms is None or positions is None:
            raise ValueError("atoms and positions cannot be None")
        if len(atoms) != len(positions):
            raise ValueError("atoms and positions must have same length")
        if len(atoms) == 0:
            raise ValueError("Empty molecule")

        cond_eval = ConditionalEval(
            ref_molec=ref_molec,
            atoms=atoms,
            positions=positions,
            condition=condition,
            num_surf_points=num_surf_points,
            pharm_multi_vector=pharm_multi_vector,
            num_processes=num_processes,
            solvent=solvent
        )
        
        res = {
            'i': i,
            'is_valid': cond_eval.is_valid,
            'is_valid_post_opt': cond_eval.is_valid_post_opt,
            'is_graph_consistent': cond_eval.is_graph_consistent,
            'smiles': cond_eval.smiles if cond_eval.is_valid else None,
            'smiles_post_opt': cond_eval.smiles_post_opt if cond_eval.is_valid_post_opt else None,
            'molblock': cond_eval.molblock if cond_eval.is_valid else None,
            'molblock_post_opt': cond_eval.molblock_post_opt if cond_eval.is_valid_post_opt else None,
            'strain_energy': cond_eval.strain_energy if cond_eval.strain_energy is not None else np.nan,
            'rmsd': cond_eval.rmsd if cond_eval.rmsd is not None else np.nan,
            'SA_score': cond_eval.SA_score if cond_eval.SA_score is not None else np.nan,
            'QED': cond_eval.QED if cond_eval.QED is not None else np.nan,
            'logP': cond_eval.logP if cond_eval.logP is not None else np.nan,
            'fsp3': cond_eval.fsp3 if cond_eval.fsp3 is not None else np.nan,
            'SA_score_post_opt': cond_eval.SA_score_post_opt if cond_eval.SA_score_post_opt is not None else np.nan,
            'QED_post_opt': cond_eval.QED_post_opt if cond_eval.QED_post_opt is not None else np.nan,
            'logP_post_opt': cond_eval.logP_post_opt if cond_eval.logP_post_opt is not None else np.nan,
            'fsp3_post_opt': cond_eval.fsp3_post_opt if cond_eval.fsp3_post_opt is not None else np.nan,
            # Conditional 3D similarity attributes
            'sim_surf_target': cond_eval.sim_surf_target if cond_eval.sim_surf_target is not None else np.nan,
            'sim_esp_target': cond_eval.sim_esp_target if cond_eval.sim_esp_target is not None else np.nan,
            'sim_pharm_target': cond_eval.sim_pharm_target if cond_eval.sim_pharm_target is not None else np.nan,
            'sim_surf_target_relax': cond_eval.sim_surf_target_relax if cond_eval.sim_surf_target_relax is not None else np.nan,
            'sim_esp_target_relax': cond_eval.sim_esp_target_relax if cond_eval.sim_esp_target_relax is not None else np.nan,
            'sim_pharm_target_relax': cond_eval.sim_pharm_target_relax if cond_eval.sim_pharm_target_relax is not None else np.nan,
            'sim_surf_target_relax_optimal': cond_eval.sim_surf_target_relax_optimal if cond_eval.sim_surf_target_relax_optimal is not None else np.nan,
            'sim_esp_target_relax_optimal': cond_eval.sim_esp_target_relax_optimal if cond_eval.sim_esp_target_relax_optimal is not None else np.nan,
            'sim_pharm_target_relax_optimal': cond_eval.sim_pharm_target_relax_optimal if cond_eval.sim_pharm_target_relax_optimal is not None else np.nan,
            'sim_surf_target_relax_esp_aligned': cond_eval.sim_surf_target_relax_esp_aligned if cond_eval.sim_surf_target_relax_esp_aligned is not None else np.nan,
            'sim_pharm_target_relax_esp_aligned': cond_eval.sim_pharm_target_relax_esp_aligned if cond_eval.sim_pharm_target_relax_esp_aligned is not None else np.nan,
            'error': None
        }
        return res
        
    except Exception as e:
        error_msg = f"Conditional evaluation failed for molecule {i}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return _create_conditional_failed_result(i, error_msg)


def _eval_consistency_single(i: int,
                             atoms: np.ndarray,
                             positions: np.ndarray,
                             surf_points: Optional[np.ndarray],
                             surf_esp: Optional[np.ndarray],
                             pharm_feats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                             pharm_multi_vector: Optional[bool],
                             probe_radius: float,
                             solvent: Optional[str],
                             num_processes: int,
                             random_molblock_charges: Optional[List[Tuple]] = None,
                             random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate a single molecule for consistency and preserve necessary attributes for the pipeline while avoiding
    pickling issues.
    """
    try:
        # Input validation
        if atoms is None or positions is None:
            raise ValueError("atoms and positions cannot be None")
        if len(atoms) != len(positions):
            raise ValueError("atoms and positions must have same length")
        if len(atoms) == 0:
            raise ValueError("Empty molecule")

        consis_eval = ConsistencyEval(
            atoms=atoms,
            positions=positions,
            surf_points=surf_points,
            surf_esp=surf_esp,
            pharm_feats=pharm_feats,
            pharm_multi_vector=pharm_multi_vector,
            probe_radius=probe_radius,
            solvent=solvent,
            num_processes=num_processes
        )

        # Compute lower bounds if available
        sim_surf_lower_bound = np.nan
        sim_esp_lower_bound = np.nan
        sim_pharm_lower_bound = np.nan

        if consis_eval.is_valid and random_molblock_charges is not None:
            (sim_surf_lower_bound,
             sim_esp_lower_bound,
             sim_pharm_lower_bound) = _compute_consistency_lower_bounds(
                consis_eval, random_molblock_charges, random_seed
            )

        # Compute upper bounds if valid post-optimization
        sim_surf_upper_bound = np.nan
        sim_esp_upper_bound = np.nan

        if consis_eval.is_valid and consis_eval.is_valid_post_opt:
            sim_surf_upper_bound, sim_esp_upper_bound = _compute_consistency_upper_bounds(
                consis_eval, num_samples=5, num_surf_points=None
            )

        res = {
            'i': i,
            'is_valid': consis_eval.is_valid,
            'is_valid_post_opt': consis_eval.is_valid_post_opt,
            'is_graph_consistent': consis_eval.is_graph_consistent,
            'smiles': consis_eval.smiles if consis_eval.is_valid else None,
            'smiles_post_opt': consis_eval.smiles_post_opt if consis_eval.is_valid_post_opt else None,
            'molblock': consis_eval.molblock if consis_eval.is_valid else None,
            'molblock_post_opt': consis_eval.molblock_post_opt if consis_eval.is_valid_post_opt else None,
            'strain_energy': consis_eval.strain_energy if consis_eval.strain_energy is not None else np.nan,
            'rmsd': consis_eval.rmsd if consis_eval.rmsd is not None else np.nan,
            'SA_score': consis_eval.SA_score if consis_eval.SA_score is not None else np.nan,
            'QED': consis_eval.QED if consis_eval.QED is not None else np.nan,
            'logP': consis_eval.logP if consis_eval.logP is not None else np.nan,
            'fsp3': consis_eval.fsp3 if consis_eval.fsp3 is not None else np.nan,
            'SA_score_post_opt': consis_eval.SA_score_post_opt if consis_eval.SA_score_post_opt is not None else np.nan,
            'QED_post_opt': consis_eval.QED_post_opt if consis_eval.QED_post_opt is not None else np.nan,
            'logP_post_opt': consis_eval.logP_post_opt if consis_eval.logP_post_opt is not None else np.nan,
            'fsp3_post_opt': consis_eval.fsp3_post_opt if consis_eval.fsp3_post_opt is not None else np.nan,
            # Consistency 3D similarity attributes
            'sim_surf_consistent': consis_eval.sim_surf_consistent if consis_eval.sim_surf_consistent is not None else np.nan,
            'sim_esp_consistent': consis_eval.sim_esp_consistent if consis_eval.sim_esp_consistent is not None else np.nan,
            'sim_pharm_consistent': consis_eval.sim_pharm_consistent if consis_eval.sim_pharm_consistent is not None else np.nan,
            'sim_surf_consistent_relax': consis_eval.sim_surf_consistent_relax if consis_eval.sim_surf_consistent_relax is not None else np.nan,
            'sim_esp_consistent_relax': consis_eval.sim_esp_consistent_relax if consis_eval.sim_esp_consistent_relax is not None else np.nan,
            'sim_pharm_consistent_relax': consis_eval.sim_pharm_consistent_relax if consis_eval.sim_pharm_consistent_relax is not None else np.nan,
            'sim_surf_consistent_relax_optimal': consis_eval.sim_surf_consistent_relax_optimal if consis_eval.sim_surf_consistent_relax_optimal is not None else np.nan,
            'sim_esp_consistent_relax_optimal': consis_eval.sim_esp_consistent_relax_optimal if consis_eval.sim_esp_consistent_relax_optimal is not None else np.nan,
            'sim_pharm_consistent_relax_optimal': consis_eval.sim_pharm_consistent_relax_optimal if consis_eval.sim_pharm_consistent_relax_optimal is not None else np.nan,
            # Lower bound similarities
            'sim_surf_lower_bound': sim_surf_lower_bound,
            'sim_esp_lower_bound': sim_esp_lower_bound,
            'sim_pharm_lower_bound': sim_pharm_lower_bound,
            # Upper bound similarities
            'sim_surf_upper_bound': sim_surf_upper_bound,
            'sim_esp_upper_bound': sim_esp_upper_bound,
            'error': None
        }
        return res

    except Exception as e:
        error_msg = f"Consistency evaluation failed for molecule {i}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return _create_consistency_failed_result(i, error_msg)
