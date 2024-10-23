"""
Evaluation using UnconditionalEvalPipeline / ConsistencyEvalPipeline for training set structures.
"""

import os
from typing import List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import pickle
import argparse

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdDetermineBonds

from shepherd_score.score.constants import ALPHA

from shepherd_score.conformer_generation import embed_conformer_from_smiles, single_point_xtb_from_xyz
from shepherd_score.container import Molecule

from shepherd_score.evaluations.evaluate import ConsistencyEvalPipeline, resample_surf_scores, get_mol_from_atom_pos


if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])
else:
    TMPDIR = Path('./')


def run_consistency_evaluation(load_dir: str,
                               condition: str,
                               solvent: Optional[str] = None,
                               training_molblock_charges=Union[List[Tuple], None],
                               num_processes: int = 4,
                               probe_radius: float = 0.6,
                               ) -> None:
    """
    Generate a benchmarking set of molecules -- both MMFF and its corresponding xtb relaxed
    structures.

    Arguments
    ---------
    load_dir : path to dir containing "samples.pickle"
    condition : str (x1x2, x1x3, x1x4)
    """
    load_dir_path = Path(load_dir)
    if not load_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {load_dir_path}')

    save_file_dir = load_dir_path / f'consis_evals'
    save_file_dir.mkdir(parents=True, exist_ok=True)

    if condition in ('x1x2', 'x1x3', 'x1x4'):
        load_file_path = load_dir_path / f'samples.pickle'
        save_file_path = save_file_dir / 'consis_eval'
    else:
        load_file_path = load_dir_path / 'samples_noisy.pickle'
        save_file_path = save_file_dir / 'consis_eval'
    print(f'Saving to {save_file_path}')
    
    with open(load_file_path, 'rb') as f:
        samples = pickle.load(f)
    print(f'Samples loaded from {load_file_path}')
    

    ls_atoms_pos = []
    if condition == 'x1x2' or condition == 'x1x3' or condition == 'x1x3x4':
        ls_surf_points = []
    if condition == 'x1x3' or condition == 'x1x3x4':
        ls_surf_esp = []
    if condition == 'x1x4' or condition == 'x1x3x4':
        ls_pharm_feats = []

    for i in range(len(samples)):
        ls_atoms_pos.append(
            (samples[i]['x1']['atoms'], samples[i]['x1']['positions'])
        )
        if condition == 'x1x2':
            ls_surf_points.append(
                samples[i]['x2']['positions']
            )
        if condition == 'x1x3' or condition == 'x1x3x4':
            ls_surf_points.append(
                samples[i]['x3']['positions']
            )
            ls_surf_esp.append(
                samples[i]['x3']['charges']
            )
        if condition == 'x1x4' or condition == 'x1x3x4':
            ls_pharm_feats.append(
                (samples[i]['x4']['types'], samples[i]['x4']['positions'], samples[i]['x4']['directions'])
            )
    
    if condition == 'x1x3x4':
        assert isinstance(ls_surf_points, list) and isinstance(ls_surf_esp, list) and isinstance(ls_pharm_feats, list)
    elif condition == 'x1x2':
        assert isinstance(ls_surf_points, list)
        ls_surf_esp = None
        ls_pharm_feats = None
    elif condition == 'x1x3':
        assert isinstance(ls_surf_points, list) and isinstance(ls_surf_esp, list)
        ls_pharm_feats = None
    elif condition == 'x1x4':
        assert isinstance(ls_pharm_feats, list)
        ls_surf_points = None
        ls_surf_esp = None

    print(f'Initializing ConsistencyEvalPipeline for {condition}.')

    # Initialize evaluation
    consis_pipeline = ConsistencyEvalPipeline(
        generated_mols=ls_atoms_pos,
        generated_surf_points=ls_surf_points,
        generated_surf_esp=ls_surf_esp,
        generated_pharm_feats=ls_pharm_feats,
        probe_radius=probe_radius,
        pharm_multi_vector=False,
        solvent=solvent,
        random_molblock_charges=training_molblock_charges
    )

    # Run evaluation
    print('Running evaluation...')
    consis_pipeline.evaluate(num_processes=num_processes, verbose=True)
    print('Finished evaluation.')

    # Save values to numpy object
    save_global_series, save_local_df = consis_pipeline.to_pandas()

    save_file_global = save_file_dir / f'consis_eval_global.pkl'
    save_global_series.to_pickle(save_file_global)
    save_file_local = save_file_dir / f'consis_eval_local.pkl'
    save_local_df.to_pickle(save_file_local)

    print(f'Finished consistency evaluation!\nSaved global attributes to: {save_file_global}')
    print(f'Saved local attributes to: {save_file_local}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path to directory to save files to.')
    parser.add_argument('--training-data', type=str, required=True,
                        help="Path to GDB training data file to randomly sample and compare scoring functions")
    parser.add_argument('--task-id', type=str, help='Task ID.')
    args = parser.parse_args()
    print(args)

    load_dir = str(args.load_dir)
    my_task_id = int(args.task_id) if args.task_id != '' else None
    training_data_file = Path(str(args.training_data))
    assert training_data_file.is_file()

    # settings for ShEPhERD generated molecules
    num_surf_points = 75 # number used to sample
    probe_radius = 0.6
    if 'moses' in load_dir:
        solvent = 'water'
        model = 'x1x3x4'
        num_processes = 16
        print('Doing Moses - aq')
    else:
        solvent = None # evaluating on GDB unconditional
        print('Doing GDB-17')

        my_task_id = int(args.task_id)

        model_types = ('x1x2', 'x1x3', 'x1x4')
        model = model_types[my_task_id]
        num_processes = 16

    load_dir_path = Path(load_dir) / model
    assert load_dir_path.is_dir()

    if 'moses' in load_dir:
        training_molblock_charges = None
    else:
        #load training data for random alignments / evaluations
        with open(training_data_file, 'rb') as f:
            training_molblock_charges = pickle.load(f)
    
    run_consistency_evaluation(
        load_dir=load_dir_path,
        condition=model,
        solvent=solvent,
        training_molblock_charges=training_molblock_charges,
        num_processes=num_processes,
        probe_radius=probe_radius
    )
