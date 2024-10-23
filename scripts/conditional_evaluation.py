"""
Script to run conditional evaluation of generated molecules.
"""

import os
from typing import Optional
from pathlib import Path
import pickle
import argparse
import open3d

import numpy as np
import pandas as pd
from rdkit import Chem

from shepherd_score.score.constants import COULOMB_SCALING
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.generate_point_cloud import get_atomic_vdw_radii, get_molecular_surface
from shepherd_score.generate_point_cloud import get_electrostatics_given_point_charges
from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb

from shepherd_score.evaluations.evaluate import ConditionalEvalPipeline

if 'TMPDIR' in os.environ:
    TMPDIR = os.environ['TMPDIR']
else:
    TMPDIR = './'


def run_conditional_eval(sample_id,
                         job_id,
                         num_tasks,
                         load_file_path,
                         solvent: Optional[str] = None,
                         num_processes: int = 1,
                         probe_radius: float = 0.6
                         ) -> None:
    """
    Run conditional evaluation and save. Split it jobs by the samples file and the job id.
    """
    load_file_path_ = Path(load_file_path)
    if not load_file_path_.is_file():
        raise ValueError(f'Provided path is not a file: {load_file_path_}')

    load_dir_path = load_file_path_.parent
    if not load_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {load_dir_path}')
    save_file_dir = load_dir_path / f'cond_eval_sample_{sample_id}'
    save_file_dir.mkdir(parents=True, exist_ok=True)
    
    with open(load_file_path_, 'rb') as f:
        samples = pickle.load(f)
        
    ref_mol = Chem.MolFromMolBlock(samples[0], removeHs=False)
    ref_partial_charges = samples[1]
    surface_points = samples[2]
    electrostatics = samples[3]
    pharm_types = samples[4]
    pharm_ancs = samples[5]
    pharm_vecs = samples[6]
    ref_molec = Molecule(ref_mol,
                         probe_radius=probe_radius,
                         partial_charges=np.array(ref_partial_charges),
                         num_surf_points=400,
                         pharm_multi_vector=False,
                         pharm_types=pharm_types,
                         pharm_ancs=pharm_ancs,
                         pharm_vecs=pharm_vecs)

    generated_mols = [(samples[-1][i]['x1']['atoms'], samples[-1][i]['x1']['positions']) for i in range(len(samples[-1]))]

    subselected_gen_mols = generated_mols[job_id:len(generated_mols):num_tasks]
    
    print(f'Starting Conditional Eval Pipeline on sample {sample_id} and job {job_id}')
    cond_pipe = ConditionalEvalPipeline(
        ref_molec=ref_molec,
        generated_mols=subselected_gen_mols,
        condition='all',
        num_surf_points=400,
        pharm_multi_vector=False,
        solvent=solvent
    )

    # Run evaluation
    cond_pipe.evaluate(num_processes=num_processes, verbose=True)

    save_global_series, save_local_df = cond_pipe.to_pandas()

    save_file_global = save_file_dir / f'cond_eval_global_{job_id}.pkl'
    save_global_series.to_pickle(save_file_global)
    save_file_local = save_file_dir / f'cond_eval_local_{job_id}.pkl'
    save_local_df.to_pickle(save_file_local)

    print(f'Finished {job_id} evaluation!\nSaved global attributes to: {save_file_global}')
    print(f'Saved local attributes to: {save_file_local}')


def run_conditional_eval_by_sample_only(
        sample_id,
        load_dir,
        conditioning_type: str,
        solvent: Optional[str] = None,
        num_processes: int = 1,
        probe_radius: float = 0.6
        ) -> None:
    """
    Run conditional evaluation and save. Do not split up each samples file by job id.
    """
    load_dir_path = Path(load_dir) / conditioning_type
    if not load_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {load_dir_path}')

    if conditioning_type == 'x2':
        condition = 'surf'
    elif conditioning_type == 'x3':
        condition = 'esp'
    elif conditioning_type == 'x4':
        condition = 'pharm'
    
    save_file_dir = load_dir_path / f'cond_evals'
    save_file_dir.mkdir(parents=True, exist_ok=True)
    
    load_file_path = load_dir_path / f'samples_{sample_id}.pickle'
    with open(load_file_path, 'rb') as f:
        samples = pickle.load(f)
        
    ref_mol = Chem.MolFromMolBlock(samples[0], removeHs=False)
    ref_partial_charges = samples[1]
    surface_points = samples[2]
    electrostatics = samples[3]
    pharm_types = samples[4]
    pharm_ancs = samples[5]
    pharm_vecs = samples[6]
    ref_molec = Molecule(ref_mol,
                         probe_radius=probe_radius,
                         partial_charges=np.array(ref_partial_charges),
                         num_surf_points=400,
                         pharm_multi_vector=False,
                         pharm_types=pharm_types,
                         pharm_ancs=pharm_ancs,
                         pharm_vecs=pharm_vecs)

    generated_mols = [(samples[-1][i]['x1']['atoms'], samples[-1][i]['x1']['positions']) for i in range(len(samples[-1]))]
    
    print(f'Starting Conditional Eval Pipeline on sample {sample_id}.')
    cond_pipe = ConditionalEvalPipeline(
        ref_molec=ref_molec,
        generated_mols=generated_mols,
        condition=condition,
        num_surf_points=400,
        pharm_multi_vector=False,
        solvent=solvent
    )

    # Run evaluation
    cond_pipe.evaluate(num_processes=num_processes, verbose=True)
    
    save_global_series, save_local_df = cond_pipe.to_pandas()

    save_file_global = save_file_dir / f'cond_eval_global.pkl'
    save_global_series.to_pickle(save_file_global)
    save_file_local = save_file_dir / f'cond_eval_local.pkl'
    save_local_df.to_pickle(save_file_local)

    print(f'Finished evaluation!\nSaved global attributes to: {save_file_global}')
    print(f'Saved local attributes to: {save_file_local}')


def run_conditional_eval_frag(job_id,
                              num_tasks,
                              load_file,
                              solvent: Optional[str] = None,
                              num_processes: int = 1,
                              probe_radius: float = 0.6
                              ) -> None:
    """
    Run conditional evaluation and save. Split it jobs by the samples file and the job id.
    """
    print('Starting Frag')
    load_file_path = Path(load_file)
    load_dir_path = load_file_path.parent
    if not load_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {load_dir_path}')
    save_file_dir = load_dir_path / 'cond_evals'
    save_file_dir.mkdir(parents=True, exist_ok=True)
    
    with open(load_file_path, 'rb') as f:
        samples = pickle.load(f)

    # Create reference mol from all fragments
    mols = [Chem.MolFromMolBlock(s, removeHs=False) for s in samples[0]]
    centers = []
    radii = []
    for mol in mols:
        centers.append(mol.GetConformer().GetPositions())
        radii.append(get_atomic_vdw_radii(mol))
    centers_comb = np.concatenate(centers)
    radii_comb = np.concatenate(radii)
    surface_points = get_molecular_surface(centers_comb, radii_comb, num_points=400, probe_radius=probe_radius)

    partial_charges = []
    esps = []
    for i, mol in enumerate(mols):
        charges = charges_from_single_point_conformer_with_xtb(mol, solvent='water', num_cores=1, temp_dir=TMPDIR)
        partial_charges.append(charges)
        esps.append(get_electrostatics_given_point_charges(charges, centers[i], surface_points))
    avg_esp = np.stack(esps).mean(axis=0)

    print('Finished generating the merged reference Molecule object.')
    
    # just choose the first molblock, it doesn't affect anything once you make the Molecule object
    ref_mol = mols[0]
    ref_partial_charges = partial_charges[0]
    # surface_points = samples[2] ignore since len=75
    # electrostatics = samples[3] ignore since len=75
    pharm_types = samples[3]
    pharm_ancs = samples[4]
    pharm_vecs = samples[5]
    ref_molec = Molecule(ref_mol,
                         probe_radius=probe_radius,
                         partial_charges=np.array(ref_partial_charges),
                        #  num_surf_points=400,
                         surface_points=surface_points, # We generate the surface
                         electrostatics=avg_esp, # We generate the esp
                         pharm_multi_vector=False,
                         pharm_types=pharm_types,
                         pharm_ancs=pharm_ancs,
                         pharm_vecs=pharm_vecs)

    generated_mols = [(samples[-1][i]['x1']['atoms'], samples[-1][i]['x1']['positions']) for i in range(len(samples[-1]))]

    subselected_gen_mols = generated_mols[job_id:len(generated_mols):num_tasks]
    
    print(f'Starting Conditional Eval Pipeline on Fragment mergining and job {job_id}')
    cond_pipe = ConditionalEvalPipeline(
        ref_molec=ref_molec,
        generated_mols=subselected_gen_mols,
        condition='all',
        num_surf_points=400,
        pharm_multi_vector=False,
        solvent=solvent
    )

    # Run evaluation
    cond_pipe.evaluate(num_processes=num_processes, verbose=True)

    save_global_series, save_local_df = cond_pipe.to_pandas()

    save_file_global = save_file_dir / f'cond_eval_global_{job_id}.pkl'
    save_global_series.to_pickle(save_file_global)
    save_file_local = save_file_dir / f'cond_eval_local_{job_id}.pkl'
    save_local_df.to_pickle(save_file_local)

    print(f'Finished {job_id} evaluation!\nSaved global attributes to: {save_file_global}')
    print(f'Saved local attributes to: {save_file_local}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-file-path', type=str, help='Path file with generated samples OR if GDB conditional samples the path to the directory.')
    parser.add_argument('--task-id', type=int, help='Task ID.')
    parser.add_argument('--num-tasks', type=int, help='Number of tasks.')
    parser.add_argument('--sample-id', type=int, required=True, help='Index used to load the file. [0,4]')
    parser.add_argument('--task', type=str, required=True, help='Choose from "NP", "frag", or "GDB"')
    args = parser.parse_args()
    print(args)

    file_path = Path(args.load_file_path)
    if not file_path.is_file():
        raise ValueError('Provided --load-file-path is not a directory.')
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)
    sample_id = int(args.sample_id)
    task = str(args.task)

    print(f'Loading from {file_path}')
    if 'NP' in task:
        run_conditional_eval(
            sample_id=sample_id,
            job_id=my_task_id,
            num_tasks=num_tasks,
            load_file_path=file_path,
            solvent='water',
            num_processes=4,
            probe_radius=0.6
        )
    
    if 'frag' in task:
        run_conditional_eval_frag(
            job_id=my_task_id,
            num_tasks=num_tasks,
            load_file=file_path,
            solvent='water',
            num_processes=4,
            probe_radius=0.6
        )

    # total evals: 60-120
    if 'GDB' in task:
        samples_numbers = np.arange(100)
        samples_to_eval = samples_numbers[my_task_id:len(samples_numbers):num_tasks]
        print(f'Running GDB_conditional for these samples:\n{samples_to_eval}')
        # Go through every assigned sample id (1-2)
        for sample_idx in samples_to_eval:
            # Go through every representation (20*3)
            for conditioning in ('x2', 'x3', 'x4'):
                # Use task ID as sample id
                print(f'Running sample {sample_idx} for {conditioning} condition.')
                run_conditional_eval_by_sample_only(sample_id=sample_idx,
                                                    load_dir=file_path,
                                                    conditioning_type=conditioning,
                                                    solvent=None,
                                                    num_processes=4,
                                                    probe_radius=0.6)
