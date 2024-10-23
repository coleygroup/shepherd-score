"""
Benchmark of UnconditionalEvalPipeline / ConsistencyEvalPipeline for training set structures.
"""

import os
from typing import List, Optional
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


def generate_benchmark_set(training_set_dir: str,
                           relevant_sets: List[int],
                           save_dir: str,
                           data_name: str,
                           num_molecs: int = 1000,
                           num_surf_points: int = 75,
                           probe_radius: float = 0.6):
    """
    Pretend that MMFF structures are "generated" and run ConsistencyEvalPipeline on it.
    """
    dir_path = Path(training_set_dir)
    save_dir_path = Path(save_dir)
    if not dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {dir_path}')
    if not save_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {save_dir_path}')

    file_paths = {p.stem.split('.')[0][-1] : p for p in dir_path.glob('*.pkl')}

    molblocks = []
    for set_ind in tqdm(relevant_sets, total=len(relevant_sets), desc='Loading data'):
        file_path = file_paths[str(set_ind)]
        with open(file_path, 'rb') as f:
            molblocks_and_charges_single = pickle.load(f)
        molblocks.extend([molblocks_and_charges[0] for molblocks_and_charges in molblocks_and_charges_single])
        del molblocks_and_charges_single

    num_total_molecs = len(molblocks)

    # Choose random indices
    rng = np.random.default_rng()
    rand_inds = rng.choice(a=num_total_molecs, size=num_molecs, replace=False)
    molblocks = [molblocks[i] for i in rand_inds]

    ls_mmff_molblocks = []
    ls_atoms_pos = []
    ls_surf_points = []
    ls_surf_esp = []
    ls_pharm_feats = []
    num_failed = 0
    for molblock in tqdm(molblocks, total=len(molblocks), desc='Generating Molecule Objects',
                         miniters=50, maxinterval=10000):
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(molblock, removeHs=False))

            mol = embed_conformer_from_smiles(smiles, MMFF_optimize=True)
            molec = Molecule(mol=mol,
                             num_surf_points=num_surf_points,
                             probe_radius=probe_radius,
                             pharm_multi_vector=False)
        except:
            num_failed += 1
            continue

        if molec.mol is not None:
            atom_id = np.array([a.GetAtomicNum() for a in molec.mol.GetAtoms()])
            atom_pos = np.array(molec.mol.GetConformer().GetPositions())
            if not (isinstance(molec.surf_pos, np.ndarray) and
                isinstance(molec.surf_esp, np.ndarray) and
                isinstance(molec.pharm_ancs, np.ndarray) and
                isinstance(molec.pharm_types, np.ndarray) and
                isinstance(molec.pharm_vecs, np.ndarray)):
                continue
        else:
            continue

        ls_mmff_molblocks.append(Chem.MolToMolBlock(molec.mol))

        ls_atoms_pos.append(
            (atom_id, atom_pos)
        )
        ls_surf_points.append(
            molec.surf_pos
        )
        ls_surf_esp.append(
            molec.surf_esp
        )
        ls_pharm_feats.append(
            (molec.pharm_types, molec.pharm_ancs, molec.pharm_vecs)
        )

    print(f'{num_failed} failed.')

    # Save representations
    save_dir_path = save_dir_path / data_name
    save_dir_path.mkdir(parents=True, exist_ok=True)

    molblock_save_path = save_dir_path / 'mmff_molblocks.pkl'
    with open(molblock_save_path, 'wb') as f:
        pickle.dump(ls_mmff_molblocks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MMFF molblocks to {molblock_save_path}')
    
    atom_pos_path = save_dir_path / 'atom_pos.pkl'
    with open(atom_pos_path, 'wb') as f:
        pickle.dump(ls_atoms_pos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MMFF atom types and positions to {atom_pos_path}')

    surf_points_path = save_dir_path / 'surfpos.pkl'
    with open(surf_points_path, 'wb') as f:
        pickle.dump(ls_surf_points, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MMFF surface points to {surf_points_path}')

    surf_esp_path = save_dir_path / 'surfesp.pkl'
    with open(surf_esp_path, 'wb') as f:
        pickle.dump(ls_surf_esp, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MMFF surface esp to {surf_esp_path}')

    pharm_feats_path = save_dir_path / 'pharmfeats.pkl'
    with open(pharm_feats_path, 'wb') as f:
        pickle.dump(ls_pharm_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MMFF surface esp to {pharm_feats_path}')

    return ls_mmff_molblocks, ls_atoms_pos, ls_surf_points, ls_surf_esp, ls_pharm_feats


def run_consistency_benchmark(save_dir: str,
                              data_name: str,
                              solvent: Optional[str] = None,
                              num_processes: int = 1,
                              probe_radius: float = 0.6,
                              ) -> None:
    """
    Generate a benchmarking set of molecules -- both MMFF and its corresponding xtb relaxed
    structures.

    Arguments
    ---------
    training_set_dir : str
    relevant_sets : List[int]
    save_file : str -- must end in .pkl in an existing directory
    num_molecs : int (default = 1000)
    """
    save_dir_path = Path(save_dir)
    if not save_dir_path.is_dir():
        raise ValueError(f'Provided path is not a directory: {save_dir_path}')

    # Load files
    atom_pos_path = save_dir_path / data_name / 'atom_pos.pkl'
    with open(str(atom_pos_path), 'rb') as f:
        ls_atoms_pos = pickle.load(f)

    surf_points_path = save_dir_path / data_name / 'surfpos.pkl'
    with open(str(surf_points_path), 'rb') as f:
        ls_surf_points = pickle.load(f)

    surf_esp_path = save_dir_path / data_name / 'surfesp.pkl'
    with open(str(surf_esp_path), 'rb') as f:
        ls_surf_esp = pickle.load(f)

    pharm_feats_path = save_dir_path / data_name / 'pharmfeats.pkl'
    with open(str(pharm_feats_path), 'rb') as f:
        ls_pharm_feats = pickle.load(f)

    # Initialize evaluation
    consis_pipeline = ConsistencyEvalPipeline(
        generated_mols=ls_atoms_pos,
        generated_surf_points=ls_surf_points,
        generated_surf_esp=ls_surf_esp,
        generated_pharm_feats=ls_pharm_feats,
        probe_radius=probe_radius,
        pharm_multi_vector=False,
        solvent=solvent
    )

    # Run evaluation
    consis_pipeline.evaluate(num_processes=num_processes, verbose=True)

    # Save values to numpy object
    benchmark_output_path = save_dir_path / data_name / f'benchmark_output.npz'
    # Save values to numpy object
    save_global_series, save_local_df = consis_pipeline.to_pandas()

    save_file_global = save_dir_path / data_name / f'benchmark_output_global.pkl'
    save_global_series.to_pickle(save_file_global)
    save_file_local = save_dir_path / data_name / f'benchmark_output_local.pkl'
    save_local_df.to_pickle(save_file_local)

    print(f'Finished consistency evaluation!\nSaved global attributes to: {save_file_global}')
    print(f'Saved local attributes to: {save_file_local}')

    print(f'Finished {data_name} benchmark!\nSaved to {benchmark_output_path}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solvent', type=str, default='',
                        help='Solvent for xtb optimization.')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of processes to use for xtb optimization.')
    parser.add_argument('--training-set-dir', type=str, default=None,
                        help='Path to directory containing pickled training data.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory to save files to.')
    parser.add_argument('--relevant-sets', type=str, required=True,
                        help='Which training sets to sample from. E.g., "1,2,3"')
    parser.add_argument('--data-name', required=True, type=str, help='Name of dataset (i.e., gdb, moses, moses_aq)')

    args = parser.parse_args()
    print(args)

    solvent = args.solvent
    if solvent == '':
        solvent = None

    num_processes = args.num_processes
    training_set_dir = args.training_set_dir
    save_dir = str(args.save_dir)
    data_name = args.data_name
    relevant_sets = str(args.relevant_sets)
    relevant_sets = [int(s) for s in relevant_sets.split(',')]
    num_molecs = 1000
    # settings for ShEPhERD generated molecules
    num_surf_points = 75
    probe_radius = 0.6

    if training_set_dir is not None:
        generate_benchmark_set(
            training_set_dir=training_set_dir,
            save_dir=save_dir,
            data_name=data_name,
            relevant_sets=relevant_sets,
            num_molecs=num_molecs,
            num_surf_points=num_surf_points, # settings for ShEPhERD generated molecules
            probe_radius=probe_radius  # settings for ShEPhERD generated molecules
        )
    else:
        if data_name in ('gdb', 'moses_aq'):
            if data_name == 'gdb':
                solvent = None
            else:
                solvent = 'water'
            print(f'Running {data_name}...')
            run_consistency_benchmark(
                save_dir=save_dir,
                data_name=data_name,
                solvent=solvent,
                num_processes=num_processes,
                probe_radius=probe_radius
            )
