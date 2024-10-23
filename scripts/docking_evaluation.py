"""
Docking benchmark:
Take 10k test molecules from MOSES and dock them against each target.
"""
import argparse
from pathlib import Path

import pickle

from rdkit import Chem

from shepherd_score.evaluations.utils.convert_data import get_smiles_from_atom_pos
from shepherd_score.evaluations.docking import docking_target_info, DockingEvalPipeline

experimental_pdb_order = ['3ny8', '7l11', '3eml', '4unn', '4rlu', '1iep', '5mo4']

docking_screen_pdb_order = ['1iep', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11']

pdb_bound_poses_order = {
    'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1': '1iep',
    'Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12' : '3eml',
    'Cc1ccc(OC[C@@H](O)[C@H](C)NC(C)C)c2c1CCC2' : '3ny8',
    'O=C(C=Cc1ccc(O)cc1)c1ccc(O)cc1O': '4rlu',
    'COc1cccc(-c2cc(-c3ccc(C(=O)O)cc3)c(C#N)c(=O)[nH]2)c1': '4unn',
    'O=C(Nc1ccc(OC(F)(F)Cl)cc1)c1cnc(N2CC[C@@H](O)C2)c(-c2ccn[nH]2)c1': '5mo4',
    'CCCOc1cc(Cl)cc(-c2cc(-c3ccccc3C#N)cn(-c3cccnc3)c2=O)c1': '7l11'
}


def docking_experiment_evaluation(pdb_id: str, sample_idx, file_path, save_path_dir: Path, my_task_id, num_tasks):
    """
    Run the docking evaluation pipeline.
    """
    docking_pipe = DockingEvalPipeline(pdb_id=pdb_id,
                                       num_processes=4,
                                       verbose=0,
                                       path_to_bin='')

    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
    
    out_dir = save_path_dir / f'{pdb_id}_{sample_idx}'
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.is_dir()
    out_file = out_dir / f'docking_eval_{my_task_id}.pkl'

    # Load all the generated samples as SMILES
    smiles_list = []
    for sample in samples[-1]:
        smiles_list.append(get_smiles_from_atom_pos(sample['x1']['atoms'], sample['x1']['positions']))
    
    this_smiles_ls = smiles_list[my_task_id:len(smiles_list):num_tasks]

    print('\n\n\nBEGIN DOCKING EVAL')
    docking_pipe.evaluate(this_smiles_ls, exhaustiveness=32, n_poses=1, protonate=False,
                          save_poses_dir_path=None, verbose=True)

    energies = [float(e) for e in docking_pipe.energies]
    smiles = docking_pipe.smiles
    buffer = docking_pipe.buffer

    with open(out_file, 'wb') as f:
        pickle.dump((pdb_id, buffer, smiles, energies), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved to {out_file}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir-path', type=str, help='Path to directory to load files.')
    parser.add_argument('--task-id', type=int, help='Task ID.')
    parser.add_argument('--num-tasks', type=int, help='Number of tasks.')
    parser.add_argument('--sample-idx', type=int, required=True, help='Index used to load the file. [0,6]')
    args = parser.parse_args()
    print(args)

    load_dir = Path(args.load_dir_path)
    if not load_dir.is_dir():
        raise ValueError('Provided --load-dir-path is not a directory.')
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)
    sample_idx = int(args.sample_idx)

    # Experimental conditioning
    working_dir = load_dir / 'PDB_analogues'
    pdb_id = experimental_pdb_order[sample_idx]

    file_path_experimental = working_dir / f'samples_{sample_idx}.pickle'
    print(f'Loading from {file_path_experimental}')
    docking_experiment_evaluation(pdb_id=pdb_id,
                                  sample_idx=sample_idx,
                                  file_path=file_path_experimental,
                                  save_path_dir=working_dir,
                                  my_task_id=my_task_id,
                                  num_tasks=num_tasks)
    
    # Docking screen conditioning
    working_dir = load_dir
    file_path_docking_screen = working_dir / f'samples_{sample_idx}.pickle'
    if 'PDB_analogues_pose' in working_dir.stem:
        with open(file_path_docking_screen, 'rb') as f:
            samples = pickle.load(f)
        ref_smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(samples[0], removeHs=True))
        pdb_id = pdb_bound_poses_order[ref_smiles]
    else:
        pdb_id = docking_screen_pdb_order[sample_idx]

    
    print(f'Loading from {file_path_docking_screen}')
    print(pdb_id)
    docking_experiment_evaluation(pdb_id=pdb_id,
                                  sample_idx=sample_idx,
                                  file_path=file_path_docking_screen,
                                  save_path_dir=working_dir,
                                  my_task_id=my_task_id,
                                  num_tasks=num_tasks)
