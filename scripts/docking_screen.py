"""
Docking benchmark:
Take 10k test molecules from MOSES and dock them against each target.
"""
import argparse
from pathlib import Path

import pickle

from shepherd_score.evaluations.docking import docking_target_info, DockingEvalPipeline


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir-path', type=str, help='Path to directory to save files to.')
    parser.add_argument('--load-file', type=str, help='Path to file containing 10k SMILES to screen.')
    parser.add_argument('--task-id', type=int, help='Task ID.')
    parser.add_argument('--num-tasks', type=int, help='Number of tasks.')
    args = parser.parse_args()
    print(args)

    save_dir = Path(args.save_dir_path)
    if not save_dir.is_dir():
        raise ValueError('Provided --save-dir-path is not a directory.')
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)
    load_file = Path(args.load_file)
    assert load_file.is_file()

    with open(load_file, 'r') as f:
        benchmark_smiles = [line.strip() for line in f]

    subset_smiles = benchmark_smiles[my_task_id:len(benchmark_smiles):num_tasks]

    # load each target with Vina oracle
    for target in docking_target_info.keys():
        dock_pipeline = DockingEvalPipeline(target, num_processes=4, verbose=1)

        energies = dock_pipeline.evaluate(subset_smiles,
                                          exhaustiveness=32,
                                          n_poses=1,
                                          protonate=False,
                                          save_poses_dir_path=None,
                                          verbose=True)
        
        smi_energies = [subset_smiles, [float(e) for e in energies]]

        # Save smiles and attributed energies
        save_dir_path = (save_dir/target)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_path = save_dir_path / f'smi_energies_{my_task_id}.pkl'
        with open(save_path, mode='wb') as f:
            pickle.dump(smi_energies, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved to {save_path}')
