"""
Autodock Vina Docking evaluation pipelines.

Requires:
- vina
- meeko
- openbabel (if protonating ligands)
"""
from typing import List, Optional, Dict, Literal, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import multiprocessing

import numpy as np
import pandas as pd

from shepherd_score.evaluations.utils.convert_data import get_smiles_from_atom_pos

from shepherd_score.evaluations.docking.docking import VinaSmiles
from shepherd_score.evaluations.docking.targets import docking_target_info


# Global variable to store VinaSmiles instance in each worker process
_worker_vina_smiles = None


def _init_worker_vina(
    receptor_pdbqt_file: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    pH: float,
    scorefunction: str,
    num_processes: int,
    verbose: int,
):
    """
    Initialize VinaSmiles instance in worker process.
    This is called once per worker when the pool is created.
    """
    global _worker_vina_smiles
    _worker_vina_smiles = VinaSmiles(
        receptor_pdbqt_file=receptor_pdbqt_file,
        center=center,
        box_size=box_size,
        pH=pH,
        scorefunction=scorefunction,
        num_processes=num_processes,
        verbose=verbose,
    )


def _eval_docking_single(
    i: int,
    smiles: str,
    exhaustiveness: int,
    n_poses: int,
    protonate: bool,
    save_poses_path: Optional[str],
) -> Dict[str, Any]:
    """
    Evaluate a single SMILES string for docking.
    
    This function is designed to be called by multiprocessing workers.
    It uses the pre-initialized VinaSmiles instance from the worker.
    """
    global _worker_vina_smiles

    if smiles is None:
        return {'i': i, 'energy': np.nan, 'error': 'SMILES is None'}

    if _worker_vina_smiles is None:
        return {'i': i, 'energy': np.nan, 'error': 'VinaSmiles not initialized in worker'}

    try:
        # Reset state before each docking call to ensure clean state
        # (set_ligand_from_string should replace previous ligand, but this ensures clean state)
        _worker_vina_smiles.state = None

        energy = _worker_vina_smiles(
            ligand_smiles=smiles,
            output_file=save_poses_path,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            protonate=protonate,
        )

        return {'i': i, 'energy': float(energy), 'error': None}
    except Exception as e:
        return {'i': i, 'energy': np.nan, 'error': str(e)}


def _unpack_eval_docking_single(args):
    """Unpacker function for multiprocessing."""
    return _eval_docking_single(*args)


class DockingEvalPipeline:

    def __init__(self,
                 pdb_id: str,
                 num_processes: int = 4,
                 docking_target_info_dict: Dict = docking_target_info,
                 verbose: int = 0,
                 path_to_bin: str = ''):
        """
        Constructor for docking evaluation pipeline. Initializes VinaSmiles with receptor pdbqt.

        Arguments
        ---------
        pdb_id : str PDB ID of receptor. Natively only supports:
            1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11
        num_processes : int (default = 4) Number of cpus to use for scoring
        docking_target_info_dict : Dict holding minimum information needed for docking.
            For example:
                {
                "1iep": {
                    "center": (15.614, 53.380, 15.455),
                    "size": (15, 15, 15),
                    "pdbqt": "path_to_file.pdbqt"
                }
        verbose : int (default = 0) Level of verbosity from vina.Vina (0 is silent)
        path_to_bin : str (default = '') path to environment bin containing `mk_prepare_ligand.py`
        """
        self.pdb_id = pdb_id
        self.path_to_bin = path_to_bin
        self.docking_target_info = docking_target_info_dict
        self.vina_smiles = None
        self.smiles = []
        self.energies = []
        self.buffer = {}
        self.num_failed = 0
        self.repeats = 0

        if pdb_id not in list(self.docking_target_info.keys()):
            raise ValueError(
                f"Provided `pdb_id` ({pdb_id}) not supported. Please choose from: {list(self.docking_target_info.keys())}."
            )

        path_to_receptor_pdbqt = Path(self.docking_target_info[self.pdb_id]['pdbqt'])
        if not path_to_receptor_pdbqt.is_file():
            raise ValueError(
                f"Provided .pdbqt file does not exist. Please check `docking_target_info_dict`. Was given: {path_to_receptor_pdbqt}"
            )

        pH = self.docking_target_info[self.pdb_id]['pH'] if 'pH' in self.docking_target_info[self.pdb_id] else 7.4

        self.receptor_pdbqt_file = str(path_to_receptor_pdbqt)
        self.center = self.docking_target_info[self.pdb_id]['center']
        self.box_size = self.docking_target_info[self.pdb_id]['size']
        self.pH = pH
        self.scorefunction = 'vina'
        self.verbose = verbose

        self.vina_smiles = VinaSmiles(
            receptor_pdbqt_file=self.receptor_pdbqt_file,
            center=self.center,
            box_size=self.box_size,
            pH=self.pH,
            scorefunction=self.scorefunction,
            num_processes=num_processes,
            verbose=self.verbose
        )


    def evaluate(self,
                 smiles_ls: List[str],
                 exhaustiveness: int = 32,
                 n_poses: int = 1,
                 protonate: bool = False,
                 save_poses_dir_path: Optional[str] = None,
                 verbose: bool = False,
                 num_workers: int = 1,
                 num_processes: int = 4,
                 *,
                 mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                 ) -> List[float]:
        """
        Loop through supplied list of SMILES strings, dock, and collect energies.

        Arguments
        ---------
        smiles_ls : List[str] list of SMILES to dock
        exhaustiveness : int (default = 32) Number of Monte Carlo simulations to run per pose
        n_poses : int (default = 1) Number of poses to save
        protonate : bool (default = False) (de-)protonate ligand with OpenBabel at pH=7.4
        save_poses_dir_path : Optional[str] (default = None) Path to directory to save docked poses.
        verbose : bool (default = False) show tqdm progress bar for each SMILES.
        num_workers : int (default = 1) number of parallel worker processes. 
            Only recommended if `smiles_ls` is > 100 due to start-up overhead of new processes.
        num_processes : int (default = 4) number of processes each worker uses internally for Vina.
            Constraint: num_workers * num_processes <= available CPUs
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.
            'spawn' is recommended for most cases.

        Returns
        -------
        List of energies (affinities) in kcal/mol
        """
        self.smiles = smiles_ls
        dir_path = None
        if save_poses_dir_path is not None:
            dir_path = Path(save_poses_dir_path)

        # Check buffer first and filter out cached SMILES
        energies = []
        indices_to_process = []
        smiles_to_process = []

        for i, smiles in enumerate(smiles_ls):
            if smiles in self.buffer:
                self.repeats += 1
                energies.append((i, self.buffer[smiles]))
            elif smiles is None:
                energies.append((i, np.nan))
                self.num_failed += 1
            else:
                indices_to_process.append(i)
                smiles_to_process.append(smiles)

        available_cpus = multiprocessing.cpu_count() or 1
        if num_processes > exhaustiveness:
            num_processes = exhaustiveness
        if num_workers < 1:
            num_workers = 1
        if num_processes < 1:
            num_processes = 1

        # Calculate max workers: num_workers * num_processes <= available_cpus
        max_workers_allowed = max(1, available_cpus // max(1, num_processes))
        if num_workers > max_workers_allowed:
            num_workers = max_workers_allowed

        if num_workers > 1 and len(smiles_to_process) > 0:
            multiprocessing.set_start_method(mp_context, force=True)

            # Prepare inputs for workers (only task-specific parameters)
            inputs = []
            for idx, smiles in zip(indices_to_process, smiles_to_process):
                save_poses_path = None
                if save_poses_dir_path is not None:
                    save_poses_path = str(dir_path / f'{self.pdb_id}_docked_{idx}{"_prot" if protonate else ""}.pdbqt')

                inputs.append((
                    idx,
                    smiles,
                    exhaustiveness,
                    n_poses,
                    protonate,
                    save_poses_path,
                ))

            # Initialize VinaSmiles once per worker using initializer
            with multiprocessing.Pool(
                num_workers,
                initializer=_init_worker_vina,
                initargs=(
                    self.receptor_pdbqt_file,
                    self.center,
                    self.box_size,
                    self.pH,
                    self.scorefunction,
                    num_processes,
                    self.verbose,
                )
            ) as pool:
                if verbose:
                    pbar = tqdm(total=len(smiles_to_process), desc=f'Docking {self.pdb_id}')
                else:
                    pbar = None

                results_iter = pool.imap_unordered(_unpack_eval_docking_single, inputs, chunksize=1)

                pending_results = {idx: None for idx in indices_to_process}

                for res in results_iter:
                    idx = res['i']
                    pending_results[idx] = res
                    energy = res['energy']
                    energies.append((idx, energy))

                    # Update buffer
                    smiles_str = smiles_ls[idx]
                    if smiles_str is not None:
                        self.buffer[smiles_str] = float(energy)
                        if np.isnan(energy):
                            self.num_failed += 1

                    if verbose and pbar is not None:
                        pbar.update(1)

                if verbose and pbar is not None:
                    pbar.close()

                for idx in indices_to_process:
                    if pending_results[idx] is None:
                        energies.append((idx, np.nan))
                        smiles_str = smiles_ls[idx]
                        if smiles_str is not None:
                            self.buffer[smiles_str] = float(np.nan)
                        self.num_failed += 1
        else:
            # Single process evaluation
            if len(smiles_to_process) > 0:
                if verbose:
                    pbar = tqdm(enumerate(zip(indices_to_process, smiles_to_process)), 
                            desc=f'Docking {self.pdb_id}', 
                            total=len(smiles_to_process))
                else:
                    pbar = enumerate(zip(indices_to_process, smiles_to_process))
                
                for _, (idx, smiles) in pbar:
                    save_poses_path = None
                    if save_poses_dir_path is not None:
                        save_poses_path = dir_path / f'{self.pdb_id}_docked{"_prot" if protonate else ""}_{idx}.pdbqt'
                    try:
                        energy = self.vina_smiles(
                            ligand_smiles=smiles,
                            output_file=save_poses_path,
                            exhaustiveness=exhaustiveness,
                            n_poses=n_poses,
                            protonate=protonate,
                        )
                        energies.append((idx, float(energy)))
                        self.buffer[smiles] = float(energy)
                    except Exception as _:
                        energies.append((idx, np.nan))
                        self.buffer[smiles] = float(np.nan)
                        self.num_failed += 1

        # Sort by original index and extract energies
        energies.sort(key=lambda x: x[0])
        self.energies = np.array([e[1] for e in energies])
        return [e[1] for e in energies]


    def benchmark(self,
                  exhaustiveness: int = 32,
                  n_poses: int = 5,
                  protonate: bool = False,
                  save_poses_dir_path: Optional[str] = None
                  ) -> float:
        """
        Run benchmark with experimental ligands.

        Arguments
        ---------
        exhaustiveness : int (default = 32) Number of Monte Carlo simulations to run per pose
        n_poses : int (default = 5) Number of poses to save
        protonate : bool (default = False) (de-)protonate ligand with OpenBabel at pH=7.4
        save_poses_dir_path : Optional[str] (default = None) Path to directory to save docked poses.

        Returns
        -------
        float : Energies (affinities) in kcal/mol
        """
        save_poses_path = None
        if save_poses_dir_path is not None:
            dir_path = Path(save_poses_dir_path)
            save_poses_path = dir_path / f"{self.pdb_id}_docked{'_prot' if protonate else ''}.pdbqt"

        best_energy = self.vina_smiles(
            self.docking_target_info[self.pdb_id]['ligand'],
            output_file=str(save_poses_path),
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            protonate=protonate,
        )
        return best_energy


    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the attributes of generated smiles and the energies to a pd.DataFrame

        Arguments
        ---------
        self

        Returns
        -------
        pd.DataFrame : attributes for each evaluated sample
        """
        global_attrs = {'smiles' : self.smiles, 'energies': self.energies}
        series_global = pd.Series(global_attrs)

        return series_global


def run_docking_benchmark(save_dir_path: str,
                          pdb_id: str,
                          num_processes: int = 4,
                          docking_target_info_dict=docking_target_info,
                          protonate: bool = False
                          ) -> None:
    """
    Run docking benchmark on experimental smiles. Uses an exhaustivness of 32 and saves the top-30
    poses to a specified location.
    
    Arguments
    ---------
    save_dir_path : str Path to save docked poses to
    pdb_id : str PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11
    num_processes : int (default = 4) Number of cpus to use for scoring
    docking_target_info_dict : Dict holding minimum information needed for docking.
        For example:
            {
            "1iep": {
                "center": (15.614, 53.380, 15.455),
                "size": (15, 15, 15),
                "pdbqt": "path_to_file.pdbqt",
                "ligand": "SMILES string of experimental ligand"
            }
    protonate : bool (default = False) whether to protonate ligands at a given pH. (Requires `"pH"`
        field to be filled out in docking_target_info_dict)

    Returns
    -------
    None
    """
    dep = DockingEvalPipeline(pdb_id=pdb_id,
                              num_processes=num_processes,
                              docking_target_info_dict=docking_target_info_dict,
                              verbose=0,
                              path_to_bin='')
    dep.benchmark(exhaustiveness=32, n_poses=30, save_poses_dir_path=save_dir_path, protonate=protonate)


def run_docking_evaluation(atoms: List[np.ndarray],
                           positions: List[np.ndarray],
                           pdb_id: str,
                           num_processes: int = 4,
                           docking_target_info_dict=docking_target_info
                           ) -> DockingEvalPipeline:
    """
    Run docking evaluation with an exhaustiveness of 32.
    
    Arguments
    ---------
    atoms : List[np.ndarray (N,)] of atomic numbers of the generated molecule or (N,M) one-hot
        encoding.
    positions : List[np.ndarray (N,3)] of coordinates for the generated molecule's atoms.
    pdb_id : str PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11
    num_processes : int (default = 4) Number of cpu's to use for Autodock Vina
    docking_target_info_dict : Dict holding minimum information needed for docking.
        For example:
            {
            "1iep": {
                "center": (15.614, 53.380, 15.455),
                "size": (15, 15, 15),
                "pdbqt": "path_to_file.pdbqt"
            }

    Returns
    -------
    DockingEvalPipeline object
        Results are found in the `buffer` attribute {'smiles' : energy}
        Or in `smiles` and `energies` which preserves the order of provided atoms/positions as a
        list.
    """
    docking_pipe = DockingEvalPipeline(pdb_id=pdb_id,
                                       num_processes=num_processes,
                                       docking_target_info_dict=docking_target_info_dict,
                                       verbose=0,
                                       path_to_bin='')

    smiles_list = []
    for sample in zip(atoms, positions):
        smiles_list.append(get_smiles_from_atom_pos(atoms=sample[0], positions=sample[1]))

    docking_pipe.evaluate(smiles_list, exhaustiveness=32, n_poses=1, protonate=False,
                          save_poses_dir_path=None, verbose=True)

    return docking_pipe


def run_docking_evaluation_from_smiles(smiles: List[str],
                                       pdb_id: str,
                                       num_processes: int = 4,
                                       docking_target_info_dict=docking_target_info
                                       ) -> DockingEvalPipeline:
    """
    Run docking evaluation with an exhaustiveness of 32.
    
    Arguments
    ---------
    smiles : List[str] list of SMILES strings. These must be valid or None.
    pdb_id : str PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11
    num_processes : int (default = 4) Number of cpu's to use for Autodock Vina
    docking_target_info_dict : Dict holding minimum information needed for docking.
        For example:
            {
            "1iep": {
                "center": (15.614, 53.380, 15.455),
                "size": (15, 15, 15),
                "pdbqt": "path_to_file.pdbqt"
            }

    Returns
    -------
    DockingEvalPipeline object
        Results are found in the `buffer` attribute {'smiles' : energy}
        Or in `smiles` and `energies` which preserves the order of provided atoms/positions as a
        list.
    """
    docking_pipe = DockingEvalPipeline(pdb_id=pdb_id,
                                       num_processes=num_processes,
                                       docking_target_info_dict=docking_target_info_dict,
                                       verbose=0,
                                       path_to_bin='')

    docking_pipe.evaluate(smiles, exhaustiveness=32, n_poses=1, protonate=False,
                          save_poses_dir_path=None, verbose=True)

    return docking_pipe
