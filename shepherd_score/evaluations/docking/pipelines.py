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
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

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
    return_best_protomer: bool,
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

        energy, docked_mol = _worker_vina_smiles(
            ligand_smiles=smiles,
            output_file=save_poses_path,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            protonate=protonate,
            return_best_protomer=True,
        )

        # Pickle the docked mol for return
        docked_mol_pickle = None
        if docked_mol is not None:
            docked_mol_pickle = pickle.dumps(docked_mol)

        return {
            'i': i,
            'energy': float(energy),
            'error': None,
            'docked_mol': docked_mol_pickle,
        }
    except Exception as e:
        return {'i': i, 'energy': np.nan, 'error': str(e), 'docked_mol': None}


def _unpack_eval_docking_single(args):
    """Unpacker function for multiprocessing."""
    return _eval_docking_single(*args)


def _eval_relax_single(
    i: int,
    mol_pickle: bytes,
    center: bool | Tuple[float, float, float],
    max_steps: int | None,
    save_poses_path: Optional[str],
) -> Dict[str, Any]:
    """
    Evaluate a single mol object for relaxation.

    This function is designed to be called by multiprocessing workers.
    It uses the pre-initialized VinaSmiles instance from the worker.
    """
    global _worker_vina_smiles

    if mol_pickle is None:
        return {'i': i, 'energy': np.nan, 'relaxed_mol': None, 'error': 'mol is None'}

    if _worker_vina_smiles is None:
        return {'i': i, 'energy': np.nan, 'relaxed_mol': None, 'error': 'VinaSmiles not initialized in worker'}

    try:
        # Unpickle the mol object
        mol = pickle.loads(mol_pickle)

        if mol is None:
            return {'i': i, 'energy': np.nan, 'relaxed_mol': None, 'error': 'mol is None after unpickling'}

        # Reset state before each optimization call
        _worker_vina_smiles.state = None

        total_energy, _, optimized_mol = _worker_vina_smiles.optimize_ligand(
            ligand=mol,
            center=center,
            max_steps=max_steps,
            output_file=save_poses_path,
        )

        # Pickle the optimized mol for return
        optimized_mol_pickle = None
        if optimized_mol is not None:
            optimized_mol_pickle = pickle.dumps(optimized_mol)

        return {
            'i': i,
            'energy': float(total_energy),
            'relaxed_mol': optimized_mol_pickle,
            'error': None
        }
    except Exception as e:
        return {'i': i, 'energy': np.nan, 'relaxed_mol': None, 'error': str(e)}


def _unpack_eval_relax_single(args):
    """Unpacker function for multiprocessing."""
    return _eval_relax_single(*args)


class DockingEvalPipeline:

    def __init__(self,
                 pdb_id: str,
                 num_processes: int = 4,
                 docking_target_info_dict: Dict = docking_target_info,
                 verbose: int = 0,
                 path_to_bin: str = ''):
        """
        Constructor for docking evaluation pipeline.

        Initializes VinaSmiles with receptor pdbqt.

        Parameters
        ----------
        pdb_id : str
            PDB ID of receptor. Natively only supports:
            1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11.
        num_processes : int, optional
            Number of CPUs to use for scoring. Default is 4.
        docking_target_info_dict : dict, optional
            Dict holding minimum information needed for docking. Example format::

                {"1iep": {"center": (15.614, 53.380, 15.455),
                          "size": (15, 15, 15),
                          "pdbqt": "path_to_file.pdbqt"}}

        verbose : int, optional
            Level of verbosity from vina.Vina (0 is silent). Default is 0.
        path_to_bin : str, optional
            Path to environment bin containing ``mk_prepare_ligand.py``. Default is ''.
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
        self.buffer_relaxed = {}
        self.relaxed_mols = []
        self.relaxed_rmsd = []

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
                 return_best_protomer: bool = False,
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
        protonate : bool (default = False) Use protonation protocol
        save_poses_dir_path : Optional[str] (default = None) Path to directory to save docked poses.
        verbose : bool (default = False) show tqdm progress bar for each SMILES.
        num_workers : int (default = 1) number of parallel worker processes.
            Only recommended if `smiles_ls` is > 100 due to start-up overhead of new processes.
        num_processes : int (default = 4) number of processes each worker uses internally for Vina.
            Constraint: num_workers * num_processes <= available CPUs
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.

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
                energies.append((i, self.buffer[smiles]['energy']))
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
                    protonate if not return_best_protomer else True,
                    return_best_protomer,
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
                    docked_mol = None
                    if res['docked_mol'] is not None:
                        try:
                            docked_mol = pickle.loads(res['docked_mol'])
                        except Exception:
                            docked_mol = None
                    if smiles_str is not None:
                        self.buffer[smiles_str] = {'energy': float(energy), 'docked_mol': docked_mol}
                        if return_best_protomer:
                            self.buffer[smiles_str].update({'protomer_smiles': None})
                            if docked_mol is not None:
                                self.buffer[smiles_str]['protomer_smiles'] = Chem.MolToSmiles(Chem.RemoveHs(docked_mol))
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
                            self.buffer[smiles_str] = {'energy': float(np.nan), 'docked_mol': None}
                            if return_best_protomer:
                                self.buffer[smiles_str].update({'protomer_smiles': None})
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
                        if return_best_protomer:
                            save_poses_path = dir_path / f'{self.pdb_id}_docked_best_prot_{idx}.pdbqt'
                        else:
                            save_poses_path = dir_path / f'{self.pdb_id}_docked{"_prot" if protonate else ""}_{idx}.pdbqt'
                    try:
                        energy, docked_mol = self.vina_smiles(
                            ligand_smiles=smiles,
                            output_file=save_poses_path,
                            exhaustiveness=exhaustiveness,
                            n_poses=n_poses,
                            protonate=protonate if not return_best_protomer else True,
                            return_best_protomer=return_best_protomer,
                        )
                        energies.append((idx, float(energy)))
                        self.buffer[smiles] = {'energy': float(energy), 'docked_mol': docked_mol}
                        if return_best_protomer:
                            self.buffer[smiles].update({'protomer_smiles': None})
                            if docked_mol is not None:
                                self.buffer[smiles]['protomer_smiles'] = Chem.MolToSmiles(Chem.RemoveHs(docked_mol))

                    except Exception:
                        energies.append((idx, np.nan))
                        self.buffer[smiles] = {'energy': float(np.nan), 'docked_mol': None}
                        if return_best_protomer:
                            self.buffer[smiles].update({'protomer_smiles': None})
                        self.num_failed += 1

        # Sort by original index and extract energies
        energies.sort(key=lambda x: x[0])
        self.energies = np.array([e[1] for e in energies])
        return [e[1] for e in energies]

    def evaluate_relax(self,
                       mol_ls: List[Chem.Mol],
                       center: bool | Tuple[float, float, float] = False,
                       max_steps: int | None = 10000,
                       save_poses_dir_path: Optional[str] = None,
                       verbose: bool = False,
                       num_workers: int = 1,
                       *,
                       mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                       ) -> List[float]:
        """
        Loop through supplied list of mol objects, optimize, and collect energies.

        Arguments
        ---------
        mol_ls : List[Chem.Mol] list of rdkit mol objects to relax
        center : bool | Tuple[float, float, float] (default = False)
            If a tuple, centers to those coordinates.
            If True, centers the ligand to the receptor's center.
            If False, does not translate the ligand from its initial conformation.
        max_steps : int | None (default = 10000) Maximum number of steps to take in the optimization.
            If None, uses the default value of 10000.
        save_poses_dir_path : Optional[str] (default = None) Path to directory to save optimized poses.
        verbose : bool (default = False) show tqdm progress bar for each mol.
        num_workers : int (default = 1) number of parallel worker processes.
        mp_context : Literal['spawn', 'forkserver'] context for multiprocessing.

        Returns
        -------
        List of energies (affinities) in kcal/mol
        """
        dir_path = None
        if save_poses_dir_path is not None:
            dir_path = Path(save_poses_dir_path)

        # Process mol objects
        energies = []
        relaxed_mols = []
        indices_to_process = []
        mols_to_process = []

        for i, mol in enumerate(mol_ls):
            if mol is None:
                energies.append((i, np.nan))
                relaxed_mols.append((i, None))
                self.num_failed += 1
            else:
                indices_to_process.append(i)
                mols_to_process.append(mol)
            self.buffer_relaxed[Chem.MolToSmiles(Chem.RemoveHs(mol))] = {'energy': np.nan, 'relaxed_mol': None, 'rmsd': np.nan}

        available_cpus = multiprocessing.cpu_count() or 1
        if num_workers < 1:
            num_workers = 1

        # Calculate max workers: num_workers * num_processes <= available_cpus
        max_workers_allowed = max(1, available_cpus)
        if num_workers > max_workers_allowed:
            num_workers = max_workers_allowed

        if num_workers > 1 and len(mols_to_process) > 0:
            multiprocessing.set_start_method(mp_context, force=True)

            # Prepare inputs for workers (pickle mol objects for multiprocessing)
            inputs = []
            for idx, mol in zip(indices_to_process, mols_to_process):
                save_poses_path = None
                if save_poses_dir_path is not None:
                    save_poses_path = str(dir_path / f'{self.pdb_id}_relaxed_{idx}.pdbqt')

                # Pickle the mol for multiprocessing
                mol_pickle = pickle.dumps(mol)

                inputs.append((
                    idx,
                    mol_pickle,
                    center,
                    max_steps,
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
                    1, # num processes = 1 since only one process is used for relaxation
                    self.verbose,
                )
            ) as pool:
                if verbose:
                    pbar = tqdm(total=len(mols_to_process), desc=f'Relaxing {self.pdb_id}')
                else:
                    pbar = None

                results_iter = pool.imap_unordered(_unpack_eval_relax_single, inputs, chunksize=1)

                pending_results = {idx: None for idx in indices_to_process}

                for res in results_iter:
                    idx = res['i']
                    pending_results[idx] = res
                    energy = res['energy']
                    energies.append((idx, energy))
                    smiles_str = Chem.MolToSmiles(Chem.RemoveHs(mols_to_process[idx]))
                    if smiles_str is not None:
                        self.buffer_relaxed[smiles_str]['energy'] = float(energy)
                        self.buffer_relaxed[smiles_str]['relaxed_mol'] = mols_to_process[idx]

                    # Unpickle the relaxed mol
                    relaxed_mol = None
                    if res['relaxed_mol'] is not None:
                        try:
                            relaxed_mol = pickle.loads(res['relaxed_mol'])
                        except Exception:
                            relaxed_mol = None
                    relaxed_mols.append((idx, relaxed_mol))
                    smiles_str = Chem.MolToSmiles(Chem.RemoveHs(mols_to_process[idx]))
                    if smiles_str is not None:
                        self.buffer_relaxed[smiles_str]['energy'] = float(energy)
                        self.buffer_relaxed[smiles_str]['relaxed_mol'] = relaxed_mol if relaxed_mol is not None else None
                        self.buffer_relaxed[smiles_str]['rmsd'] = AllChem.CalcRMS(
                            self.vina_smiles._center_ligand(mols_to_process[idx], self.center),
                            relaxed_mol
                        )
                    if np.isnan(energy):
                        self.num_failed += 1

                    if verbose and pbar is not None:
                        pbar.update(1)

                if verbose and pbar is not None:
                    pbar.close()

                for idx in indices_to_process:
                    if pending_results[idx] is None:
                        energies.append((idx, np.nan))
                        relaxed_mols.append((idx, None))
                        self.num_failed += 1
        else:
            # Single process evaluation
            if len(mols_to_process) > 0:
                if verbose:
                    pbar = tqdm(enumerate(zip(indices_to_process, mols_to_process)),
                            desc=f'Relaxing {self.pdb_id}',
                            total=len(mols_to_process))
                else:
                    pbar = enumerate(zip(indices_to_process, mols_to_process))

                for _, (idx, mol) in pbar:
                    save_poses_path = None
                    if save_poses_dir_path is not None:
                        save_poses_path = dir_path / f'{self.pdb_id}_relaxed_{idx}.pdbqt'
                    try:
                        total_energy, _, optimized_mol = self.vina_smiles.optimize_ligand(
                            ligand=mol,
                            center=center,
                            max_steps=max_steps,
                            output_file=save_poses_path,
                        )
                        energies.append((idx, float(total_energy)))
                        relaxed_mols.append((idx, optimized_mol))
                    except Exception:
                        energies.append((idx, np.nan))
                        relaxed_mols.append((idx, None))
                        self.num_failed += 1

                    smiles_str = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    if smiles_str is not None:
                        self.buffer_relaxed[smiles_str]['energy'] = float(total_energy)
                        self.buffer_relaxed[smiles_str]['relaxed_mol'] = optimized_mol if optimized_mol is not None else None
                        self.buffer_relaxed[smiles_str]['rmsd'] = AllChem.CalcRMS(
                            self.vina_smiles._center_ligand(mol, self.center),
                            optimized_mol
                        )

        # Sort by original index and extract energies and relaxed mols
        energies.sort(key=lambda x: x[0])
        relaxed_mols.sort(key=lambda x: x[0])
        return np.array([e[1] for e in energies])

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

        best_energy, docked_ligand = self.vina_smiles(
            self.docking_target_info[self.pdb_id]['ligand'],
            output_file=str(save_poses_path),
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            protonate=protonate,
        )
        return best_energy, docked_ligand

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the attributes of generated smiles and the energies to a pd.DataFrame

        Returns
        -------
        pd.DataFrame : attributes for each evaluated sample
        """
        global_attrs = {'smiles' : self.smiles, 'energies': self.energies}
        series_global = pd.Series(global_attrs)

        return series_global

    def to_pandas_relaxed(self) -> pd.DataFrame:
        """
        Convert the attributes of relaxed mols and the energies to a pd.DataFrame

        Returns
        -------
        pd.DataFrame : attributes for each relaxed sample
        """
        df_relaxed = pd.DataFrame(self.buffer_relaxed)
        return df_relaxed


def run_docking_benchmark(save_dir_path: str,
                          pdb_id: str,
                          num_processes: int = 4,
                          docking_target_info_dict=docking_target_info,
                          protonate: bool = False
                          ) -> None:
    """
    Run docking benchmark on experimental SMILES.

    Uses an exhaustiveness of 32 and saves the top-30 poses to a specified location.

    Parameters
    ----------
    save_dir_path : str
        Path to save docked poses to.
    pdb_id : str
        PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11.
    num_processes : int, optional
        Number of CPUs to use for scoring. Default is 4.
    docking_target_info_dict : dict, optional
        Dict holding minimum information needed for docking. Example format::

            {"1iep": {"center": (15.614, 53.380, 15.455),
                      "size": (15, 15, 15),
                      "pdbqt": "path_to_file.pdbqt",
                      "ligand": "SMILES string of experimental ligand"}}

    protonate : bool, optional
        Whether to protonate ligands at a given pH. Requires ``"pH"`` field to be
        filled out in docking_target_info_dict. Default is ``False``.

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
                           docking_target_info_dict=docking_target_info,
                           exhaustiveness: int = 32,
                           n_poses: int = 1,
                           protonate: bool = False,
                           save_poses_dir_path: Optional[str] = None,
                           verbose: bool = True,
                           num_workers: int = 1,
                           *,
                           mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                           ) -> DockingEvalPipeline:
    """
    Run docking evaluation with an exhaustiveness of 32.

    Parameters
    ----------
    atoms : list
        List of np.ndarray (N,) of atomic numbers of the generated molecule or (N, M)
        one-hot encoding.
    positions : list
        List of np.ndarray (N, 3) of coordinates for the generated molecule's atoms.
    pdb_id : str
        PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11.
    num_processes : int, optional
        Number of CPUs to use for Autodock Vina. Default is 4.
    docking_target_info_dict : dict, optional
        Dict holding minimum information needed for docking. Example format::

            {"1iep": {"center": (15.614, 53.380, 15.455),
                      "size": (15, 15, 15),
                      "pdbqt": "path_to_file.pdbqt"}}

    exhaustiveness : int, optional
        Number of Monte Carlo simulations to run per pose. Default is 32.
    n_poses : int, optional
        Number of poses to save. Default is 1.
    protonate : bool, optional
        Use protonation protocol. Default is ``False``.
    save_poses_dir_path : str, optional
        Path to directory to save docked poses. Default is ``None``.
    verbose : bool, optional
        Show tqdm progress bar for each SMILES. Default is ``True``.
    num_workers : int, optional
        Number of parallel worker processes. Default is 1.
    mp_context : str, optional
        Context for multiprocessing. One of 'spawn' or 'forkserver'. Default is 'spawn'.

    Returns
    -------
    DockingEvalPipeline
        Results are found in the ``buffer`` attribute {'smiles': energy} or in ``smiles``
        and ``energies`` which preserves the order of provided atoms/positions as a list.
    """
    docking_pipe = DockingEvalPipeline(pdb_id=pdb_id,
                                       num_processes=num_processes,
                                       docking_target_info_dict=docking_target_info_dict,
                                       verbose=0,
                                       path_to_bin='')

    smiles_list = []
    for sample in zip(atoms, positions):
        smiles_list.append(get_smiles_from_atom_pos(atoms=sample[0], positions=sample[1]))

    docking_pipe.evaluate(smiles_list, exhaustiveness=exhaustiveness, n_poses=n_poses,
                          protonate=protonate, save_poses_dir_path=save_poses_dir_path,
                          verbose=verbose, num_workers=num_workers, num_processes=num_processes,
                          mp_context=mp_context)

    return docking_pipe


def run_docking_evaluation_from_smiles(smiles: List[str],
                                       pdb_id: str,
                                       num_processes: int = 4,
                                       docking_target_info_dict=docking_target_info,
                                       exhaustiveness: int = 32,
                                       n_poses: int = 1,
                                       protonate: bool = False,
                                       save_poses_dir_path: Optional[str] = None,
                                       verbose: bool = True,
                                       num_workers: int = 1,
                                       *,
                                       mp_context: Literal['spawn', 'forkserver'] = 'spawn'
                                       ) -> DockingEvalPipeline:
    """
    Run docking evaluation with an exhaustiveness of 32.

    Parameters
    ----------
    smiles : list
        List of SMILES strings. These must be valid or ``None``.
    pdb_id : str
        PDB ID of receptor. Natively only supports:
        1iep, 3eml, 3ny8, 4rlu, 4unn, 5mo4, 7l11.
    num_processes : int, optional
        Number of CPUs to use for Autodock Vina. Default is 4.
    docking_target_info_dict : dict, optional
        Dict holding minimum information needed for docking. Example format::

            {"1iep": {"center": (15.614, 53.380, 15.455),
                      "size": (15, 15, 15),
                      "pdbqt": "path_to_file.pdbqt"}}

    exhaustiveness : int, optional
        Number of Monte Carlo simulations to run per pose. Default is 32.
    n_poses : int, optional
        Number of poses to save. Default is 1.
    protonate : bool, optional
        Use protonation protocol. Default is ``False``.
    save_poses_dir_path : str, optional
        Path to directory to save docked poses. Default is ``None``.
    verbose : bool, optional
        Show tqdm progress bar for each SMILES. Default is ``True``.
    num_workers : int, optional
        Number of parallel worker processes. Default is 1.
    mp_context : str, optional
        Context for multiprocessing. One of 'spawn' or 'forkserver'. Default is 'spawn'.

    Returns
    -------
    DockingEvalPipeline
        Results are found in the ``buffer`` attribute {'smiles': energy} or in ``smiles``
        and ``energies`` which preserves the order of provided SMILES as a list.
    """
    docking_pipe = DockingEvalPipeline(pdb_id=pdb_id,
                                       num_processes=num_processes,
                                       docking_target_info_dict=docking_target_info_dict,
                                       verbose=0,
                                       path_to_bin='')

    docking_pipe.evaluate(smiles, exhaustiveness=exhaustiveness, n_poses=n_poses,
                          protonate=protonate, save_poses_dir_path=save_poses_dir_path,
                          verbose=verbose, num_workers=num_workers, num_processes=num_processes,
                          mp_context=mp_context)

    return docking_pipe
