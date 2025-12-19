"""
Autodock Vina Docking evaluation pipelines.

Requires:
- vina
- meeko
- openbabel (if protonating ligands)
"""
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from shepherd_score.evaluations.utils.convert_data import get_smiles_from_atom_pos

from shepherd_score.evaluations.docking.docking import VinaSmiles
from shepherd_score.evaluations.docking.targets import docking_target_info

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

        self.vina_smiles = VinaSmiles(
            receptor_pdbqt_file=path_to_receptor_pdbqt,
            center=self.docking_target_info[self.pdb_id]['center'],
            box_size=self.docking_target_info[self.pdb_id]['size'],
            pH=pH,
            scorefunction='vina',
            num_processes=num_processes,
            verbose=verbose
        )


    def evaluate(self,
                 smiles_ls: List[str],
                 exhaustiveness: int = 32,
                 n_poses: int = 1,
                 protonate: bool = False,
                 save_poses_dir_path: Optional[str] = None,
                 verbose = False
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

        Returns
        -------
        List of energies (affinities) in kcal/mol
        """
        save_poses_path = None
        self.smiles = smiles_ls

        if save_poses_dir_path is not None:
            dir_path = Path(save_poses_dir_path)

        energies = []
        if verbose:
            pbar = tqdm(enumerate(smiles_ls), desc=f'Docking {self.pdb_id}', total=len(smiles_ls))
        else:
            pbar = enumerate(smiles_ls)
        for i, smiles in pbar:
            if smiles in self.buffer:
                self.num_failed += 1
                self.repeats += 1
                energies.append(self.buffer[smiles])
                continue
            if smiles is None:
                energies.append(np.nan)
                self.num_failed += 1
                continue

            if save_poses_dir_path is not None:
                save_poses_path = dir_path / f'{self.pdb_id}_docked{"_prot" if protonate else ""}_{i}.pdbqt'
            try:
                energies.append(
                    self.vina_smiles(
                        ligand_smiles=smiles,
                        output_file=save_poses_path,
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                        protonate=protonate,
                        path_to_bin=self.path_to_bin,
                    )
                )
                self.buffer[smiles] = float(energies[-1])
            except Exception as _:
                energies.append(np.nan)
                self.buffer[smiles] = float(energies[-1])

        self.energies = np.array(energies)
        return energies


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
            path_to_bin=self.path_to_bin,
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
