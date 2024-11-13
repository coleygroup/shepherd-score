"""
Autodock Vina Docking evaluation pipeline.

Adapted from Therapeutic Data Commons (TDC).
Huang et al. (2021) https://arxiv.org/abs/2102.09548

Requires:
- vina
- meeko
- openbabel (if protonating ligands)
"""
import os
import subprocess
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.evaluations.utils.convert_data import get_smiles_from_atom_pos

try:
    from vina import Vina
except:
    raise ImportError(
        "Please install vina following guidance in https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
    )

# Get the directory of the current file
CURRENT_DIR = Path(__file__).parent

TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])
    

# Ligands from PDB
## Docking information from TDC
## https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py
docking_target_info = {
    "1iep": {
        "center": (15.61389189189189, 53.38013513513513, 15.454837837837842),
        "size": (15, 15, 15),
        "ligand": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", # STI, Imatinib
        "pH": 6.5,
        "pdbqt": CURRENT_DIR / f'pdbs/1iep.pdbqt'
    },
    "3eml": {
        "center": (-9.063639999999998, -7.1446, 55.86259999999999),
        "size": (15, 15, 15),
        "ligand": "c1cc(oc1)c2nc3nc(nc(n3n2)N)NCCc4ccc(cc4)O", # ZMA
        "pH": 6.5,
        "pdbqt": CURRENT_DIR / f'pdbs/3eml.pdbqt'
    },
    "3ny8": {
        "center": (2.2488, 4.68495, 51.39820000000001),
        "size": (15, 15, 15),
        "ligand": "Cc1ccc(c2c1CCC2)OC[C@H]([C@H](C)NC(C)C)O", # JRZ
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / f'pdbs/3ny8.pdbqt'
    },
    "4rlu": {
        "center": (-0.7359999999999999, 22.75547368421052, -31.2368947368421),
        "size": (15, 15, 15),
        "ligand": "c1cc(ccc1C=CC(=O)c2ccc(cc2O)O)O", # HCC
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / f'pdbs/4rlu.pdbqt'
    },
    "4unn": {
        "center": (5.684346153846153, 18.191769230769232, -7.37157692307692),
        "size": (15, 15, 15),
        "ligand": "COc1cccc(c1)C2=CC(=C(C(=O)N2)C#N)c3ccc(cc3)C(=O)O", # QZZ
        "pH": 7.4, # non reported
        "pdbqt": CURRENT_DIR / f'pdbs/4unn.pdbqt'
    },
    "5mo4": {
        "center": (-44.901709677419355, 20.490354838709674, 8.483354838709678),
        "size": (15, 15, 15),
        "ligand": "c1cc(ccc1NC(=O)c2cc(c(nc2)N3CC[C@H](C3)O)c4ccn[nH]4)OC(F)(F)Cl", # AY7, asciminib
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / f'pdbs/5mo4.pdbqt'
    },
    "7l11": {
        "center": (-21.814812500000006, -4.216062499999999, -27.983781250000),
        "size": (15, 15, 15),
        "ligand": "CCCOc1cc(cc(c1)Cl)C2=CC(=CN(C2=O)c3cccnc3)c4ccccc4C#N", # XF1
        "pH": 6.0,
        "pdbqt": CURRENT_DIR / f'pdbs/7l11.pdbqt'
    },
}


def protonate_smiles(smiles: str,
                     pH: float = 7.4,
                     path_to_bin: str = ''
                     ) -> str:
    """
    Protonate SMILES string with OpenBabel at given pH.

    Adapted from DockString:
    https://github.com/dockstring/dockstring/blob/main/dockstring/utils.py#L330

    Arguments
    ---------
    smiles : str SMILES string of molecule to be protonated
    pH : float (default = 7.4) pH at which the molecule should be protonated
    path_to_bin : str (default = '') path to environment bin containing `mk_prepare_ligand.py`

    Returns
    -------
    SMILES string of protonated structure
    """

    # cmd list format raises errors, therefore one string
    cmd = f'{path_to_bin}obabel -:"{smiles}" -ismi -ocan -p {pH}'
    cmd_return = subprocess.run(cmd, capture_output=True, shell=True)
    output = cmd_return.stdout.decode('utf-8')

    if cmd_return.returncode != 0:
        raise ValueError(f'Could not protonate SMILES: {smiles}')

    return output.strip()


class VinaSmiles:
    """
    Perform docking search from a SMILES string.

    Adapted from TDC.
    """
    def __init__(self,
                 receptor_pdbqt_file: str,
                 center: Tuple[float],
                 box_size: Tuple[float],
                 pH: float = 7.4,
                 scorefunction: str = "vina",
                 num_processes: int = 4,
                 verbose: int = 0):
        """
        Constructs Vinva scoring function with receptor.

        Arguments
        ---------
        receptor_pdbqt_file : str path to .pdbqt file of receptor.
        center : Tuple[float] (len=3) coordinates for the center of the pocket.
        box_size : Tuple[float](len=3) box edge lengths of pocket.
        pH : float Experimental pH used for crystal structure elucidation.
        scorefunction : str (default=vina) name of scoring function to use with Vina. 'vina' or 'ad4'
        num_processes : int (default=2) Number of cpus to use for scoring
        verbose : int (default = 0) Level of verbosity from vina.Vina (0 is silent)
        """
        self.v = Vina(sf_name=scorefunction, seed=987654321, verbosity=verbose, cpu=num_processes)
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size
        self.pH = pH
        self.v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
        try:
            self.v.compute_vina_maps(center=self.center, box_size=self.box_size)
        except:
            raise ValueError(
                "Cannot compute the affinity map, please check center and box_size"
            )


    def __call__(self,
                 ligand_smiles: str,
                 output_file: Optional[str] = None,
                 exhaustiveness: int = 8,
                 n_poses: int = 5,
                 protonate: bool = False,
                 path_to_bin: str = ''
                 ) -> float:
        """
        Score ligand by docking in receptor.

        Arguments
        ---------
        ligand_smiles : str SMILES of ligand to dock.
        output_file : Optional[str] path to save docked poses.
        exhaustiveness : int (default = 8) Number of Monte Carlo simulations to run per pose.
        n_poses : int (default = 5) Number of poses to save.
        protonate : bool (default = False) (de-)protonate ligand with OpenBabel at pH=7.4
        path_to_bin : str (default = '') path to environment bin containing `mk_prepare_ligand.py`

        Returns
        -------
        float : energy (affinity) in kcal/mol
        """
        try:
            if protonate:
                ligand_smiles = protonate_smiles(ligand_smiles, pH=self.pH, path_to_bin=path_to_bin)
            m = Chem.MolFromSmiles(ligand_smiles)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m, randomSeed=123456789)
            AllChem.MMFFOptimizeMolecule(m)
            rand = str((os.getpid() * int(time.time())) % 123456789)
            temp_mol_file = TMPDIR / f'__temp_{rand}.mol'
            print(Chem.MolToMolBlock(m), file=open(str(temp_mol_file), "w+"))
            temp_pdbqt = TMPDIR / f'__temp_{rand}.pdbqt'
            os.system(f"{path_to_bin}mk_prepare_ligand.py -i {temp_mol_file} -o {temp_pdbqt}")
            self.v.set_ligand_from_file(str(temp_pdbqt))
            self.v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            if output_file is not None:
                self.v.write_poses(str(output_file), n_poses=n_poses, overwrite=True)
            energy = self.v.score()[0]
            os.system(f"rm {temp_mol_file} {temp_pdbqt}")
        except Exception as e:
            print(e)
            return np.nan
        return energy


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

        if path_to_bin == '':
            result = subprocess.run(['which', 'mk_prepare_ligand.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = result.stdout.decode().strip()

            if not stdout:
                raise FileNotFoundError(
                    f"`mk_prepare_ligand.py` that should have been installed with with meeko was not found."+\
                    "Install and run `which mk_prepare_ligand.py` and supply to `path_to_bin` argument."
                )

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
            except:
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

