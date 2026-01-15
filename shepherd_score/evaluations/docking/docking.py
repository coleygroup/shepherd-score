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
import time
from typing import Tuple, Optional, Literal, List
from pathlib import Path
import uuid

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.conformer_generation import update_mol_coordinates
from shepherd_score.protonation.protonate import protonate_smiles

try:
    from vina import Vina
except ImportError:
    raise ImportError(
        "Please install vina following guidance in https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
    )

try:
    from meeko import MoleculePreparation
    from meeko import PDBQTWriterLegacy
    from meeko import PDBQTMolecule
    from meeko import RDKitMolCreate

except ImportError:
    raise ImportError(
        "Please install meeko following guidance in https://meeko.readthedocs.io/en/release-doc/installation.html"
    )

def embed_conformer_from_smiles_fixed(
    smiles: str,
    attempts: int=50,
    MMFF_optimize: bool=True,
    random_seed: int=123456789,
) -> Chem.Mol:
    """
    Embeds a mol object into a 3D RDKit mol object with ETKDG (and optional MMFF94)
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=random_seed, maxAttempts=attempts)
    if MMFF_optimize:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol

class VinaBase:
    """
    Base class for Vina scoring function.
    """
    def __init__(
        self,
        receptor_pdbqt_file: str,
        center: Tuple[float],
        box_size: Tuple[float],
        pH: float = 7.4,
        scorefunction: str = "vina",
        num_processes: int = 4,
        verbose: int = 0,
        *,
        protonate_method: Literal['openbabel', 'molscrub', 'chemaxon'] = 'molscrub',
        path_to_bin: str = '',
        cxcalc_exe: str | None = None,
        molconvert_exe: str | None = None,
        chemaxon_license_path: str | None = None,
    ):
        """
        Constructs Vina scoring function with receptor.

        Arguments
        ---------
        receptor_pdbqt_file : str path to .pdbqt file of receptor.
        center : Tuple[float] (len=3) coordinates for the center of the pocket.
        box_size : Tuple[float](len=3) box edge lengths of pocket.
        pH : float Experimental pH used for crystal structure elucidation.
        scorefunction : str (default=vina) name of scoring function to use with Vina. 'vina' or 'ad4'
        num_processes : int (default=2) Number of cpus to use for scoring
        verbose : int (default = 0) Level of verbosity from vina.Vina (0 is silent)

        protonate_method : Literal['openbabel', 'molscrub', 'chemaxon'] (default = 'molscrub') method to use for protonation
        cxcalc_exe : str | None (default = None) path to cxcalc executable
        molconvert_exe : str | None (default = None) path to molconvert executable
        chemaxon_license_path : str | None (default = None) path to chemaxon license file
        """
        self.v = Vina(sf_name=scorefunction, seed=987654321, verbosity=verbose, cpu=num_processes)
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size
        self.pH = pH
        self.v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
        try:
            self.v.compute_vina_maps(center=self.center, box_size=self.box_size)
        except Exception:
            raise ValueError(
                "Cannot compute the affinity map, please check center and box_size"
            )
        self.mk_prep_ligand = MoleculePreparation()

        self.protonate_method = protonate_method
        self.path_to_bin = path_to_bin
        self.cxcalc_exe = cxcalc_exe
        self.molconvert_exe = molconvert_exe
        self.chemaxon_license_path = chemaxon_license_path

        self.state = None

    def load_ligand_from_smiles(
        self,
        ligand_smiles: str,
        protonate: bool = False,
        return_all: bool = False,
    ) -> List[Chem.Mol]:
        """
        Load ligand SMILES string into Vina.
        """
        if protonate:
            protomers = protonate_smiles(
                smiles=ligand_smiles,
                pH=self.pH,
                method=self.protonate_method,
                path_to_bin=self.path_to_bin,
                cxcalc_exe=self.cxcalc_exe,
                molconvert_exe=self.molconvert_exe,
                chemaxon_license_path=self.chemaxon_license_path,
            )
            if return_all:
                ligand_smiles = protomers
            else:
                ligand_smiles = [protomers[0]]
        else:
            ligand_smiles = [ligand_smiles]

        mols = []
        for smi in ligand_smiles:
            m = embed_conformer_from_smiles_fixed(smi, MMFF_optimize=True, random_seed=123456789)
            if m is not None:
                mols.append(m)
        return mols

    def load_ligand_from_sdf(
        self,
        sdf_file: str,
    ) -> Chem.Mol:
        """
        Load ligand from SDF file into Vina.
        """
        mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]

        if mol.GetNumConformers() == 0:
            mols = self.load_ligand_from_smiles(Chem.MolToSmiles(mol))
            if len(mols) > 0:
                mol = mols[0]
            else:
                raise ValueError(
                    f"Failed to load SDF file and could not embed conformer: {sdf_file}"
                )
        return mol

    def _prep_ligand(
        self,
        ligand: Chem.Mol,
    ) -> str | None:
        """
        Prepare ligand for docking.
        """
        try:
            molsetup = self.mk_prep_ligand.prepare(ligand)[0]
            ligand_pdbqt_string, was_successful, error_message = PDBQTWriterLegacy.write_string(molsetup)
            if not was_successful:
                print(error_message)
                return None
            return ligand_pdbqt_string
        except Exception as e:
            print(e)
            return None

    def _center_ligand(
        self,
        ligand: Chem.Mol,
        center: Tuple[float, float, float],
    ) -> Chem.Mol:
        """
        Centers a ligand conformer's center of mass to a given center.
        Returns a centered copy of the ligand.
        """
        ligand_com = ligand.GetConformer().GetPositions().mean(axis=0)
        recentered_coords = ligand.GetConformer().GetPositions() - ligand_com + center
        ligand = update_mol_coordinates(ligand, recentered_coords)
        return ligand

    def dock_ligand(
        self,
        ligand: Chem.Mol,
        output_file: Optional[str] = None,
        exhaustiveness: int = 8,
        n_poses: int = 5,
    ) -> Tuple[np.float64, np.float64, Chem.Mol] | None:
        """
        Given a ligand, do a global optimization and return the best energy and optionally the pose.

        Arguments
        ---------
        ligand : Chem.Mol ligand to dock.
        output_file : Optional[str] path to save docked poses.
        exhaustiveness : int (default = 8) Number of Monte Carlo simulations to run per pose.
        n_poses : int (default = 5) Number of poses to save.

        Returns
        -------
        Tuple
            total_energy : np.float64 energy of the best pose (kcal/mol)
            torsion_energy : np.float64 torsion energy of the best pose (kcal/mol)
            docked_rdmol : Chem.Mol rdkit mol of best docked pose
        """
        try:
            ligand_pdbqt_string = self._prep_ligand(ligand)
            self.v.set_ligand_from_string(ligand_pdbqt_string)
            self.v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            self.state = 'docked'
            if output_file is not None:
                self.v.write_poses(str(output_file), n_poses=n_poses, overwrite=True)

            # Extract docked poses to rdkit mols
            vina_output_string = self.v.poses()
            docked_pdbqt_mols = PDBQTMolecule(vina_output_string, is_dlg=False, skip_typing=False)
            docked_rdmol = RDKitMolCreate.from_pdbqt_mol(docked_pdbqt_mols)[0]

        except Exception as e:
            print(e)
            return np.nan, np.nan, None
        (total_energy, _, _, _, _, _, torsion_energy, _) = self.v.score()
        return total_energy, torsion_energy, docked_rdmol

    def score_ligand(
        self,
        ligand: Chem.Mol,
        center: bool | Tuple[float, float, float] = False,
    ) -> Tuple[np.float64, np.float64]:
        """
        Scores a given ligand's pose in it's current conformation (e.g., no optimization).

        Arguments
        ---------
        ligand : Chem.Mol ligand to score.
        center : bool | Tuple[float, float, float] | None (default = False)
            If a tuple, centers to those coordinates.
            If True, centers the ligand to the receptor's center.
            If False, does not translate the ligand from its initial conformation.

        Returns
        -------
        (total_energy, torsion_energy) : Tuple[np.float64, np.float64]
            Energies in kcal/mol
        """
        if center is True:
            ligand = self._center_ligand(ligand, self.center)
        elif isinstance(center, tuple):
            ligand = self._center_ligand(ligand, center)

        try:
            ligand_pdbqt_string = self._prep_ligand(ligand)
            self.v.set_ligand_from_string(ligand_pdbqt_string)
            (total_energy, _, _, _, _, _, torsion_energy, _) = self.v.score()
            self.state = None
        except Exception as e:
            print(e)
            return np.nan, np.nan
        return total_energy, torsion_energy

    def optimize_ligand(
        self,
        ligand: Chem.Mol,
        center: bool | Tuple[float, float, float] = False,
        max_steps: int | None = 10000,
        output_file: Optional[str] = None,
    ) -> Tuple[np.float64, np.float64, Chem.Mol]:
        """
        Locally optimize loaded ligand pose.

        Arguments
        ---------
        ligand : Chem.Mol ligand to optimize.
        center : bool | Tuple[float, float, float] | None (default = False)
            If a tuple, centers to those coordinates.
            If True, centers the ligand to the receptor's center.
            If False, does not translate the ligand from its initial conformation.
        max_steps : int | None (default = 10000) Maximum number of steps to take in the optimization.
            If None, uses the default value of 10000.
        output_file : Optional[str] path to save optimized pose.

        Returns
        -------
        Tuple
            total_energy : np.float64 energy of the best pose (kcal/mol)
            torsion_energy : np.float64 torsion energy of the best pose (kcal/mol)
            optimized_rdmol : Chem.Mol rdkit mol of optimized pose
        """
        if center is True:
            ligand = self._center_ligand(ligand, self.center)
        elif isinstance(center, tuple):
            ligand = self._center_ligand(ligand, center)

        _used_temp_file = False

        try:
            ligand_pdbqt_string = self._prep_ligand(ligand)
            self.v.set_ligand_from_string(ligand_pdbqt_string)
            (total_energy, _, _, _, _, _, torsion_energy, _) = self.v.optimize(
                max_steps=max_steps if max_steps is not None else 0,
            )
            self.state = 'optimized'
            if output_file is not None:
                self.v.write_pose(output_file, overwrite=True)

            if output_file is None:
                _used_temp_file = True
                _file_name = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
                _file_name = f'{_file_name}_optimized.pdbqt'
                if os.environ.get('TMPDIR', None) is not None:
                    _dir_path = os.environ['TMPDIR']
                elif os.environ.get('/tmp', None) is not None:
                    _dir_path = '/tmp'
                else:
                    _dir_path = './'
                temp_output_file = str(Path(_dir_path) / _file_name)
                self.v.write_pose(temp_output_file, overwrite=True)

            if output_file is None:
                output_file = temp_output_file

            pdbqt_mol_opt = PDBQTMolecule.from_file(output_file)
            rdkitmol_opt = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol_opt)[0]

            if _used_temp_file:
                os.remove(output_file)

        except Exception as e:
            print(e)
            return np.nan, np.nan, None
        return total_energy, torsion_energy, rdkitmol_opt

    def save_pose_to_file(self, output_file: str, n_poses: int = 1):
        if self.state is None:
            print("Cannot save pose in state None. Run docking or optimization first.")
            return
        if self.state == 'docked':
            self.v.write_poses(output_file, n_poses=n_poses, overwrite=True)
        elif self.state == 'optimized':
            self.v.write_pose(output_file, overwrite=True)


class VinaSmiles(VinaBase):
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
                 verbose: int = 0,
                 *,
                 protonate_method: Literal['openbabel', 'molscrub', 'chemaxon'] = 'molscrub',
                 cxcalc_exe: str | None = None,
                 molconvert_exe: str | None = None,
                 chemaxon_license_path: str | None = None,
                 ):
        """
        Constructs Vina scoring function with receptor.

        Arguments
        ---------
        receptor_pdbqt_file : str path to .pdbqt file of receptor.
        center : Tuple[float] (len=3) coordinates for the center of the pocket.
        box_size : Tuple[float](len=3) box edge lengths of pocket.
        pH : float Experimental pH used for crystal structure elucidation.
        scorefunction : str (default=vina) name of scoring function to use with Vina. 'vina' or 'ad4'
        num_processes : int (default=2) Number of cpus to use for scoring
        verbose : int (default = 0) Level of verbosity from vina.Vina (0 is silent)

        protonate_method : Literal['openbabel', 'molscrub', 'chemaxon'] (default = 'molscrub') method to use for protonation
        cxcalc_exe : str | None (default = None) path to cxcalc executable
        molconvert_exe : str | None (default = None) path to molconvert executable
        chemaxon_license_path : str | None (default = None) path to chemaxon license file
        """
        super().__init__(
            receptor_pdbqt_file=receptor_pdbqt_file,
            center=center,
            box_size=box_size,
            pH=pH,
            scorefunction=scorefunction,
            num_processes=num_processes,
            verbose=verbose,
            protonate_method=protonate_method,
            cxcalc_exe=cxcalc_exe,
            molconvert_exe=molconvert_exe,
            chemaxon_license_path=chemaxon_license_path,
        )


    def __call__(self,
                 ligand_smiles: str,
                 output_file: Optional[str] = None,
                 exhaustiveness: int = 8,
                 n_poses: int = 5,
                 protonate: bool = False,
                 return_best_protomer: bool = False,
                 ) -> Tuple[float, Chem.Mol]:
        """
        Score ligand by docking in receptor.

        Arguments
        ---------
        ligand_smiles : str SMILES of ligand to dock.
        output_file : Optional[str] path to save docked poses.
        exhaustiveness : int (default = 8) Number of Monte Carlo simulations to run per pose.
        n_poses : int (default = 5) Number of poses to save.
        protonate : bool (default = False) (de-)protonate ligand with OpenBabel at pH=7.4
        return_best_protomer: bool (default = False) Evaluate all protomers and return the best
        energy and pose / protomer which may be different from the input SMILES.

        Returns
        -------
        Tuple
            float : energy (affinity) in kcal/mol
            Chem.Mol : docked ligand
        """
        if not return_best_protomer:
            ligand = self.load_ligand_from_smiles(ligand_smiles, protonate=protonate, return_all=False)
            total_energy, _, docked_mol = self.dock_ligand(
                ligand=ligand,
                output_file=output_file,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
            )
            return total_energy, docked_mol
        else:
            protomers = self.load_ligand_from_smiles(ligand_smiles, protonate=protonate, return_all=True)
            best_energy = np.inf
            best_protomer = None
            for protomer in protomers:
                total_energy, _, docked_mol = self.dock_ligand(
                    ligand=protomer,
                    output_file=output_file,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_protomer = docked_mol
            return best_energy, best_protomer
