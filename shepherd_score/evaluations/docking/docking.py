"""AutoDock Vina docking evaluation pipeline.

VinaSmiles class adapted from Therapeutic Data Commons (TDC) [1].

Requires: vina, meeko; openbabel only if protonating ligands.

References
----------
.. [1] Huang et al. (2021) https://arxiv.org/abs/2102.09548
"""
import os
import time
from typing import Tuple, Optional, Literal, List, Union
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
    """Embed SMILES into a 3D RDKit mol with ETKDG and optional MMFF94.

    Parameters
    ----------
    smiles : str
        SMILES string.
    attempts : int, optional
        Max embedding attempts. Default is 50.
    MMFF_optimize : bool, optional
        Run MMFF94 optimization. Default is True.
    random_seed : int, optional
        Random seed for embedding. Default is 123456789.

    Returns
    -------
    Chem.Mol
        Molecule with 3D conformer.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=random_seed, maxAttempts=attempts)
    if MMFF_optimize:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol

class VinaBase:
    """Base class for Vina scoring and docking."""

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
        cxcalc_exe: Optional[str] = None,
        molconvert_exe: Optional[str] = None,
        chemaxon_license_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        receptor_pdbqt_file : str
            Path to receptor PDBQT file.
        center : tuple of float, length 3
            Pocket center coordinates.
        box_size : tuple of float, length 3
            Search box edge lengths.
        pH : float, optional
            pH for protonation. Default is 7.4.
        scorefunction : str, optional
            'vina' or 'ad4'. Default is 'vina'.
        num_processes : int, optional
            CPUs for scoring. Default is 4.
        verbose : int, optional
            Vina verbosity (0 = silent). Default is 0.
        protonate_method : {'openbabel', 'molscrub', 'chemaxon'}, optional
            Protonation method. Default is 'molscrub'.
        path_to_bin : str, optional
            Path to OpenBabel binaries. Default is ''.
        cxcalc_exe : str or None, optional
            Path to cxcalc. Default is None.
        molconvert_exe : str or None, optional
            Path to molconvert. Default is None.
        chemaxon_license_path : str or None, optional
            Path to ChemAxon license. Default is None.
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
        """Load ligand from SMILES; optionally protonate and embed.

        Parameters
        ----------
        ligand_smiles : str
            SMILES string.
        protonate : bool, optional
            Protonate at instance pH. Default is False.
        return_all : bool, optional
            If True and protonate=True, return all protomers. Default is False.

        Returns
        -------
        list of Chem.Mol
            RDKit mols with 3D conformers.
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
        """Load ligand from SDF; embed from SMILES if no conformer.

        Parameters
        ----------
        sdf_file : str
            Path to SDF file.

        Returns
        -------
        Chem.Mol
            Molecule with 3D coords.

        Raises
        ------
        ValueError
            If SDF has no conformer and embedding fails.
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
    ) -> Optional[str]:
        """Convert ligand to PDBQT string for Vina. Returns None on failure."""
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
        """Center ligand COM to given coordinates; return a copy.

        Parameters
        ----------
        ligand : Chem.Mol
            Molecule with conformer.
        center : tuple of float, length 3
            Target center (x, y, z).

        Returns
        -------
        Chem.Mol
            Copy with updated coordinates.
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
    ) -> Optional[Tuple[np.float64, np.float64, Chem.Mol]]:
        """Given a ligand, do a global optimization and return the best energy and optionally the
        pose.

        Parameters
        ----------
        ligand : Chem.Mol
            Ligand to dock.
        output_file : str or None, optional
            Path to save poses. Default is None.
        exhaustiveness : int, optional
            Monte Carlo runs per pose. Default is 8.
        n_poses : int, optional
            Number of poses to save. Default is 5.

        Returns
        -------
        tuple or None
            (total_energy, torsion_energy, docked_mol) in kcal/mol, or None on failure.
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
        center: Union[bool, Tuple[float, float, float]] = False,
    ) -> Tuple[np.float64, np.float64]:
        """Score ligand in current conformation (no optimization).

        Parameters
        ----------
        ligand : Chem.Mol
            Ligand to score.
        center : bool or tuple of float, optional
            If True, center to receptor box. If tuple (x,y,z), center there.
            If False, use current coords. Default is False.

        Returns
        -------
        tuple of np.float64
            (total_energy, torsion_energy) in kcal/mol.
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
        center: Union[bool, Tuple[float, float, float]] = False,
        max_steps: Optional[int] = 10000,
        output_file: Optional[str] = None,
    ) -> Tuple[np.float64, np.float64, Chem.Mol]:
        """Locally optimize ligand pose in the binding site.

        Parameters
        ----------
        ligand : Chem.Mol
            Ligand to optimize.
        center : bool or tuple of float, optional
            If True, center to receptor box. If tuple (x,y,z), center there.
            If False, use current coords. Default is False.
        max_steps : int or None, optional
            Max optimization steps. None uses Vina default. Default is 10000.
        output_file : str or None, optional
            Path to save optimized pose. Default is None.

        Returns
        -------
        tuple
            (total_energy, torsion_energy, optimized_mol) in kcal/mol.
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
        """Write current pose(s) to file.

        Parameters
        ----------
        output_file : str
            Output path.
        n_poses : int, optional
            Number of poses (only when state is 'docked'). Default is 1.
        """
        if self.state is None:
            print("Cannot save pose in state None. Run docking or optimization first.")
            return
        if self.state == 'docked':
            self.v.write_poses(output_file, n_poses=n_poses, overwrite=True)
        elif self.state == 'optimized':
            self.v.write_pose(output_file, overwrite=True)


class VinaSmiles(VinaBase):
    """Docking from SMILES (embed + optional protonation). Adapted from TDC."""

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
                 cxcalc_exe: Optional[str] = None,
                 molconvert_exe: Optional[str] = None,
                 chemaxon_license_path: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        receptor_pdbqt_file : str
            Path to receptor PDBQT file.
        center : tuple of float, length 3
            Pocket center coordinates.
        box_size : tuple of float, length 3
            Search box edge lengths.
        pH : float, optional
            pH for protonation. Default is 7.4.
        scorefunction : str, optional
            'vina' or 'ad4'. Default is 'vina'.
        num_processes : int, optional
            CPUs for scoring. Default is 4.
        verbose : int, optional
            Vina verbosity (0 = silent). Default is 0.
        protonate_method : {'openbabel', 'molscrub', 'chemaxon'}, optional
            Protonation method. Default is 'molscrub'.
        cxcalc_exe : str or None, optional
            Path to cxcalc. Default is None.
        molconvert_exe : str or None, optional
            Path to molconvert. Default is None.
        chemaxon_license_path : str or None, optional
            Path to ChemAxon license. Default is None.
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
        """Dock ligand SMILES in receptor; return best energy and pose.

        Parameters
        ----------
        ligand_smiles : str
            SMILES of ligand to dock.
        output_file : str or None, optional
            Path to save poses. Default is None.
        exhaustiveness : int, optional
            Monte Carlo runs per pose. Default is 8.
        n_poses : int, optional
            Number of poses to save. Default is 5.
        protonate : bool, optional
            Protonate at instance pH. Default is False.
        return_best_protomer : bool, optional
            If True, dock all protomers and return best by energy. Default is False.
            Returned SMILES may be different from the input SMILES due to protonation.

        Returns
        -------
        tuple
            (energy in kcal/mol, docked Chem.Mol).
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
