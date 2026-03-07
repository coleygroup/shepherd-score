"""Ligand-protein interaction analysis using ProLIF.

Requires ProLIF and MDAnalysis.
"""

from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import prolif as plf
import MDAnalysis as mda


DEFAULT_INTERACTION_TYPES = [
    "Hydrophobic",
    "HBDonor",
    "HBAcceptor",
    "PiStacking",
    "Anionic",
    "Cationic",
    "CationPi",
    "PiCation",
    'XBAcceptor',
    'XBDonor'
]


@lru_cache(maxsize=32)
def _load_protein_mol_cached(pdb_path_resolved: str) -> plf.Molecule:
    """Load protein as ProLIF Molecule. Cached by resolved path.

    Parameters
    ----------
    pdb_path_resolved : str
        Fully resolved (absolute) path to the PDB file.

    Returns
    -------
    plf.Molecule or None
        Protein as ProLIF Molecule, or None if file does not exist.
    """
    if not Path(pdb_path_resolved).exists():
        return None
    u = mda.Universe(pdb_path_resolved)
    return plf.Molecule.from_mda(u)


def add_Hs_to_ligand_from_sdf(sdf_file: str) -> Chem.Mol:
    """Load ligand from SDF and add explicit hydrogens with coordinates.
    Expects exactly one molecule in the SDF.

    Parameters
    ----------
    sdf_file : str
        Path to SDF file.

    Returns
    -------
    Chem.Mol
        Molecule with explicit H and 3D coords.
    """
    with Chem.SDMolSupplier(sdf_file) as suppl:
        mol = next(suppl)
    mol_w_h = Chem.AddHs(mol, addCoords=True)
    return mol_w_h


def get_prolif_fingerprint(ligand_sdf_path: str,
                           protein_pdb_path: str,
                           verbose: bool = False
                           ) -> Tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]:
    """Compute ProLIF interaction fingerprint for one ligand-protein pair.

    Parameters
    ----------
    ligand_sdf_path : str
        Path to ligand SDF (single molecule).
    protein_pdb_path : str
        Path to protonated protein PDB.
    verbose : bool, optional
        Show progress. Default is False.

    Returns
    -------
    tuple
        (fingerprint, ligand ProLIF molecule, protein ProLIF molecule).
    """
    p_path = Path(protein_pdb_path).resolve()
    assert p_path.is_file()
    l_path = Path(ligand_sdf_path)
    assert l_path.is_file()

    # Prep Ligand
    rdkit_mol = add_Hs_to_ligand_from_sdf(l_path)
    ligand_mol = plf.Molecule.from_rdkit(rdkit_mol)
    protein_mol = _load_protein_mol_cached(str(p_path))

    fp = plf.Fingerprint(DEFAULT_INTERACTION_TYPES)
    fp.run_from_iterable([ligand_mol], protein_mol, progress=verbose)

    return fp, ligand_mol, protein_mol


class Interactions:
    """ProLIF interaction fingerprints for multiple ligands vs one protein."""

    def __init__(
        self,
        protein_pdb_path: str,
        interaction_types: List[str] = DEFAULT_INTERACTION_TYPES,
        ref_ligand: Optional[Chem.Mol] = None,
    ):
        """
        Parameters
        ----------
        protein_pdb_path : str
            Path to protonated protein PDB file.
        interaction_types : list of str, optional
            ProLIF interaction types. Default are: %s.
        ref_ligand : Chem.Mol, optional
            Reference ligand (docked) for similarity. Default is None.
        """ % (DEFAULT_INTERACTION_TYPES,)
        self.protein_pdb_path = protein_pdb_path
        p_path = Path(protein_pdb_path).resolve()
        assert p_path.is_file()

        self.protein_mol = _load_protein_mol_cached(str(p_path))

        self.fp = plf.Fingerprint(interactions=interaction_types, count=False)

        self.ref_ligand_mol = None
        self.ref_ligand_fp = None
        self.fp_ref = None
        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

    def _get_ref_ligand_fp(self, ref_ligand: Chem.Mol) -> plf.Fingerprint:
        """Compute and cache fingerprint for the reference ligand."""
        if self.fp_ref is None:
            self.fp_ref = plf.Fingerprint(list(self.fp.interactions), count=False)
        ref_ligand_mol = plf.Molecule.from_rdkit(ref_ligand)
        self.ref_ligand_fp = self.fp_ref.run_from_iterable(
            [ref_ligand_mol], self.protein_mol, progress=False)

    def get_fingerprints(self, ligands: List[Chem.Mol]) -> Dict[int, plf.Fingerprint]:
        """Compute fingerprints for all ligands (must have explicit H).

        Parameters
        ----------
        ligands : list of Chem.Mol
            RDKit molecules with 3D coords.

        Returns
        -------
        plf.Fingerprint
            Fingerprint object with results for all poses.
        """
        plf_ligands = [plf.Molecule.from_rdkit(ligand) for ligand in ligands]
        self.fp.run_from_iterable(plf_ligands, self.protein_mol, progress=True)
        return self.fp

    def to_pandas(self, ref_ligand: Optional[Chem.Mol] = None) -> pd.DataFrame:
        """Export fingerprints to a DataFrame.
        If ``Interactions`` was initialized with a ``ref_ligand``, this method will use it.


        Parameters
        ----------
        ref_ligand : Chem.Mol, optional
            If given, include reference pose as index=-1 and align column levels.

        Returns
        -------
        pd.DataFrame
            Interaction matrix (poses x residue-interaction).
        """
        df = self.fp.to_dataframe(index_col="Pose")

        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

        if self.ref_ligand_fp is not None:
            df_ref = self.ref_ligand_fp.to_dataframe(index_col="Pose")
            df_ref.rename(index={0: -1}, inplace=True)
            # set the ligand name to be the same as poses
            df_ref.rename(columns={'UNL1': df.columns.levels[0][0]}, inplace=True)
            df_ref_poses = (
                pd.concat([df_ref, df])
                .fillna(False)
                .sort_index(
                    axis=1,
                    level=1,
                    key=lambda index: [plf.ResidueId.from_string(x) for x in index],
                )
            )
            return df_ref_poses
        return df

    def get_fingerprint_similarity(self, ref_ligand: Optional[Chem.Mol] = None) -> np.ndarray:
        """Tanimoto similarity of each pose to the reference fingerprint.
        If ``Interactions`` was initialized with a reference ligand, this method will use it.
        Otherwise, ``ref_ligand`` must be provided.

        Parameters
        ----------
        ref_ligand : Chem.Mol, optional
            Override reference ligand; must be set if not provided at init.

        Returns
        -------
        np.ndarray
            Similarities, one per pose (excluding reference).
        """
        if self.ref_ligand_fp is None and ref_ligand is not None:
            raise ValueError(
                "Reference ligand fingerprint is not set. Please set a reference ligand."
            )
        df_ref_poses = self.to_pandas(ref_ligand)

        bitvectors = plf.to_bitvectors(df_ref_poses)
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(bitvectors[0], bitvectors[1:])
        return np.array(tanimoto_sims)
