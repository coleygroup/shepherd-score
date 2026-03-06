"""
Protein and ligand preperation functions.
Clustering of pharmacophores.

Requires Biopython, ProLIF, MDAnalysis, PDB2PQR.
"""

from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import prolif as plf
import MDAnalysis as mda

# from MDAnalysis.topology.guessers import guess_types

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
    """Load protein as ProLIF Molecule. Cached by resolved path to avoid repeated PDB load."""
    if not Path(pdb_path_resolved).exists():
        return None
    u = mda.Universe(pdb_path_resolved)
    return plf.Molecule.from_mda(u)


def add_Hs_to_ligand_from_sdf(sdf_file: str) -> Chem.Mol:
    """
    Loads molecule from SDF file and adds hydrogens with geometry.
    Assumes only ONE ligand in the sdf file.

    Arguments
    ---------
    sdf_file : path to sdf file holding ligand.

    Returns
    -------
    rdkit Mol object containing conformer with explicit hydrogens and attributed geometry.
    """
    with Chem.SDMolSupplier(sdf_file) as suppl:
        mol = next(suppl)
    mol_w_h = Chem.AddHs(mol, addCoords=True)
    return mol_w_h


def get_prolif_fingerprint(ligand_sdf_path: str,
                           protein_pdb_path: str,
                           verbose: bool = False
                           ) -> Tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]:
    """
    Generate a ProLIF fingerprint from a ligand SDF file and protein (protonated) pdb file.

    Arguments
    ---------
    ligand_sdf_path : str path to sdf file holding ligand.
    protein_pdb_path : str path to pdb file holding protonated protein.
    verbose : bool (default = False)

    Returns
    -------
    Tuple
        fp : ProLIF Fingerprint object
        ligand_mol : ProLIF Molecule object of ligand
        protein_mol : ProLIF Molecule object of protein
    """
    p_path  = Path(protein_pdb_path)
    assert p_path.is_file()
    l_path = Path(ligand_sdf_path)
    assert l_path.is_file()

    # Prep Ligand
    rdkit_mol = add_Hs_to_ligand_from_sdf(l_path)
    ligand_mol = plf.Molecule.from_rdkit(rdkit_mol)

    # Load protein
    # u = mda.Universe(p_path)
    # # Guess elements from atom names
    # elements = guess_types(u.atoms.names)
    # # Assign the guessed elements to the AtomGroup
    # u.add_TopologyAttr('elements', elements)
    # u.atoms.guess_bonds() # Guess connectivity
    # protein_mol = plf.Molecule.from_mda(u)

    protein_mol = _load_protein_mol_cached(p_path)

    fp = plf.Fingerprint(
        [
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
    )
    fp.run_from_iterable([ligand_mol], protein_mol, progress=verbose)

    return fp, ligand_mol, protein_mol


class Interactions:
    def __init__(
        self,
        protein_pdb_path: str,
        interaction_types: List[str] = DEFAULT_INTERACTION_TYPES,
        ref_ligand: Chem.Mol | None = None,
    ):
        f"""
        Initialize the Interactions class.

        Arguments
        ---------
        protein_pdb_path : str path to pdb file holding protonated protein.
        interaction_types : List[str] list of interaction types to include in the fingerprint.
            Default is {DEFAULT_INTERACTION_TYPES}.
        ref_ligand : Chem.Mol (default = None) Reference ligand to use for interaction similarity.
            In its docked pose.
        """
        self.protein_pdb_path = protein_pdb_path
        p_path  = Path(protein_pdb_path)
        assert p_path.is_file()

        # u = mda.Universe(p_path)
        # # Guess elements from atom names
        # elements = guess_types(u.atoms.names)
        # # Assign the guessed elements to the AtomGroup
        # u.add_TopologyAttr('elements', elements)
        # u.atoms.guess_bonds() # Guess connectivity
        # self.protein_mol = plf.Molecule.from_mda(u)
        self.protein_mol = _load_protein_mol_cached(p_path)

        self.fp = plf.Fingerprint(interactions=interaction_types, count=False)

        self.ref_ligand_mol = None
        self.ref_ligand_fp = None
        self.fp_ref = None
        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

    def _get_ref_ligand_fp(self, ref_ligand: Chem.Mol) -> plf.Fingerprint:
        """
        Get the fingerprint of the reference ligand.
        """
        if self.fp_ref is None:
            self.fp_ref = plf.Fingerprint(list(self.fp.interactions), count=False)
        ref_ligand_mol = plf.Molecule.from_rdkit(ref_ligand)
        self.ref_ligand_fp = self.fp_ref.run_from_iterable(
            [ref_ligand_mol], self.protein_mol, progress=False)

    def get_fingerprints(self, ligands: List[Chem.Mol]) -> Dict[int, plf.Fingerprint]:
        """
        Get fingerprint for a list of ligands. Assumes ligands have explicit hydrogens.
        """
        plf_ligands = [plf.Molecule.from_rdkit(ligand) for ligand in ligands]
        self.fp.run_from_iterable(plf_ligands, self.protein_mol, progress=True)
        return self.fp

    def to_pandas(self, ref_ligand: Chem.Mol | None = None) -> pd.DataFrame:
        """
        Convert the fingerprints to a pandas dataframe.
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


    def get_fingerprint_similarity(self, ref_ligand: Chem.Mol | None = None) -> np.ndarray:
        """
        Get the similarity between the fingerprint of the reference ligand and the ligand.
        """
        if self.ref_ligand_fp is None and ref_ligand is not None:
            raise ValueError(
                "Reference ligand fingerprint is not set. Please set a reference ligand."
            )
        df_ref_poses = self.to_pandas(ref_ligand)

        bitvectors = plf.to_bitvectors(df_ref_poses)
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(bitvectors[0], bitvectors[1:])
        return np.array(tanimoto_sims)
