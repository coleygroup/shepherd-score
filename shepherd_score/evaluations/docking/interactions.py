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
        interaction_count: bool = True
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
        interaction_count : bool, optional
            If True, count interactions instead of binary presence. Default is True.

        Examples
        --------
        After running a docking pipeline, use a reference ligand (docked or
        crystal pose) and compute fingerprints for all docked poses:

        >>> docking_pipe = DockingEvalPipeline(...)
        >>> docking_pipe.evaluate(...)
        >>> ref_mol = ...  # docked or crystal pose
        >>> interactions = Interactions(
        ...     "<path to pdb>",
        ...     ref_ligand=ref_mol,
        ... )
        >>> interactions.get_fingerprints(
        ...     docking_pipe.to_pandas().docked_mol.dropna()
        ... )
        >>> interactions.get_fingerprint_similarity()
        """ % (DEFAULT_INTERACTION_TYPES,)
        self.protein_pdb_path = protein_pdb_path
        p_path = Path(protein_pdb_path).resolve()
        assert p_path.is_file()

        self.protein_mol = _load_protein_mol_cached(str(p_path))

        self.fp = plf.Fingerprint(interactions=interaction_types, count=interaction_count)

        self.ref_ligand_mol = None
        self.ref_ligand_fp = None
        self.fp_ref = None
        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

    def _get_ref_ligand_fp(self, ref_ligand: Chem.Mol) -> plf.Fingerprint:
        """Compute and cache fingerprint for the reference ligand."""
        if self.fp_ref is None:
            self.fp_ref = plf.Fingerprint(list(self.fp.interactions), count=self.fp.count)
        self.ref_ligand_mol = plf.Molecule.from_rdkit(ref_ligand)
        self.ref_ligand_fp = self.fp_ref.run_from_iterable(
            [self.ref_ligand_mol], self.protein_mol, progress=False)

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

    def to_pandas(
        self,
        ref_ligand: Optional[Chem.Mol] = None,
        interaction_count: Optional[bool] = None
    ) -> pd.DataFrame:
        """Export fingerprints to a DataFrame.
        If ``Interactions`` was initialized with a ``ref_ligand``, this method will use it.


        Parameters
        ----------
        ref_ligand : Chem.Mol, optional
            If given, include reference pose as index=-1 and align column levels.
        interaction_count : bool, optional
            If True, count interactions instead of binary presence. Default is True.

        Returns
        -------
        pd.DataFrame
            Interaction matrix (poses x residue-interaction).
        """
        df = self.fp.to_dataframe(index_col="Pose", count=interaction_count)

        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

        if self.ref_ligand_fp is not None:
            df_ref = self.ref_ligand_fp.to_dataframe(index_col="Pose", count=interaction_count)
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

    def get_fingerprint_similarity(
        self,
        ref_ligand: Optional[Chem.Mol] = None,
        interaction_count: bool = True
    ) -> np.ndarray:
        """Tanimoto similarity of each pose to the reference fingerprint.
        Symmetric: both extra interactions (not in reference) and missing interactions
        (not in pose) reduce the score.
        If ``Interactions`` was initialized with a reference ligand, this method will use it.
        Otherwise, ``ref_ligand`` must be provided.

        Parameters
        ----------
        ref_ligand : Chem.Mol, optional
            Override reference ligand; must be set if not provided at init.
        interaction_count : bool
            If True, count interactions instead of binary presence. Default is True.

        Returns
        -------
        np.ndarray
            Similarities, one per pose (excluding reference).
        """
        if self.ref_ligand_fp is None and ref_ligand is None:
            raise ValueError(
                "Reference ligand fingerprint is not set. Please set a reference ligand."
            )
        df_ref_poses = self.to_pandas(ref_ligand, interaction_count)

        bitvectors = plf.to_bitvectors(df_ref_poses)
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(bitvectors[0], bitvectors[1:])
        return np.array(tanimoto_sims)

    def get_interaction_recovery(
        self,
        ref_ligand: Optional[Chem.Mol] = None
    ) -> np.ndarray:
        """Interaction count recovery of each pose relative to the reference.
        Asymmetric: only missing reference interactions reduce the score;
        extra interactions in the pose are ignored.
        If ``Interactions`` was initialized with a reference ligand, this method will use it.
        Otherwise, ``ref_ligand`` must be provided.

        Parameters
        ----------
        ref_ligand : Chem.Mol, optional
            Override reference ligand; must be set if not provided at init.

        Returns
        -------
        np.ndarray
            Recovery fraction per pose, one per pose (excluding reference).

        References
        ----------
        .. [1] Errington D. et al.J Cheminform. 2025. 17(1):76. doi: 10.1186/s13321-025-01011-6
        """
        if self.ref_ligand_fp is None and ref_ligand is None:
            raise ValueError(
                "Reference ligand fingerprint is not set. Please set a reference ligand."
            )
        if ref_ligand is not None:
            self._get_ref_ligand_fp(ref_ligand)

        ref_df = plf.to_dataframe(self.ref_ligand_fp.ifp, self.fp_ref.interactions, count=True, index_col="Pose")
        ref_counts = ref_df.droplevel("ligand", axis=1).to_dict("records")[0]
        total_ref_count = sum(ref_counts.values())

        other_df = plf.to_dataframe(self.fp.ifp, self.fp.interactions, count=True, index_col="Pose")
        all_pose_counts = other_df.droplevel("ligand", axis=1).to_dict("records")
        return np.array([
            sum(min(ref_counts[k], pose_counts.get(k, 0)) for k in ref_counts)
            / total_ref_count
            for pose_counts in all_pose_counts
        ])
