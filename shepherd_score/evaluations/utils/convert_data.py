"""
Helper functions to convert different data types.
"""

import os
from typing import Union, Tuple, List
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
import rdkit.Chem.rdDetermineBonds
import pandas as pd


def write_xyz_file(atomic_numbers: np.ndarray,
                   positions: np.ndarray,
                   path_to_file: Union[str, None] = None
                   ) -> str:
    """
    Writes an xyz file of an atomistic structure, given np.ndarray of atomic numbers and coordinates.
    
    Arguments
    ---------
    atomic_numbers : np.ndarray of shape (N,) containing atomic numbers
    positions : np.ndarray of shape (N,3) containing atomic coordinates
    path_to_file : str specifying file path -- e.g. path_to_file = 'examples/molecule.xyz'. If None, then no output file is written.
    
    Returns
    -------
    str : xyz block
    """
    N = atomic_numbers.shape[0]

    xyz = ''
    xyz += f'{N}\n\n'
    for i in range(0,N):
        a = int(atomic_numbers[i])
        p = positions[i]
        xyz+= f'{rdkit.Chem.Atom(a).GetSymbol()} {str(p[0].round(3))} {str(p[1].round(3))} {str(p[2].round(3))}\n'
    xyz+= '\n'

    if path_to_file is not None:
        with open(f'{path_to_file}', 'w') as f:
            f.write(xyz)
    return xyz


def extract_mol_from_xyz_block(xyz_block: str,
                               charge: int = 0,
                               verbose: bool = False
                               ) -> rdkit.Chem.Mol:
    """
    Attempts to extract a mol object from an xyz block.

    Assumes that the xyz structure has hydrogens included explicitly.

    Arguments
    ---------
    xyz_block: str containing atomistic structure in xyz format.
    charge: int specifying the expected (overall) charge of the structure.
    verbose: bool indicating whether to print error statements upon extraction failure

    Returns
    -------
    rdkit.Chem.rdchem.Mol object if successful, None otherwise
    """
    mol = rdkit.Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        if verbose:
            print('Mol object could not be extracted')
        return None

    try:
        rdkit.Chem.rdDetermineBonds.DetermineBonds(mol, charge=charge, embedChiral=True)
    except Exception as e:
        if verbose:
            print(e)
        return None

    num_radicals = sum([a.GetNumRadicalElectrons() for a in mol.GetAtoms()])
    if num_radicals != 0:
        if verbose:
            print('Extracted molecule has radical electrons')
        return None

    mol.UpdatePropertyCache()
    rdkit.Chem.GetSymmSSSR(mol)

    if '.' in Chem.MolToSmiles(mol):
        if verbose:
            print('Mol object was extracted but contained multiple molecules')
        return None

    num_formal_chg = 0
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            num_formal_chg += 1
        if num_formal_chg > 6:
            return None

    return mol


def get_mol_from_atom_pos(atoms: np.ndarray,
                          positions: np.ndarray
                          ) -> Tuple[Union[Chem.Mol, None], int, str]:
    """
    Try to get a RDKit mol object from atom and coordinate arrays.

    Arguments
    ---------
    atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot
        encoding.
    positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.

    Returns
    -------
    Tuple
        mol : Chem.Mol or None
        charge : int overall charge of molecule
        xyz_block : str
    """
    if len(atoms.shape) == 2:
        atomic_nums = np.argmin(np.abs(atoms - 1.0), axis = -1)
    else:
        atomic_nums = atoms
    xyz_block = write_xyz_file(atomic_nums, positions)

    for charge in [0, 1, -1, 2, -2]:
        try:
            mol = extract_mol_from_xyz_block(xyz_block=xyz_block, charge=charge)
        except Exception as e:
            mol = None

        if mol is not None:
            break
    else:
        charge = 0
    return mol, charge, xyz_block


def get_smiles_from_atom_pos(atoms: np.ndarray,
                             positions: np.ndarray
                             ) -> Union[str, None]:
    """
    Try to get a SMILES string from atom and coordinate arrays.

    Arguments
    ---------
    atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot
        encoding.
    positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.

    Returns
    -------
    SMILES str or None
    """
    mol, _, _ = get_mol_from_atom_pos(atoms=atoms, positions=positions)
    smiles = None
    if mol is not None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    return smiles



def load_npz_to_df(npz_path: Union[Path, str],
                   file_id: bool
                   ) -> pd.DataFrame:
    """
    Function to load a single npz file and return a dataframe with expanded zero-dimensional arrays.
    This works specifically for files generated by ConditionalEvalPipeline.
    """
    data = np.load(npz_path, allow_pickle=True)
    df_dict = {}

    # Find the first non-zero dimensional array length (assumed to be N_i)
    length = None
    for key, arr in data.items():
        
        if arr.ndim == 1 and len(arr) < 50:  # Non-zero dimensional array
            length = len(arr)
            break
    
    # Ensure we have a valid length for the file
    if length is None:
        raise ValueError(f"No 1D array found in {npz_path}")
    
    # Fill in the dictionary with arrays
    for key, arr in data.items():
        if key in ('ref_surf_resampling_scores', 'ref_surf_esp_resampling_scores', 'ref_mol_morgan_fp'):
            continue
        if arr.ndim == 0:  # Zero-dimensional array
            df_dict[key] = np.repeat(arr, length)  # Repeat value to match length N_i
        elif arr.ndim == 1 and len(arr) == length:  # 1D arrays with length N_i
            df_dict[key] = arr
        else:
            raise ValueError(f"Inconsistent array length for {key} in {npz_path}")
    
    if file_id is not None:
        df_dict['file_id'] = np.repeat(file_id, length)
    
    return pd.DataFrame(df_dict)


def collate_npz_files(npz_files: List[Union[str, Path]], 
                      include_file_id: bool
                      ) -> pd.DataFrame:
    """
    Function to collate all npz files into a single dataframe.

    Arguments
    ---------
    npz_files : list of file paths
    include_file_id : bool Whether to include a column called "file_id" that groups together
        rows that came from the same file.
    
    Returns
    -------
    pd.DataFrame : rows are each sample, columns are each property, and it repeats any 0d arrays.
    """
    dfs = []
    for i, npz_file in enumerate(npz_files):
        if include_file_id:
            df = load_npz_to_df(npz_file, file_id=i)
        else:
            df = load_npz_to_df(npz_file, file_id=None)
        dfs.append(df)
    
    # Concatenate all dataframes
    return pd.concat(dfs, ignore_index=True)
