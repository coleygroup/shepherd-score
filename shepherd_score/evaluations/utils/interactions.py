"""
Protein and ligand preperation functions.
Clustering of pharmacophores.

Requires Biopython, ProLIF, MDAnalysis.
"""

from pathlib import Path
from typing import Union, Tuple

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from rdkit import Chem
import prolif as plf
import MDAnalysis as mda
from sklearn.cluster import DBSCAN

from MDAnalysis.topology.guessers import guess_types


def remove_ligand_from_protein(pdb_file: str) -> None:
    """
    Remove ligand from protein to get apo structure. 
    Generate new apo pdb file.

    Saves a pdb file with "_apo" appended to the original name.

    Returns
    -------
    None
    """
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # Get the first model
    model = structure[0]

    # Iterate over chains
    for chain in model:
        # Iterate over residues
        residues = list(chain)
        for i, residue in enumerate(residues):
            # Check if the residue is a ligand (HETATM record)
            if residue.id[0] == ' ':
                continue  # Skip standard amino acids
            elif residue.id[0][0] == '*':
                continue  # Skip water molecules
            else:
                # Remove the ligand residue
                chain.detach_child(residue.id)

    load_file_path = Path(pdb_file)
    if 'bound' in load_file_path.name:
        save_file_name = load_file_path.name.replace('bound', 'apo')
    else:
        save_file_name = load_file_path.stem + '_apo.pdb'
    
    save_file = str(load_file_path.parent / save_file_name)

    # Write the apo protein structure to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(save_file)


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
    u = mda.Universe(p_path)
    # Guess elements from atom names
    elements = guess_types(u.atoms.names)
    # Assign the guessed elements to the AtomGroup
    u.add_TopologyAttr('elements', elements)
    u.atoms.guess_bonds() # Guess connectivity
    protein_mol = plf.Molecule.from_mda(u)

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


def cluster_pharm_type(combined_pharm_dict: dict,
                       pharm_type: str,
                       min_dist: float = 2,
                       min_num_points: int = 2,
                       keep_non_clustered: bool = False
                       ) -> dict:
    """
    Given a dictionary of pharmacophores, cluster and get centroids of the pharmacophores if
    `min_num_points` lie within `min_dist`
    """
    flattened_matrices = np.array(combined_pharm_dict[pharm_type]['P'])
    # Convert to a numpy array for clustering
    X = np.array(flattened_matrices)

    # Apply DBSCAN (set eps as the maximum distance for neighboring points)
    dbscan = DBSCAN(eps=min_dist, min_samples=min_num_points)  # Adjust eps and min_samples as needed
    dbscan.fit(X)

    # Get the cluster labels (noise points are labeled -1)
    labels = dbscan.labels_

    # Get unique clusters (ignoring noise points, if any)
    unique_clusters = set(labels) - {-1}

    # Compute centroids for each cluster
    centroids = {}
    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        centroids[cluster] = np.mean(cluster_points, axis=0)

    vector_matrix = np.array(combined_pharm_dict[pharm_type]['V'])

    # Average the vectors
    cluster_vecs = {}
    for i in centroids.keys():
        total_v = np.zeros(3)
        inds = np.where(np.array(labels == i))[0]
        if pharm_type.lower() == 'aromatic':
            avg_vec = average_unit_vectors(vector_matrix[inds])
        elif pharm_type.lower() == 'hydrophobe':
            pass
        else:
            for ii in inds:
                total_v += vector_matrix[ii]
            avg_vec = total_v / len(inds)
        if pharm_type.lower() == 'hydrophobe':
            cluster_vecs[i] = total_v
        else:
            cluster_vecs[i] = avg_vec / np.linalg.norm(avg_vec)

    clustered_pharms = {pharm_type : {'P': [], 'V': []}}
    for i in range(len(centroids)):
        clustered_pharms[pharm_type]['P'].append(tuple(centroids[i]))
        clustered_pharms[pharm_type]['V'].append(tuple(cluster_vecs[i]))

    if keep_non_clustered:
        noise_points = X[labels == -1]
        noise_vectors = vector_matrix[labels == -1]
        for i, noise_point in enumerate(noise_points):
            clustered_pharms[pharm_type]['P'].append(tuple(noise_point))
            clustered_pharms[pharm_type]['V'].append(tuple(noise_vectors[i]))

    return clustered_pharms


def average_unit_vectors(vectors):
    """
    Averages unit vectors, treating antiparallel vectors as aligned.

    Parameters:
        vectors (np.ndarray): A 2D array of shape (n, 3), where each row is a unit vector in 3D space.

    Returns:
        np.ndarray: The averaged unit vector.
    """
    # Use the first vector as a reference to align the others
    ref_vector = vectors[0]
    
    # Align all vectors to the reference direction
    aligned_vectors = []
    for vec in vectors:
        if np.dot(vec, ref_vector) < 0:  # Check if the vector is antiparallel
            aligned_vectors.append(-vec)  # Flip the vector
        else:
            aligned_vectors.append(vec)
    
    # Compute the average of the aligned vectors
    aligned_vectors = np.array(aligned_vectors)
    avg_vector = np.mean(aligned_vectors, axis=0)
    
    # Normalize the result to make it a unit vector
    avg_unit_vector = avg_vector / np.linalg.norm(avg_vector)
    
    return avg_unit_vector
