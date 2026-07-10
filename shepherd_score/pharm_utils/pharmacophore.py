"""
Generate pharmacophores from a RDKit conformer.

Parts of code adapted from Francois Berenger / Tsuda Lab and RDKit.

References:

- Tsuda Lab: https://github.com/tsudalab/ACP4/blob/master/bin/acp4_ph4.py
  (From https://doi.org/10.1021/acs.jcim.2c01623)
- RDKit: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/FeatDirUtilsRD.py
- RDKit: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/ShowFeats.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Union

import numpy as np
from scipy.spatial import distance, Delaunay

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# pharmacophores
from shepherd_score.pharm_utils.pharmvec import GetDonorFeatVects, GetAcceptorFeatVects, GetAromaticFeatVects, GetHalogenFeatVects
from shepherd_score.score.constants import P_TYPES

PT = Chem.GetPeriodicTable()

feature_colors = {
  'Donor': (0, 1, 1),
  'Acceptor': (1, 0, 1),
  'NegIonizable': (1, 0, 0),
  'Anion': (1,0,0),
  'PosIonizable': (0, 0, 1),
  'Cation': (0,0,1),
  'ZnBinder': (1, .5, .5),
  'Zn': (1, .5, .5),
  'Aromatic': (1, .8, .2),
  'LumpedHydrophobe': (.5, .25, 0),
  'Hydrophobe': (.5, .25, 0),
  'Halogen': (.13, .55, .13),
  'Dummy': (0., .4, .55)
}

# Below is used to get hydrophobic groups
#### From https://github.com/tsudalab/ACP4/blob/master/bin/acp4_ph4.py ####
#### Credit to Francois Berenger and Tsuda Lab ####
#### https://doi.org/10.1021/acs.jcim.2c01623 ####

# These are the same as Pharmer / Pmapper
__hydrophobic_smarts = [
    "a1aaaaa1",
    "a1aaaa1",
    # branched terminals as one point
    "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
    "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    # simple rings only; need to combine points to get good results for 3d structures
    "[C&r3]1~[C&r3]~[C&r3]1",
    "[C&r4]1~[C&r4]~[C&r4]~[C&r4]1",
    "[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]1",
    "[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]1",
    "[C&r7]1~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]1",
    "[C&r8]1~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]1",
    # aliphatic chains
    "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    "[$([CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    # sulfur (apparently)
    "[$([S]~[#6])&!$(S~[!#6])]"
]

def pattern_of_smarts(s):
    return Chem.MolFromSmarts(s)

__hydrophobic_patterns = list(map(pattern_of_smarts, __hydrophobic_smarts))

# geometric center of a matched pattern
def __average_match(conf, matched_pattern):
    avg_x = 0.0
    avg_y = 0.0
    avg_z = 0.0
    count = float(len(matched_pattern))
    for i in matched_pattern:
        xyz = conf.GetAtomPosition(i)
        avg_x += xyz.x
        avg_y += xyz.y
        avg_z += xyz.z
    center = (avg_x / count,
              avg_y / count,
              avg_z / count)
    return center

def __find_matches(mol, patterns, return_atom_ids: bool = False):
    res = []
    conf = mol.GetConformer()
    for pat in patterns:
        # get all matches for that pattern
        matched = mol.GetSubstructMatches(pat)
        for m in matched:
            # get the center of each matched group
            avg = __average_match(conf, m)
            if return_atom_ids:
                # Derive aromaticity from the matched atoms themselves rather than the
                # SMARTS text, so it stays correct if __hydrophobic_smarts is edited.
                is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in m)
                res.append((avg, set(m), is_aromatic))
            else:
                res.append(avg)
    return res

def __average(vecs):
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    n = float(len(vecs))
    for (x, y, z) in vecs:
        sum_x += x
        sum_y += y
        sum_z += z
    return (sum_x / n,
            sum_y / n,
            sum_z / n)

def _rdkit_point3d_to_tuple(point: Chem.Geometry.Point3D):
    """
    Convert an rdkit Point3D to a tuple.

    For reasons I can not explain, it's 1000x faster to convert this way instead of
    calling tuple(point)

    def pt_to_tuple(pt):
        return (pt.x, pt.y, pt.z)
    %timeit tuple(pt)
    %timeit pt_to_tuple(pt)

    Gives:
    527 μs ± 16.1 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    252 ns ± 1.77 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    """
    return (point.x, point.y, point.z)

def _copy_point3d(point: Chem.Geometry.Point3D) -> Chem.Geometry.Point3D:
    """
    Independent copy of a Point3D, ~6x faster than copy.deepcopy(point) since it
    skips the generic (memo dict, __reduce_ex__ lookup, ...) deepcopy machinery
    for what is just 3 floats.
    """
    return Chem.rdGeometry.Point3D(point.x, point.y, point.z)

def find_hydrophobes(mol: rdkit.Chem.rdchem.Mol,
                     cluster_hydrophobic: bool = True,
                     return_atom_ids: bool = False):
    """
    Find hydrophobes and cluster them.

    Arguments
    ---------
    mol : rdkit Mol object with a conformer.
    cluster_hydrophobic : bool (default=True) to cluster hydrophobic atoms if they fall within 2A.
    return_atom_ids : bool (default=False)
        When True, returns a list of ``(center_tuple, aromatic_atom_ids_set)`` pairs instead of
        plain center tuples. ``aromatic_atom_ids_set`` is the union of atom indices from
        aromatic-ring matches that ended up in the cluster; it is an empty set for clusters
        composed entirely of non-aromatic matches.

    Returns
    -------
    list of tuples containing coordinates for the locations for each hydrophobe,
    or (when ``return_atom_ids=True``) a list of ``(center, set[int])`` pairs.
    """
    all_hydrophobes = __find_matches(mol, __hydrophobic_patterns,
                                     return_atom_ids=return_atom_ids)
    if not cluster_hydrophobic:
        return all_hydrophobes

    # Extract centers regardless of return_atom_ids; when True each entry is
    # (center, atom_ids, is_aromatic), otherwise each entry is already the center tuple.
    centers = [h[0] for h in all_hydrophobes] if return_atom_ids else all_hydrophobes
    n = len(all_hydrophobes)
    idx2cluster = list(range(n))
    if n > 1:
        # Precompute all pairwise distances in one vectorized call
        within_cutoff = distance.squareform(distance.pdist(np.asarray(centers))) <= 2.0
        for i in range(n):
            cluster_id = idx2cluster[i]
            for j in range(i + 1, n):
                if within_cutoff[i, j]:
                    idx2cluster[j] = cluster_id

    grouped = []
    for cid in set(idx2cluster):
        group_centers = []
        aromatic_ids = set()
        for i, h in enumerate(all_hydrophobes):
            if idx2cluster[i] != cid:
                continue
            group_centers.append(centers[i])
            if return_atom_ids and h[2]:  # h[2] is is_aromatic
                aromatic_ids |= h[1]
        avg = __average(group_centers)
        grouped.append((avg, aromatic_ids) if return_atom_ids else avg)
    return grouped

### End Tsuda Lab code


def _get_points_fibonacci(num_samples):
    """
    Generate points on unit sphere using fibonacci approach.
    Adapted from Morfeus:
    https://github.com/digital-chemistry-laboratory/morfeus/blob/main/morfeus/geometry.py

    Parameters
    ----------
    num_samples : int
        Number of points to sample from the surface of a sphere

    Returns
    -------
    np.ndarray (num_samples,3)
        Coordinates of the sampled points.
    """
    offset = 2.0 / num_samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(num_samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - np.square(y))
    phi = np.mod((i + 1), num_samples) * increment
    x = np.cos(phi) * r
    z = np.sin(phi) * r

    points = np.column_stack((x, y, z))
    return points


def __outside_hull(sample_points: np.ndarray,
                   hull: Delaunay
                   ) -> np.ndarray:
    """
    Test if points in `sample_points` are outside of the convex hull formed by the atoms.

    Arguments
    ---------
    sample_points : (N,3) np.ndarray of the points to check if outside the "interior" of the molecule.
    hull : scipy.spatial.Delaunay object initialized by the positions of the atoms of the molecule.

    Returns
    -------
    (N,) np.ndarray of booleans describing if sample_points are outside of the convex hull
    """
    return hull.find_simplex(sample_points) < 0


def __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx):
    """
    Check if at least 2% of sampled points fall within a surface-accessible volume of the molecule.
     This is 2% of the original 200 points (4 points).
    Currently using SAS with a probe radius of 0.8A rather than vdW volume. vdW volume will fail to
     exclude buried pharmacophores. Also experimented with checking if the interaction points fell
     within a convex hull and buried volume with Morpheus which both had limited improvements.

    Arguments
    ---------
    interaction_sphere : np.ndarray (M, 3) of points to check accessibility of a potentially
                    interacting atom. M <= 200
    atom_pos : np.ndarray (N, 3) Positions of atoms in molecule.
    radii : np.ndarray (N,) vdW radii for each corresponding atom.
    mask_atom_idx : np.ndarray of bool (N,) contains atom indices to ignore if the interaction
                    points are within their SA volumes. For example, the acceptor atom or the
                    donating hydrogens.

    Returns
    -------
    bool
    """
    # compute distances from each sampled point to all atoms (except excluded)
    dist_matrix = distance.cdist(interaction_sphere, atom_pos[mask_atom_idx])
    mask = np.all(dist_matrix >= radii + 0.8, axis=1) # mask for points within vdW + probe radius
    interaction_sphere = interaction_sphere[mask]

    # if hull is not None:
    #      # If you actually want to include this, then only compute Delaunay ONCE per molecule (outside this func).
    #     hull = Delaunay(mol.GetConformer().GetPositions())
    #     sas_mask = np.all(dist_matrix[mask] >= radii + 0.8, axis=1) # points within SAS defined volume
    #     hull_mask = __outside_hull(interaction_sphere, hull).astype(bool) # points within hull
    #     interaction_sphere = interaction_sphere[hull_mask | sas_mask]

    num_accessible = len(interaction_sphere) # number of non-colliding points
    if num_accessible > 4: # at least 2% accessible from initial total 200 points
        return True
    else:
        return False


def _is_donator_accessible(mol: rdkit.Chem.rdchem.Mol,
                           hydrogens: Union[List[rdkit.Chem.rdchem.Atom], None],
                           pharm_pos: Tuple,
                           unit_vec: Tuple,
                           ) -> bool:
    """
    Check accessbility of donator atoms inspired by protocol of Pharao.
    DOI: 10.1016/j.jmgm.2008.04.003
    Check whether at least 2% of the points sampled on a sphere of 1.8A radius is accessible.
        i.e., beyond the SAS
    Arguments
    ---------
    mol : rdkit Mol with conformer
    pharm_pos : tuple holding coords of anchor point
    unit_vec : tuple holding coords of releative unit vector
    num_nbrs : int of the number of neighbors to the acceptor (heavy + hydr)

    Returns
    -------
    bool
    """
    if hydrogens is None:
        hyd_atom_ids = []
    else:
        hyd_atom_ids = [h.GetIdx() for h in hydrogens]
    hyd_atom_ids_set = set(hyd_atom_ids)
    radii = np.array([PT.GetRvdw(atom.GetAtomicNum()) for i, atom in enumerate(mol.GetAtoms()) if i not in hyd_atom_ids_set])

    # Pharmacophore position is about 1.2A in direction of vector
    pharm_pos = np.array(pharm_pos) + 1.2*np.array(unit_vec)

    # unit sphere
    interaction_sphere = _get_points_fibonacci(200)
    interaction_radius = 1.8 # angstroms
    interaction_sphere *= interaction_radius
    interaction_sphere += pharm_pos # move to position of pharmacophore

    atom_pos = mol.GetConformer().GetPositions()
    # don't include the hydrogens themselves
    mask_atom_idx = np.isin(np.arange(len(atom_pos)), hyd_atom_ids, invert=True)
    return __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx)


def _is_acceptor_accessible(mol: rdkit.Chem.rdchem.Mol,
                            acceptor_atom: rdkit.Chem.rdchem.Atom,
                            pharm_pos: Tuple,
                            unit_vec: Tuple,
                            num_nbrs: int,
                            ) -> bool:
    """
    Check accessbility of acceptor atoms inspired by protocol of Pharao.
    DOI: 10.1016/j.jmgm.2008.04.003
    Check whether at least 2% of the points sampled on a sphere of 1.8A radius is accessible.
        i.e., beyond the SAS

    Arguments
    ---------
    mol : rdkit Mol with conformer
    acceptor_atom : rdkit Atom from mol that is the acceptor
    pharm_pos : tuple holding coords of anchor point
    unit_vec : tuple holding coords of releative unit vector
    num_nbrs : int of the number of neighbors to the acceptor (heavy + hydr)

    Returns
    -------
    bool
    """
    acceptor_atom_id = acceptor_atom.GetIdx()
    radii = np.array([PT.GetRvdw(atom.GetAtomicNum()) for i, atom in enumerate(mol.GetAtoms()) if i != acceptor_atom_id])

    pharm_pos = np.array(pharm_pos)

    # unit sphere
    interaction_sphere = _get_points_fibonacci(200)

    # mask out irrelevant parts of the sphere
    if num_nbrs >= 3:
        # hemisphere
        vec = np.array(unit_vec)
        inds = np.where(np.dot(vec, interaction_sphere.T) > 0)[0]
        interaction_sphere = interaction_sphere[inds]
    elif num_nbrs == 2:
        # Little more than a hemisphere, sqrt(2)/2 = -0.7071 -> 180+45 deg
        vec = np.array(unit_vec)
        inds = np.where(np.dot(vec, interaction_sphere.T) > -0.7071)[0]
        interaction_sphere = interaction_sphere[inds]
    # otherwise full sphere

    interaction_radius = 1.8 # angstroms
    interaction_sphere *= interaction_radius
    interaction_sphere += pharm_pos # move to position of pharmacophore

    atom_pos = mol.GetConformer().GetPositions()
    # don't include the atom itself
    mask_atom_idx = np.where(np.arange(len(atom_pos)) != acceptor_atom_id)[0]
    return __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx)


### From rdkit:
# https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/FeatDirUtilsRD.py
# https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/ShowFeats.py


def _average_vectors(vectors: List):
    """
    Arguments
    ---------
    vectors : List of rdkit geometry point3d objects. These should be unit vectors.

    Returns
    -------
    rdkit.Geometry.rdGeometry.Point3D object that is an average of the provided vectors.
    """
    avg_vec = 0
    for v in vectors:
        if avg_vec == 0:
            avg_vec = _copy_point3d(v)
        else:
            avg_vec += v
    avg_vec.Normalize()
    return avg_vec


# Lazily create and cache the feature factory
_cached_factory: rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeatureFactory | None = (
    None
)

def get_pharmacophores_dict(mol: rdkit.Chem.rdchem.Mol,
                            multi_vector: bool = True,
                            exclude: List[int] = [],
                            check_access: bool = False,
                            scale: float = 1.0,
                            return_atom_ids: bool = False,
                            ) -> Dict:
    """
    Get the positions of pharmacophore anchors and their associated unit vectors.

    Returns a dictionary. Adapted from rdkit.Chem.Features.ShowFeats.ShowMolFeats.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit Mol object with a conformer.
    multi_vector : bool, optional
        Whether to represent pharmacophores with multiple vectors. Default is ``True``.
    exclude : list, optional
        List of atom indices to not include as a HBD. Default is [].
    check_access : bool, optional
        Check if HBD/HBA are accessible to the molecular surface. Default is ``False``.
    scale : float, optional
        Length of the vector in Angstroms. Default is 1.0.
    return_atom_ids : bool, optional
        When ``True``, each family sub-dict also contains an ``'A'`` key holding a list of
        atom-id sets (one set per emitted pharmacophore, aligned with ``'P'``). For
        hydrophobes, only aromatic-ring-derived clusters carry non-empty sets; aliphatic
        clusters have empty sets. Default is ``False``.

    Returns
    -------
    dict
        Dictionary with format ``{'FeatureName': {'P': [(anchor coord), ...],
        'V': [(rel. vec), ...]}}``.  When ``return_atom_ids=True``, each entry also has
        ``'A': [set_of_int, ...]`` aligned with ``'P'``.
    """
    global _cached_factory
    pharmacophores = {}

    if _cached_factory is None:
        dirname = os.path.dirname(__file__)
        fdef_file = os.path.join(dirname, "smarts_features.fdef")
        _cached_factory = AllChem.BuildFeatureFactory(fdef_file)

    mol_feats = _cached_factory.GetFeaturesForMol(mol)
    conf = mol.GetConformer()

    # Filter only these for rdkit processing, we will compute hydrophobes later
    keep = ('Aromatic', 'ZnBinder', 'Donor', 'Acceptor', 'Cation', 'Anion', 'Halogen')

    # Non-hydrophobe pharmacophore processing
    for feat in mol_feats:
        family = feat.GetFamily() # type of pharmacophore
        if family not in keep:
          continue
        if family not in pharmacophores:
            pharmacophores[family] = {'P': [], 'V': []}
            if return_atom_ids:
                pharmacophores[family]['A'] = []

        if return_atom_ids:
            feat_atom_ids = set(feat.GetAtomIds())

        if family == 'Aromatic':
            pos = feat.GetPos()
            anchor, vec = GetAromaticFeatVects(conf = conf,
                                               featAtoms = feat.GetAtomIds(),
                                               featLoc = pos,
                                               return_both = multi_vector,
                                               scale = scale)
            if not multi_vector:
                anchor = anchor[0]
                vec = vec[0]

        elif family == 'Donor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                # Multivector by default
                anchor, vec, hydrogen_list = GetDonorFeatVects(conf = conf,
                                                               featAtoms = aids,
                                                               scale = scale,
                                                               exclude = exclude)
                if vec is not None and len(vec) > 1:
                    avg_vec = _average_vectors(vec)
                else:
                    if vec is None:
                        avg_vec = None
                    else:
                        avg_vec = _copy_point3d(vec[0])

                if check_access:
                    if anchor is None or avg_vec is None:
                        continue
                    # Convert Point3D -> tuple; see _rdkit_point3d_to_tuple.
                    anchor_pt = anchor if not isinstance(anchor, list) else anchor[0]
                    if not _is_donator_accessible(mol = mol,
                                                  hydrogens = hydrogen_list,
                                                  pharm_pos = _rdkit_point3d_to_tuple(anchor_pt),
                                                  unit_vec = _rdkit_point3d_to_tuple(avg_vec)
                                                  ):
                        continue # don't keep this pharmacophore

                # If only one vector per pharmacophore
                if not multi_vector and anchor is not None:
                    anchor = anchor[0]
                    vec = _copy_point3d(avg_vec)

        elif family == 'Acceptor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                # Multivector by default
                anchor, vec = GetAcceptorFeatVects(conf = conf,
                                                   featAtoms = aids,
                                                   scale = scale)

                if vec is not None and len(vec) > 1:
                    avg_vec = _average_vectors(vec)
                else:
                    if vec is None:
                        avg_vec = None
                    else:
                        avg_vec = _copy_point3d(vec[0])

                if check_access:
                    if anchor is None or avg_vec is None:
                        continue
                    numNbrs = len(featAtom.GetNeighbors())
                    anchor_pt = anchor if not isinstance(anchor, list) else anchor[0]
                    if not _is_acceptor_accessible(mol = mol,
                                                   acceptor_atom = featAtom,
                                                   pharm_pos = _rdkit_point3d_to_tuple(anchor_pt),
                                                   unit_vec = _rdkit_point3d_to_tuple(avg_vec),
                                                   num_nbrs = numNbrs):
                        continue # don't keep this pharmacophore

                # If only one vector per pharmacophore
                if not multi_vector and anchor is not None:
                    anchor = anchor[0]
                    vec = _copy_point3d(avg_vec)

        elif family == 'Halogen':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                anchor, vec = GetHalogenFeatVects(conf = conf,
                                                  featAtoms = aids,
                                                  scale = scale)
                anchor = anchor[0]
                vec = vec[0]

        else:
            anchor = feat.GetPos()
            vec = Chem.rdGeometry.Point3D(0,0,0)

        if anchor is not None and vec is not None:
            if isinstance(anchor, list):
                pharmacophores[family]['P'].extend(_rdkit_point3d_to_tuple(x) for x in anchor)
                pharmacophores[family]['V'].extend(_rdkit_point3d_to_tuple(x) for x in vec)
                if return_atom_ids:
                    pharmacophores[family]['A'].extend(feat_atom_ids for _ in anchor)
            else:
                pharmacophores[family]['P'].append(_rdkit_point3d_to_tuple(anchor))
                pharmacophores[family]['V'].append(_rdkit_point3d_to_tuple(vec))
                if return_atom_ids:
                    pharmacophores[family]['A'].append(feat_atom_ids)

    # Hydrophobe processing
    hydrophobes_raw = find_hydrophobes(mol=mol, cluster_hydrophobic=True,
                                       return_atom_ids=return_atom_ids)
    if return_atom_ids:
        hydrophobe_centers = [entry[0] for entry in hydrophobes_raw]
        hydrophobe_atom_ids = [entry[1] for entry in hydrophobes_raw]
    else:
        hydrophobe_centers = hydrophobes_raw
        hydrophobe_atom_ids = None
    pharmacophores['Hydrophobe'] = {
        'P': hydrophobe_centers,
        'V': [(0, 0, 0)] * len(hydrophobe_centers),
    }
    if return_atom_ids:
        pharmacophores['Hydrophobe']['A'] = hydrophobe_atom_ids
    return pharmacophores


_RING_PRIORITY_TYPE_INDICES = frozenset({
    P_TYPES.index('Aromatic'),
    P_TYPES.index('Hydrophobe'),
})


def _heavy_atoms_in_ring(mol: rdkit.Chem.rdchem.Mol, ring: Tuple[int, ...]) -> set[int]:
    return {i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}


def _max_priority_atoms_in_shared_rings(ring_heavy_sets: List[set],
                                        atom_ids: set[int],
                                        priority_atoms: set[int]) -> int:
    """
    Return the maximum number of priority atoms found in any single ring that
    overlaps ``atom_ids``.

    ``ring_heavy_sets`` is the list of per-ring heavy-atom sets, precomputed once
    per molecule (see :func:`priority_pharm_labels`) so that ring info is not
    rebuilt for every pharmacophore.
    """
    if not atom_ids or not priority_atoms:
        return 0
    best = 0
    for heavy in ring_heavy_sets:
        if not (heavy & atom_ids):
            continue
        best = max(best, len(heavy & priority_atoms))
    return best


def priority_pharm_labels(mol: rdkit.Chem.rdchem.Mol,
                          atom_ids_per_pharm: List[set],
                          pharm_types: np.ndarray,
                          priority_atoms: Iterable[int],
                          min_ring_priority_atoms: int = 3) -> np.ndarray:
    """
    Compute a 0/1 priority label for each pharmacophore.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule used to resolve ring membership for aromatic/hydrophobe labels.
    atom_ids_per_pharm : list of sets
        One set of atom indices per pharmacophore, aligned with the X/P/V arrays.
    pharm_types : np.ndarray
        Pharmacophore type indices aligned with ``atom_ids_per_pharm`` (same order
        as ``P_TYPES``).
    priority_atoms : iterable of int
        Atom indices considered "priority".
    min_ring_priority_atoms : int, optional
        Minimum number of heavy ring atoms that must also be in ``priority_atoms``
        before an aromatic or aromatic-derived hydrophobe is labeled 1. Use ``1`` to
        treat any single priority atom in the ring as sufficient. Default is ``3``.

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        1 where the pharmacophore is priority, else 0. Non-ring pharmacophores use
        simple atom-id intersection. Aromatic and aromatic-derived hydrophobe
        pharmacophores additionally require at least ``min_ring_priority_atoms`` heavy
        atoms from a shared ring to appear in ``priority_atoms``.
    """
    priority = {int(a) for a in priority_atoms}
    ring_heavy_sets: Optional[List[set]] = None
    labels = []
    for aids, pharm_type in zip(atom_ids_per_pharm, pharm_types):
        if not aids or priority.isdisjoint(aids):
            labels.append(0)
            continue
        if int(pharm_type) in _RING_PRIORITY_TYPE_INDICES:
            if ring_heavy_sets is None:
                ring_heavy_sets = [_heavy_atoms_in_ring(mol, ring)
                                   for ring in mol.GetRingInfo().AtomRings()]
            ring_priority_count = _max_priority_atoms_in_shared_rings(
                ring_heavy_sets, aids, priority)
            labels.append(1 if ring_priority_count >= min_ring_priority_atoms else 0)
        else:
            labels.append(1)
    return np.array(labels, dtype=np.int64)


@dataclass(eq=False)
class Pharmacophore:
    """
    Container for the pharmacophores extracted from a molecule.

    For backwards compatibility with the original 3-tuple return of
    :func:`get_pharmacophores`, instances unpack and index as ``(types, positions,
    vectors)``::

        X, P, V = get_pharmacophores(mol)   # still works

    Named attributes (``.types``, ``.positions``, ``.vectors``) are also available.
    When built with ``return_atom_ids=True``, the per-pharmacophore atom-id sets are
    retained on ``.atom_ids`` and priority labels can be computed lazily — for any
    number of priority-atom sets or thresholds — without re-extracting pharmacophores.

    Attributes
    ----------
    types : np.ndarray
        Pharmacophore type indices (``P_TYPES`` order), shape (N,).
    positions : np.ndarray
        Anchor positions, shape (N, 3).
    vectors : np.ndarray
        Relative unit vectors, shape (N, 3).
    mol : rdkit.Chem.Mol or None
        Source molecule, needed for ring-aware :meth:`priority_labels`.
    atom_ids : list of set or None
        Per-pharmacophore atom-id sets aligned with ``types``; ``None`` unless the
        container was built with ``return_atom_ids=True``.
    labels : np.ndarray or None
        Priority labels aligned with ``types``, populated when ``get_pharmacophores``
        is called with ``priority_atoms``; ``None`` otherwise.
    """
    types: np.ndarray
    positions: np.ndarray
    vectors: np.ndarray
    mol: Optional[rdkit.Chem.rdchem.Mol] = None
    atom_ids: Optional[List[set]] = None
    labels: Optional[np.ndarray] = None

    def _as_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.types, self.positions, self.vectors)

    def __iter__(self):
        return iter(self._as_tuple())

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx):
        return self._as_tuple()[idx]

    def priority_labels(self,
                        priority_atoms: Iterable[int],
                        min_ring_priority_atoms: int = 3) -> np.ndarray:
        """
        Compute a 0/1 priority label per pharmacophore against ``priority_atoms``.

        Requires the container to have been built with ``return_atom_ids=True``.
        See :func:`priority_pharm_labels` for the labeling semantics.

        Parameters
        ----------
        priority_atoms : iterable of int
            Atom indices considered "priority".
        min_ring_priority_atoms : int, optional
            Minimum heavy ring atoms in ``priority_atoms`` before an aromatic or
            aromatic-derived hydrophobe is labeled 1. Default is ``3``.

        Returns
        -------
        np.ndarray, shape (N,), dtype int64
        """
        if self.atom_ids is None:
            raise ValueError(
                "priority_labels requires per-pharmacophore atom ids; rebuild with "
                "get_pharmacophores(..., return_atom_ids=True)."
            )
        return priority_pharm_labels(self.mol,
                                     self.atom_ids,
                                     self.types,
                                     priority_atoms,
                                     min_ring_priority_atoms=min_ring_priority_atoms)


def get_pharmacophores(mol: rdkit.Chem.rdchem.Mol,
                       multi_vector: bool = True,
                       exclude: List[int] = [],
                       check_access: bool = False,
                       scale: float = 1.0,
                       return_atom_ids: bool = False,
                       priority_atoms: Optional[Iterable[int]] = None,
                       min_ring_priority_atoms: int = 3,
                       ) -> Pharmacophore:
    """
    Get the identity, anchor positions, and relative unit vectors for each pharmacophore.

    Pharmacophore ordering for indexing:
    ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Cation', 'Anion', 'ZnBinder')

    Notes
    -----
    The ``check_access`` parameter is currently based on whether interaction points sampled
    from a sphere's surface with a radius of 1.8A from the acceptor/donor atom falls outside
    the solvent accessible surface defined by the vdW radius + 0.8A of the neighboring atoms.
    This works for buried acceptors/donors, but may be prone to false positives. For example,
    CN(C)C would have its sole HBA rejected. Other approaches such as buried volume should
    be considered in the future.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit Mol object with conformer.
    multi_vector : bool, optional
        Whether to represent pharmacophores with multiple vectors. Default is ``True``.
    exclude : list, optional
        List of hydrogen indices to not include as a HBD. Default is [].
    check_access : bool, optional
        Check if HBD/HBA are accessible to the molecular surface. Default is ``False``.
    scale : float, optional
        Length of a pharmacophore vector in Angstroms. Default is 1.0.
    return_atom_ids : bool, optional
        When ``True``, the returned :class:`Pharmacophore` retains the per-pharmacophore
        atom-id sets on ``.atom_ids``, enabling lazy priority labeling via
        :meth:`Pharmacophore.priority_labels`. Implied by ``priority_atoms``.
        Default is ``False``.
    priority_atoms : iterable of int, optional
        Atom indices considered "priority". When provided, priority labels are computed
        in this single call and stored on ``.labels`` of the returned container (atom
        ids are retained automatically). See :meth:`Pharmacophore.priority_labels` for
        the labeling semantics. Default is ``None``.
    min_ring_priority_atoms : int, optional
        Only used when ``priority_atoms`` is provided. Minimum heavy ring atoms in
        ``priority_atoms`` before an aromatic or aromatic-derived hydrophobe is labeled
        1. Set to ``1`` to label any pharmacophore whose ring shares a single priority
        atom. Default is ``3``.

    Returns
    -------
    Pharmacophore
        Container that unpacks as the original ``(X, P, V)`` 3-tuple for backwards
        compatibility, where:

        - ``X`` (``.types``): pharmacophore type indices (``P_TYPES`` order), shape (N,).
        - ``P`` (``.positions``): anchor positions, shape (N, 3).
        - ``V`` (``.vectors``): relative unit vectors, shape (N, 3); adding P and V
          gives the extended point.

        When ``return_atom_ids=True`` (or ``priority_atoms`` is given), ``.atom_ids``
        holds per-pharmacophore atom-id sets and ``.priority_labels(priority_atoms, ...)``
        computes 0/1 priority labels. When ``priority_atoms`` is given, ``.labels`` is
        also populated in this call.
    """
    return_atom_ids = return_atom_ids or (priority_atoms is not None)
    pharmacophores_dict = get_pharmacophores_dict(mol=mol,
                                                  multi_vector=multi_vector,
                                                  check_access=check_access,
                                                  scale=scale,
                                                  exclude=exclude,
                                                  return_atom_ids=return_atom_ids)
    N = sum(len(pharmacophores_dict[family]['P']) for family in pharmacophores_dict)
    X = np.empty((N,), dtype=np.int64)
    P = np.empty((N, 3), dtype=np.float64)
    V = np.empty((N, 3), dtype=np.float64)
    atom_ids_per_pharm: Optional[List[set]] = [] if return_atom_ids else None
    start_idx = 0
    for family in pharmacophores_dict:
        this_len = len(pharmacophores_dict[family]['P'])
        if this_len == 0:
            continue
        end_idx = start_idx + this_len
        X[start_idx:end_idx] = P_TYPES.index(family)
        P[start_idx:end_idx, :] = pharmacophores_dict[family]['P']
        V[start_idx:end_idx, :] = pharmacophores_dict[family]['V']
        if return_atom_ids:
            atom_ids_per_pharm.extend(pharmacophores_dict[family]['A'])
        start_idx = end_idx

    pharm = Pharmacophore(types=X, positions=P, vectors=V,
                           mol=mol if return_atom_ids else None,
                           atom_ids=atom_ids_per_pharm)
    if priority_atoms is not None:
        pharm.labels = pharm.priority_labels(
            priority_atoms, min_ring_priority_atoms=min_ring_priority_atoms)
    return pharm
