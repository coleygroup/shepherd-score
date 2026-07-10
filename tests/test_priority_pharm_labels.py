"""Tests for priority_atoms labeling on ring-derived pharmacophores."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score.score.constants import P_TYPES


def _embed_pyridine():
    mol = Chem.MolFromSmiles('c1ccncc1')
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _ring_heavy_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1 and a.IsInRing()]


def _labels_by_type(mol, priority_atoms, **kwargs):
    pharm = get_pharmacophores(mol, multi_vector=False, return_atom_ids=True)
    labels = pharm.priority_labels(priority_atoms, **kwargs)
    by_type = {}
    for idx, label in zip(pharm.types, labels):
        name = P_TYPES[int(idx)]
        by_type.setdefault(name, []).append(int(label))
    return by_type


def test_single_ring_atom_does_not_label_aromatic_or_hydrophobe():
    mol = _embed_pyridine()
    ring_atoms = _ring_heavy_indices(mol)
    n_idx = next(a.GetIdx() for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 7 and a.IsInRing())

    by_type = _labels_by_type(mol, [n_idx])
    assert by_type['Aromatic'] == [0]
    assert by_type['Hydrophobe'] == [0]
    assert by_type['Acceptor'] == [1]

    by_type = _labels_by_type(mol, [ring_atoms[0]])
    assert by_type['Aromatic'] == [0]
    assert by_type['Hydrophobe'] == [0]


def test_three_ring_atoms_labels_aromatic_and_hydrophobe():
    mol = _embed_pyridine()
    ring_atoms = _ring_heavy_indices(mol)

    by_type = _labels_by_type(mol, ring_atoms[:3])
    assert by_type['Aromatic'] == [1]
    assert by_type['Hydrophobe'] == [1]


def test_min_ring_priority_atoms_one_is_looser():
    mol = _embed_pyridine()
    ring_atoms = _ring_heavy_indices(mol)

    by_type = _labels_by_type(mol, [ring_atoms[0]], min_ring_priority_atoms=1)
    assert by_type['Aromatic'] == [1]
    assert by_type['Hydrophobe'] == [1]


def test_single_call_priority_atoms_matches_lazy():
    """priority_atoms in one call stores .labels equal to the lazy computation."""
    mol = _embed_pyridine()
    ring_atoms = _ring_heavy_indices(mol)

    one_call = get_pharmacophores(mol, multi_vector=False, priority_atoms=ring_atoms[:3])
    lazy = get_pharmacophores(mol, multi_vector=False, return_atom_ids=True)

    assert one_call.labels is not None
    np.testing.assert_array_equal(one_call.labels, lazy.priority_labels(ring_atoms[:3]))
    # still unpacks as the original 3-tuple
    x, p, v = one_call
    np.testing.assert_array_equal(x, one_call.types)
