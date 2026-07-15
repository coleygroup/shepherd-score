"""
Tests for the container ``Molecule`` / ``MoleculePair`` abstractions in
``shepherd_score.container._core``:

- ``Surface`` and ``Pharmacophore`` object storage with backwards-compatible
  ``surf_pos``/``surf_esp``/``probe_radius`` and ``pharm_*`` property access.
- ``AlignmentResult`` dict backing ``transform_*``/``sim_aligned_*`` properties (with
  setters, as ``MoleculePairBatch`` relies on external writes).
- ``get_positions``/``get_charges`` (no-H) helpers.
- ``center_to`` in-place mutation through the new properties.
- Single-pair ``align_with_*`` / ``score_with_*`` end-to-end (previously untested
  directly).
"""
import numpy as np
import pytest

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from shepherd_score.pharm_utils.pharmacophore import Pharmacophore
from shepherd_score.container import (
    Molecule,
    MoleculePair,
    Surface,
    AlignmentResult,
)
from .utils import _configure_jax_platform

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(smiles, seed=0xf00d):
    """Embed a SMILES into a 3D RDKit mol (with hydrogens)."""
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(m, params)
    return m


def _mol(smiles, num_surf_points=80, pharm_multi_vector=False, seed=0xf00d, **kwargs):
    return Molecule(_embed(smiles, seed=seed),
                    num_surf_points=num_surf_points,
                    pharm_multi_vector=pharm_multi_vector,
                    **kwargs)


def _pair(smiles_ref='c1ccccc1O', smiles_fit='c1ccccc1N', num_surf_points=75):
    ref = _mol(smiles_ref, num_surf_points=num_surf_points, seed=1)
    fit = _mol(smiles_fit, num_surf_points=num_surf_points, seed=2)
    return MoleculePair(ref, fit, num_surf_points=num_surf_points, do_center=True)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

def test_surface_profile_backward_compat():
    mol = _mol('c1ccccc1O', num_surf_points=75)
    assert isinstance(mol.surface, Surface)
    assert mol.surf_pos.shape == (75, 3)
    assert mol.surf_esp.shape == (75,)
    assert mol.probe_radius == 1.2
    assert mol.surface.positions is mol.surf_pos
    assert mol.surface.esp is mol.surf_esp


def test_surface_profile_none_when_no_points():
    mol = Molecule(_embed('CCO'))  # no num_surf_points / density
    assert mol.surf_pos is None
    assert mol.surf_esp is None
    assert isinstance(mol.surface, Surface)
    assert mol.probe_radius == 1.2


def test_probe_radius_custom_and_settable():
    mol = _mol('CCO', num_surf_points=40, probe_radius=1.0)
    assert mol.probe_radius == 1.0
    mol.probe_radius = 0.9
    assert mol.surface.probe_radius == 0.9


# ---------------------------------------------------------------------------
# Pharmacophore object storage
# ---------------------------------------------------------------------------

def test_pharmacophore_object_and_tuple_unpack():
    mol = _mol('c1ccccc1O')
    assert isinstance(mol.pharmacophore, Pharmacophore)
    X, P, V = mol.pharmacophore  # 3-tuple unpack still works
    assert mol.pharm_types is X
    assert mol.pharm_ancs is P
    assert mol.pharm_vecs is V
    assert P.shape[1] == 3 and V.shape[1] == 3
    assert X.shape[0] == P.shape[0] == V.shape[0]


def test_pharmacophore_none_when_not_requested():
    mol = Molecule(_embed('c1ccccc1O'))  # pharm_multi_vector defaults to None
    assert mol.pharmacophore is None
    assert mol.pharm_types is None
    assert mol.pharm_ancs is None
    assert mol.pharm_vecs is None


def test_molecule_accepts_pharm_arrays_directly():
    src = _mol('c1ccccc1O')
    types, ancs, vecs = src.pharm_types, src.pharm_ancs, src.pharm_vecs
    mol = Molecule(_embed('c1ccccc1O'),
                   pharm_types=types, pharm_ancs=ancs, pharm_vecs=vecs)
    assert isinstance(mol.pharmacophore, Pharmacophore)
    assert np.array_equal(mol.pharm_types, types)
    assert np.array_equal(mol.pharm_ancs, ancs)
    assert np.array_equal(mol.pharm_vecs, vecs)


def test_get_pharmacophore_passthrough_atom_ids_and_priority():
    mol = _mol('c1ccccc1O')
    # atom-id retention unlocks priority labeling from the Molecule
    mol.get_pharmacophore(multi_vector=False, return_atom_ids=True)
    assert mol.pharmacophore.atom_ids is not None
    labels = mol.pharmacophore.priority_labels([0, 1, 2])
    assert labels.shape == (mol.pharm_types.shape[0],)
    assert labels.dtype == np.int64

    # priority_atoms populates .labels in a single call, default behavior otherwise same
    mol.get_pharmacophore(multi_vector=False, priority_atoms=[0, 1, 2])
    assert mol.pharmacophore.labels is not None
    assert mol.pharmacophore.labels.shape == (mol.pharm_types.shape[0],)


# ---------------------------------------------------------------------------
# get_positions / get_charges helpers
# ---------------------------------------------------------------------------

def test_get_positions_and_charges_no_H():
    mol = _mol('c1ccccc1O', num_surf_points=40)
    n_heavy = mol.atom_pos.shape[0]
    n_all = mol.mol.GetConformer().GetPositions().shape[0]
    assert n_all > n_heavy  # has hydrogens

    assert np.array_equal(mol.get_positions(no_H=True), mol.atom_pos)
    assert mol.get_positions(no_H=False).shape == (n_all, 3)

    assert mol.get_charges(no_H=True).shape == (n_heavy,)
    assert np.array_equal(mol.get_charges(no_H=True),
                          mol.partial_charges[mol._nonH_atoms_idx])
    assert mol.get_charges(no_H=False).shape == (n_all,)
    assert np.array_equal(mol.get_charges(no_H=False), mol.partial_charges)


# ---------------------------------------------------------------------------
# center_to in-place mutation through properties
# ---------------------------------------------------------------------------

def test_center_to_shifts_all_profiles():
    mol = _mol('c1ccccc1O', num_surf_points=40)
    atom0 = mol.atom_pos.copy()
    surf0 = mol.surf_pos.copy()
    anc0 = mol.pharm_ancs.copy()
    shift = atom0.mean(0)

    mol.center_to(shift)

    assert np.allclose(mol.atom_pos, atom0 - shift)
    assert np.allclose(mol.surf_pos, surf0 - shift)
    assert np.allclose(mol.pharm_ancs, anc0 - shift)
    # atom_pos is now centered at origin
    assert np.allclose(mol.atom_pos.mean(0), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# AlignmentResult / MoleculePair alignment-result properties
# ---------------------------------------------------------------------------

def test_alignment_result_dataclass_defaults():
    ar = AlignmentResult()
    assert ar.score is None
    assert np.array_equal(ar.transform, np.eye(4))


def test_molecule_pair_alignment_defaults():
    mp = _pair()
    for name in ('vol', 'vol_noH', 'surf', 'esp',
                 'vol_esp', 'vol_esp_noH', 'esp_combo', 'pharm'):
        assert np.array_equal(getattr(mp, f'transform_{name}'), np.eye(4))
        assert getattr(mp, f'sim_aligned_{name}') is None


def test_molecule_pair_alignment_setters():
    """Mirror how MoleculePairBatch writes results back onto each pair."""
    mp = _pair()
    T = np.eye(4)
    T[0, 3] = 5.0
    mp.transform_vol_noH = T
    mp.sim_aligned_vol_noH = np.float32(0.42)
    assert np.array_equal(mp.transform_vol_noH, T)
    assert np.isclose(float(mp.sim_aligned_vol_noH), 0.42)
    # other modes remain untouched
    assert mp.sim_aligned_pharm is None
    assert np.array_equal(mp.transform_pharm, np.eye(4))


# ---------------------------------------------------------------------------
# Single-pair alignment + scoring end-to-end (torch/np)
# ---------------------------------------------------------------------------

@pytest.mark.torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires PyTorch")
def test_single_pair_align_all_modes():
    mp = _pair()

    aligned = mp.align_with_vol(no_H=True, num_repeats=5, max_num_steps=30)
    assert aligned.shape[1] == 3
    assert mp.sim_aligned_vol_noH is not None
    assert mp.transform_vol_noH.shape == (4, 4)

    mp.align_with_vol_esp(lam=0.1, no_H=True, num_repeats=5, max_num_steps=30)
    assert mp.sim_aligned_vol_esp_noH is not None

    mp.align_with_surf(alpha=0.81, num_repeats=5, max_num_steps=30)
    assert mp.sim_aligned_surf is not None

    mp.align_with_esp(alpha=0.81, num_repeats=5, max_num_steps=30)
    assert mp.sim_aligned_esp is not None

    mp.align_with_esp_combo(alpha=0.81, num_repeats=5, max_num_steps=30)
    assert mp.sim_aligned_esp_combo is not None

    anchors, vectors = mp.align_with_pharm(num_repeats=5, max_num_steps=30)
    assert anchors.shape[1] == 3 and vectors.shape[1] == 3
    assert mp.sim_aligned_pharm is not None

    # all similarity scores in a sane range
    for name in ('vol_noH', 'vol_esp_noH', 'surf', 'esp', 'esp_combo', 'pharm'):
        score = float(getattr(mp, f'sim_aligned_{name}'))
        assert -1e-4 <= score <= 1.0 + 1e-4


@pytest.mark.torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires PyTorch")
def test_single_pair_score_np_matches_torch():
    mp = _pair()
    assert np.isclose(mp.score_with_surf(alpha=0.81, use='np'),
                      mp.score_with_surf(alpha=0.81, use='torch'), atol=1e-4)
    assert np.isclose(mp.score_with_esp(alpha=0.81, use='np'),
                      mp.score_with_esp(alpha=0.81, use='torch'), atol=1e-4)
    assert np.isclose(mp.score_with_pharm(use='np'),
                      mp.score_with_pharm(use='torch'), atol=1e-4)


# ---------------------------------------------------------------------------
# JAX path (exercises _require_jax + use_jax branch)
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_single_pair_align_vol_jax():
    # Configure JAX platform before import to avoid GPU initialization errors.
    _configure_jax_platform()
    pytest.importorskip("jax")
    mp = _pair()
    aligned = mp.align_with_vol(no_H=True, num_repeats=5, max_num_steps=30, use_jax=True)
    assert aligned.shape[1] == 3
    assert mp.sim_aligned_vol_noH is not None
    assert mp.transform_vol_noH.shape == (4, 4)
