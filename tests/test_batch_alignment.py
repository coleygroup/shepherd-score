"""
Tests for MoleculePairBatch and masked JAX alignment functions.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from shepherd_score.score.gaussian_overlap_jax import get_overlap_jax, get_overlap_jax_mask
from shepherd_score.score.electrostatic_scoring_jax import get_overlap_esp_jax, get_overlap_esp_jax_mask
from shepherd_score.alignment._jax import optimize_ROCS_overlay_jax, optimize_ROCS_overlay_jax_mask
from shepherd_score.alignment._jax import optimize_ROCS_esp_overlay_jax, optimize_ROCS_esp_overlay_jax_mask
from shepherd_score.score.pharmacophore_scoring_jax import (
    get_overlap_pharm_jax_vectorized,
    get_overlap_pharm_jax_vectorized_mask,
)
from shepherd_score.alignment._jax import (
    optimize_pharm_overlay_jax_vectorized,
    optimize_pharm_overlay_jax_vectorized_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mol_pair(smiles_ref, smiles_fit):
    """Return a MoleculePair from two SMILES strings."""
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import MoleculePair

    def _mol(smi):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        return m

    return MoleculePair(_mol(smiles_ref), _mol(smiles_fit))


def _make_mol_pair_with_pharm(smiles_ref, smiles_fit):
    """Return a MoleculePair where both Molecule objects have pharmacophore data."""
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule, MoleculePair

    def _mol_obj(smi):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        return Molecule(m, pharm_multi_vector=False)

    ref_mol = _mol_obj(smiles_ref)
    fit_mol = _mol_obj(smiles_fit)
    return MoleculePair(ref_mol, fit_mol)


# ---------------------------------------------------------------------------
# masked scoring accuracy
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_scoring_matches_unmasked():
    """Padding + mask should give the same overlap as the unpadded version."""
    rng = np.random.default_rng(42)
    n_a, n_b = 10, 8
    alpha = 0.81

    pts_a = rng.standard_normal((n_a, 3)).astype(np.float32)
    pts_b = rng.standard_normal((n_b, 3)).astype(np.float32)

    score_ref = float(get_overlap_jax(jnp.array(pts_a), jnp.array(pts_b), alpha))

    # Pad to 15 x 15
    pad = 15
    pts_a_pad = np.zeros((pad, 3), dtype=np.float32)
    pts_a_pad[:n_a] = pts_a
    pts_b_pad = np.zeros((pad, 3), dtype=np.float32)
    pts_b_pad[:n_b] = pts_b
    mask_a = np.zeros(pad, dtype=np.float32)
    mask_a[:n_a] = 1.0
    mask_b = np.zeros(pad, dtype=np.float32)
    mask_b[:n_b] = 1.0

    score_masked = float(get_overlap_jax_mask(
        jnp.array(pts_a_pad), jnp.array(pts_b_pad),
        jnp.array(mask_a), jnp.array(mask_b), alpha
    ))

    assert abs(score_ref - score_masked) < 1e-6, (
        f"Masked score {score_masked:.6f} != unmasked {score_ref:.6f}"
    )


# ---------------------------------------------------------------------------
# masked alignment accuracy
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_alignment_matches_unmasked():
    """optimize_ROCS_overlay_jax_mask should reach similar score as unmasked."""
    rng = np.random.default_rng(0)
    n_ref, n_fit = 12, 9
    alpha = 0.81

    ref_pts = rng.standard_normal((n_ref, 3)).astype(np.float32)
    fit_pts = rng.standard_normal((n_fit, 3)).astype(np.float32)

    # Unmasked alignment
    _, _, score_unmasked = optimize_ROCS_overlay_jax(
        jnp.array(ref_pts), jnp.array(fit_pts), alpha, num_repeats=10
    )

    # Padded masked alignment
    pad_ref, pad_fit = 20, 20
    ref_pad = np.zeros((pad_ref, 3), dtype=np.float32)
    ref_pad[:n_ref] = ref_pts
    fit_pad = np.zeros((pad_fit, 3), dtype=np.float32)
    fit_pad[:n_fit] = fit_pts
    mask_ref = np.zeros(pad_ref, dtype=np.float32)
    mask_ref[:n_ref] = 1.0
    mask_fit = np.zeros(pad_fit, dtype=np.float32)
    mask_fit[:n_fit] = 1.0

    _, _, score_masked = optimize_ROCS_overlay_jax_mask(
        jnp.array(ref_pad), jnp.array(fit_pad),
        jnp.array(mask_ref), jnp.array(mask_fit),
        alpha, num_repeats=10
    )

    assert abs(float(score_unmasked) - float(score_masked)) < 1e-5, (
        f"Masked score {float(score_masked):.4f} differs too much from "
        f"unmasked {float(score_unmasked):.4f}"
    )


# ---------------------------------------------------------------------------
# MoleculePairBatch end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_molecule_pair_batch_align_with_vol():
    """MoleculePairBatch.align_with_vol stores scores and returns aligned coords."""
    from shepherd_score.container import MoleculePairBatch

    smiles_pairs = [
        ("CCO", "CCC"),
        ("c1ccccc1", "c1ccncc1"),
        ("CC(=O)O", "CC(=O)N"),
    ]

    pairs = [_make_mol_pair(ref_smi, fit_smi) for ref_smi, fit_smi in smiles_pairs]
    batch = MoleculePairBatch(pairs)

    scores, aligned_list = batch.align_with_vol(no_H=True, num_repeats=10, max_num_steps=50)

    assert len(aligned_list) == len(pairs), "Should return one array per pair"
    assert scores.shape == (len(pairs),), "scores array should have one entry per pair"

    for i, (pair, aligned) in enumerate(zip(pairs, aligned_list)):
        # Score should be stored in-place
        assert pair.sim_aligned_vol_noH is not None, f"Pair {i}: score not stored"
        score = float(pair.sim_aligned_vol_noH)
        assert 0.0 <= score <= 1.0, f"Pair {i}: score {score:.4f} out of [0,1]"

        # Shape check: unpadded aligned coords
        expected_n = pair.fit_molec.atom_pos.shape[0]
        assert aligned.shape == (expected_n, 3), (
            f"Pair {i}: aligned shape {aligned.shape} != ({expected_n}, 3)"
        )


# ---------------------------------------------------------------------------
# masked pharmacophore scoring matches unmasked
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_pharm_scoring_matches_unmasked():
    """Padding + mask should give the same pharmacophore overlap as the unpadded version."""
    rng = np.random.default_rng(7)
    n1, n2 = 6, 5
    n_types = 8  # real types (0-7); 8 = Dummy

    ptypes_1 = rng.integers(0, n_types, size=n1).astype(np.int32)
    ptypes_2 = rng.integers(0, n_types, size=n2).astype(np.int32)
    ancs_1 = rng.standard_normal((n1, 3)).astype(np.float32)
    ancs_2 = rng.standard_normal((n2, 3)).astype(np.float32)
    vecs_1 = rng.standard_normal((n1, 3)).astype(np.float32)
    vecs_2 = rng.standard_normal((n2, 3)).astype(np.float32)

    score_ref = float(get_overlap_pharm_jax_vectorized(
        jnp.array(ptypes_1), jnp.array(ptypes_2),
        jnp.array(ancs_1), jnp.array(ancs_2),
        jnp.array(vecs_1), jnp.array(vecs_2),
    ))

    # Pad to larger size
    pad1, pad2 = 10, 10
    DUMMY = 8

    pt1_pad = np.full(pad1, DUMMY, dtype=np.int32)
    pt1_pad[:n1] = ptypes_1
    pt2_pad = np.full(pad2, DUMMY, dtype=np.int32)
    pt2_pad[:n2] = ptypes_2

    a1_pad = np.zeros((pad1, 3), dtype=np.float32)
    a1_pad[:n1] = ancs_1
    a2_pad = np.zeros((pad2, 3), dtype=np.float32)
    a2_pad[:n2] = ancs_2

    v1_pad = np.zeros((pad1, 3), dtype=np.float32)
    v1_pad[:n1] = vecs_1
    v2_pad = np.zeros((pad2, 3), dtype=np.float32)
    v2_pad[:n2] = vecs_2

    mask1 = np.zeros(pad1, dtype=np.float32)
    mask1[:n1] = 1.0
    mask2 = np.zeros(pad2, dtype=np.float32)
    mask2[:n2] = 1.0

    score_masked = float(get_overlap_pharm_jax_vectorized_mask(
        jnp.array(pt1_pad), jnp.array(pt2_pad),
        jnp.array(a1_pad), jnp.array(a2_pad),
        jnp.array(v1_pad), jnp.array(v2_pad),
        jnp.array(mask1), jnp.array(mask2),
    ))

    assert abs(score_ref - score_masked) < 1e-6, (
        f"Masked pharm score {score_masked:.6f} != unmasked {score_ref:.6f}"
    )


# ---------------------------------------------------------------------------
# masked pharmacophore alignment matches unmasked
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_pharm_alignment_matches_unmasked():
    """optimize_pharm_overlay_jax_vectorized_mask should reach similar score as unmasked."""
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule

    m = Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)OC(C(F)C(=O)O)CCC")
    m2 = Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)OCCCC")
    m = Chem.AddHs(m)
    m2 = Chem.AddHs(m2)
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    AllChem.EmbedMolecule(m2, AllChem.ETKDGv3())
    mol = Molecule(m, pharm_multi_vector=False)
    mol2 = Molecule(m2, pharm_multi_vector=False)
    pt = jnp.array(mol.pharm_types)
    ancs = jnp.array(mol.pharm_ancs)
    vecs = jnp.array(mol.pharm_vecs)
    pt2 = jnp.array(mol2.pharm_types)
    ancs2 = jnp.array(mol2.pharm_ancs)
    vecs2 = jnp.array(mol2.pharm_vecs)

    # Unmasked
    _, _, _, score_unmasked = optimize_pharm_overlay_jax_vectorized(
        pt, pt2, ancs, ancs2, vecs, vecs2, num_repeats=10, max_num_steps=50
    )

    # Pad
    pad = mol.pharm_types.shape[0]
    n2 = mol2.pharm_types.shape[0]
    DUMMY = 8

    pt_pad = np.full(pad, DUMMY, dtype=np.int32)
    pt_pad[:n2] = np.array(pt2)
    ancs_pad = np.zeros((pad, 3), dtype=np.float32)
    ancs_pad[:n2] = np.array(ancs2)
    vecs_pad = np.zeros((pad, 3), dtype=np.float32)
    vecs_pad[:n2] = np.array(vecs2)
    mask1 = np.ones(pad, dtype=np.float32)
    mask2 = np.zeros(pad, dtype=np.float32)
    mask2[:n2] = 1.0

    _, _, _, score_masked = optimize_pharm_overlay_jax_vectorized_mask(
        pt, jnp.array(pt_pad),
        ancs, jnp.array(ancs_pad),
        vecs, jnp.array(vecs_pad),
        mask1, mask2,
        num_repeats=10, max_num_steps=50,
        init_ref_anchors=np.array(ancs),
        init_fit_anchors=np.array(ancs),
    )

    assert abs(float(score_unmasked) - float(score_masked)) < 1e-5, (
        f"Masked pharm alignment score {float(score_masked):.4f} differs too much "
        f"from unmasked {float(score_unmasked):.4f}"
    )


# ---------------------------------------------------------------------------
# MoleculePairBatch.align_with_pharm end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_molecule_pair_batch_align_with_pharm():
    """MoleculePairBatch.align_with_pharm stores scores and returns aligned coords."""
    from shepherd_score.container import MoleculePairBatch

    smiles_pairs = [
        # Use molecules with clear pharmacophores (aromatic, HBD, HBA)
        ("CC(=O)Nc1ccc(cc1)O", "CC(=O)Nc1ccc(cc1)N"),   # acetaminophen vs 4-aminoacetanilide
        ("OC(=O)c1ccccc1N", "OC(=O)c1ccccc1O"),           # anthranilic vs salicylic acid
        ("Cc1ccc(cc1)S(N)(=O)=O", "Cc1ccc(cc1)C(=O)N"),   # sulfonamide vs toluamide
    ]

    pairs = [_make_mol_pair_with_pharm(ref_smi, fit_smi) for ref_smi, fit_smi in smiles_pairs]
    batch = MoleculePairBatch(pairs)

    scores, ancs_list, vecs_list = batch.align_with_pharm(num_repeats=10, max_num_steps=50)

    assert len(ancs_list) == len(pairs), "Should return one anchors array per pair"
    assert len(vecs_list) == len(pairs), "Should return one vectors array per pair"
    assert scores.shape == (len(pairs),), "scores array should have one entry per pair"

    for i, (pair, ancs, vecs) in enumerate(zip(pairs, ancs_list, vecs_list)):
        assert pair.sim_aligned_pharm is not None, f"Pair {i}: score not stored"
        score = float(pair.sim_aligned_pharm)
        assert 0.0 <= score <= 1.0, f"Pair {i}: score {score:.4f} out of [0,1]"

        expected_n = pair.fit_molec.pharm_types.shape[0]
        assert ancs.shape == (expected_n, 3), (
            f"Pair {i}: anchors shape {ancs.shape} != ({expected_n}, 3)"
        )
        assert vecs.shape == (expected_n, 3), (
            f"Pair {i}: vectors shape {vecs.shape} != ({expected_n}, 3)"
        )


# ---------------------------------------------------------------------------
# masked ESP scoring accuracy
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_esp_scoring_matches_unmasked():
    """Padding + mask should give the same ESP overlap as the unpadded version."""
    rng = np.random.default_rng(13)
    n_a, n_b = 10, 8
    alpha = 0.81
    lam = 0.1

    pts_a = rng.standard_normal((n_a, 3)).astype(np.float32)
    pts_b = rng.standard_normal((n_b, 3)).astype(np.float32)
    ch_a = rng.standard_normal(n_a).astype(np.float32)
    ch_b = rng.standard_normal(n_b).astype(np.float32)

    score_ref = float(get_overlap_esp_jax(jnp.array(pts_a), jnp.array(pts_b),
                                          jnp.array(ch_a), jnp.array(ch_b), alpha, lam))

    # Pad to 15 x 15
    pad = 15
    pts_a_pad = np.zeros((pad, 3), dtype=np.float32)
    pts_a_pad[:n_a] = pts_a
    pts_b_pad = np.zeros((pad, 3), dtype=np.float32)
    pts_b_pad[:n_b] = pts_b
    ch_a_pad = np.zeros(pad, dtype=np.float32)
    ch_a_pad[:n_a] = ch_a
    ch_b_pad = np.zeros(pad, dtype=np.float32)
    ch_b_pad[:n_b] = ch_b
    mask_a = np.zeros(pad, dtype=np.float32)
    mask_a[:n_a] = 1.0
    mask_b = np.zeros(pad, dtype=np.float32)
    mask_b[:n_b] = 1.0

    score_masked = float(get_overlap_esp_jax_mask(
        jnp.array(pts_a_pad), jnp.array(pts_b_pad),
        jnp.array(ch_a_pad), jnp.array(ch_b_pad),
        jnp.array(mask_a), jnp.array(mask_b), alpha, lam
    ))

    assert abs(score_ref - score_masked) < 1e-6, (
        f"Masked ESP score {score_masked:.6f} != unmasked {score_ref:.6f}"
    )


# ---------------------------------------------------------------------------
# masked ESP alignment accuracy
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_masked_esp_alignment_matches_unmasked():
    """optimize_ROCS_esp_overlay_jax_mask should reach similar score as unmasked."""
    rng = np.random.default_rng(5)
    n_ref, n_fit = 12, 9
    alpha = 0.81
    lam = 0.1

    ref_pts = rng.standard_normal((n_ref, 3)).astype(np.float32)
    fit_pts = rng.standard_normal((n_fit, 3)).astype(np.float32)
    ref_ch = rng.standard_normal(n_ref).astype(np.float32)
    fit_ch = rng.standard_normal(n_fit).astype(np.float32)

    # Unmasked alignment
    _, _, score_unmasked = optimize_ROCS_esp_overlay_jax(
        jnp.array(ref_pts), jnp.array(fit_pts),
        jnp.array(ref_ch), jnp.array(fit_ch),
        alpha, lam, num_repeats=10
    )

    # Padded masked alignment
    pad_ref, pad_fit = 20, 20
    ref_pad = np.zeros((pad_ref, 3), dtype=np.float32)
    ref_pad[:n_ref] = ref_pts
    fit_pad = np.zeros((pad_fit, 3), dtype=np.float32)
    fit_pad[:n_fit] = fit_pts
    ref_ch_pad = np.zeros(pad_ref, dtype=np.float32)
    ref_ch_pad[:n_ref] = ref_ch
    fit_ch_pad = np.zeros(pad_fit, dtype=np.float32)
    fit_ch_pad[:n_fit] = fit_ch
    mask_ref = np.zeros(pad_ref, dtype=np.float32)
    mask_ref[:n_ref] = 1.0
    mask_fit = np.zeros(pad_fit, dtype=np.float32)
    mask_fit[:n_fit] = 1.0

    _, _, score_masked = optimize_ROCS_esp_overlay_jax_mask(
        jnp.array(ref_pad), jnp.array(fit_pad),
        jnp.array(ref_ch_pad), jnp.array(fit_ch_pad),
        jnp.array(mask_ref), jnp.array(mask_fit),
        alpha, lam, num_repeats=10
    )

    assert abs(float(score_unmasked) - float(score_masked)) < 1e-5, (
        f"Masked ESP alignment score {float(score_masked):.4f} differs too much from "
        f"unmasked {float(score_unmasked):.4f}"
    )


# ---------------------------------------------------------------------------
# MoleculePairBatch.align_with_vol_esp end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.jax
def test_molecule_pair_batch_align_with_vol_esp():
    """MoleculePairBatch.align_with_vol_esp stores scores and returns aligned coords."""
    from shepherd_score.container import MoleculePairBatch

    smiles_pairs = [
        ("CCO", "CCC"),
        ("c1ccccc1", "c1ccncc1"),
        ("CC(=O)O", "CC(=O)N"),
    ]

    pairs = [_make_mol_pair(ref_smi, fit_smi) for ref_smi, fit_smi in smiles_pairs]
    batch = MoleculePairBatch(pairs)

    scores, aligned_list = batch.align_with_vol_esp(lam=0.1, no_H=True,
                                                     num_repeats=10, max_num_steps=50)

    assert len(aligned_list) == len(pairs), "Should return one array per pair"
    assert scores.shape == (len(pairs),), "scores array should have one entry per pair"

    for i, (pair, aligned) in enumerate(zip(pairs, aligned_list)):
        assert pair.sim_aligned_vol_esp_noH is not None, f"Pair {i}: score not stored"
        score = float(pair.sim_aligned_vol_esp_noH)
        assert 0.0 <= score <= 1.0, f"Pair {i}: score {score:.4f} out of [0,1]"

        expected_n = pair.fit_molec.atom_pos.shape[0]
        assert aligned.shape == (expected_n, 3), (
            f"Pair {i}: aligned shape {aligned.shape} != ({expected_n}, 3)"
        )
