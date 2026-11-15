"""The pharmacophore ``extended_points`` objective has no kernel, so the batched path hands off
to the eager autograd driver (``accel/batch/aligners_legacy.py``). These tests check that the
hand-off happens, that the flags reach the driver, and that the three settings are three
different objectives.
"""
import warnings

import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccc2O"]


def _mol(smiles, seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        assert AllChem.EmbedMolecule(rd, p) == 0, smiles
    return Molecule(rd, pharm_multi_vector=False)


def _run(mols, **kw):
    """Align mols[0] against every mol and return the score vector."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    pairs = [MoleculePair(mols[0], f, do_center=True, device=torch.device("cpu"))
             for f in mols]
    MoleculePairBatch(pairs).align_with_pharm(backend="numba", **kw)
    return [float(p.sim_aligned_pharm) for p in pairs]


@pytest.fixture(scope="module")
def mols():
    return [_mol(s, i) for i, s in enumerate(SMILES)]


@pytest.mark.parametrize("flags,expect_legacy", [
    ({}, False),
    ({"extended_points": False}, False),
    ({"extended_points": True}, True),
    ({"extended_points": True, "only_extended": True}, True),
])
def test_extended_points_routes_to_the_kernel_free_driver(flags, expect_legacy, mols,
                                                          monkeypatch):
    """``extended_points=True`` must leave the kernel path and anything else must stay on it."""
    from shepherd_score.accel.batch import aligners_legacy

    calls = []
    real = aligners_legacy._align_batch_pharm_extended

    def spy(spec, pairs, params, **kw):
        calls.append(dict(params))
        return real(spec, pairs, params, **kw)

    monkeypatch.setattr(aligners_legacy, "_align_batch_pharm_extended", spy)
    _run(mols, **flags)

    assert bool(calls) is expect_legacy, \
        f"flags {flags}: legacy driver called {len(calls)} times, expected {expect_legacy}"
    if expect_legacy:
        assert calls[0]["extended_points"] is True
        assert calls[0].get("only_extended", False) is flags.get("only_extended", False), \
            "only_extended did not reach the driver"


def test_the_three_settings_are_three_different_objectives(mols):
    """Plain, extended and only-extended must score differently on a distinct pair."""
    plain = _run(mols)
    ext = _run(mols, extended_points=True)
    only = _run(mols, extended_points=True, only_extended=True)

    for name, vec in (("plain", plain), ("extended", ext), ("only_extended", only)):
        assert len(vec) == len(SMILES) and all(np.isfinite(vec)), f"{name}: {vec}"

    # row 0 is the self-copy and scores 1.0 under all three; compare the rest
    for a, b, label in ((plain, ext, "plain vs extended"),
                        (plain, only, "plain vs only_extended"),
                        (ext, only, "extended vs only_extended")):
        assert not np.allclose(a[1:], b[1:], atol=1e-6), f"{label} scored the same"


@pytest.mark.parametrize("flags", [
    {"extended_points": True},
    {"extended_points": True, "only_extended": True},
])
def test_self_copy_is_one_on_the_extended_objective(flags, mols):
    """A molecule against a copy of itself scores 1.0 on the extended objective."""
    assert np.isclose(_run(mols, **flags)[0], 1.0, atol=1e-4)


def test_extended_points_composes_with_trans_init(mols):
    """The kernel-free driver also takes the translation-seeded coarse grid."""
    ext = _run(mols, extended_points=True)
    ti = _run(mols, extended_points=True, trans_init=True)
    assert np.isclose(ti[0], 1.0, atol=1e-4), f"trans_init self-copy {ti[0]}"
    assert all(np.isfinite(ti)), ti
    assert np.allclose(ext, ti, atol=5e-3), \
        f"trans_init moved the extended objective off its basin: {ext} vs {ti}"
