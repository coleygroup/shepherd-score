"""Reference-layer gates for the PyTorch alignment modes, at the ``optimize_se3`` / mode level:

- the autograd gradient matches finite differences at a non-identity pose,
- the optimizer recovers a planted SE(3) pose, and its returned transform reproduces its score,
- a molecule that keeps an H after ``RemoveHs`` (deuterium) does not desync per-atom fields from
  their centers, and aligns to itself,
- Tversky asymmetry: a small query scores higher as the reference than as the fit.

Self-overlap = 1.0 for every registry mode and both similarities lives in
``test_modes.py::test_every_mode_scores_one_on_itself``. The real Fukui field needs xTB, so a
deterministic synthetic signed field is injected via ``Molecule(fukui=...)``.
"""
import warnings

import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.alignment import optimize_se3
from shepherd_score.alignment._optimize import tree_map
from shepherd_score.alignment.utils.se3 import get_SE3_transform
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.modes import get_mode
from .utils import align_per_pair, mode_loss

CPU = torch.device("cpu")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
METHYL_SALICYLATE = "COC(=O)c1ccccc1O"
DEUTERATED = "[2H]OC(=O)c1ccccc1"          # the deuterium survives RemoveHs
TVERSKY = (0.95, 0.05)


def _synthetic_fukui(rd):
    """Deterministic signed per-atom field in with-H order; a function of atom order so copies match."""
    z = np.array([a.GetAtomicNum() for a in rd.GetAtoms()], dtype=np.float32)
    return (0.2 * ((z % 4) - 1.5)).astype(np.float32)   # values in {-0.3, -0.1, 0.1, 0.3}


def _molecule(rd):
    return Molecule(rd, pharm_multi_vector=False, fukui=_synthetic_fukui(rd))


def _embed(smiles):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=42)
    if rd is None:
        pytest.skip(f"embedding returned None for {smiles!r}")
    return rd


def _inputs(mode, molec):
    """``mode.inputs`` as float32 tensors; integer arrays (labels, types) keep their dtype."""
    def convert(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        return t.to(torch.float32) if t.is_floating_point() else t
    return tree_map(convert, mode.inputs(molec))


# id -> (registered mode name, parameters). A new reference mode adds a row here.
CASES = {
    "vol_tversky":       ("vol", dict(similarity=TVERSKY)),
    "vol_esp_tversky":   ("vol_esp", dict(lam=0.1, similarity=TVERSKY)),
    "vol_lipo":          ("vol_lipo", {}),
    "vol_lipo_tversky":  ("vol_lipo", dict(similarity=TVERSKY)),
    "vol_fukui":         ("vol_fukui", {}),
    "vol_atomtype":      ("vol_atomtype", {}),
    "vol_color_tversky": ("vol_color", dict(similarity=TVERSKY, color_similarity="tversky")),
}


def _case(case_id):
    name, params = CASES[case_id]
    return get_mode(name, **params)


@pytest.fixture(scope="module")
def pair_mols():
    return _molecule(_embed(ASPIRIN)), _molecule(_embed(METHYL_SALICYLATE))


# --- autograd matches finite differences at a non-identity pose (distinct pair) ---------------
@pytest.mark.parametrize("name", list(CASES))
def test_autograd_matches_finite_difference(name, pair_mols):
    mode = _case(name)
    ref, fit = (_inputs(mode, m) for m in pair_mols)

    def obj(se3):
        return mode_loss(mode, se3, ref, fit)

    # small rotation quat (r, i, j, k) + translation, so the gradient is genuinely nonzero
    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    obj(se3).backward()
    ana = se3.grad.clone()
    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data)
        d[i] = eps
        num[i] = (obj(se3.data + d) - obj(se3.data - d)) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"{name}: autograd {ana} vs FD {num}"


# --- the optimizer recovers a planted pose (self pair, fit moved by a known SE(3)) ------------
@pytest.mark.parametrize("name", list(CASES))
def test_optimizer_recovers_planted_pose(name, pair_mols):
    mode = _case(name)
    ref = _inputs(mode, pair_mols[0])
    planted = get_SE3_transform(torch.tensor([0.924, 0.383, 0.0, 0.0, 1.0, -0.5, 0.3]))
    fit = mode.pose(ref, planted)
    _, transform, score = optimize_se3(mode, ref, fit, num_repeats=20, max_num_steps=200)
    assert float(score) > 0.95, f"{name}: planted-pose recovery only reached {float(score):.3f}"
    # the returned transform must reproduce the reported score, not just the scalar
    rescored = float(mode.score(ref, mode.pose(fit, transform)))
    assert np.isclose(rescored, float(score), atol=1e-4), f"{name}: rescored {rescored} vs {float(score)}"


# --- retained-H molecule ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def deuterated():
    m = Chem.AddHs(Chem.MolFromSmiles(DEUTERATED))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    assert AllChem.EmbedMolecule(m, params) == 0
    AllChem.MMFFOptimizeMolecule(m)
    mol = _molecule(m)
    assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
        "test premise broken: this molecule no longer retains an H after RemoveHs"
    return m


@pytest.mark.parametrize("name, getter", [
    ("vol_lipo", lambda m: m.get_lipophilicity(no_H=True)),
    ("vol_fukui", lambda m: m.get_fukui(no_H=True)),
    ("vol_atomtype", lambda m: m.get_atomic_numbers(True)),
])
def test_retained_h_field_follows_heavy_atoms(name, getter, deuterated):
    """The per-atom field has one value per true heavy atom (N-1), not per shape center (N),
    and the mode pairs the two channels without a broadcast error."""
    mol = _molecule(deuterated)
    field = getter(mol)
    assert len(field) == len(mol._nonH_atoms_idx) != mol.atom_pos.shape[0]
    mode = _case(name)
    x = _inputs(mode, mol)
    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1])
    assert 0.0 <= float(1 - mode_loss(mode, se3, x, x)) <= 1.0


@pytest.mark.parametrize("mode", ["vol_lipo", "vol_lipo_tversky", "vol_fukui", "vol_esp_tversky",
                                  "vol_mr", "vol_atomtype"])
def test_retained_h_self_alignment(mode, deuterated):
    """End to end through ``MoleculePair.align_with_*``: no N vs N-1 desync, self-copy ~1.0."""
    pair = MoleculePair(_molecule(deuterated), _molecule(Chem.Mol(deuterated)),
                        do_center=True, device=CPU)
    score = align_per_pair(pair, mode, num_repeats=10, max_num_steps=100)
    assert np.isclose(score, 1.0, atol=1e-3), f"{mode} retained-H self-overlap {score} != 1.000"


# --- Tversky asymmetry (fits-inside) ----------------------------------------------------------
@pytest.mark.parametrize("mode, small, big", [
    ("vol_tversky", "c1ccccc1", "c1ccc2ccccc2c1"),
    ("vol_esp_tversky", "Oc1ccccc1", "Oc1ccc2ccccc2c1"),
])
def test_tversky_asymmetry_fits_inside(mode, small, big):
    """A small query contained in a larger molecule scores higher as the reference than as the fit."""
    s, b = _embed(small), _embed(big)

    def run(ref, fit):
        pair = MoleculePair(Molecule(ref), Molecule(fit), do_center=True, device=CPU)
        return align_per_pair(pair, mode, num_repeats=30, max_num_steps=200)

    fwd, rev = run(s, b), run(b, s)
    assert fwd > rev, f"{mode}: expected fits-inside fwd ({fwd:.3f}) > reverse ({rev:.3f})"
