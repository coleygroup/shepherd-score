"""Reference-mode test template, for a mode whose inputs ``tests/modes/test_reference_modes.py``
cannot build (surfaces, caller-supplied clouds). Prefer adding a row to that file's ``CASES``
and parametrize lists; see the design-scoring-mode SKILL.md "Validate" section.

Copy this to ``tests/modes/test_<mode>.py``, replace ``YOURMODE`` with the name registered in
``shepherd_score.modes.MODES``, and fill in the ``# TODO`` markers.

Gates 1-3 apply to every reference mode. Gate 4 (retained H) applies only if the mode reads a
per-atom field; gate 5 (asymmetry) only if it is meant to be used with a Tversky similarity.
Delete the gates that do not apply. All remaining gates must pass before handing off to
``accelerate-scoring-mode``.

Filled-in examples:
  * ``tests/modes/test_reference_modes.py`` -- the same gates, parametrized over the vol-family modes,
    with a per-atom field injected through ``Molecule(fukui=...)`` so no xTB binary is needed.
  * ``tests/modes/test_vol_color_avoid.py`` -- a mode with a caller-supplied avoid cloud, in its own file.
  * ``tests/modes/test_modes.py``     -- every registered mode on itself, Tversky slots, a user-defined
    ``Mode`` subclass and ``blend``.

This file is valid Python but does not run until YOURMODE exists. Keep it under
``.claude/skills/``; do not ship it as a live test.
"""
import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from shepherd_score.alignment import optimize_se3
from shepherd_score.alignment._optimize import tree_map
from shepherd_score.alignment.utils.se3 import get_SE3_transform
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.modes import get_mode

from .._support.utils import mode_loss


MODE = "YOURMODE"           # name registered in shepherd_score.modes.MODES
SLOT = MODE                 # _ALIGN_KEYS slot written by align_with_<MODE> (vol/vol_esp use "<mode>_noH")
PARAMS = {}                 # TODO: non-default mode parameters, e.g. {"lam": 0.1}
FAST = dict(num_repeats=20, max_num_steps=200)


def _embed(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0
    return m


def _molecule(m):
    """A Molecule carrying what YOURMODE reads.

    TODO: pass the features the mode needs (``num_surf_points``, ``pharm_multi_vector``,
    ``charge_model="mmff"``, ...). For a field that needs xTB, inject a deterministic synthetic
    field through the constructor (see ``_molecule`` in ``tests/modes/test_reference_modes.py``); the overlap math does not
    depend on how the field was produced.
    """
    return Molecule(Chem.Mol(m), charge_model="mmff")


def _make_pair(smi="c1ccccc1CCO", fit_smi=None):
    """(ref, fit) pair. With ``fit_smi=None`` the fit is a copy of the reference.

    TODO: pass ``num_surf_points`` to MoleculePair as well for surface modes; the pair checks
    its own value, not the molecules'.
    """
    ref = _embed(smi)
    fit = ref if fit_smi is None else _embed(fit_smi)
    return MoleculePair(_molecule(ref), _molecule(fit), device=torch.device("cpu"))


def _tensors(mode, molec):
    """``mode.inputs`` as float32 tensors; integer arrays (labels) keep their dtype."""
    def convert(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        return t.to(torch.float32) if t.is_floating_point() else t
    return tree_map(convert, mode.inputs(molec))


# --- Gate 1: self-score ------------------------------------------------------------------------
def test_self_score():
    """The fixed-pose self-score, then the optimizer from the identity. Adapt the expected value
    for objectives that are not normalized overlaps (penalties, agreement terms)."""
    pair = _make_pair()
    assert np.isclose(float(pair.score(MODE, **PARAMS)), 1.0, atol=1e-5)
    getattr(pair, f"align_with_{MODE}")(**FAST)
    score = getattr(pair, f"sim_aligned_{SLOT}")
    assert np.isclose(float(score), 1.0, atol=1e-3), f"optimized self-score {score}"


# --- Gate 2: autograd matches central finite differences ---------------------------------------
def test_autograd_matches_finite_difference():
    """At a non-identity pose on two different molecules. At the identity of a self-pair the
    gradient is ~0 and the comparison passes trivially. SE(3) helpers work in float32, so the
    finite-difference step and tolerance are loose."""
    pair = _make_pair(fit_smi="c1ccccc1CCN")   # TODO: a second molecule the mode can score
    mode = get_mode(MODE, **PARAMS)
    ref, fit = _tensors(mode, pair.ref_molec), _tensors(mode, pair.fit_molec)

    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    mode_loss(mode, se3, ref, fit).backward()
    ana = se3.grad.clone()
    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data); d[i] = eps
        hi = mode_loss(mode, se3.data + d, ref, fit)
        lo = mode_loss(mode, se3.data - d, ref, fit)
        num[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"autograd {ana} vs FD {num}"


# --- Gate 3: the optimizer recovers a planted pose ---------------------------------------------
def test_optimizer_recovers_planted_pose():
    """Move the fit's conformer by a known SE(3) and rebuild the Molecule, so every feature is
    recomputed in the moved frame; then align and rescore the returned transform directly."""
    m = _embed("c1ccccc1CCO")
    moved = Chem.Mol(m)
    T = get_SE3_transform(torch.tensor([0.924, 0.383, 0.0, 0.0, 1.0, -0.5, 0.3])).numpy()
    rdMolTransforms.TransformConformer(moved.GetConformer(), T.astype(np.float64))
    pair = MoleculePair(_molecule(m), _molecule(moved), device=torch.device("cpu"))

    res = pair.align(MODE, **FAST, **PARAMS)
    assert float(res.score) > 0.95, f"planted-pose recovery only reached {float(res.score):.3f}"

    mode = get_mode(MODE, **PARAMS)
    ref, fit = _tensors(mode, pair.ref_molec), _tensors(mode, pair.fit_molec)
    posed = mode.pose(fit, torch.as_tensor(res.transform, dtype=torch.float32))
    assert np.isclose(float(mode.score(ref, posed)), float(res.score), atol=1e-4)


# --- Gate 4: retained-H molecule (per-atom fields only) ---------------------------------------
def test_retained_h_molecule():
    """``Chem.RemoveHs`` keeps isotope-labelled hydrogen, so ``atom_pos`` on a deuterated molecule
    is longer than the strict-heavy set (``_nonH_atoms_idx``) that per-atom fields use. Field
    positions must come from the same indices. Plain molecules pass gates 1-3 even with this bug.
    Delete this gate if the mode reads no per-atom field."""
    mol = _molecule(_embed("[2H]OC(=O)c1ccccc1"))
    assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
        "test premise broken: this molecule no longer retains an H after RemoveHs"
    pytest.fail("fill in the field checks for YOURMODE, or delete this gate")
    # field = mol.get_YOURFEATURE(no_H=True)
    # assert len(field) == len(mol._nonH_atoms_idx) != mol.atom_pos.shape[0]
    # inputs = get_mode(MODE, **PARAMS).inputs(mol)          # positions and values must agree
    # ...then score the deuterated molecule against itself: pair.score(MODE) == 1


# --- Gate 5: asymmetry (Tversky only) ---------------------------------------------------------
def test_tversky_is_asymmetric():
    """With a != b the score is not symmetric: score(A, B) != score(B, A). Gaussian-overlap
    Tversky can exceed one; pharmacophore Tversky clamps at one. Delete this gate for a mode
    that is only used with Tanimoto."""
    small, large = _embed("c1ccccc1"), _embed("c1ccccc1CCCCO")
    ab = MoleculePair(_molecule(small), _molecule(large), device=torch.device("cpu"))
    ba = MoleculePair(_molecule(large), _molecule(small), device=torch.device("cpu"))
    s_ab = float(ab.score(MODE, similarity="tversky", **PARAMS))
    s_ba = float(ba.score(MODE, similarity="tversky", **PARAMS))
    assert not np.isclose(s_ab, s_ba, atol=1e-3), (s_ab, s_ba)
