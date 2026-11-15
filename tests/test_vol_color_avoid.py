"""Reference-mode gates for ``vol_color_avoid``: the ``vol_color`` shape + color blend minus the
``vol_avoid`` linear hard-sphere penalty against a fixed avoid cloud in the reference frame,

    (1 - color_weight) * shape + color_weight * color - avoid_weight * penalty.

The mode reads no new per-atom field (shape atoms, stored pharmacophores and a caller-supplied
cloud), so the retained-H gate of the template does not apply. Tversky asymmetry belongs to the
underlying shape and color terms; here it is checked only through the decomposition.
"""
import copy

import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from shepherd_score.alignment._optimize import tree_map
from shepherd_score.alignment.utils.se3 import get_SE3_transform
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.modes import Avoid, get_mode, vol_color, vol_color_avoid

from .utils import mode_loss


MODE = "vol_color_avoid"
SEED = 0
FAST = dict(num_repeats=20, max_num_steps=200)


def _embed(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0
    return m


def _molecule(m):
    return Molecule(Chem.Mol(m), charge_model="mmff", pharm_multi_vector=False)


def _make_pair(smi="c1ccccc1CCO", fit_smi=None):
    ref = _embed(smi)
    fit = ref if fit_smi is None else _embed(fit_smi)
    return MoleculePair(_molecule(ref), _molecule(fit), device=torch.device("cpu"))


def _far_cloud(pair):
    return pair.ref_molec.atom_pos + 40.0


def _near_cloud(pair):
    """Points on one side of the reference, inside its shape, so the penalty is active."""
    pos = pair.ref_molec.atom_pos
    c = pos.mean(0)
    axis = np.linalg.svd(pos - c)[2][0]
    side = pos[(pos - c) @ axis > 0]
    return side + 0.3 * axis


def _tensors(mode, molec):
    def convert(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        return t.to(torch.float32) if t.is_floating_point() else t
    return tree_map(convert, mode.inputs(molec))


def _penalty(pair, transform, cloud):
    """The raw penalty of the fit's heavy atoms under ``transform``."""
    avoid = Avoid(cloud)
    fit = _tensors(avoid, pair.fit_molec)
    posed = avoid.pose(fit, torch.as_tensor(transform, dtype=torch.float32))
    return float(avoid.score(None, posed))


# --- Gate 1: self-score ------------------------------------------------------------------------
@pytest.mark.parametrize("similarity", ["tanimoto", "tversky"])
def test_self_score(similarity):
    """With the cloud out of reach the penalty is 0, so the self-score is vol_color's: 1."""
    pair = _make_pair()
    far = _far_cloud(pair)
    assert np.isclose(float(pair.score(MODE, avoid_points=far, similarity=similarity)), 1.0, atol=1e-5)
    pair.align_with_vol_color_avoid(far, similarity=similarity, **FAST)
    slot = "sim_aligned_vol_color_avoid" + ("_tversky" if similarity == "tversky" else "")
    assert np.isclose(float(getattr(pair, slot)), 1.0, atol=1e-3)


def test_self_score_with_active_penalty():
    """At the identity a cloud inside the molecule lowers the score by exactly the weighted
    penalty (not a unit self-score: the objective is not a normalized overlap)."""
    pair = _make_pair()
    near = _near_cloud(pair)
    p = _penalty(pair, np.eye(4), near)
    assert p > 0.5
    s = float(pair.score(MODE, avoid_points=near, avoid_weight=0.25))
    assert np.isclose(s, 1.0 - 0.25 * p, atol=1e-5)


# --- Decomposition into the existing modes ------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {},
    {"color_weight": 0.3, "avoid_weight": 0.5, "avoid_min_dist": 3.0},
    {"similarity": "tversky"},
    {"similarity": "tanimoto", "color_similarity": "tversky_ref", "directionless": False},
])
def test_is_vol_color_minus_weighted_penalty(kw):
    pair = _make_pair(fit_smi="c1ccccc1CCN")
    near = _near_cloud(pair)
    color_kw = {k: v for k, v in kw.items() if k not in ("avoid_weight", "avoid_min_dist")}
    expected = (float(pair.score(vol_color(**color_kw)))
                - kw.get("avoid_weight", 1.0)
                * float(pair.score(Avoid(near, min_dist=kw.get("avoid_min_dist", 2.0)))))
    got = float(pair.score(MODE, avoid_points=near, **kw))
    assert np.isclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("no_H", [True, False])
def test_is_built_from_vol_color(no_H):
    """``vol_color_avoid`` is ``vol_color``'s terms plus the penalty, and ``no_H`` reaches the
    shape term and the penalty alike (all-atom scoring on the reference paths)."""
    pair = _make_pair(fit_smi="c1ccccc1CCN")
    near = _near_cloud(pair)
    mode = vol_color_avoid(near, no_H=no_H)
    base = vol_color(no_H=no_H)
    assert [(w, type(t), getattr(t, "no_H", None)) for w, t in mode.terms[:2]] == \
           [(w, type(t), getattr(t, "no_H", None)) for w, t in base.terms]
    assert mode.terms[0][1].no_H is no_H and mode.terms[2][1].no_H is no_H
    expected = (float(pair.score(base))
                - float(pair.score(Avoid(near, min_dist=2.0, no_H=no_H))))
    assert np.isclose(float(pair.score(mode)), expected, atol=1e-6)


def test_requires_avoid_points():
    with pytest.raises(ValueError, match="avoid_points"):
        vol_color_avoid()
    with pytest.raises(ValueError, match="avoid_points"):
        _make_pair().align(MODE)


# --- Gate 2: autograd matches central finite differences ---------------------------------------
def test_autograd_matches_finite_difference():
    """Distinct molecules, non-identity pose, active penalty (the relu kinks are measure zero;
    the assertion below checks no pair sits within the FD step of one)."""
    pair = _make_pair(fit_smi="c1ccccc1CCN")
    near = _near_cloud(pair)
    mode = get_mode(MODE, avoid_points=near)
    ref, fit = _tensors(mode, pair.ref_molec), _tensors(mode, pair.fit_molec)

    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    posed = mode.pose(fit, get_SE3_transform(se3.detach()))[2]['points']
    d = torch.cdist(posed, torch.as_tensor(near, dtype=torch.float32))
    assert (d < 2.0).any(), "test premise broken: penalty inactive at the test pose"
    assert ((d - 2.0).abs() > 1e-2).all(), "a pair sits on the hard-sphere boundary"

    mode_loss(mode, se3, ref, fit).backward()
    ana = se3.grad.clone()
    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        step = torch.zeros_like(se3.data)
        step[i] = eps
        num[i] = (mode_loss(mode, se3.data + step, ref, fit)
                  - mode_loss(mode, se3.data - step, ref, fit)) / (2 * eps)
    assert torch.allclose(ana, num, atol=5e-3), f"autograd {ana} vs FD {num}"


# --- Gate 3: the optimizer recovers a planted pose ---------------------------------------------
def test_optimizer_recovers_planted_pose():
    m = _embed("c1ccccc1CCO")
    moved = Chem.Mol(m)
    T = get_SE3_transform(torch.tensor([0.924, 0.383, 0.0, 0.0, 1.0, -0.5, 0.3])).numpy()
    rdMolTransforms.TransformConformer(moved.GetConformer(), T.astype(np.float64))
    pair = MoleculePair(_molecule(m), _molecule(moved), device=torch.device("cpu"))
    far = _far_cloud(pair)

    res = pair.align(MODE, avoid_points=far, **FAST)
    assert float(res.score) > 0.95, f"planted-pose recovery only reached {float(res.score):.3f}"

    mode = get_mode(MODE, avoid_points=far)
    ref, fit = _tensors(mode, pair.ref_molec), _tensors(mode, pair.fit_molec)
    posed = mode.pose(fit, torch.as_tensor(res.transform, dtype=torch.float32))
    assert np.isclose(float(mode.score(ref, posed)), float(res.score), atol=1e-4)


# --- The penalty steers the optimizer -----------------------------------------------------------
def test_penalty_moves_the_pose_away_from_the_cloud():
    """Against the same pair, vol_color_avoid ends with a smaller penalty than vol_color's pose
    and a better vol_color_avoid objective; its reported score rescores from its transform."""
    torch.manual_seed(SEED)
    pair = _make_pair(fit_smi="c1ccccc1CCN")
    near = _near_cloud(pair)
    pair.align_with_vol_color(**FAST)
    pair.align_with_vol_color_avoid(near, **FAST)
    p_color = _penalty(pair, pair.transform_vol_color, near)
    p_avoid = _penalty(pair, pair.transform_vol_color_avoid, near)
    assert p_avoid < p_color, (p_avoid, p_color)

    mode = get_mode(MODE, avoid_points=near)
    ref, fit = _tensors(mode, pair.ref_molec), _tensors(mode, pair.fit_molec)

    def objective(T):
        return float(mode.score(ref, mode.pose(fit, torch.as_tensor(T, dtype=torch.float32))))
    assert np.isclose(objective(pair.transform_vol_color_avoid),
                      float(pair.sim_aligned_vol_color_avoid), atol=1e-4)
    assert objective(pair.transform_vol_color_avoid) >= objective(pair.transform_vol_color) - 1e-4


# --- Gate 4: determinism ----------------------------------------------------------------------
def test_deterministic_given_seed():
    results = []
    for _ in range(2):
        torch.manual_seed(SEED)
        pair = _make_pair(fit_smi="c1ccccc1CCN")
        pair.align_with_vol_color_avoid(_near_cloud(pair))
        results.append((float(pair.sim_aligned_vol_color_avoid), pair.transform_vol_color_avoid))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])


def test_tversky_stores_in_the_tversky_slot():
    """A Tversky similarity lands in the vol_color_avoid_tversky slot, as for the other modes,
    and leaves the Tanimoto slot untouched."""
    pair = _make_pair(fit_smi="c1ccccc1CCN")
    far = _far_cloud(pair)
    torch.manual_seed(SEED)
    res = pair.align(MODE, avoid_points=far, similarity="tversky", num_repeats=8, max_num_steps=40)
    torch.manual_seed(SEED)
    pair.align_with_vol_color_avoid(far, similarity="tversky", num_repeats=8, max_num_steps=40)
    assert np.isclose(float(res.score), float(pair.sim_aligned_vol_color_avoid_tversky), atol=1e-6)
    assert pair.sim_aligned_vol_color_avoid is None


@pytest.mark.parametrize("name", ["vol_color", "vol_color_avoid"])
def test_missing_pharmacophores_error_names_the_mode(name):
    """A Combined mode's error names its label, not the class ('Combined needs ...')."""
    m = _embed("c1ccccc1CCO")
    pair = MoleculePair(Molecule(Chem.Mol(m)), Molecule(Chem.Mol(m)), device=torch.device("cpu"))
    kw = {"avoid_points": pair.ref_molec.atom_pos + 40.0} if name.endswith("avoid") else {}
    with pytest.raises(ValueError, match=f"^{name} needs pharmacophores"):
        pair.score(name, **kw)


def test_avoid_cloud_is_used_as_is():
    """``do_center`` moves the reference but never the avoid cloud: the batch path's channel
    reader returns the cloud unchanged (only reordered), and ``align`` and ``score`` see the
    same cloud, so scoring the aligned pose reproduces the alignment's score."""
    from shepherd_score.accel.channels import CHANNELS
    m = _embed("c1ccccc1CCO")
    pair = MoleculePair(_molecule(m), _molecule(m), do_center=True, device=torch.device("cpu"))
    cloud = _near_cloud(pair).astype(np.float32)            # already in the centered frame
    pair.avoid_points = cloud
    got = CHANNELS["avoid"].read(pair)
    np.testing.assert_array_equal(got[np.lexsort(got.T)], cloud[np.lexsort(cloud.T)])

    kw = dict(avoid_points=cloud, avoid_weight=1.0, avoid_min_dist=2.0)
    res = pair.align("vol_avoid", num_repeats=4, max_num_steps=50, **kw)
    posed = copy.deepcopy(pair)
    posed.fit_molec = pair.get_transformed_molecule(res.transform)
    assert abs(float(posed.score("vol_avoid", **kw)) - float(res.score)) < 1e-3
