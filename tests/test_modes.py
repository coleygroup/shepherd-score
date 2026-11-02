"""``MoleculePair.align`` / ``score`` with named and user-defined modes, and the ``similarity``
argument that replaces the separate ``*_tversky`` modes."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
import torch

from shepherd_score.container import MoleculePairBatch
from shepherd_score.alignment import optimize_ROCS_overlay
from shepherd_score.modes import MODES, Avoid, Combined, Mode, Vol, VolAvoid, blend, get_mode, vol_color
from shepherd_score.score.gaussian_overlap import VAB_2nd_order
from shepherd_score.score.pharmacophore_scoring import tversky_func
from shepherd_score.score.similarity import similarity_from_overlaps

from .legacy_fixture import make_pair, load_molecules, NUM_SURF_POINTS

FAST = dict(num_repeats=8, max_num_steps=40)


def _self_pair():
    from shepherd_score.container import MoleculePair
    m = load_molecules()[0]
    m.fukui = np.linspace(-0.2, 0.2, m.mol.GetNumAtoms())   # the real field needs xTB
    return MoleculePair(m, m, num_surf_points=NUM_SURF_POINTS, device=torch.device('cpu'))


def _far_cloud(pair):
    return pair.ref_molec.atom_pos + 40.0


@pytest.mark.parametrize('name', sorted(MODES))
@pytest.mark.parametrize('similarity', ['tanimoto', 'tversky'])
def test_every_mode_scores_one_on_itself(name, similarity):
    pair = _self_pair()
    kw = {'avoid_points': _far_cloud(pair)} if name == 'vol_avoid' else {}
    assert np.isclose(pair.score(name, similarity=similarity, **kw), 1.0, atol=1e-5)
    res = pair.align(name, similarity=similarity, **FAST, **kw)
    assert np.isclose(res.score, 1.0, atol=5e-3)
    assert res.transform.shape == (4, 4)


@pytest.mark.parametrize('name, kw', [
    ('vol', {}),
    ('vol_esp', {}),
    ('surf', {'alpha': 0.81}),
    ('surf_esp', {'alpha': 0.81}),
    ('vol_lipo', {}),
    ('vol_color', {}),
    ('vol_and_surf_esp', {'alpha': 0.81}),
])
def test_similarity_argument_stores_in_tversky_slot(name, kw):
    pair = make_pair(0, 1)
    assert not hasattr(pair, f'align_with_{name}_tversky')
    res = pair.align(name, similarity='tversky', **FAST, **kw)
    getattr(pair, f'align_with_{name}')(similarity='tversky', **FAST, **kw)
    assert np.allclose(res.score, getattr(pair, f'sim_aligned_{name}_tversky'), atol=1e-6)
    assert np.allclose(res.transform, getattr(pair, f'transform_{name}_tversky'), atol=1e-5)


def test_pharm_similarity_keeps_its_slot():
    pair = make_pair(0, 1)
    res = pair.align('pharm', similarity='tversky', **FAST)
    pair.align_with_pharm(similarity='tversky', **FAST)
    assert np.allclose(res.score, pair.sim_aligned_pharm, atol=1e-6)


def test_vol_color_similarity_covers_both_channels():
    pair = make_pair(0, 1)
    pair.align_with_vol_color(similarity=(0.7, 0.3), **FAST)
    both = pair.align(vol_color(similarity=(0.7, 0.3), color_similarity=(0.7, 0.3)), **FAST)
    assert np.allclose(pair.sim_aligned_vol_color_tversky, both.score, atol=1e-6)
    # color-only Tversky keeps the Tanimoto slot
    pair.align_with_vol_color(color_similarity='tversky', **FAST)
    color = pair.align(vol_color(color_similarity='tversky'), **FAST)
    assert np.allclose(pair.sim_aligned_vol_color, color.score, atol=1e-6)


def test_pharm_weights_match_names():
    pair = make_pair(0, 1)
    for name, weights in [('tversky', (0.95, 0.05)), ('tversky_ref', (1.0, 0.0))]:
        assert np.isclose(pair.score('pharm', similarity=name),
                          pair.score('pharm', similarity=weights), atol=1e-6)
    assert np.isfinite(pair.score('pharm', similarity=(0.7, 0.3)))


def _near_cloud(pair):
    rng = np.random.default_rng(0)
    return (pair.ref_molec.atom_pos.mean(0) + [3.0, 0, 0] + rng.normal(size=(20, 3))).astype(np.float32)


def test_vol_avoid_matches_the_old_optimizer():
    pair = make_pair(0, 1)
    avoid = _near_cloud(pair)
    t = pair._to_tensor
    _, se3, ref = optimize_ROCS_overlay(t(pair.ref_molec.get_positions(True)),
                                        t(pair.fit_molec.get_positions(True)), 0.81,
                                        avoid_points=t(avoid), num_repeats=16, max_num_steps=50)
    pair.align_with_vol_avoid(avoid, use_analytical=False, num_repeats=16, max_num_steps=50)
    assert np.isclose(pair.sim_aligned_vol_avoid, float(ref), atol=1e-6)
    assert np.allclose(pair.transform_vol_avoid, se3.numpy(), atol=1e-4)
    pair.align_with_vol_avoid(avoid, num_repeats=16, max_num_steps=50)    # analytical
    assert np.isclose(pair.sim_aligned_vol_avoid, float(ref), atol=1e-4)


def test_vol_avoid_is_vol_minus_weighted_penalty():
    pair = make_pair(0, 1)
    avoid = _near_cloud(pair)
    blended = Combined(terms=((1.0, Vol()), (-0.5, Avoid(avoid, min_dist=3.0))))
    assert np.isclose(pair.score(VolAvoid(avoid_points=avoid, avoid_weight=0.5, avoid_min_dist=3.0)),
                      pair.score(blended), atol=1e-5)
    assert np.isclose(pair.score(VolAvoid(avoid_points=_far_cloud(pair))), pair.score('vol'), atol=1e-6)
    with pytest.raises(ValueError, match='avoid_points'):
        VolAvoid()


def test_combined_poses_shared_arrays_once():
    pair = make_pair(0, 1)
    mode = Combined(terms=((1.0, Vol()), (-1.0, Avoid(_near_cloud(pair)))))
    ref, fit = pair._mode_inputs(mode, pair._as_tensor)
    assert fit[0]['points'] is fit[1]['points']
    posed = mode.pose(fit, torch.eye(4, device=fit[0]['points'].device))
    assert posed[0]['points'] is posed[1]['points']
    # vol_color's terms read different arrays and stay separate
    ref, fit = pair._mode_inputs(vol_color(), pair._as_tensor)
    assert fit[0]['points'] is not fit[1]['anchors']


def test_tversky_name_and_weights_agree():
    pair = make_pair(0, 1)
    assert np.isclose(pair.score('vol', similarity='tversky'), pair.score('vol', similarity=(0.95, 0.05)))
    assert not np.isclose(pair.score('vol', similarity='tversky'), pair.score('vol', similarity='tversky_fit'))


def test_tversky_matches_sigma_form_when_weights_sum_to_one():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(9, 3, generator=g), torch.randn(7, 3, generator=g) + 0.3
    AB, AA, BB = VAB_2nd_order(a, b, 0.81), VAB_2nd_order(a, a, 0.81), VAB_2nd_order(b, b, 0.81)
    for sigma in (0.95, 0.5, 0.05):
        # tversky_func clamps at 1 (asymmetric weights can exceed it); the weights form does not.
        assert torch.isclose(similarity_from_overlaps(AB, AA, BB, (sigma, 1 - sigma)).clamp_max(1.0),
                             tversky_func(AB, AA, BB, sigma))


def test_score_matches_legacy_scorers():
    pair = make_pair(0, 1)
    assert np.isclose(pair.score('surf'), pair.score_with_surf(0.81), atol=1e-5)
    assert np.isclose(pair.score('surf_esp', lam=0.3), pair.score_with_esp(0.81, lam=0.3), atol=1e-5)
    assert np.isclose(pair.score('pharm'), pair.score_with_pharm(), atol=1e-5)


def test_backends():
    pair = make_pair(0, 1)
    auto = pair.align('vol', **FAST)
    assert np.allclose(auto.score, pair.align('vol', backend='analytical', **FAST).score)
    torch_res = pair.align('vol', backend='torch', **FAST)
    assert np.isclose(auto.score, torch_res.score, atol=1e-3)
    with pytest.raises(ValueError, match='not available'):
        pair.align('vol', similarity='tversky', backend='analytical')


def test_bad_arguments():
    with pytest.raises(ValueError, match='Unknown mode'):
        get_mode('volume')
    with pytest.raises(TypeError):
        get_mode('vol', lam=0.3)
    with pytest.raises(TypeError):
        get_mode(Vol(), alpha=1.0)
    with pytest.raises(ValueError, match='Unknown similarity'):
        make_pair().score('vol', similarity='dice')


@dataclass(frozen=True)
class HeavyAtomCount(Mode):
    """A user mode defined outside the package: Gaussian overlap of nitrogen and oxygen atoms."""
    name: ClassVar[str] = 'n_o'
    alpha: float = 0.5
    similarity: str = 'tanimoto'

    def inputs(self, mol):
        z = mol.get_atomic_numbers(no_H=False)
        return {'points': mol.get_positions(no_H=False)[(z == 7) | (z == 8)]}

    def overlap(self, a, b):
        return VAB_2nd_order(a['points'], b['points'], self.alpha)


def test_user_defined_mode_and_blend():
    pair = make_pair(0, 1)
    res = pair.align(HeavyAtomCount(), **FAST)
    assert 0 < res.score <= 1
    mixed = blend(Vol(), HeavyAtomCount(similarity='tversky'), 0.3, 'vol_n_o')
    assert isinstance(mixed, Combined)
    res = pair.align(mixed, **FAST)
    assert 0 < res.score <= 1 and len(res.aligned) == 2


def test_batch_align_dispatches_similarity():
    pairs = [make_pair(0, 1), make_pair(2, 3)]
    MoleculePairBatch(pairs).align('vol', similarity='tversky', backend='numba')
    assert all(p.sim_aligned_vol_tversky is not None for p in pairs)
    assert all(p.sim_aligned_vol_noH is None for p in pairs)


def test_batch_similarity_runs_the_tversky_driver():
    batch = MoleculePairBatch([make_pair(0, 1), make_pair(2, 3)])
    assert not hasattr(batch, 'align_with_vol_tversky')
    scores, aligned = batch.align_with_vol(backend='numba', similarity=(0.7, 0.3),
                                           return_aligned=True)
    assert all(a is not None for a in aligned)
    ref, _ = batch._align_mode('vol_tversky', backend='numba', tversky_alpha=0.7, tversky_beta=0.3)
    assert np.allclose(scores, ref, atol=1e-6)
    assert np.allclose(scores, [p.sim_aligned_vol_tversky for p in batch.pairs], atol=1e-6)


def test_batch_align_runs_registry_tversky_modes():
    batch = MoleculePairBatch([make_pair(0, 1), make_pair(2, 3)])
    scores, _ = batch.align('vol_tversky', backend='numba', tversky_alpha=0.7, tversky_beta=0.3)
    ref, _ = batch.align_with_vol(backend='numba', similarity=(0.7, 0.3))
    assert np.allclose(scores, ref, atol=1e-6)
    scores, *_ = batch.align('pharm_tversky', backend='numba')
    assert np.allclose(scores, [p.sim_aligned_pharm_tversky for p in batch.pairs], atol=1e-6)
    with pytest.raises(ValueError, match='already a Tversky mode'):
        batch.align('vol_tversky', similarity='tversky')


def test_batch_tversky_keeps_the_heavy_atom_guard():
    batch = MoleculePairBatch([make_pair(0, 1)])
    with pytest.raises(NotImplementedError, match='heavy atoms only'):
        batch.align_with_vol(backend='numba', similarity='tversky', no_H=False)
