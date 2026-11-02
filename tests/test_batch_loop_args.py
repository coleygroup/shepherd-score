"""MoleculePairBatch forwards num_repeats / lr / trans_init to the batched drivers, resolves
``None`` to the registry defaults and ``screen_lr`` (accel) or 50 / 0.1 / 200 (jax, torch), and warns when
``backend='jax'`` has no JAX kernel."""
import warnings

import pytest

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from shepherd_score.accel._modes import SPECS
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch


def _backend():
    """The default batched backend, skipping when it is not installed."""
    import torch
    if torch.cuda.is_available():
        pytest.importorskip("triton")
        return 'triton'
    pytest.importorskip("numba")
    return 'numba'


def _mol(smiles, seed):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(m, params)
    return Molecule(m, num_surf_points=40, charge_model='mmff', pharm_multi_vector=False)


def _batch():
    return MoleculePairBatch([MoleculePair(_mol('c1ccccc1O', 1), _mol('c1ccccc1N', 2),
                                           num_surf_points=40)])


class _Recorded(Exception):
    pass


def _record(monkeypatch, mode):
    """Replace ``accel.batch.aligners._align_batch_<mode>`` with a stub that records its keywords."""
    from shepherd_score.accel.batch import aligners
    seen = {}

    def stub(pairs, **kw):
        seen.update(kw)
        raise _Recorded

    monkeypatch.setattr(aligners, f'_align_batch_{mode}', stub)
    return seen


@pytest.mark.parametrize('mode', ['vol', 'surf'])
def test_explicit_loop_args_reach_driver(monkeypatch, mode):
    seen = _record(monkeypatch, mode)
    kw = {'alpha': 0.81} if mode == 'surf' else {}
    with pytest.raises(_Recorded):
        getattr(_batch(), f'align_with_{mode}')(
            num_repeats=7, lr=0.2, trans_init=True, max_num_steps=11, backend=_backend(), **kw)
    assert seen['num_repeats'] == 7
    assert seen['lr'] == 0.2
    assert seen['trans_init'] is True
    assert seen['steps_fine'] == 11


@pytest.mark.parametrize('mode', ['vol', 'surf', 'vol_esp', 'vol_color'])
def test_none_loop_args_take_registry_defaults(monkeypatch, mode):
    from shepherd_score.accel.drivers import engine
    seen = []
    real = engine.align

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(engine, 'align', spy)
    kw = {'alpha': 0.81} if mode == 'surf' else {}
    getattr(_batch(), f'align_with_{mode}')(backend=_backend(), **kw)
    spec = SPECS[mode]
    assert seen
    assert {c['num_seeds'] for c in seen} == {spec.seeds}
    assert {c['steps_fine'] for c in seen} == {spec.steps}
    assert {c['lr'] for c in seen} == {spec.screen_lr}


def test_none_lr_matches_main():
    """An unset accel ``lr`` is 0.1, as on main, except for the six modes main left at 0.075."""
    at_0075 = {'vol', 'surf', 'vol_tversky', 'vol_esp_tversky', 'surf_tversky',
               'surf_esp_tversky'}
    assert {n for n, s in SPECS.items() if s.screen_lr == 0.075} == at_0075
    assert all(s.screen_lr == 0.1 for n, s in SPECS.items() if n not in at_0075)


def test_engine_honors_num_repeats(monkeypatch):
    from shepherd_score.accel.drivers import engine
    seeds = []
    real = engine.align

    def spy(*a, **kw):
        seeds.append(kw['num_seeds'])
        return real(*a, **kw)

    monkeypatch.setattr(engine, 'align', spy)
    _batch().align_with_vol(num_repeats=3, max_num_steps=2, backend=_backend())
    assert seeds and set(seeds) == {3}


def test_jax_backend_warns_without_jax_kernel():
    with pytest.warns(RuntimeWarning, match='no JAX implementation'):
        _batch().align_with_vol_color(backend='jax', num_repeats=1, max_num_steps=1)
    pytest.importorskip('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _batch().align_with_vol(backend='jax', num_repeats=1, max_num_steps=1)


def _spy_pair_align(monkeypatch):
    calls = []
    real = MoleculePair.align

    def spy(self, mode, **kw):
        calls.append((mode, kw))
        return real(self, mode, **kw)

    monkeypatch.setattr(MoleculePair, 'align', spy)
    return calls


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_reference_backends_default_to_50_200(monkeypatch, backend):
    """``jax`` / ``torch`` take MoleculePair's 50 seeds / lr 0.1 / 200 steps, not the registry's."""
    calls = _spy_pair_align(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _batch().align_with_vol_lipo(backend=backend)
    (mode, kw), = calls
    assert (kw['num_repeats'], kw['lr'], kw['max_num_steps']) == (50, 0.1, 200)


def test_jax_vol_defaults_and_alpha(monkeypatch):
    """The padded JAX ``vol`` / ``vol_esp`` paths take 50 / 0.1 / 200 and pass ``alpha`` through."""
    pytest.importorskip('jax')
    import shepherd_score.alignment_jax as aj
    for name in ('optimize_ROCS_overlay_jax_mask', 'optimize_ROCS_esp_overlay_jax_mask'):
        seen = {}

        def stub(**kw):
            seen.update(kw)
            raise _Recorded

        monkeypatch.setattr(aj, name, stub)
        method = 'align_with_vol' if 'esp' not in name else 'align_with_vol_esp'
        with pytest.raises(_Recorded):
            getattr(_batch(), method)(backend='jax', alpha=0.5)
        assert seen['alpha'] == 0.5
        assert (seen['num_repeats'], seen['lr'], seen['max_num_steps']) == (50, 0.1, 200)


def test_vol_esp_forwards_num_workers(monkeypatch):
    from shepherd_score.accel import cpu_pool
    pytest.importorskip('numba')
    seen = {}

    def stub(mode, pairs, num_workers, kw):
        seen.update(mode=mode, num_workers=num_workers)
        raise _Recorded

    monkeypatch.setattr(cpu_pool, 'align_pairs', stub)
    with pytest.raises(_Recorded):
        _batch().align_with_vol_esp(backend='numba', num_workers=2)
    assert seen == {'mode': 'vol_esp', 'num_workers': 2}


def test_vol_avoid_reference_path_alpha_and_aligned(monkeypatch):
    calls = _spy_pair_align(monkeypatch)
    batch = _batch()
    cloud = batch.pairs[0].ref_molec.atom_pos + 30.0
    scores, aligned = batch.align_with_vol_avoid(cloud, alpha=0.6, backend='torch',
                                                 num_repeats=2, max_num_steps=3,
                                                 return_aligned=True)
    (mode, kw), = calls
    assert mode == 'vol_avoid' and kw['alpha'] == 0.6
    assert aligned[0].shape == batch.pairs[0].fit_molec.atom_pos.shape
    assert scores[0] == pytest.approx(float(batch.pairs[0].sim_aligned_vol_avoid))
    _, none = batch.align_with_vol_avoid(cloud, backend='torch', num_repeats=2, max_num_steps=3,
                                         return_aligned=False)
    assert none == [None]


def test_tversky_accepts_trans_init():
    """The Tversky registry modes take ``trans_init`` like every other mode."""
    scores, _ = _batch().align_with_vol(similarity='tversky', trans_init=True, num_repeats=2,
                                        max_num_steps=3, backend=_backend())
    assert scores.shape == (1,) and scores[0] > 0
