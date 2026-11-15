"""Accel parity gates for ``vol_color_avoid`` and its Tversky variant:

    (1 - color_weight) * shape + color_weight * color - avoid_weight * penalty

Layered as in the accelerate-scoring-mode gates: fixed-pose value and gradient against the
PyTorch reference, the fused shape+color launch against the separate kernels (and its size
fallback), the CUDA graph against the eager loop, then the optimizer (batched vs per-pair,
rescoring of the returned transform, Triton vs numba) and the screen path (array vs object, the
avoid cloud following the centered query, multi-GPU workers). Tests that apply to both run once
per similarity (``SIMS``). The decomposition into ``vol_color`` and the ``vol_avoid`` penalty is
checked on the reference (``test_vol_color_avoid.py``); matching the reference at fixed poses
carries it over.
"""
import copy
import dataclasses
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from . import _vca as V

MODE = "vol_color_avoid"
MODE_T = "vol_color_avoid_tversky"
TV = (0.95, 0.05)                    # the spec's tversky_alpha / tversky_beta defaults
#: non-default weights, so a swapped or mis-signed weight cannot pass
KW = dict(color_weight=0.3, avoid_weight=0.7, avoid_min_dist=2.5)
PAIRS = [(0, 1), (2, 3), (4, 0)]
#: Tanimoto and the named Tversky weights; the fixed-pose value also runs explicit weights
SIMS = ["tanimoto", TV]


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")


def _device(device):
    if device == "cuda":
        _cuda()
    return device


def _mode_of(sim):
    return MODE if sim == "tanimoto" else MODE_T


def _sim_kw(sim):
    """Engine parameters selecting ``sim`` on the spec ``_mode_of(sim)``."""
    return {} if sim == "tanimoto" else dict(tversky_alpha=sim[0], tversky_beta=sim[1])


def _ref_kw(sim):
    """The reference mode's keywords for ``sim``."""
    return {} if sim == "tanimoto" else dict(similarity=tuple(sim))


def _spec(mode=MODE, **over):
    from shepherd_score.accel._modes import SPECS
    return dataclasses.replace(SPECS[mode], **over) if over else SPECS[mode]


def _params(spec, **kw):
    p = dict(spec.params)
    p.update(kw)
    p["lr"] = float(spec.screen_lr)
    return p


@pytest.fixture(scope="module")
def pairs():
    return [V.pair(V.SMILES[i], V.SMILES[j]) for i, j in PAIRS]


@pytest.fixture(scope="module")
def clouds(pairs):
    # different K per pair, so the avoid pad is exercised alongside real-count masks
    return [V.near_cloud(p.ref_molec, n=n) for p, n in zip(pairs, (3, 6, 2))]


def _problem(pairs, clouds, device, sim="tanimoto", spec=None, num_seeds=4):
    import shepherd_score.accel.drivers.engine as EN
    spec = spec or _spec(_mode_of(sim))
    return EN.assemble(spec, V.chans(pairs, clouds, device),
                       params=_params(spec, **KW, **_sim_kw(sim)), num_seeds=num_seeds)


def _jitter(pr, seed=5, sq=0.3, st=0.5):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g).to(pr.q.device, pr.q.dtype)
    P = pr.q.shape[0]
    return F.normalize(pr.q + sq * rn(P, 4), dim=1), pr.t + st * rn(P, 3)


def _batch_pairs(dev):
    ps = [V.pair(V.SMILES[i], V.SMILES[j], device=dev) for i, j in PAIRS]
    return ps, [V.near_cloud(p.ref_molec, n=n) for p, n in zip(ps, (3, 6, 2))]


def _batch_similarity(sim):
    """What ``MoleculePairBatch.align_with_vol_color_avoid`` takes for ``sim``."""
    return "tanimoto" if sim == "tanimoto" else "tversky"

# 1. fixed-pose value and gradient against the reference


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("sim", SIMS + [(0.7, 0.3)])
def test_fixed_pose_value_matches_reference(pairs, clouds, device, sim):
    import shepherd_score.accel.drivers.engine as EN
    pr = _problem(pairs, clouds, _device(device), sim)
    q, t = _jitter(pr)
    sc = EN._score_poses(pr, q, t).cpu()
    ref = torch.tensor([V.ref_score_at(pairs[k // pr.P], clouds[k // pr.P], q[k], t[k],
                                       **_ref_kw(sim), **KW) for k in range(q.shape[0])])
    assert (ref < 0.5).any() and (sc.std() > 0.05), "fixture does not exercise the penalty"
    assert torch.allclose(sc, ref, rtol=1e-5, atol=1e-4), (sc - ref).abs().max()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("sim", SIMS)
def test_fine_step_gradient_follows_the_reported_score(pairs, clouds, device, sim):
    """The engine's blended descent gradient equals the derivative of the score it reports, on
    the tangent directions where that score is smooth (the hinge has kinks)."""
    import shepherd_score.accel.drivers.engine as EN
    pr = _problem(pairs, clouds, _device(device), sim, num_seeds=8)
    q, t = _jitter(pr)
    st = EN._State(q, t, 0.1)
    EN._step_generic(pr, st, update=False)
    g = torch.Generator().manual_seed(7)
    rn = lambda *s: torch.randn(*s, generator=g).to(q.device, q.dtype)
    P = q.shape[0]
    uq = rn(P, 4); uq = uq - q * (q * uq).sum(1, keepdim=True)
    ut = rn(P, 3)
    d_eng = -((st.gq * uq).sum(1) + (st.gt * ut).sum(1)).double()

    def fd(h):
        fp = EN._score_poses(pr, F.normalize(q + h * uq, dim=1), t + h * ut)
        fm = EN._score_poses(pr, F.normalize(q - h * uq, dim=1), t - h * ut)
        return ((fp - fm) / (2 * h)).double()
    f1, f2 = fd(1e-3), fd(1e-4)
    smooth = (f1 - f2).abs() <= 0.01 * f2.abs() + 1e-3
    assert smooth.double().mean() >= 0.5, "too few smooth directions to test"
    rel = (d_eng - f1).abs() / (f1.abs() + 1e-3)
    assert (rel[smooth] < 0.01).double().mean() >= 0.95, f"median rel err {rel[smooth].median():.2e}"

# 2. fused shape+color launch (CUDA)


@pytest.mark.cuda
@pytest.mark.parametrize("sim", SIMS)
def test_fused_pair_runs_and_matches_separate_kernels(pairs, clouds, sim):
    _cuda()
    import shepherd_score.accel.drivers.engine as EN
    import shepherd_score.accel.drivers.terms as T
    pr = _problem(pairs, clouds, "cuda", sim, num_seeds=8)
    assert pr.spec.fused_pair
    q, t = _jitter(pr)
    assert T.evaluate_fused_pair(pr.terms[0].spec, pr.terms[1].spec, pr.terms[0].inputs,
                                 pr.terms[1].inputs, q, t, pr.params) is not None, \
        "fused launch did not apply; the comparison below would be separate vs separate"
    out = []
    for fp in (True, False):
        pr.spec = _spec(_mode_of(sim), fused_pair=fp)
        st = EN._State(q, t, 0.1)
        s = EN._step_generic(pr, st, update=False)
        out.append((s, st.gq.clone(), st.gt.clone()))
    for a, b in zip(*out):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-5), (a - b).abs().max()


@pytest.mark.cuda
def test_fused_pair_falls_back_past_its_tile(pairs, clouds):
    """Past ``VOL_COLOR_FUSED_MAX_PAD`` the fused launch declines and the separate kernels run,
    matching the CPU step on the same (over-padded) inputs."""
    _cuda()
    import shepherd_score.accel.drivers.engine as EN
    import shepherd_score.accel.drivers.terms as T
    from shepherd_score.accel.drivers.engine import Batch
    from shepherd_score.accel.kernels.vol_color_triton import VOL_COLOR_FUSED_MAX_PAD

    def wide(device):
        ch = V.chans(pairs, clouds, device)
        a = ch["atoms"]
        w = VOL_COLOR_FUSED_MAX_PAD + 5 - a.ref.shape[1]
        ch["atoms"] = Batch(F.pad(a.ref, (0, 0, 0, w)), F.pad(a.fit, (0, 0, 0, w)),
                            a.n_real, a.m_real)
        return ch
    sp = _spec()
    res = []
    for dev in ("cuda", "cpu"):
        pr = EN.assemble(sp, wide(dev), params=_params(sp, **KW), num_seeds=4)
        if dev == "cuda":
            assert T.evaluate_fused_pair(pr.terms[0].spec, pr.terms[1].spec, pr.terms[0].inputs,
                                         pr.terms[1].inputs, pr.q, pr.t, pr.params) is None
        st = EN._State(pr.q, pr.t, 0.1)
        res.append((EN._step_generic(pr, st, update=False).cpu(), st.gq.cpu(), st.gt.cpu()))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max()

# 3. CUDA graph vs eager


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [MODE, "vol_color_avoid_tversky", "vol_avoid"])
def test_graph_runs_and_matches_eager(pairs, clouds, mode):
    """Matched seeds and a patience large enough that neither loop stops early: the graph and
    the eager loop return the same scores. The graph must actually have been captured."""
    _cuda()
    import shepherd_score.accel.drivers.engine as EN
    from shepherd_score.accel._modes import SPECS
    from shepherd_score.accel.drivers import _graphed
    ch = V.chans(pairs, clouds, "cuda")
    sp = SPECS[mode]
    assert sp.graph_budget is not None
    seeds = None
    out = {}
    for gb in (sp.graph_budget, None):
        s2 = dataclasses.replace(sp, graph_budget=gb)
        p = _params(s2, **{k: v for k, v in KW.items() if k in s2.params})
        if seeds is None:
            pr = EN.assemble(s2, ch, params=p, num_seeds=16)
            seeds = (pr.q.view(len(pairs), -1, 4).clone(), pr.t.view(len(pairs), -1, 3).clone())
        _graphed.reset_graph_cache()
        s, q, t = EN.align(s2, ch, params=p, num_seeds=16, steps_fine=40, lr=s2.screen_lr,
                           early_stop_patience=10_000, seeds=seeds)
        out[gb is not None] = s.cpu()
        if gb is not None:
            assert len(_graphed._FINE_GRAPH_CACHE) == 1, "graph capture fell back to eager"
    assert torch.allclose(out[True], out[False], atol=1e-4), (out[True], out[False])

# 4. optimizer


def test_numba_batched_matches_per_pair_engine():
    """``MoleculePairBatch`` (numba) equals the engine run directly on the pair's channels."""
    from shepherd_score.container import MoleculePairBatch
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS, SPECS
    from ._engine import run
    sp = SPECS[MODE]
    p0 = V.pair(V.SMILES[0], V.SMILES[1])
    avoid = V.near_cloud(p0.ref_molec, n=4)
    sc_pp, _, _ = run(MODE, V.chans([p0], [avoid]), num_seeds=MODE_SEEDS[MODE],
                      steps_fine=MODE_STEPS[MODE], lr=sp.screen_lr,
                      early_stop_patience=sp.patience, **KW)
    b = MoleculePairBatch([V.pair(V.SMILES[0], V.SMILES[1])])
    sc_b, _ = b.align_with_vol_color_avoid(avoid, backend="numba", **KW)
    assert abs(float(sc_pp[0]) - float(sc_b[0])) < 1e-4, (float(sc_pp[0]), float(sc_b[0]))


@pytest.mark.parametrize("backend", ["numba", pytest.param("triton", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("sim", SIMS)
def test_returned_transform_rescores_with_the_reference(backend, sim):
    """The batch fills only its similarity's slots, and the stored transform, applied to the
    fit, re-scores to the stored similarity under the PyTorch reference mode."""
    from shepherd_score.container import MoleculePairBatch
    from shepherd_score.modes import vol_color_avoid
    dev = "cuda" if backend == "triton" else "cpu"
    _device(dev)
    ps, avoid = _batch_pairs(dev)
    sc, _ = MoleculePairBatch(ps).align_with_vol_color_avoid(
        avoid, backend=backend, similarity=_batch_similarity(sim), **KW)
    slot, other = _mode_of(sim), (MODE_T if sim == "tanimoto" else MODE)
    for p, cl, s in zip(ps, avoid, sc):
        assert getattr(p, f"sim_aligned_{other}") is None
        assert abs(float(getattr(p, f"sim_aligned_{slot}")) - float(s)) < 1e-6
        mode = vol_color_avoid(avoid_points=cl, **_ref_kw(sim), **KW)
        ref, fit = p._mode_inputs(mode, p._as_tensor)
        se3 = torch.as_tensor(np.asarray(getattr(p, f"transform_{slot}")), dtype=torch.float32)
        posed = mode.pose(fit, se3.to(fit[0]["points"].device))
        with torch.no_grad():
            r = float(mode.score(ref, posed))
        assert abs(r - float(s)) < 1e-4, (r, float(s))


def test_batch_tversky_torch_fallback_uses_the_same_slots():
    """The per-pair PyTorch fallback runs the reference with the spec's Tversky weights."""
    from shepherd_score.container import MoleculePairBatch
    p0 = V.pair(V.SMILES[0], V.SMILES[1])
    cl = V.near_cloud(p0.ref_molec, n=4)
    sc, _ = MoleculePairBatch([p0]).align_with_vol_color_avoid(
        cl, backend="torch", similarity="tversky", num_repeats=4, max_num_steps=10, **KW)
    assert p0.sim_aligned_vol_color_avoid is None
    assert abs(float(p0.sim_aligned_vol_color_avoid_tversky) - float(sc[0])) < 1e-6


@pytest.mark.parametrize("mode", [MODE, "vol_avoid"])
def test_a_new_cloud_replaces_the_cached_one(mode):
    """Re-aligning the same pairs with a different cloud scores the new cloud, not the tensor
    cached by the previous call (it once silently reused the first cloud). The strong cloud
    must also lower the score."""
    from shepherd_score.container import MoleculePairBatch
    ps = [V.pair(V.SMILES[0], V.SMILES[1])]
    b = MoleculePairBatch(ps)
    run = getattr(b, f"align_with_{mode}")
    far = V.far_cloud(ps[0].ref_molec)
    strong = np.asarray(ps[0].ref_molec.atom_pos, dtype=np.float32)
    s_far, _ = run(far, backend="numba")
    s_strong, _ = run(strong, backend="numba")
    fresh, _ = getattr(MoleculePairBatch([V.pair(V.SMILES[0], V.SMILES[1])]),
                       f"align_with_{mode}")(strong, backend="numba")
    assert float(s_strong[0]) < float(s_far[0]) - 0.05
    assert np.isclose(float(s_strong[0]), float(fresh[0]), atol=1e-6)


def test_self_copy_far_cloud_scores_one():
    from shepherd_score.container import MoleculePairBatch
    p = V.pair(V.SMILES[0], V.SMILES[0])
    sc, _ = MoleculePairBatch([p]).align_with_vol_color_avoid(V.far_cloud(p.ref_molec),
                                                              backend="numba")
    assert np.isclose(float(sc[0]), 1.0, atol=1e-2), float(sc[0])


@pytest.mark.cuda
@pytest.mark.parametrize("sim", SIMS)
def test_triton_matches_numba_end_to_end(sim):
    _cuda()
    from shepherd_score.container import MoleculePairBatch
    out = {}
    for backend, dev in (("numba", "cpu"), ("triton", "cuda")):
        ps, avoid = _batch_pairs(dev)
        sc, _ = MoleculePairBatch(ps).align_with_vol_color_avoid(
            avoid, backend=backend, similarity=_batch_similarity(sim), **KW)
        out[backend] = np.asarray(sc, dtype=np.float64)
    assert np.allclose(out["numba"], out["triton"], atol=1e-2), out

# 5. screen path


_LIB = V.SMILES + ("CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    from shepherd_score.screen import ProfileStore
    p = os.path.join(tmp_path_factory.mktemp("vca"), "lib.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=[MODE, "vol_avoid"], dtype="float32",
                             pre_centered=True, canonical=False) as st:
        for i, s in enumerate(_LIB):
            st.add(V.mol(s, seed=i), id=i)
    return p


def _moved(m, shift):
    """A copy of ``m`` translated by ``shift`` (``center_to`` subtracts its argument)."""
    m = copy.deepcopy(m)
    m.center_to(-np.asarray(shift, dtype=np.float64))
    return m


def _screen_scores(store_path, backend, mode=MODE, shift=None, cloud=None, center_cloud=True,
                   **kw):
    """Hits for SMILES[0] with a near cloud; ``shift`` moves the query and its cloud together.
    The screen centers the query (pre-centered store) and never moves the cloud, so by default
    the cloud is centered the same way (the documented ``cloud - query.atom_pos.mean(0)``);
    ``center_cloud=False`` passes it raw."""
    from shepherd_score.screen import ProfileStore, screen
    from shepherd_score.accel._modes import SPECS
    q = V.mol(V.SMILES[0])
    cloud = V.near_cloud(q, n=4) if cloud is None else cloud
    if shift is not None:
        # float64, so moving the cloud loses nothing: a float32 cloud 30 A out rounds by ~1e-6 A,
        # which the translation-seeded slow path can amplify to ~1e-3 in a score.
        q, cloud = _moved(q, shift), np.asarray(cloud, np.float64) + np.asarray(shift)
    if center_cloud:
        cloud = np.asarray(cloud, np.float64) - np.asarray(q.atom_pos, np.float64).mean(0)
    mkw = {k: v for k, v in KW.items() if k in SPECS[mode].params}
    with ProfileStore.open(store_path) as st:
        hits = screen(q, st, mode=mode, backend=backend, top_k=len(_LIB),
                      avoid_points=cloud, **mkw, **kw)
    return hits


_FAR = (30.0, -20.0, 25.0)            # far enough that an unshifted cloud would never intrude


def _by_id(hits):
    return {h.id: h.score for h in hits}


def _same_hits(a, b, tol=1e-4):
    a, b = _by_id(a), _by_id(b)
    assert a.keys() == b.keys()
    for i in a:
        assert abs(a[i] - b[i]) <= tol, (i, a[i], b[i])


def test_screen_requires_avoid_points(store_path):
    from shepherd_score.screen import ProfileStore, screen
    with ProfileStore.open(store_path) as st:
        with pytest.raises(ValueError, match="vol_color_avoid requires an explicit avoid_points"):
            screen(V.mol(V.SMILES[0]), st, mode=MODE, backend="numba")


@pytest.mark.parametrize("backend", ["numba", pytest.param("triton", marks=pytest.mark.cuda)])
def test_screen_array_path_matches_object_path(monkeypatch, store_path, backend):
    import shepherd_score.screen as screenmod
    from shepherd_score.accel.batch import _arrays
    if backend == "triton":
        _cuda()
    if MODE not in screenmod._ARRAY_MODES:
        pytest.skip(f"{MODE} has no array path")
    res = {}
    for enabled in (True, False):
        monkeypatch.setattr(_arrays, "ENABLED", enabled, raising=True)
        assert screenmod._use_arrays(MODE) is enabled
        res[enabled] = _screen_scores(store_path, backend)
    a, b = res[True], res[False]
    assert len(a) == len(b) == len(_LIB)
    assert [(h.id, h.score) for h in a] == [(h.id, h.score) for h in b]     # bit-identical
    for ha, hb in zip(a, b):
        assert np.array_equal(np.asarray(ha.transform), np.asarray(hb.transform))


_PATHS = ["arrays", "objects", "slow"]


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("mode", [MODE, "vol_avoid"])
def test_screen_uses_the_cloud_as_is(monkeypatch, store_path, path, mode):
    """``avoid_points`` is used as-is, in the frame the (centered) query is scored in. Moving the
    query and its cloud 30+ A and centering the cloud by the query's centroid gives the same
    hits; passing the moved cloud raw leaves it behind, so the penalty vanishes (the screen
    does not shift it). Numba covers every path; Triton runs through the multi-GPU test."""
    import shepherd_score.screen as screenmod
    from shepherd_score.accel.batch import _arrays
    backend = "numba"
    if path == "arrays" and mode not in screenmod._ARRAY_MODES:
        pytest.skip(f"{mode} has no array path")
    monkeypatch.setattr(_arrays, "ENABLED", path == "arrays", raising=True)
    kw = {"trans_init": True} if path == "slow" else {}     # trans_init takes the slow path
    here = _screen_scores(store_path, backend, mode, **kw)
    moved = _screen_scores(store_path, backend, mode, shift=_FAR, **kw)
    _same_hits(here, moved, tol=1e-6)
    q = V.mol(V.SMILES[0])
    free = _screen_scores(store_path, backend, mode, cloud=V.far_cloud(q), **kw)
    assert max(abs(_by_id(free)[i] - s) for i, s in _by_id(moved).items()) > 1e-2, \
        "the cloud does not penalize; the test cannot tell a used cloud from a lost one"
    raw = _screen_scores(store_path, backend, mode, shift=_FAR, center_cloud=False, **kw)
    _same_hits(raw, free, tol=1e-6)                     # not shifted: 30+ A away, no penalty


def test_screen_many_shares_one_cloud(store_path):
    """One cloud for two queries serves each unchanged: ``screen_many`` hits equal per-query
    ``screen`` hits with the same cloud."""
    from shepherd_score.screen import ProfileStore, screen, screen_many
    from shepherd_score.accel._modes import SPECS
    m0, m1 = V.mol(V.SMILES[0]), V.mol(V.SMILES[1])
    cloud = V.near_cloud(m0, n=4) - np.asarray(m0.atom_pos).mean(0)     # m0's centered frame
    mkw = {k: v for k, v in KW.items() if k in SPECS[MODE].params}
    with ProfileStore.open(store_path) as st:
        both = screen_many([m0, m1], st, mode=MODE, backend="numba", top_k=len(_LIB),
                           avoid_points=cloud, **mkw)
        for m, hits in zip((m0, m1), both):
            alone = screen(m, st, mode=MODE, backend="numba", top_k=len(_LIB),
                           avoid_points=cloud, **mkw)
            _same_hits(hits, alone, tol=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [MODE, "vol_avoid"])
def test_screen_multi_gpu_uses_the_cloud(store_path, mode):
    """The ``ndev>1`` workers get the query's cloud (they once dropped it entirely): two GPUs
    on the moved scene match one GPU on the unmoved one."""
    _cuda()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    from shepherd_score.screen import close_multigpu_pool
    one = _screen_scores(store_path, "triton", mode)
    try:
        two = _screen_scores(store_path, "triton", mode, shift=_FAR, ndev=2)
    finally:
        close_multigpu_pool()
    _same_hits(one, two)          # 1e-4: one GPU vs two

# 6. Tversky-only checks (the shared gates above run for both similarities)


def test_tversky_spec_defaults_match_the_named_similarity():
    from shepherd_score.accel._modes import SPECS
    from shepherd_score.score.similarity import tversky_weights
    sp = SPECS[MODE_T]
    assert (sp.params["tversky_alpha"], sp.params["tversky_beta"]) == tversky_weights("tversky")
    assert [t.reduction for t in sp.terms] == ["tversky", "tversky", "raw"]
    assert sp.fused_pair and sp.graph_budget is not None


def test_tversky_screen_runs_from_a_vol_color_avoid_store(store_path):
    """A store built for vol_color_avoid also serves its Tversky variant (same channels).
    Array-vs-object parity for it is covered by the registry-driven test_screen_arrays."""
    from shepherd_score.screen import ProfileStore, screen
    with ProfileStore.open(store_path) as st:
        assert st.supports(MODE_T)
    q = V.mol(V.SMILES[0])
    cloud = V.near_cloud(q, n=4)
    with ProfileStore.open(store_path) as st:
        hits = screen(q, st, mode=MODE_T, backend="numba", top_k=3, avoid_points=cloud, **KW)
    assert len(hits) == 3 and hits[0].score >= hits[-1].score
