"""The multi-GPU screen worker (``_screen_worker``) must dispatch per mode, selecting the same
builder/aligner pair as ``_run_shards_inproc``. The worker is driven in process with its
GPU-facing edges stubbed, so this runs without a GPU; the one end-to-end check needs two.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import shepherd_score.screen as scr                               # noqa: E402

#: Only ``steps_fine`` is read unconditionally by the real aligners; the recorders ignore it.
_BATCH_KW = {"steps_fine": 1}


class _FakeStore:
    """One empty shard, never canonical. The builders are stubbed, so the arrays are unused."""

    canonical = False
    path = "<fake>"

    @staticmethod
    def open(path):
        return _FakeStore()

    def read_shard(self, idx):
        return {"start": 0, "n": 1}, {}

    def __len__(self):
        return 1


def _install_recorders(mp, seen, seen_kw=None):
    """Point every builder and aligner at a recorder that reports which mode's entry ran.

    The tables and the two module globals are replaced together, so a dispatch mistake
    fails on the assertion rather than inside a Triton kernel a CPU box cannot reach.
    """
    seen_kw = [] if seen_kw is None else seen_kw
    def builder(key):
        def build(arrs, device):
            seen.append(("build", key))
            # the builder's contract is ``(ids, {channel: (flat, off)})``
            return "ids", {"c0": ("flat", "off")}
        return build

    def aligner(key):
        # ``*a``: the recorder must survive any call shape, or a regression would surface as
        # a TypeError rather than as the assertion below
        def align(*a, **k):
            seen.append(("align", key))
            # keep batch_kw so a test can inspect it (the canonical store's const_seeds)
            if len(a) >= 3 and isinstance(a[2], dict):
                seen_kw.append(a[2])
            return np.zeros(1), np.zeros((1, 4, 4), dtype=np.float32)
        return align

    mp.setattr(scr, "_ARRAY_BUILDERS", {m: builder(m) for m in scr._ARRAY_MODES}, raising=True)
    mp.setattr(scr, "_ARRAY_ALIGNERS", {m: aligner(m) for m in scr._ARRAY_MODES}, raising=True)
    mp.setattr(scr, "_build_fit_arrays_vol", builder("vol"), raising=True)
    mp.setattr(scr, "_align_fast_arrays", aligner("vol"), raising=False)   # may not exist
    mp.setattr(scr, "_accumulate_arrays", lambda *a, **k: None, raising=True)
    mp.setattr(scr, "_ref_tensors_from_arrays",
               lambda ra, mode, device: {"_ref_xyz_t": None}, raising=True)


def _run_worker(mp, mode, share=None, store_cls=_FakeStore, ref_arrays=None, seen_kw=None):
    """Drive ``_screen_worker`` for one mode, by queue or by ``share`` list; return what it selected."""
    seen = []
    _install_recorders(mp, seen, seen_kw)
    mp.setattr(scr, "ProfileStore", store_cls, raising=True)
    mp.setattr(torch.cuda, "set_device", lambda *a, **k: None, raising=True)
    mp.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=True)
    # _cap_threads calls the process-global torch.set_num_threads; the worker is in-process here
    import shepherd_score.accel.multi_gpu as mg
    mp.setattr(mg, "_cap_threads", lambda threads: None, raising=True)
    # on a CPU-only box the canonical seed set is built on the CPU instead of cuda:<rank>
    import shepherd_score.accel.drivers._common as _dc
    _real_seeds = _dc.canonical_seed_quats
    mp.setattr(_dc, "canonical_seed_quats",
               lambda xyz, n, k, device: _real_seeds(xyz, n, k, "cpu"), raising=True)

    class _ShardQ:
        def __init__(self):
            self._q = [0, None]                 # one shard, then the stop sentinel

        def get(self):
            return self._q.pop(0)

    class _OutConn:                             # stands in for the worker's result pipe
        def __init__(self):
            self.msgs = []

        def send(self, msg):
            self.msgs.append(msg)

    out = _OutConn()
    scr._screen_worker(0, 1, "<fake>", [{} if ref_arrays is None else ref_arrays], mode,
                       _BATCH_KW, 1, _ShardQ() if share is None else share, out)

    assert out.msgs, f"_screen_worker returned nothing for mode={mode}"
    msg = out.msgs[0]
    if len(msg) == 3 and msg[1] == "__ERR__":
        pytest.fail(f"_screen_worker raised for mode={mode}:\n{msg[2]}")
    return seen


def _run_inproc(mp, mode, store=None, ref_arrays=None, seen_kw=None):
    """Drive ``_run_shards_inproc`` over the same one shard and return what it selected."""
    seen = []
    _install_recorders(mp, seen, seen_kw)
    scr._run_shards_inproc(_FakeStore() if store is None else store, [0],
                           [{} if ref_arrays is None else ref_arrays], mode, torch.device("cpu"),
                           1, _BATCH_KW, {}, "numba", True, False, None, False, 1)
    return seen


class _CanonicalFakeStore(_FakeStore):
    """The same one empty shard, flagged canonical, so the vol screen may use constant seeds."""

    canonical = True

    @staticmethod
    def open(path):
        return _CanonicalFakeStore()


def test_worker_uses_the_canonical_stores_constant_seeds_like_the_inproc_loop(monkeypatch):
    """On a canonical store the worker passes the same ``const_seeds`` as the in-process loop."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((12, 3)).astype(np.float32)
    kw_w, kw_i = [], []
    with monkeypatch.context() as mp:
        _run_worker(mp, "vol", share=[0], store_cls=_CanonicalFakeStore,
                    ref_arrays={"atoms": xyz}, seen_kw=kw_w)
    with monkeypatch.context() as mp:
        _run_inproc(mp, "vol", store=_CanonicalFakeStore(), ref_arrays={"atoms": xyz}, seen_kw=kw_i)
    assert kw_w and kw_i, "the recorders saw no batch_kw"
    assert "const_seeds" in kw_i[0], "the in-process loop no longer sets const_seeds"
    assert "const_seeds" in kw_w[0], "the multi-GPU worker runs per-molecule seeds on a canonical store"
    a, b = kw_w[0]["const_seeds"], kw_i[0]["const_seeds"]
    assert np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)), \
        "worker and in-process loop derived different constant seeds from the same query"
    assert "steps_fine" in kw_w[0] and kw_w[0] is not _BATCH_KW, "batch_kw must be copied, not mutated"
    with monkeypatch.context() as mp:                  # a non-canonical store: no const_seeds
        kw_n = []
        _run_worker(mp, "vol", share=[0], ref_arrays={"atoms": xyz}, seen_kw=kw_n)
    assert kw_n and "const_seeds" not in kw_n[0]


def test_array_tables_cover_every_array_mode():
    """The tables ``_array_dispatch`` reads must cover every array mode."""
    assert set(scr._ARRAY_BUILDERS) == set(scr._ARRAY_MODES)
    assert set(scr._ARRAY_ALIGNERS) == set(scr._ARRAY_MODES)
    for mode in scr._ARRAY_MODES:
        assert scr._array_dispatch(mode) == (scr._ARRAY_BUILDERS[mode],
                                             scr._ARRAY_ALIGNERS[mode])


@pytest.mark.parametrize("mode", scr._ARRAY_MODES)
def test_worker_selects_the_same_pair_as_the_inproc_driver(monkeypatch, mode):
    """The ndev>1 worker must pick the mode's own builder and aligner."""
    with monkeypatch.context() as mp:
        worker_seen = _run_worker(mp, mode)
    with monkeypatch.context() as mp:
        inproc_seen = _run_inproc(mp, mode)

    assert worker_seen == [("build", mode), ("align", mode)], \
        f"the ndev>1 worker selected {worker_seen} for mode={mode}"
    assert worker_seen == inproc_seen, \
        f"worker selected {worker_seen}, in-process driver selected {inproc_seen}"


@pytest.mark.parametrize("mode", scr._ARRAY_MODES)
def test_worker_static_share_dispatches_like_the_queue_form(monkeypatch, mode):
    """The list-share form dispatches once per shard like the queue form; an empty share still reports."""
    with monkeypatch.context() as mp:
        seen = _run_worker(mp, mode, share=[0, 0])
    assert seen == [("build", mode), ("align", mode)] * 2
    with monkeypatch.context() as mp:
        assert _run_worker(mp, mode, share=[]) == []


def test_use_arrays_is_on_for_every_mode_this_file_drives():
    """The dispatch under test sits behind ``_use_arrays``, which must be on here."""
    from shepherd_score.accel.batch import _arrays
    assert _arrays.ENABLED, "the array path is disabled; the dispatch test would be vacuous"
    for mode in scr._ARRAY_MODES:
        assert scr._use_arrays(mode) is True

# End-to-end, and genuinely GPU-only: it needs TWO devices for _screen_many_multigpu to run.

_SMILES = ["CCO", "C1CCCCC1", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1",
           "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
           "C" * 18, "C" * 26, "C" * 34]


def _require_two_gpus():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("ndev>1 needs two CUDA devices")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


@pytest.fixture(scope="module")
def two_gpu_store(tmp_path_factory):
    """Same construction as tests/screen/test_screen_arrays.py: real conformers, synthetic surfaces."""
    _require_two_gpus()             # skip BEFORE embedding nine conformers on a CPU-only box
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule

    mols = []
    for i, smi in enumerate(_SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = i
        assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
        rng = np.random.default_rng(i)
        mols.append(Molecule(m,
                             surface_points=rng.standard_normal((64, 3)).astype(np.float32) * 3.0,
                             electrostatics=rng.standard_normal((64,)).astype(np.float32),
                             pharm_multi_vector=False))
    p = os.path.join(tmp_path_factory.mktemp("ndev"), "lib.fss")
    # shard_size=3 over 9 molecules so both workers actually get shards off the queue.
    with scr.ProfileStore.create(p, num_surf_points=64, modes=("vol", "vol_color"),
                                 dtype="float32", pre_centered=True, shard_size=3) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)
    return p, mols


@pytest.mark.cuda
def test_two_gpu_screen_matches_the_single_process_screen(two_gpu_store):
    """ndev=2 must match ndev=1, and vol must differ from vol_color so the check is not vacuous."""
    _require_two_gpus()
    store_path, mols = two_gpu_store
    query = mols[1]
    n = len(scr.ProfileStore.open(store_path))

    one = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol_color",
                     backend="triton", top_k=n, max_num_steps=30)
    two = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol_color",
                     backend="triton", top_k=n, max_num_steps=30, ndev=2)
    vol = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol",
                     backend="triton", top_k=n, max_num_steps=30)

    got = {h.id: h.score for h in two}
    want = {h.id: h.score for h in one}
    volw = {h.id: h.score for h in vol}
    assert set(got) == set(want)
    assert max(abs(got[i] - want[i]) for i in want) < 1e-6, \
        f"ndev=2 disagrees with ndev=1: max|delta| {max(abs(got[i] - want[i]) for i in want):.4e}"
    assert max(abs(volw[i] - want[i]) for i in want) > 1e-3, \
        "vol and vol_color score this library identically; the check above is vacuous"


@pytest.mark.cuda
def test_two_gpu_screen_raises_when_a_worker_dies(two_gpu_store, monkeypatch):
    """A dead screen worker raises at once instead of hanging, and the broken pool is dropped."""
    _require_two_gpus()
    store_path, mols = two_gpu_store
    pool = scr._mgpu_pool(2, 1)
    pool["procs"][1].kill()
    pool["procs"][1].join()
    monkeypatch.setattr(scr, "_mgpu_pool", lambda ndev, threads: pool)
    with pytest.raises(RuntimeError, match="died without replying"):
        scr.screen(mols[1], scr.ProfileStore.open(store_path), mode="vol", backend="triton",
                   top_k=3, max_num_steps=10, ndev=2)
    assert scr._MGPU_POOL is None
    assert all(not p.is_alive() for p in pool["procs"])
