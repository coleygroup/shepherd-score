"""Gate for the screen's pipeline levers: pose-capped sub-batching, deferred canonical
composition, pinned shard upload and the memory-mappable shard format. Each must give the same
answer; the pose cap also decides which fine loop runs, so it must be a function of the band
alone. CPU-only tests run everywhere; the rest need CUDA + Triton.
"""
import os

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402

from shepherd_score.container import Molecule                   # noqa: E402
from shepherd_score.screen import ProfileStore, screen          # noqa: E402
import shepherd_score.screen as scr                             # noqa: E402


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


# Alkanes so the library crosses _band_key's multiples of 16 (see test_screen_arrays.py).
_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"] + \
          ["C" * n for n in (12, 18, 20, 26, 30, 34)]


def _build_molecule(smi, seed, S=64):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    return Molecule(m,
                    surface_points=rng.standard_normal((S, 3)).astype(np.float32) * 3.0,
                    electrostatics=rng.standard_normal((S,)).astype(np.float32),
                    pharm_multi_vector=False)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


@pytest.fixture(scope="module")
def canon_store(tmp_path_factory, molecules):
    p = os.path.join(tmp_path_factory.mktemp("pipeline"), "canon.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float16",
                             pre_centered=True, canonical=True, shard_size=4) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p

# 1) deferred canonical composition  (no GPU needed -- it is pure heap bookkeeping)

def test_deferred_composition_matches_composing_at_offer_time():
    """Composing at ``sorted()`` must equal composing at offer time, with a non-identity rotation."""
    rng = np.random.default_rng(0)
    K = 25
    T = rng.standard_normal((K, 4, 4)).astype(np.float32)
    # proper rotations via QR, so `rot` is the kind of matrix a canonical store writes
    R = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(K)]).astype(np.float32)
    R[np.linalg.det(R) < 0] *= -1.0
    scores = rng.random(K)

    heap = scr._TopK(5)
    for i in range(K):
        heap.offer_row(float(scores[i]), i, T, i, R)
    got = {h.id: np.asarray(h.transform) for h in heap.sorted()}

    ref = scr._TopK(5)
    for i in range(K):
        if len(ref.heap) < ref.k or float(scores[i]) > ref.heap[0][0]:
            ref._push(float(scores[i]), i, scr._compose_rot(T[i], R[i]))
    want = {h.id: np.asarray(h.transform) for h in ref.sorted()}

    assert set(got) == set(want)
    for k in want:
        assert np.array_equal(got[k], want[k]), f"transform {k} differs"
        assert not np.array_equal(got[k][:3, :3], T[k][:3, :3]), \
            "composition was a no-op -- this test would not catch its removal"


def test_deferred_composition_is_a_noop_without_a_rotation():
    """A non-canonical store passes ``rot=None`` and must still get the raw row back."""
    rng = np.random.default_rng(1)
    T = rng.standard_normal((8, 4, 4)).astype(np.float32)
    heap = scr._TopK(4)
    for i in range(8):
        heap.offer_row(float(i), i, T, i, None)
    for h in heap.sorted():
        assert np.array_equal(np.asarray(h.transform), T[h.id])


def test_raw_and_merge_raw_round_trip_composed_transforms():
    """``raw()`` crosses a process boundary in multi_gpu, so it must not emit pending tuples."""
    rng = np.random.default_rng(2)
    T = rng.standard_normal((6, 4, 4)).astype(np.float32)
    R = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(6)]).astype(np.float32)
    a = scr._TopK(3)
    for i in range(6):
        a.offer_row(float(i), i, T, i, R)
    raw = a.raw()
    for (_s, _i, t) in raw:
        assert isinstance(t, np.ndarray), "raw() leaked a deferred (transform, rot) tuple"
    b = scr._TopK(3)
    b.merge_raw(raw)
    assert [np.asarray(h.transform).tolist() for h in b.sorted()] == \
           [np.asarray(h.transform).tolist() for h in a.sorted()]

# 2) device-side float16 widening

def test_to_device_matches_host_widening_exactly():
    """float16 -> float32 is exact, so widening on the device must give the same bits."""
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((257, 3)) * 10).astype(np.float16)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    got = scr._to_device(a, dev, dtype=torch.float32)
    want = torch.as_tensor(a, dtype=torch.float32, device=dev)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    # already float32 -> passed straight through, not re-cast
    b = a.astype(np.float32)
    assert torch.equal(scr._to_device(b, dev, dtype=torch.float32),
                       torch.as_tensor(b, device=dev))

# 3) pose cap

@pytest.mark.cuda
def test_pose_cap_bounds_the_subbatch_and_only_when_asked():
    """``pose_cap`` bounds the sub-batch and is opt-in; without it the schedule is memory-derived."""
    _require_fast_cuda()
    from shepherd_score.accel.batch import _pad as padmod

    dev = torch.device("cuda")
    seen = []

    def process(start, k):
        seen.append(k)
        z = torch.zeros(k, device=dev)
        return z, torch.zeros(k, 4, device=dev), torch.zeros(k, 3, device=dev)

    cap, seeds, K = 640, 10, 1000
    try:
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        padmod._subbatched_align(process, K, key=("t", 1, 1, seeds), device=dev,
                                 pose_cap=cap, seeds=seeds)
        assert seen, "process never ran"
        assert max(seen) <= cap // seeds, f"chunk {max(seen)} exceeds the {cap // seeds} cap"
        assert sum(seen) == K

        seen.clear()
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        padmod._subbatched_align(process, K, key=("t", 1, 1, seeds), device=dev)
        assert max(seen) > cap // seeds, "cap applied without pose_cap= -- it is not opt-in"
    finally:
        padmod._PAIR_FOOTPRINT_BYTES.clear()


@pytest.mark.cuda
def test_pose_cap_graphs_every_chunk_on_a_narrow_fixture(monkeypatch, canon_store, molecules):
    """With the cap every fine-loop call on this band-48 fixture reaches the CUDA graph."""
    _require_fast_cuda()
    from shepherd_score.accel.batch import _arrays
    from shepherd_score.accel.drivers import engine as enginemod
    from shepherd_score.accel.drivers._graphed import reset_graph_cache
    from shepherd_score.accel.batch import _pad as padmod

    monkeypatch.setattr(_arrays, "ENABLED", True)
    seen = {"calls": 0, "graphed": 0}
    # one fine loop serves every mode: the seams are the engine entry and the shared graph runner
    _align, _rgf = enginemod.align, enginemod.run_graphed

    def cfa(*a, **k):
        seen["calls"] += 1
        return _align(*a, **k)

    def rgf(*a, **k):
        seen["graphed"] += 1
        return _rgf(*a, **k)

    monkeypatch.setattr(enginemod, "align", cfa)
    monkeypatch.setattr(enginemod, "run_graphed", rgf)

    store = ProfileStore.open(canon_store)
    q = molecules[0]

    # precondition: every molecule pads to band 48; above that a capped chunk legitimately runs
    # eager and graphed == calls would be a false claim
    _heavy = max(int((m.atom_pos_noH if hasattr(m, "atom_pos_noH") else m.atom_pos).shape[0])
                 for m in molecules + [q])
    assert _heavy <= 48, (
        f"fixture has a {_heavy}-heavy-atom molecule; above 48 the band exceeds graph_cap and "
        "eager chunks are CORRECT. Re-derive the expectation instead of widening this bound.")

    reset_graph_cache()
    hits = screen(q, store, mode="vol", backend="triton", top_k=5)
    assert seen["calls"] > 0, "the fine loop never ran -- the spy missed the call site"
    assert seen["graphed"] == seen["calls"], \
        f"{seen['calls'] - seen['graphed']} of {seen['calls']} fine-loop calls skipped the graph"

    # the cap is a schedule, not a score: a different (smaller) cap must give the same answer
    n = len(store)
    a = np.full(n, np.nan); screen(q, store, mode="vol", backend="triton", top_k=5, scores_out=a)
    old = padmod._FINE_CHUNK_POSES
    try:
        padmod._FINE_CHUNK_POSES = max(10, old // 8)
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        reset_graph_cache()
        b = np.full(n, np.nan)
        hits2 = screen(q, store, mode="vol", backend="triton", top_k=5, scores_out=b)
    finally:
        padmod._FINE_CHUNK_POSES = old
        padmod._PAIR_FOOTPRINT_BYTES.clear()
    ok = ~(np.isnan(a) | np.isnan(b))
    assert np.array_equal(a[ok], b[ok]), "the sub-batch size changed a score"
    assert [h.id for h in hits] == [h.id for h in hits2]

# 4) upload-ahead

@pytest.mark.cuda
def test_shard_upload_stages_through_pinned_memory(monkeypatch, canon_store):
    """The shard upload must stage through pinned memory and widen float16 on the device, exactly."""
    _require_fast_cuda()
    store = ProfileStore.open(canon_store)
    _, arrs = store.read_shard(0)
    dev = torch.device("cuda")
    scr._PIN_STAGE.clear()

    pinned = []
    real_empty = torch.empty

    def spy(*a, **k):
        if k.get("pin_memory"):
            pinned.append(True)
        return real_empty(*a, **k)

    monkeypatch.setattr(torch, "empty", spy)
    # the builder returns (ids, {channel: (flat, off)}) -- one dict, whatever the mode
    ids, fit = scr._build_fit_arrays_vol(arrs, dev)
    pos, off = fit["atoms"]
    monkeypatch.undo()
    assert pinned, "the shard upload did not stage through pinned memory"
    assert pos.is_cuda and pos.dtype == torch.float32
    assert off.is_cuda and off.dtype == torch.int64
    # exact against the host-widened path it replaces
    want = torch.as_tensor(arrs["atom_pos"], dtype=torch.float32, device=dev)
    assert torch.equal(pos, want)
    assert torch.equal(off, torch.as_tensor(arrs["atom_off"], dtype=torch.long, device=dev))


@pytest.mark.cuda
def test_memory_mapped_shards_match_the_zip_format(tmp_path, molecules):
    """The .npy shard format must read back the same bytes and screen to the same scores as .npz."""
    out = {}
    for fmt in ("npz", "npy"):
        p = os.path.join(str(tmp_path), f"{fmt}.fss")
        with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float16",
                                 pre_centered=True, canonical=True, shard_size=4,
                                 shard_format=fmt) as st:
            for i, m in enumerate(molecules):
                st.add(m, id=i)
        out[fmt] = ProfileStore.open(p)
    a, b = out["npz"], out["npy"]
    assert a.num_shards == b.num_shards > 1
    for i in range(a.num_shards):
        _, ra = a.read_shard(i)
        _, rb = b.read_shard(i)
        assert set(ra) == set(rb)
        for k in ra:
            assert np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])), k

    _require_fast_cuda()
    from shepherd_score.accel.batch import _arrays
    from shepherd_score.accel.drivers._graphed import reset_graph_cache
    prev, _arrays.ENABLED = _arrays.ENABLED, True
    try:
        q = molecules[0]
        sa = np.full(len(a), np.nan); reset_graph_cache()
        ha = screen(q, a, mode="vol", backend="triton", top_k=5, scores_out=sa)
        sb = np.full(len(b), np.nan); reset_graph_cache()
        hb = screen(q, b, mode="vol", backend="triton", top_k=5, scores_out=sb)
    finally:
        _arrays.ENABLED = prev
    ok = ~(np.isnan(sa) | np.isnan(sb))
    assert np.array_equal(sa[ok], sb[ok]), "the shard format changed a score"
    assert [h.id for h in ha] == [h.id for h in hb]
