"""Tests for the out-of-core streaming screen (``shepherd_score.screen``).

Store round-trip (pure numpy) always runs; end-to-end alignment (rdkit + torch + numba) checks
self-copy -> 1.0 and equivalence to the in-memory MoleculePairBatch path.
"""
import os

import numpy as np
import pytest

from shepherd_score.screen import MoleculeProfile, ProfileStore, screen


# --------------------------------------------------------------------------- #
# Layer 1 -- store round-trip (numpy only)
# --------------------------------------------------------------------------- #
def _synthetic_profile(seed, *, with_full_charges, n_heavy=5, n_all=8, S=64, P=4):
    rng = np.random.default_rng(seed)
    atom_pos = rng.standard_normal((n_heavy, 3)).astype(np.float32)
    surf = rng.standard_normal((S, 3)).astype(np.float32)
    esp = rng.standard_normal((S,)).astype(np.float32)
    pharm_types = rng.integers(0, 8, size=P).astype(np.int32)
    pharm_ancs = rng.standard_normal((P, 3)).astype(np.float32)
    pharm_vecs = rng.standard_normal((P, 3)).astype(np.float32)
    if with_full_charges:                       # esp_combo-style with-H arrays
        charges = rng.standard_normal((n_all,)).astype(np.float32)
        radii = (rng.random(n_all) + 1.0).astype(np.float32)
        centers = rng.standard_normal((n_all, 3)).astype(np.float32)
        nonH = np.sort(rng.choice(n_all, size=n_heavy, replace=False)).astype(np.int64)
        return MoleculeProfile(atom_pos=atom_pos, surf_pos=surf, surf_esp=esp,
                               partial_charges=charges, radii=radii, nonH_atoms_idx=nonH,
                               pharm_types=pharm_types, pharm_ancs=pharm_ancs,
                               pharm_vecs=pharm_vecs, centers_w_H=centers, id=seed)
    charges = rng.standard_normal((n_heavy,)).astype(np.float32)   # heavy-only
    return MoleculeProfile(atom_pos=atom_pos, surf_pos=surf, surf_esp=esp,
                           partial_charges=charges, pharm_types=pharm_types,
                           pharm_ancs=pharm_ancs, pharm_vecs=pharm_vecs, id=seed)


def test_store_roundtrip_all_modes(tmp_path):
    """float32, pre_centered=False -> reloaded arrays are bit-exact."""
    store_path = os.path.join(tmp_path, "lib.fss")
    profs = [_synthetic_profile(i, with_full_charges=True) for i in range(7)]
    with ProfileStore.create(store_path, num_surf_points=64,
                             modes=("vol", "vol_esp", "surf", "esp", "pharm", "esp_combo"),
                             dtype="float32", shard_size=3, pre_centered=False) as store:
        for p in profs:
            store.add_profile(p)

    store = ProfileStore.open(store_path)
    assert len(store) == 7
    assert store.num_surf_points == 64
    for md in ("vol", "vol_esp", "surf", "esp", "pharm", "esp_combo"):
        assert store.supports(md)

    reloaded = [p for shard in store.iter_shards() for p in shard]
    assert len(reloaded) == 7
    for orig, got in zip(profs, reloaded):
        assert got.id == orig.id
        np.testing.assert_array_equal(got.atom_pos, orig.atom_pos)
        np.testing.assert_array_equal(got.surf_pos, orig.surf_pos)
        np.testing.assert_array_equal(got.surf_esp, orig.surf_esp)
        np.testing.assert_array_equal(got.partial_charges, orig.partial_charges)
        np.testing.assert_array_equal(got._nonH_atoms_idx, orig._nonH_atoms_idx)
        np.testing.assert_array_equal(got.radii, orig.radii)
        np.testing.assert_array_equal(got.pharm_types, orig.pharm_types)
        np.testing.assert_array_equal(got.pharm_ancs, orig.pharm_ancs)
        np.testing.assert_array_equal(got.pharm_vecs, orig.pharm_vecs)
        np.testing.assert_array_equal(got.mol.GetConformer().GetPositions(),
                                      orig.mol.GetConformer().GetPositions())
        # vol_esp heavy-charge selection round-trips through the stored with-H index
        np.testing.assert_array_equal(got.partial_charges[got._nonH_atoms_idx],
                                      orig.partial_charges[orig._nonH_atoms_idx])


def test_store_heavy_charges_identity_index(tmp_path):
    """A vol_esp-only store keeps heavy charges with an identity nonH index."""
    store_path = os.path.join(tmp_path, "lib.fss")
    profs = [_synthetic_profile(i, with_full_charges=False) for i in range(4)]
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol", "vol_esp"),
                             dtype="float32", pre_centered=False) as store:
        for p in profs:
            store.add_profile(p)
    store = ProfileStore.open(store_path)
    assert not store.schema["with_H"]
    assert not store.supports("esp")           # no surfaces stored
    reloaded = [p for shard in store.iter_shards() for p in shard]
    for orig, got in zip(profs, reloaded):
        np.testing.assert_array_equal(got.partial_charges, orig.partial_charges)
        np.testing.assert_array_equal(got._nonH_atoms_idx, np.arange(len(orig.partial_charges)))


def test_store_pre_centered(tmp_path):
    """pre_centered=True centers on the heavy-atom COM without changing geometry (canonical=False isolates that)."""
    store_path = os.path.join(tmp_path, "lib.fss")
    p = _synthetic_profile(0, with_full_charges=False)
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        store.add_profile(p)
    got = next(iter(ProfileStore.open(store_path).iter_shards()))[0]
    np.testing.assert_allclose(got.atom_pos.mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(got.atom_pos - got.atom_pos.mean(0),
                               p.atom_pos - p.atom_pos.mean(0), atol=1e-6)


def test_store_is_canonical_by_default_only_when_it_serves_vol(tmp_path):
    """``canonical=None`` resolves True only for a pre-centered store serving a constant-seed mode."""
    p = _synthetic_profile(0, with_full_charges=False)

    vol = os.path.join(tmp_path, "vol.fss")
    with ProfileStore.create(vol, num_surf_points=64, modes=("vol",), dtype="float32") as store:
        store.add_profile(p)
    st = ProfileStore.open(vol)
    assert st.canonical is True
    got = next(iter(st.iter_shards()))[0]
    np.testing.assert_allclose(got.atom_pos.mean(0), 0.0, atol=1e-5)
    dist = lambda x: np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)   # noqa: E731
    np.testing.assert_allclose(dist(got.atom_pos), dist(p.atom_pos), atol=1e-4)

    surf = os.path.join(tmp_path, "surf.fss")
    with ProfileStore.create(surf, num_surf_points=64, modes=("surf",), dtype="float32") as store:
        store.add_profile(p)
    assert ProfileStore.open(surf).canonical is False

    with pytest.raises(ValueError):
        ProfileStore.create(os.path.join(tmp_path, "bad.fss"), num_surf_points=64,
                            modes=("vol",), pre_centered=False, canonical=True)


def test_float16_store_is_lossy_but_close(tmp_path):
    store_path = os.path.join(tmp_path, "lib.fss")
    p = _synthetic_profile(0, with_full_charges=False)
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol", "surf"),
                             dtype="float16", pre_centered=False) as store:
        store.add_profile(p)
    got = next(iter(ProfileStore.open(store_path).iter_shards()))[0]
    assert got.atom_pos.dtype == np.float32                 # upcast in RAM
    np.testing.assert_allclose(got.atom_pos, p.atom_pos, atol=2e-2)


def test_screen_rejects_unsupported_mode(tmp_path):
    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol",),
                             pre_centered=False) as store:
        store.add_profile(_synthetic_profile(0, with_full_charges=False))
    store = ProfileStore.open(store_path)

    class _Q:                                  # query stand-in; never aligned
        num_surf_points = 64
    with pytest.raises(ValueError, match="does not support"):
        screen(_Q(), store, mode="esp", backend="numba")


def test_screen_guards(tmp_path):
    """no_H=False is rejected (heavy-atom only)."""
    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64,
                             modes=("vol", "esp", "esp_combo"), pre_centered=False) as store:
        store.add_profile(_synthetic_profile(0, with_full_charges=True))
    store = ProfileStore.open(store_path)

    class _Q:
        num_surf_points = 64
    with pytest.raises(ValueError, match="heavy atoms only"):
        screen(_Q(), store, mode="vol", backend="numba", no_H=False)


# --------------------------------------------------------------------------- #
# Layer 2 -- end-to-end alignment (rdkit + torch + numba)
# --------------------------------------------------------------------------- #
rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from rdkit import Chem                          # noqa: E402
from rdkit.Chem import AllChem                  # noqa: E402

from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch  # noqa: E402
from shepherd_score.score.constants import ALPHA  # noqa: E402
from shepherd_score.screen import screen_many  # noqa: E402

_SMILES = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1O"]


def _build_molecule(smi, seed, S=64):
    """Real RDKit conformer + a *synthetic* surface (so the test needs no Open3D)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    surf = rng.standard_normal((S, 3)).astype(np.float32) * 3.0
    esp = rng.standard_normal((S,)).astype(np.float32)
    return Molecule(m, surface_points=surf, electrostatics=esp, pharm_multi_vector=False)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


_MODE_KW = {"vol": {}, "vol_esp": {"lam": 0.1}, "esp": {}, "pharm": {}, "vol_color": {}}


@pytest.mark.parametrize("mode", ["vol", "vol_esp", "pharm", "esp", "vol_color"])
def test_self_screen_recovers_one(tmp_path_factory, molecules, mode):
    """Screening a query that is also IN the library recovers it at score ~1.0."""
    store_path = os.path.join(tmp_path_factory.mktemp(mode), "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64,
                             modes=("vol", "vol_esp", "esp", "pharm", "vol_color"),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)

    query = molecules[1]                        # also stored at id=1
    hits = screen(query, ProfileStore.open(store_path), mode=mode, backend="numba",
                  num_repeats=5, max_num_steps=50, top_k=len(molecules), **_MODE_KW[mode])
    assert len(hits) == len(molecules)
    by_id = {h.id: h.score for h in hits}
    assert by_id[1] == pytest.approx(1.0, abs=1e-2)         # self-copy optimum
    assert hits[0].id == 1                                  # and it ranks first
    assert hits[0].transform.shape == (4, 4)


def test_from_molecule_smoke(molecules):
    """MoleculeProfile.from_molecule extracts the right arrays and drops the Mol."""
    prof = MoleculeProfile.from_molecule(molecules[0], modes=("surf", "esp", "pharm"))
    assert prof.mol is None
    np.testing.assert_array_equal(prof.surf_pos, molecules[0].surf_pos)
    np.testing.assert_array_equal(prof.surf_esp, molecules[0].surf_esp)
    np.testing.assert_array_equal(prof.atom_pos, molecules[0].atom_pos.astype(np.float32))
    assert prof.pharm_types is not None


def test_screen_does_not_mutate_query(tmp_path, molecules):
    """Centering operates on a copy; the caller's query is never mutated."""
    query = molecules[0]
    before = query.atom_pos.copy()
    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol",),
                             dtype="float32", pre_centered=False) as store:
        for i, m in enumerate(molecules[1:]):
            store.add(m, id=i)
    screen(query, ProfileStore.open(store_path), mode="vol", backend="numba",
           num_repeats=5, max_num_steps=30, top_k=3)          # do_center defaults True here
    np.testing.assert_array_equal(query.atom_pos, before)


def test_stream_matches_in_memory_vol(tmp_path, molecules):
    """Streaming scores equal the in-memory MoleculePairBatch path (numba, deterministic seeds)."""
    query = molecules[0]
    lib = molecules[1:]

    pairs = [MoleculePair(query, m, do_center=False) for m in lib]
    torch.manual_seed(0)
    ref_scores, _ = MoleculePairBatch(pairs).align_with_vol(
        backend="numba", num_repeats=5, max_num_steps=50)

    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol",),
                             dtype="float32", pre_centered=False) as store:
        for i, m in enumerate(lib):
            store.add(m, id=i)

    store = ProfileStore.open(store_path)
    out = np.full(len(store), np.nan)
    torch.manual_seed(0)
    screen(query, store, mode="vol", backend="numba", do_center=False,
           num_repeats=5, max_num_steps=50, top_k=len(store), scores_out=out)

    np.testing.assert_allclose(out, ref_scores, atol=1e-4)


def test_fast_path_matches_object_path(tmp_path, molecules):
    """The fast path on a pre-centered store matches MoleculePairs built from the same profiles."""
    import copy
    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol", "esp"),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    query = molecules[0]

    fast = screen(query, store, mode="vol", backend="numba", max_num_steps=50,
                  top_k=len(molecules))
    fast_by_id = {h.id: h.score for h in fast}

    cq = copy.deepcopy(query)
    cq.center_to(cq.atom_pos.mean(0))                       # match pre-centered profiles
    profiles = [p for shard in store.iter_shards() for p in shard]
    pairs = [MoleculePair(cq, p, do_center=False) for p in profiles]
    ref_scores, _ = MoleculePairBatch(pairs).align_with_vol(backend="numba", max_num_steps=50)
    for p, rs in zip(profiles, ref_scores):
        assert fast_by_id[p.id] == pytest.approx(float(rs), abs=1e-4)


def test_vol_color_store_and_fast_matches_object(tmp_path, molecules):
    """vol_color screens from a pharm store and its fast path matches the per-pair object path."""
    import copy
    store_path = os.path.join(tmp_path, "lib.fss")
    # canonical=False: a canonical store would run constant seeds on the fast path only, so the
    # two legs would legitimately differ; this test is about the array->kernel plumbing.
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_color",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    assert store.supports("vol_color") and store.supports("vol")
    assert not store.supports("esp")                       # no surfaces stored

    query = molecules[1]
    fast = screen(query, store, mode="vol_color", backend="numba", color_weight=0.5,
                  num_repeats=5, max_num_steps=50, top_k=len(molecules))
    fast_by_id = {h.id: h.score for h in fast}
    assert fast_by_id[1] == pytest.approx(1.0, abs=1e-2)   # self-copy optimum

    cq = copy.deepcopy(query)
    cq.center_to(cq.atom_pos.mean(0))
    profiles = [p for shard in store.iter_shards() for p in shard]
    pairs = [MoleculePair(cq, p, do_center=False) for p in profiles]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_color(
        backend="numba", color_weight=0.5, num_repeats=5, max_num_steps=50)
    for p, rs in zip(profiles, ref):
        assert fast_by_id[p.id] == pytest.approx(float(rs), abs=1e-4)


def test_vol_tversky_stream_matches_object(tmp_path, molecules):
    """vol_tversky screens from an atoms-only store and matches the per-pair object path."""
    import copy
    store_path = os.path.join(tmp_path, "lib.fss")
    # canonical=False: a canonical store would run constant seeds on the fast path only, so the
    # two legs would legitimately differ; this test is about the array->kernel plumbing.
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_tversky",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    assert store.supports("vol_tversky") and store.supports("vol")

    query = molecules[1]
    hits = screen(query, store, mode="vol_tversky", backend="numba",
                  num_repeats=5, max_num_steps=50, top_k=len(molecules))
    by_id = {h.id: h.score for h in hits}
    assert by_id[1] == pytest.approx(1.0, abs=1e-2)        # self fits perfectly inside self

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    profiles = [p for shard in store.iter_shards() for p in shard]
    pairs = [MoleculePair(cq, p, do_center=False) for p in profiles]
    ref, _ = MoleculePairBatch(pairs).align_with_vol(
        similarity="tversky", backend="numba", num_repeats=5, max_num_steps=50)
    for p, rs in zip(profiles, ref):
        assert by_id[p.id] == pytest.approx(float(rs), abs=1e-4)


def test_vol_esp_tversky_stream_matches_object(tmp_path, molecules):
    """vol_esp_tversky screens from a vol_esp-style store and matches the per-pair object path."""
    import copy, glob
    store_path = os.path.join(tmp_path, "lib.fss")
    # canonical=False: a canonical store would run constant seeds on the fast path only, so the
    # two legs would legitimately differ; this test is about the array->kernel plumbing.
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_esp_tversky",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    # Reuses vol_esp's store schema (heavy centers + charges); no new per-molecule data.
    assert store.supports("vol_esp_tversky") and store.supports("vol_esp")
    assert store.schema["charges"]
    # the heavy charges reached disk; read through the store rather than re-deriving the layout
    d = store.read_shard(0)[1]
    assert "charges" in d

    query = molecules[1]                                 # also stored at id=1
    hits = screen(query, store, mode="vol_esp_tversky", backend="numba", lam=0.1,
                  num_repeats=16, max_num_steps=50, top_k=len(molecules))
    by_id = {h.id: h.score for h in hits}
    assert by_id[1] == pytest.approx(1.0, abs=1e-2)      # self fits perfectly inside self (Tversky(A,A)=1)

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    def _c(m):
        c = copy.deepcopy(m); c.center_to(c.atom_pos.mean(0)); return c
    pairs = [MoleculePair(cq, _c(m), do_center=False) for m in molecules]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_esp(
        similarity="tversky", backend="numba", lam=0.1, num_repeats=16, max_num_steps=50)
    # identical inputs to one driver, so abs=1e-4; the Tversky score is not clamped to [0, 1]
    for i, rs in enumerate(ref):
        assert by_id[i] == pytest.approx(float(rs), abs=1e-4)


def test_vol_lipo_stream_matches_object(tmp_path, molecules):
    """vol_lipo stores its variable-length lipo arrays and streams to the object path's scores."""
    import copy, glob
    store_path = os.path.join(tmp_path, "lib.fss")
    # canonical=False: a canonical store would run constant seeds on the fast path only, so the
    # two legs would legitimately differ; this test is about the array->kernel plumbing.
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_lipo",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    assert store.supports("vol_lipo") and store.supports("vol")
    assert store.schema["lipophilicity"]
    # the lipo set reached disk; read through the store rather than re-deriving the layout
    d = store.read_shard(0)[1]
    assert {"lipo_off", "lipo_pos", "lipophilicity"} <= set(d)

    query = molecules[1]
    hits = screen(query, store, mode="vol_lipo", backend="numba",
                  num_repeats=16, max_num_steps=50, top_k=len(molecules))
    by_id = {h.id: h.score for h in hits}
    assert by_id[1] == pytest.approx(1.0, abs=1e-2)        # self-overlay optimum (shape+lipo)

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    profiles = [p for shard in store.iter_shards() for p in shard]
    pairs = [MoleculePair(cq, p, do_center=False) for p in profiles]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_lipo(
        backend="numba", num_repeats=16, max_num_steps=50)
    # abs=1e-2: the query is re-centered independently (fp noise), which can flip a near-tie seed
    # in the two-channel objective; wrong or empty lipo data would move scores far more
    for p, rs in zip(profiles, ref):
        assert by_id[p.id] == pytest.approx(float(rs), abs=1e-2)


def test_screen_many_equals_per_query_and_streams_once(tmp_path_factory, molecules):
    """A panel screen (one read per shard) equals per-query screens across multiple shards."""
    store_path = os.path.join(tmp_path_factory.mktemp("panel"), "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol", "esp"),
                             dtype="float32", pre_centered=True, shard_size=2) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    assert store.num_shards == 2                            # 4 mols / shard_size 2

    qs = [molecules[0], molecules[2]]
    panel = screen_many(qs, store, mode="esp", backend="numba", num_repeats=5,
                        max_num_steps=50, top_k=len(molecules))
    assert len(panel) == 2
    for j, q in enumerate(qs):
        single = screen(q, store, mode="esp", backend="numba", num_repeats=5,
                        max_num_steps=50, top_k=len(molecules))
        assert ({h.id: round(h.score, 5) for h in panel[j]}
                == {h.id: round(h.score, 5) for h in single})


def test_panel_scores_out(tmp_path, molecules):
    """screen_many writes a per-query full score vector in library order."""
    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol",),
                             dtype="float32", pre_centered=True, shard_size=2) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    qs = [molecules[1], molecules[3]]
    outs = [np.full(len(store), np.nan) for _ in qs]
    panel = screen_many(qs, store, mode="vol", backend="numba", max_num_steps=50,
                        top_k=len(molecules), scores_out=outs)
    for j in range(len(qs)):
        assert not np.isnan(outs[j]).any()
        by_id = {h.id: h.score for h in panel[j]}
        for i in range(len(store)):
            assert outs[j][i] == pytest.approx(by_id[i], abs=1e-6)
    # query j=0 is molecules[1] -> its own id (1) scores ~1.0
    assert outs[0][1] == pytest.approx(1.0, abs=1e-2)


def test_vol_esp_stream_retained_h(tmp_path):
    """vol_esp streams a molecule whose RemoveHs retains an H, matching the in-memory path."""
    import copy

    # [2H]OC(=O)c1ccccc1 keeps its deuterium after RemoveHs -> atom_pos (10) != heavy (9).
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", "[2H]OC(=O)c1ccccc1",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    mols = [_build_molecule(s, seed=i) for i, s in enumerate(smis)]
    retained = [m for m in mols if len(m.atom_pos) != len(m._nonH_atoms_idx)]
    assert retained, "test premise broken: no molecule retains an H after RemoveHs"

    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol", "vol_esp"),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)

    # the store gained the heavy offset + strict-heavy centers; read through the store
    import glob
    d = store.read_shard(0)[1]
    assert "heavy_off" in d and "xyz_noH" in d

    query = mols[0]                                  # clean query
    hits = screen(query, ProfileStore.open(store_path), mode="vol_esp", backend="numba",
                  num_repeats=5, max_num_steps=50, top_k=len(mols), lam=0.1)
    assert len(hits) == len(mols)
    fast_by_id = {h.id: h.score for h in hits}

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    def _c(m):
        c = copy.deepcopy(m); c.center_to(c.atom_pos.mean(0)); return c
    pairs = [MoleculePair(cq, _c(m), do_center=False) for m in mols]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_esp(
        backend="numba", lam=0.1, num_repeats=5, max_num_steps=50)
    for i, rs in enumerate(ref):                     # incl. the retained-H molecule
        assert fast_by_id[i] == pytest.approx(float(rs), abs=1e-4)


@pytest.mark.parametrize("modes", [("vol_esp",), ("vol_and_surf_esp",)])
def test_reload_retained_h_keeps_heavy_arrays_aligned(tmp_path, modes):
    """Profiles read back from a store keep their heavy charges, ``nonH`` index and heavy centers
    when an earlier molecule's RemoveHs retained an H (which shifts the heavy offsets)."""
    smis = ["[2H]OC(=O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O"]
    mols = [_build_molecule(s, seed=i) for i, s in enumerate(smis)]
    assert len(mols[0].atom_pos) != len(mols[0]._nonH_atoms_idx)

    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=modes,
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)
    profiles = ProfileStore.open(store_path).read_profiles(0)

    with_h = "vol_and_surf_esp" in modes
    for m, p in zip(mols, profiles):
        if with_h:
            np.testing.assert_allclose(p.partial_charges, m.partial_charges, atol=1e-6)
            np.testing.assert_array_equal(p._nonH_atoms_idx, m._nonH_atoms_idx)
        else:
            np.testing.assert_allclose(p.partial_charges,
                                       m.partial_charges[m._nonH_atoms_idx], atol=1e-6)
        heavy = p.atom_pos if p.atom_pos_noH is None else p.atom_pos_noH
        assert len(heavy) == len(m._nonH_atoms_idx)


def test_vol_lipo_stream_retained_h(tmp_path):
    """vol_lipo stores the RemoveHs and true-heavy bases independently when an H is retained."""
    import copy, glob

    # [2H]OC(=O)c1ccccc1 keeps its deuterium after RemoveHs -> atom_pos (10) != heavy (9).
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", "[2H]OC(=O)c1ccccc1",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    mols = [_build_molecule(s, seed=i) for i, s in enumerate(smis)]
    retained = [m for m in mols if len(m.atom_pos) != len(m._nonH_atoms_idx)]
    assert retained, "test premise broken: no molecule retains an H after RemoveHs"

    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_lipo",),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)

    # the lipo offset table diverges from atom_off on the retained-H molecule
    d = store.read_shard(0)[1]
    assert {"lipo_off", "lipo_pos", "lipophilicity"} <= set(d)
    atom_lens = np.diff(d["atom_off"])
    lipo_lens = np.diff(d["lipo_off"])
    assert (atom_lens != lipo_lens).any(), "retained-H molecule should diverge atom_off vs lipo_off"

    query = mols[0]                                  # clean query
    hits = screen(query, ProfileStore.open(store_path), mode="vol_lipo", backend="numba",
                  num_repeats=16, max_num_steps=50, top_k=len(mols))
    assert len(hits) == len(mols)
    fast_by_id = {h.id: h.score for h in hits}

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    def _c(m):
        c = copy.deepcopy(m); c.center_to(c.atom_pos.mean(0)); return c
    pairs = [MoleculePair(cq, _c(m), do_center=False) for m in mols]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_lipo(
        backend="numba", num_repeats=16, max_num_steps=50)
    for i, rs in enumerate(ref):                     # incl. the retained-H molecule
        assert fast_by_id[i] == pytest.approx(float(rs), abs=1e-2)


# --------------------------------------------------------------------------- #
# vol_fukui -- the same store/fast-path plumbing as vol_lipo, with a deterministic synthetic
# Fukui field injected via Molecule(fukui=...) so no xTB is needed.
# --------------------------------------------------------------------------- #
def _synthetic_fukui(rd):
    """Deterministic signed per-atom field (with-H order) standing in for the xTB Fukui descriptor."""
    z = np.array([a.GetAtomicNum() for a in rd.GetAtoms()], dtype=np.float32)
    return (0.2 * ((z % 4) - 1.5)).astype(np.float32)


def _build_molecule_fukui(smi, seed, S=64):
    """As ``_build_molecule`` but injects the synthetic Fukui field so no xTB binary is needed."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    surf = rng.standard_normal((S, 3)).astype(np.float32) * 3.0
    esp = rng.standard_normal((S,)).astype(np.float32)
    return Molecule(m, surface_points=surf, electrostatics=esp, pharm_multi_vector=False,
                    fukui=_synthetic_fukui(m))


def test_vol_fukui_stream_matches_object(tmp_path):
    """vol_fukui stores its variable-length Fukui arrays and streams to the object path's scores."""
    import copy, glob
    smis = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1O"]
    mols = [_build_molecule_fukui(s, seed=i) for i, s in enumerate(smis)]

    store_path = os.path.join(tmp_path, "lib.fss")
    # canonical=False: a canonical store would run constant seeds on the fast path only, so the
    # two legs would legitimately differ; this test is about the array->kernel plumbing.
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_fukui",),
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)
    store = ProfileStore.open(store_path)
    assert store.supports("vol_fukui") and store.supports("vol")
    assert store.schema["fukui"]
    # the Fukui set reached disk; read through the store rather than re-deriving the layout
    d = store.read_shard(0)[1]
    assert {"fukui_off", "fukui_pos", "fukui"} <= set(d)

    query = mols[1]
    hits = screen(query, store, mode="vol_fukui", backend="numba",
                  num_repeats=16, max_num_steps=50, top_k=len(mols))
    by_id = {h.id: h.score for h in hits}
    assert by_id[1] == pytest.approx(1.0, abs=1e-2)        # self-overlay optimum (shape+fukui)

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    profiles = [p for shard in store.iter_shards() for p in shard]
    pairs = [MoleculePair(cq, p, do_center=False) for p in profiles]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_fukui(
        backend="numba", num_repeats=16, max_num_steps=50)
    # abs=1e-2: independent re-centring can flip a near-tie seed; wrong Fukui data moves far more
    for p, rs in zip(profiles, ref):
        assert by_id[p.id] == pytest.approx(float(rs), abs=1e-2)


def test_vol_fukui_stream_retained_h(tmp_path):
    """vol_fukui stores the RemoveHs and true-heavy bases independently when an H is retained."""
    import copy, glob

    # [2H]OC(=O)c1ccccc1 keeps its deuterium after RemoveHs -> atom_pos (10) != heavy (9).
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", "[2H]OC(=O)c1ccccc1",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    mols = [_build_molecule_fukui(s, seed=i) for i, s in enumerate(smis)]
    retained = [m for m in mols if len(m.atom_pos) != len(m._nonH_atoms_idx)]
    assert retained, "test premise broken: no molecule retains an H after RemoveHs"

    store_path = os.path.join(tmp_path, "lib.fss")
    with ProfileStore.create(store_path, num_surf_points=64, modes=("vol_fukui",),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)

    # the Fukui offset table diverges from atom_off on the retained-H molecule
    d = store.read_shard(0)[1]
    assert {"fukui_off", "fukui_pos", "fukui"} <= set(d)
    atom_lens = np.diff(d["atom_off"])
    fukui_lens = np.diff(d["fukui_off"])
    assert (atom_lens != fukui_lens).any(), "retained-H molecule should diverge atom_off vs fukui_off"

    query = mols[0]                                  # clean query
    hits = screen(query, ProfileStore.open(store_path), mode="vol_fukui", backend="numba",
                  num_repeats=16, max_num_steps=50, top_k=len(mols))
    assert len(hits) == len(mols)
    fast_by_id = {h.id: h.score for h in hits}

    cq = copy.deepcopy(query); cq.center_to(cq.atom_pos.mean(0))
    def _c(m):
        c = copy.deepcopy(m); c.center_to(c.atom_pos.mean(0)); return c
    pairs = [MoleculePair(cq, _c(m), do_center=False) for m in mols]
    ref, _ = MoleculePairBatch(pairs).align_with_vol_fukui(
        backend="numba", num_repeats=16, max_num_steps=50)
    for i, rs in enumerate(ref):                     # incl. the retained-H molecule
        assert fast_by_id[i] == pytest.approx(float(rs), abs=1e-2)
