"""Parity gate for the array-native screen path (``FSS_SCREEN_ARRAYS=1``).

For every mode in ``screen.py::_ARRAY_MODES`` the array path must produce byte-identical scores
to the object path. Spies assert which builder ran, so a gate that silently routed both legs
the same way could not pass. The library spans several pad bands (see ``_SMILES``).
"""
import os

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402

import shepherd_score.screen as screenmod                       # noqa: E402
from shepherd_score.accel._modes import CANONICAL_MODES, uses_avoid  # noqa: E402
from shepherd_score.container import Molecule                   # noqa: E402
from shepherd_score.screen import ProfileStore, screen          # noqa: E402

_BAND = 16          # mirrors accel.batch._pad._BAND; asserted against the real one below

#: Every mode-shaped thing in this file derives from this tuple; snapshotted at import because
#: ``pytest.mark.parametrize`` needs a concrete sequence.
_MODES = tuple(screenmod._ARRAY_MODES)

#: The kwargs each mode requires of ``screen()``: spec parameters defaulting to ``None`` must be
#: supplied, and the avoid modes additionally need their avoid cloud.
_REQUIRED_VALUES = {"lam": 0.1, "alpha": 0.81}
_MODE_KW = {}
for _m, _sp in screenmod._SPECS.items():
    _kw = {k: _REQUIRED_VALUES[k] for k, v in _sp.params.items() if v is None}
    if uses_avoid(_m):
        _kw["avoid_points"] = np.array([[6.0, 0.0, 0.0], [6.0, 2.0, 0.0]], dtype=np.float32)
    if _kw:
        _MODE_KW[_m] = _kw

#: Modes whose self-copy cannot reach ~1.0 on this fixture, derived from the spec: a value-only
#: agreement term (the synthetic surface is not the one implied by the atoms) or a subtracted
#: penalty term (the avoid modes).
_SELF_SCORE_NOT_ONE = {m for m, sp in screenmod._SPECS.items()
                       if any(t.reduction == "agreement" for t in sp.terms)
                       or any(isinstance(t.weight, str) and t.weight.startswith("-")
                              for t in sp.terms)}


def _scored_asymmetrically(mode: str) -> bool:
    """Whether ``mode`` uses an asymmetric Tversky reduction, so the self-copy need not rank first."""
    kw = screenmod._fast_batch_kwargs(mode, dict(_MODE_KW.get(mode, {})))
    return kw.get("tversky_alpha") != kw.get("tversky_beta")


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


def _require_numba():
    """``screen()`` raises ImportError without numba on the fast CPU path."""
    pytest.importorskip("numba")


# Drug-like molecules (band 16) plus n-alkanes so the library also occupies bands 32 and 48;
# "C" * n has exactly n heavy atoms.
_SMILES = [
    "CCO",                                  # 3
    "C1CCCCC1",                             # 6
    "c1ccccc1O",                            # 7
    "CC(=O)Nc1ccc(O)cc1",                   # 11
    "CC(=O)Oc1ccccc1C(=O)O",                # 13
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",           # 15
    "C" * 18,                               # 18 -> band 32
    "C" * 22,                               # 22 -> band 32
    "C" * 26,                               # 26 -> band 32
    "C" * 30,                               # 30 -> band 32
    "C" * 34,                               # 34 -> band 48
    "C" * 36,                               # 36 -> band 48
]


def _build_molecule(smi, seed, S=64):
    """Real RDKit conformer with a synthetic surface (no Open3D) and a synthetic with-H Fukui field (no xtb)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    surf = rng.standard_normal((S, 3)).astype(np.float32) * 3.0
    esp = rng.standard_normal((S,)).astype(np.float32)
    fukui = rng.standard_normal((m.GetNumAtoms(),)).astype(np.float32) * 0.1
    return Molecule(m, surface_points=surf, electrostatics=esp, pharm_multi_vector=False,
                    fukui=fukui)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


@pytest.fixture(scope="module")
def store_path(tmp_path_factory, molecules):
    """One non-canonical store serving every array mode, so both parity legs run the same seeds."""
    p = os.path.join(tmp_path_factory.mktemp("arrays"), "lib.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=_MODES,
                             dtype="float32", pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def _screen_recording(monkeypatch, store_path, query, *, enabled, mode="vol", steps=30,
                      backend="triton"):
    """Run one screen with the array path forced on/off, recording which builder ran."""
    from shepherd_score.accel.batch import _arrays

    monkeypatch.setattr(_arrays, "ENABLED", enabled, raising=True)
    # the seam must bite for this mode, or both legs would take the same path and the parity
    # assertion would compare the object path with itself
    assert screenmod._use_arrays(mode) is enabled, (
        f"_arrays.ENABLED={enabled} but _use_arrays({mode!r}) is "
        f"{screenmod._use_arrays(mode)}; the two legs would take the same path")

    seen = {"arrays": 0, "objects": 0, "buckets": 0}
    real_obj = screenmod._build_fit_fast_pairs
    real_spans = _arrays.plan_spans

    def _count_arr(real):
        def f(*a, **k):
            seen["arrays"] += 1
            return real(*a, **k)
        return f

    def spy_obj(*a, **k):
        seen["objects"] += 1
        return real_obj(*a, **k)

    real_spans_multi = _arrays.plan_spans_multi

    def spy_spans(*a, **k):
        order, buckets = real_spans(*a, **k)
        seen["buckets"] = max(seen["buckets"], len(buckets))
        return order, buckets

    def spy_spans_multi(*a, **k):
        # vol_and_surf_esp keys six dims and plans through plan_spans_multi, not plan_spans
        buckets = real_spans_multi(*a, **k)
        seen["buckets"] = max(seen["buckets"], len(buckets))
        return buckets

    # Every array builder is wrapped and the _ARRAY_BUILDERS table re-pointed at the wrappers:
    # the dispatch reads the table, so patching only the module globals would count zero.
    for _fn in dict.fromkeys(screenmod._ARRAY_BUILDERS.values()):
        monkeypatch.setattr(screenmod, _fn.__name__, _count_arr(_fn), raising=True)
    monkeypatch.setattr(screenmod, "_ARRAY_BUILDERS",
                        {m: _count_arr(fn) for m, fn in screenmod._ARRAY_BUILDERS.items()},
                        raising=True)
    monkeypatch.setattr(screenmod, "_build_fit_fast_pairs", spy_obj, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans", spy_spans, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans_multi", spy_spans_multi, raising=True)

    n = len(ProfileStore.open(store_path))
    scores = np.full(n, np.nan, dtype=float)
    hits = screen(query, ProfileStore.open(store_path), mode=mode, backend=backend,
                  top_k=n, max_num_steps=steps, scores_out=scores, **_MODE_KW.get(mode, {}))
    return scores, hits, seen


def _assert_paths_diverged(seen_on, seen_off):
    """The two legs really ran different builders."""
    assert seen_on["arrays"] > 0, "ENABLED=True did not reach the array builder"
    assert seen_on["objects"] == 0, "ENABLED=True still built _FastPair objects"
    assert seen_off["objects"] > 0, "ENABLED=False did not reach the object builder"
    assert seen_off["arrays"] == 0, "ENABLED=False still reached the array builder"
    assert seen_on["buckets"] >= 1, "plan_spans never ran on the array path"


def _assert_bit_identical(s_on, h_on, s_off, h_off):
    # --- scores are real, not a degenerate all-equal / all-NaN vector -----------------------
    assert np.isfinite(s_on).all(), "array path produced non-finite scores"
    assert np.isfinite(s_off).all(), "object path produced non-finite scores"
    assert s_on.max() - s_on.min() > 1e-6, "scores are degenerate; parity would be vacuous"

    # --- the gate itself --------------------------------------------------------------------
    np.testing.assert_array_equal(s_on, s_off)
    assert [h.id for h in h_on] == [h.id for h in h_off], "top-k identity/order diverged"
    np.testing.assert_array_equal(np.array([h.score for h in h_on], dtype=float),
                                  np.array([h.score for h in h_off], dtype=float))


def _assert_self_copy(mode, scores, hits):
    """The query is library id=1: it ranks first (symmetric modes) and scores ~1.0 unless excused."""
    if not _scored_asymmetrically(mode):
        assert hits[0].id == 1, "query is in the library at id=1 and must rank first"
    if mode not in _SELF_SCORE_NOT_ONE:
        assert scores[1] == pytest.approx(1.0, abs=1e-2), "self-copy must score ~1.0"


def test_library_spans_multiple_pad_bands(molecules):
    """The fixture must cross ``_band_key`` boundaries or the span arithmetic goes untested."""
    from shepherd_score.accel.batch._pad import _BAND as REAL_BAND, _band_key

    assert REAL_BAND == _BAND, "band width changed upstream; revisit the library sizes"
    sizes = [int(m.atom_pos.shape[0]) for m in molecules]
    bands = {_band_key(n) for n in sizes}
    assert len(set(sizes)) >= 6, f"library sizes are not varied enough: {sorted(set(sizes))}"
    assert len(bands) >= 3, f"library occupies only bands {sorted(bands)}; need >=3"


def test_store_fixture_serves_every_array_mode(store_path):
    """The store must serve every array mode, and every array mode must also be a fast mode."""
    store = ProfileStore.open(store_path)
    assert set(screenmod._ARRAY_MODES) == set(_MODES), \
        "_ARRAY_MODES changed after import; the store was built for the import-time tuple"
    for mode in screenmod._ARRAY_MODES:
        assert store.supports(mode), (
            f"the fixture store was built for {store.modes} and cannot serve {mode!r}; "
            f"its parity test would fail in _resolve_screen, not on parity")
        assert mode in screenmod._FAST_MODES, (
            f"{mode!r} is an array mode but not a fast mode; the ENABLED=False leg would not be "
            f"the object path this file compares against")


def test_the_self_copy_rank_anchor_is_not_excused_away():
    """``_scored_asymmetrically`` must not excuse every mode, and never ``vol``."""
    excused = [m for m in _MODES if _scored_asymmetrically(m)]
    assert not _scored_asymmetrically("vol"), "vol is a symmetric Tanimoto; nothing excuses it"
    assert len(excused) < len(_MODES), \
        f"every array mode is excused from the rank anchor ({excused}); it now asserts nothing"


@pytest.mark.cuda
@pytest.mark.parametrize("mode", _MODES)
def test_array_path_is_bit_identical_to_object_path(monkeypatch, store_path, molecules, mode):
    """The array path is a re-expression of the object path, so scores must match EXACTLY."""
    _require_fast_cuda()
    query = molecules[1]

    with monkeypatch.context() as mp:
        s_on, h_on, seen_on = _screen_recording(mp, store_path, query, enabled=True, mode=mode)
    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _screen_recording(mp, store_path, query, enabled=False, mode=mode)

    _assert_paths_diverged(seen_on, seen_off)
    _assert_bit_identical(s_on, h_on, s_off, h_off)


@pytest.mark.parametrize("mode", _MODES)
def test_array_path_is_bit_identical_to_object_path_cpu(monkeypatch, store_path, molecules, mode):
    """The same gate on the CPU route; the self-copy anchor rides along on the ON leg."""
    _require_numba()
    query = molecules[1]

    with monkeypatch.context() as mp:
        s_on, h_on, seen_on = _screen_recording(mp, store_path, query, enabled=True, mode=mode,
                                                backend="numba")
    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _screen_recording(mp, store_path, query, enabled=False,
                                                   mode=mode, backend="numba")

    _assert_paths_diverged(seen_on, seen_off)
    _assert_bit_identical(s_on, h_on, s_off, h_off)
    _assert_self_copy(mode, s_on, h_on)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", _MODES)
def test_array_path_recovers_the_self_copy(monkeypatch, store_path, molecules, mode):
    """The same anchor on the GPU route, where the scores come out of the triton kernels."""
    _require_fast_cuda()
    with monkeypatch.context() as mp:
        scores, hits, seen = _screen_recording(mp, store_path, molecules[1],
                                               enabled=True, mode=mode)
    assert seen["arrays"] > 0
    _assert_self_copy(mode, scores, hits)


def test_use_arrays_gates_on_mode_and_reads_enabled_live(monkeypatch):
    """``_use_arrays`` gates on both the mode and the live ``ENABLED`` flag."""
    from shepherd_score.accel.batch import _arrays

    # the non-array modes are derived, not named, so the negative control cannot rot
    non_array = tuple(m for m in CANONICAL_MODES if m not in screenmod._ARRAY_MODES)

    monkeypatch.setattr(_arrays, "ENABLED", True, raising=True)
    for mode in screenmod._ARRAY_MODES:
        assert screenmod._use_arrays(mode) is True
    for mode in non_array + ("__not_a_mode__",):
        assert screenmod._use_arrays(mode) is False, f"{mode} has no array-native aligner"
    assert set(screenmod._ARRAY_BUILDERS) == set(screenmod._ARRAY_MODES), \
        "a mode in _ARRAY_MODES with no builder entry would KeyError mid-screen"

    monkeypatch.setattr(_arrays, "ENABLED", False, raising=True)
    for mode in screenmod._ARRAY_MODES:
        assert screenmod._use_arrays(mode) is False, "ENABLED is read at call time, not import"

# Canonical-frame stores: the transform must come back in the molecule's frame, not the store's.
# Scores and ranking are correct either way, so the check re-scores each returned pose on the
# molecule's own centered coordinates; a pose in the wrong frame does not reproduce its score.

def _canonical_store(tmp_path, molecules):
    p = os.path.join(tmp_path, "canon.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float32",
                             pre_centered=True, canonical=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def _plain_store(tmp_path, molecules):
    """The control leg: same library, canonical explicitly off (the default would be on)."""
    p = os.path.join(tmp_path, "plain.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float32",
                             pre_centered=True, canonical=False) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def test_canonical_store_records_its_rotation(tmp_path, molecules):
    """The store must carry ``rot`` and it must be a proper rotation."""
    store = ProfileStore.open(_canonical_store(str(tmp_path), molecules))
    assert store.canonical is True
    _, arrs = store.read_shard(0)
    R = np.asarray(arrs["rot"])
    assert R.shape == (len(molecules), 3, 3)
    assert R.dtype == np.float32                       # float32 regardless of store dtype
    assert np.allclose(np.matmul(R, np.transpose(R, (0, 2, 1))),
                       np.eye(3)[None], atol=2e-5), "rot is not orthogonal"
    assert np.all(np.linalg.det(R) > 0), "rot contains a reflection"


def _rescore_hits(hits, molecules, query):
    """``(id, reported, re-scored)`` per hit: apply its own transform and recompute the Tanimoto."""
    from shepherd_score.score.gaussian_overlap import shape_tanimoto
    q = np.asarray(query.atom_pos, dtype=np.float64)
    qt = torch.as_tensor(q - q.mean(0), dtype=torch.float64)
    out = []
    for h in hits:
        fit = np.asarray(molecules[h.id].atom_pos, dtype=np.float64)
        fit = fit - fit.mean(0)
        T = np.asarray(h.transform, dtype=np.float64)
        posed = fit @ T[:3, :3].T + T[:3, 3]
        out.append((h.id, h.score, float(shape_tanimoto(qt, torch.as_tensor(posed), 0.81))))
    return out


@pytest.mark.parametrize("canonical", [False, True])
def test_transform_is_in_the_molecule_frame_cpu(tmp_path, molecules, canonical):
    """A returned pose must reproduce its own score on both store kinds (CPU route)."""
    path = _canonical_store(str(tmp_path), molecules) if canonical         else _plain_store(str(tmp_path), molecules)
    hits = screen(molecules[0], ProfileStore.open(path), mode="vol", backend="torch", top_k=4)
    assert hits, "screen returned nothing"
    for mol_id, reported, rescored in _rescore_hits(hits, molecules, molecules[0]):
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f} "
            f"-- the transform is not in the molecule's frame (canonical={canonical})")


def test_canonical_store_screens_without_the_array_path(monkeypatch, tmp_path, molecules):
    """A canonical store must not require the array path (``const_seeds`` is array-only)."""
    from shepherd_score.accel.batch import _arrays
    monkeypatch.setattr(_arrays, "ENABLED", False, raising=True)
    hits = screen(molecules[0], ProfileStore.open(_canonical_store(str(tmp_path), molecules)),
                  mode="vol", backend="torch", top_k=3)
    assert len(hits) == 3


@pytest.mark.cuda
def test_canonical_transform_is_in_the_molecule_frame(tmp_path, molecules):
    """GPU twin of the frame check: the pose comes from the array path's SE(3) epilogue."""
    _require_fast_cuda()
    path = _canonical_store(str(tmp_path), molecules)
    hits = screen(molecules[0], ProfileStore.open(path), mode="vol", backend="triton", top_k=5)
    assert hits, "screen returned nothing"
    for mol_id, reported, rescored in _rescore_hits(hits, molecules, molecules[0]):
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f} "
            f"-- the transform is not in the molecule's frame")

# vol_esp on a canonical store with a molecule whose Chem.RemoveHs retains an H: ``xyz_noH`` is
# materialised only for that molecule and must be rotated like ``atom_pos``. A score comparison
# cannot see a missing rotation; only re-scoring the returned pose can.

_RETAINED_H_SMI = "[2H]OC(=O)c1ccccc1"          # the deuterium survives RemoveHs


def _esp_molecules():
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", _RETAINED_H_SMI,
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    mols = [_build_molecule(s, seed=i) for i, s in enumerate(smis)]
    retained = [i for i, m in enumerate(mols) if len(m.atom_pos) != len(m._nonH_atoms_idx)]
    assert retained, "test premise broken: no molecule retains an H after RemoveHs"
    return mols, set(retained)


def _rescore_hits_vol_esp(hits, molecules, query, lam=0.1):
    """``(id, reported, re-scored)`` per vol_esp hit; ``get_overlap_esp`` already returns a Tanimoto."""
    from shepherd_score.score.electrostatic_scoring import get_overlap_esp

    def heavy(m):
        x = m.mol.GetConformer().GetPositions()[m._nonH_atoms_idx]
        return np.ascontiguousarray(x, dtype=np.float64)

    q = heavy(query); q = q - np.asarray(query.atom_pos, dtype=np.float64).mean(0)
    qt = torch.as_tensor(q)
    qc = torch.as_tensor(np.asarray(query.get_charges(no_H=True), dtype=np.float64))
    out = []
    for h in hits:
        m = molecules[h.id]
        fit = heavy(m) - np.asarray(m.atom_pos, dtype=np.float64).mean(0)
        T = np.asarray(h.transform, dtype=np.float64)
        posed = torch.as_tensor(fit @ T[:3, :3].T + T[:3, 3])
        fc = torch.as_tensor(np.asarray(m.get_charges(no_H=True), dtype=np.float64))
        out.append((h.id, h.score,
                    float(get_overlap_esp(qt, posed, qc, fc, alpha=0.81, lam=lam))))
    return out


@pytest.mark.parametrize("canonical", [False, True])
def test_vol_esp_retained_h_transform_is_in_the_molecule_frame(tmp_path, canonical):
    """A vol_esp pose reproduces its own score on both store kinds, retained-H molecule included."""
    mols, retained = _esp_molecules()
    p = os.path.join(str(tmp_path), f"esp_{int(canonical)}.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol", "vol_esp"),
                             dtype="float32", pre_centered=True, canonical=canonical) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)

    opened = ProfileStore.open(p)
    assert opened.canonical is canonical
    assert "xyz_noH" in opened.read_shard(0)[1], \
        "premise broken: the strict-heavy center array was not stored"

    hits = screen(mols[0], ProfileStore.open(p), mode="vol_esp", backend="torch",
                  lam=0.1, top_k=len(mols))
    assert hits, "screen returned nothing"
    seen = set()
    for mol_id, reported, rescored in _rescore_hits_vol_esp(hits, mols, mols[0]):
        seen.add(mol_id)
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f} "
            f"-- the transform is not in the molecule's frame (canonical={canonical}"
            f"{', RETAINED-H' if mol_id in retained else ''})")
    assert retained <= seen, "the retained-H molecule was not among the hits checked"
