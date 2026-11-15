"""The mode registry (``accel/_modes.py``) is the single source of truth for alignment modes:
its internal invariants hold and every consumer (aligners, cpu_pool, multi_gpu, screen)
agrees with it.
"""
import pytest

from shepherd_score.accel import _modes as M


def test_canonical_modes_consistent():
    assert len(M.CANONICAL_MODES) == 23   # 10 core + 9 SI experimental modes + vol_fukui + vol_avoid + vol_color_avoid(_tversky)
    assert tuple(M.MODE_ATTRS) == M.CANONICAL_MODES
    # every mode has both a seed and a step default
    assert set(M.MODE_SEEDS) == set(M.CANONICAL_MODES)
    assert set(M.MODE_STEPS) == set(M.CANONICAL_MODES)
    # attrs are (transform, score) pairs of non-empty distinct strings
    for m, (tf, sc) in M.MODE_ATTRS.items():
        assert tf.startswith("transform_") and sc.startswith("sim_aligned_") and tf != sc


def test_legacy_aliases_resolve_to_canonical():
    assert M.LEGACY_MODE_ALIASES == {"esp": "surf_esp", "esp_combo": "vol_and_surf_esp"}
    for legacy, canon in M.LEGACY_MODE_ALIASES.items():
        assert canon in M.CANONICAL_MODES
        assert legacy not in M.CANONICAL_MODES          # legacy names are NOT themselves canonical
        assert M.canonical(legacy) == canon
    assert M.canonical("vol") == "vol"                  # unknown/canonical passes through


def test_process_modes_match_mode_spec():
    """PROCESS_MODES equals the ``_MODE_SPEC`` keys and each spec's ``out`` matches MODE_ATTRS."""
    pytest.importorskip("torch")
    from shepherd_score.accel.batch import _MODE_SPEC
    assert tuple(_MODE_SPEC) == M.PROCESS_MODES
    assert set(M.PROCESS_MODES) <= set(M.CANONICAL_MODES)
    for m in M.PROCESS_MODES:
        assert _MODE_SPEC[m]["out"] == M.MODE_ATTRS[m]


def test_aligners_seeds_steps_are_registry():
    pytest.importorskip("torch")
    from shepherd_score.accel.batch import aligners
    assert aligners._MODE_SEEDS is M.MODE_SEEDS
    assert aligners._MODE_STEPS is M.MODE_STEPS


def test_cpu_pool_wired_to_registry():
    pytest.importorskip("torch")
    from shepherd_score.accel import cpu_pool
    assert cpu_pool.POOL_MODES == M.PROCESS_MODES
    assert cpu_pool._LEGACY_MODE_ALIASES == M.LEGACY_MODE_ALIASES


def test_multi_gpu_attr_maps_derived_from_registry():
    from shepherd_score.accel import multi_gpu
    assert tuple(multi_gpu._SCORE_ATTR) == M.PROCESS_MODES
    assert multi_gpu._LEGACY_MODE_ALIASES == M.LEGACY_MODE_ALIASES
    for m in M.PROCESS_MODES:
        assert multi_gpu._TRANSFORM_ATTR[m] == M.MODE_ATTRS[m][0]
        assert multi_gpu._SCORE_ATTR[m] == M.MODE_ATTRS[m][1]


def test_screen_attr_maps_cover_all_modes():
    pytest.importorskip("rdkit")
    from shepherd_score import screen
    assert tuple(screen._SCORE_ATTR) == M.CANONICAL_MODES
    assert tuple(screen._VALID_MODES) == M.CANONICAL_MODES
    assert screen._canon_mode("esp_combo") == "vol_and_surf_esp"
    for m, (tf, sc) in M.MODE_ATTRS.items():
        assert screen._TRANSFORM_ATTR[m] == tf and screen._SCORE_ATTR[m] == sc


def test_moleculepair_batch_aligner_binds_are_registry_driven():
    """``_align_batch_<mode>`` is bound on MoleculePair for every canonical mode and legacy alias."""
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    from shepherd_score.container._core import MoleculePair
    from shepherd_score.accel import batch as _ba
    for m in M.CANONICAL_MODES:
        assert getattr(MoleculePair, "_align_batch_" + m) is getattr(_ba, "_align_batch_" + m), m
    for legacy, canon in M.LEGACY_MODE_ALIASES.items():
        assert getattr(MoleculePair, "_align_batch_" + legacy) is getattr(_ba, "_align_batch_" + canon), legacy


def test_moleculepair_init_result_slots_cover_registry():
    """``MoleculePair.__init__`` pre-initialises every MODE_ATTRS slot and the legacy no_H extras."""
    import numpy as np
    pytest.importorskip("torch")
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem
    from shepherd_score.container import MoleculePair

    def _mol(smi):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        return m

    mp = MoleculePair(_mol("CCO"), _mol("CCN"))  # num_surf_points=None -> no Open3D
    eye = np.eye(4)
    for tf, sc in M.MODE_ATTRS.values():
        assert np.array_equal(getattr(mp, tf), eye), tf
        assert getattr(mp, sc) is None, sc
    for tf, sc in (("transform_vol", "sim_aligned_vol"), ("transform_vol_esp", "sim_aligned_vol_esp")):
        assert np.array_equal(getattr(mp, tf), eye), tf
        assert getattr(mp, sc) is None, sc
    # legacy-name reads resolve to canonical storage via the property aliases
    assert np.array_equal(mp.transform_esp, mp.transform_surf_esp)
    assert np.array_equal(mp.transform_esp_combo, mp.transform_vol_and_surf_esp)
