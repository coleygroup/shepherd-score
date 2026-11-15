"""``cpu_threads=`` on ``MoleculePairBatch`` and ``screen``: the cap is in force during the
call, both thread counts are restored after it (also on error), results do not depend on it,
and impossible requests fail with a message naming the fix."""
import os

import numpy as np
import pytest
import torch

numba = pytest.importorskip("numba")

from . import _vca as V                                          # noqa: E402

MODE = "vol_color_avoid"


def _batch(cpu_threads=None):
    from shepherd_score.container import MoleculePairBatch
    ps = [V.pair(V.SMILES[i], V.SMILES[j]) for i, j in [(0, 1), (2, 3), (4, 0)]]
    return MoleculePairBatch(ps, cpu_threads=cpu_threads), [V.near_cloud(p.ref_molec, n=4)
                                                           for p in ps]


def _pool():
    return int(numba.config.NUMBA_NUM_THREADS)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_rejects_non_positive_integers(bad):
    from shepherd_score.container import MoleculePairBatch
    with pytest.raises(ValueError, match="cpu_threads"):
        MoleculePairBatch([], cpu_threads=bad)


def test_above_the_pool_names_numba_num_threads():
    from shepherd_score.accel._threads import cpu_threads
    with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
        with cpu_threads(_pool() + 1):
            pass


def test_cap_holds_during_the_call_and_is_restored(monkeypatch):
    if _pool() < 2:
        pytest.skip("needs a numba pool of at least 2")
    from shepherd_score.accel.batch import aligners
    seen = {}
    real = aligners._align_batch_vol_color_avoid

    def spy(*a, **k):
        seen["numba"] = numba.get_num_threads()
        return real(*a, **k)
    monkeypatch.setattr(aligners, "_align_batch_vol_color_avoid", spy)
    before = (numba.get_num_threads(), torch.get_num_threads())
    b, clouds = _batch(cpu_threads=2)
    b.align_with_vol_color_avoid(clouds, backend="numba")
    assert seen["numba"] == 2
    assert (numba.get_num_threads(), torch.get_num_threads()) == before


def test_counts_restored_when_the_call_raises():
    from shepherd_score.accel._threads import cpu_threads
    before = (numba.get_num_threads(), torch.get_num_threads())
    with pytest.raises(RuntimeError):
        with cpu_threads(1):
            raise RuntimeError("boom")
    assert (numba.get_num_threads(), torch.get_num_threads()) == before


def test_scores_do_not_depend_on_the_cap():
    """Each pose is one prange iteration, so the thread count cannot change any result."""
    b1, clouds = _batch(cpu_threads=1)
    s1, _ = b1.align_with_vol_color_avoid(clouds, backend="numba")
    b2, _ = _batch()
    s2, _ = b2.align_with_vol_color_avoid(clouds, backend="numba")
    assert np.array_equal(s1, s2)
    for p, q in zip(b1.pairs, b2.pairs):
        assert np.array_equal(p.transform_vol_color_avoid, q.transform_vol_color_avoid)


def test_conflicts_with_num_workers():
    b, clouds = _batch(cpu_threads=2)
    with pytest.raises(ValueError, match="num_workers"):
        b.align_with_vol_color_avoid(clouds, backend="numba", num_workers=2)


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    from shepherd_score.screen import ProfileStore
    p = os.path.join(tmp_path_factory.mktemp("threads"), "lib.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=[MODE], dtype="float32",
                             pre_centered=True, canonical=False) as st:
        for i, s in enumerate(V.SMILES):
            st.add(V.mol(s, seed=i), id=i)
    return p


def test_screen_cap_gives_the_same_hits_and_restores(store_path):
    from shepherd_score.screen import ProfileStore, screen
    q = V.mol(V.SMILES[0])
    cloud = V.near_cloud(q, n=4)
    before = (numba.get_num_threads(), torch.get_num_threads())
    with ProfileStore.open(store_path) as st:
        a = screen(q, st, mode=MODE, backend="numba", avoid_points=cloud, cpu_threads=1)
        b = screen(q, st, mode=MODE, backend="numba", avoid_points=cloud)
    assert [(h.id, h.score) for h in a] == [(h.id, h.score) for h in b]
    assert (numba.get_num_threads(), torch.get_num_threads()) == before


def test_screen_rejects_cpu_threads_with_ndev(store_path):
    from shepherd_score.screen import ProfileStore, screen
    with ProfileStore.open(store_path) as st:
        with pytest.raises(ValueError, match="worker_threads"):
            screen(V.mol(V.SMILES[0]), st, mode=MODE, avoid_points=np.zeros((1, 3)),
                   ndev=2, cpu_threads=2)
