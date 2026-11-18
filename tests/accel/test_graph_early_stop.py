"""The CUDA-graph replay loop's early stop uses the caller's patience, exactly as the eager and
fused CPU loops do, and a cached graph takes each call's patience rather than the one it was
captured with. CPU-only: a stand-in graph replays a scripted per-seed score trajectory through
the real ``_GraphedFineBase.run`` / ``run_graphed`` code.
"""
import pytest

rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from shepherd_score.accel.drivers import _graphed                  # noqa: E402

STEPS = 30
SEEDS = 2


class _ScriptedGraph:
    def __init__(self, owner):
        self.owner = owner

    def replay(self):
        self.owner.tick()


class _Scripted(_graphed._GraphedFineBase):
    """``best`` follows ``trajectory[i]`` after replay ``i`` (the last row repeats)."""

    def __init__(self, trajectory):
        self.trajectory = [torch.as_tensor(r, dtype=torch.float32) for r in trajectory]
        self._best = torch.empty(len(trajectory[0]))
        self.replays = 0
        super().__init__(STEPS)

    def capture(self, *inputs):
        self._reset()
        self.graph = _ScriptedGraph(self)

    def tick(self):
        row = self.trajectory[min(self.replays, len(self.trajectory) - 1)]
        torch.maximum(self._best, row, out=self._best)
        self.replays += 1

    def _load(self, *x):
        pass

    def _reset(self):
        self.replays = 0
        self._best.fill_(-float("inf"))

    def _result(self):
        return self._best, None, None

    @property
    def best(self):
        return self._best


# Two pairs x two seeds. Both pairs reach their best on replay 1 and never improve again.
FLAT = [[0.5, 0.4, 0.6, 0.3]]
# Pair 1 improves once more at replay 16, after two stalled checks (replays 6 and 11).
LATE = [[0.5, 0.4, 0.6, 0.3]] * 15 + [[0.5, 0.4, 0.7, 0.3]]


def _run(key, trajectory, patience, store):
    def make():
        gf = _Scripted(trajectory)
        store.append(gf)
        return gf
    _graphed.run_graphed(make, key, (), es_patience=patience, es_tol=1e-5, es_seeds=SEEDS)
    return store[0].replays


@pytest.fixture
def key():
    k = ("test_graph_early_stop", object())
    yield k
    _graphed._FINE_GRAPH_CACHE.pop(k, None)


@pytest.mark.parametrize("patience, expected", [(2, 11), (4, 21), (5, 26), (0, STEPS)])
def test_replay_loop_stops_at_the_eager_checks(key, patience, expected):
    # Checks fall after replays 1, 6, 11, ...; the first only seeds the baseline, so a fully
    # stalled bucket stops after 1 + 5 * patience replays. Patience 0 replays the full budget.
    assert _run(key, FLAT, patience, []) == expected


def test_late_improvement_is_lost_at_patience_two_and_kept_at_four(key):
    store = []
    assert _run(key, LATE, 2, store) == 11
    assert float(store[0].best.view(-1, SEEDS).amax(dim=1)[1]) == pytest.approx(0.6)
    _graphed._FINE_GRAPH_CACHE.pop(key, None)
    store = []
    # The improvement at replay 16 resets the count; only 3 stalled checks (21, 26, 30) fit.
    assert _run(key, LATE, 4, store) == STEPS
    assert float(store[0].best.view(-1, SEEDS).amax(dim=1)[1]) == pytest.approx(0.7)


def test_cached_graph_takes_each_calls_patience(key):
    store = []
    assert _run(key, FLAT, 2, store) == 11            # captures the graph
    assert len(store) == 1
    assert _run(key, FLAT, 4, store) == 21            # cache hit: same object, new patience
    assert len(store) == 1
    assert _run(key, FLAT, 0, store) == STEPS
    assert store[0].es_patience == 0
