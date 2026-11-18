"""The shard-parallel CPU driver keeps its forked pool across calls against the same library.

Runs in a subprocess: forking a process that has already run a numba prange, as this pytest
process has by collection time, can abort the child under libgomp.
"""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("numba")
pytest.importorskip("torch")
pytest.importorskip("rdkit")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT = r'''
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from shepherd_score.container import Molecule
import shepherd_score.accel.screen_parallel as sp

smiles = ["CCO", "C1CCCCC1", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1",
          "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
mols = []
for i, smi in enumerate(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = i
    assert AllChem.EmbedMolecule(m, params) == 0, smi
    rng = np.random.default_rng(i)
    mols.append(Molecule(m, surface_points=rng.standard_normal((64, 3)).astype(np.float32) * 3.0,
                         electrostatics=rng.standard_normal((64,)).astype(np.float32),
                         pharm_multi_vector=False))
query, lib = mols[1], mols
a = sp.screen_parallel(query, lib, "vol", n_workers=2)
pool1 = sp._POOL["pool"]
b = sp.screen_parallel(query, lib, "vol", n_workers=2)          # same list object: reused
pool2 = sp._POOL["pool"]
c = sp.screen_parallel(query, list(lib), "vol", n_workers=2)    # another list: forked afresh
pool3 = sp._POOL["pool"]
d = sp.screen_parallel(query, lib, "vol", n_workers=3)          # another worker count: afresh
pool4 = sp._POOL["pool"]
sp.screen_parallel_close()
print(json.dumps(dict(a=a, b=b, c=c, d=d, reused=pool1 is pool2, refreshed=pool3 is not pool2,
                      resized=pool4 is not pool3, closed=sp._POOL is None and sp._LIBRARY == [])))
'''


@pytest.mark.skipif(os.name != "posix", reason="screen_parallel forks; POSIX only")
def test_pool_is_reused_for_the_same_library_and_refreshed_for_another():
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True,
                       timeout=900, cwd=_ROOT, env=env)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["reused"], "a second call against the same library forked a new pool"
    assert out["refreshed"], "a different library object was served by the old pool"
    assert out["resized"], "a different worker count was served by the old pool"
    assert out["closed"]
    assert len(out["a"]) == 6 and all(isinstance(s, float) for s in out["a"])
    # a, b and c align identical shard batches and must agree bit for bit; d splits into three
    # shards, and the padded batch composition feeds the seed frame, so d is only close
    assert out["a"] == out["b"], "the reused pool returned different scores"
    assert out["a"] == out["c"], "a fresh pool for the same library returned different scores"
    assert max(abs(x - y) for x, y in zip(out["a"], out["d"])) < 1e-3, \
        "a different shard split moved a score by more than the batch-composition level"


def test_shards_are_strided_and_cover_the_library():
    """Worker w gets w, w+k, w+2k, ... so compound ensembles are spread across workers."""
    from shepherd_score.accel.screen_parallel import _chunks
    ch = _chunks(10, 4)
    assert [list(r) for r in ch] == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
    assert sorted(i for r in _chunks(1000, 7) for i in r) == list(range(1000))
    assert len(_chunks(3, 8)) == 3, "never more chunks than molecules"


def test_physical_core_selection_takes_one_cpu_per_sibling_group(tmp_path, monkeypatch):
    from shepherd_score.accel import screen_parallel as sp
    if not os.path.isdir("/sys/devices/system/cpu/cpu0/topology"):
        pytest.skip("no CPU topology in /sys on this platform")
    allowed = set(os.sched_getaffinity(0))
    cores = sp._physical_cores(allowed)
    assert cores and set(cores) <= allowed and len(cores) == len(set(cores))
    assert len(cores) <= len(allowed)
