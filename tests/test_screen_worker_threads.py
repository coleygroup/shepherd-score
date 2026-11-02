"""Multi-GPU host thread budgets are explicit and restored after pool creation."""
import importlib
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('requested,expected', [(None,20),(1,1),(3,3)])
def test_worker_thread_budget(monkeypatch, requested, expected):
    screen=importlib.import_module('shepherd_score.screen')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)   # CI runs a CPU-only torch
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:set(range(40)),raising=False)
    monkeypatch.setattr(screen.ProfileStore,'open',lambda _:SimpleNamespace(num_shards=2))
    monkeypatch.setenv('OMP_NUM_THREADS','7')
    class PoolReached(Exception):
        pass
    def pool(ndev,threads):
        assert ndev==2 and threads==expected
        assert os.environ['OMP_NUM_THREADS']==str(expected)
        raise PoolReached
    monkeypatch.setattr(screen,'_mgpu_pool',pool)
    with pytest.raises(PoolReached):
        screen._screen_many_multigpu([], 'unused','vol',2,{},100,False,
                                     worker_threads=requested)
    assert os.environ['OMP_NUM_THREADS']=='7'
