"""Frozen configurations must bypass tuning and refuse missing shapes."""
import json
import pytest

triton = pytest.importorskip('triton')
import triton.language as tl
torch = pytest.importorskip('torch')
pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]

from shepherd_score.accel.kernels.tuning import autotune, export_configurations


@triton.jit
def _copy(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y+i, tl.load(X+i, i<N, other=0), i<N)


def test_replay_and_missing_shape(tmp_path, monkeypatch):
    monkeypatch.delenv('FSS_TRITON_CONFIGS', raising=False)
    configs = [triton.Config({'BLOCK':n}, num_warps=1) for n in (32,64)]
    tuned = autotune(configs=configs, key=['N'])(_copy)
    x = torch.arange(16, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    tuned[(1,)](x,y,16)
    path = tmp_path/'launches.json'
    export_configurations(path)
    monkeypatch.setenv('FSS_TRITON_CONFIGS', str(path))
    frozen = autotune(configs=configs, key=['N'])(_copy)
    monkeypatch.setattr(frozen, '_bench', lambda *a,**k: pytest.fail('replay retuned'))
    frozen[(1,)](x,y,16)
    assert torch.equal(x,y)
    with pytest.raises(ValueError, match='Missing frozen'):
        frozen[(1,)](x,y,15)
    data = json.loads(path.read_text());data['hardware']['gpu']='different GPU'
    bad = tmp_path/'different.json';bad.write_text(json.dumps(data))
    monkeypatch.setenv('FSS_TRITON_CONFIGS',str(bad))
    wrong = autotune(configs=configs,key=['N'])(_copy)
    with pytest.raises(ValueError,match='recorded GPU'):
        wrong[(1,)](x,y,16)
