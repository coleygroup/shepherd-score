"""Splitting a launch across CUDA grid slices must not change the answer.

``drivers/terms._chunked`` slices at a pointer-offset ceiling; the per-molecule arguments
(coordinates, ``N_real`` / ``M_real``) must be sliced with the molecules. Driven with a tiny
stand-in limit so it runs on CPU.
"""
import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")


def _fake_kernel(A, B, q, t, *, alpha, N_real, M_real, NEED_GRAD=True, **extra):
    """Kernel stand-in whose value depends on ``N_real``, so a misaligned slice shows up."""
    K = q.shape[0]
    S = K // A.shape[0]
    n = N_real.to(torch.float32).repeat_interleave(S)
    m = M_real.to(torch.float32).repeat_interleave(S)
    V = n * 1000.0 + m + A[:, 0, 0].repeat_interleave(S) * 1e6
    return V, torch.zeros_like(q), torch.zeros_like(t)


@pytest.mark.parametrize("S", [1, 4])
@pytest.mark.parametrize("n_mol", [10, 100, 257])
def test_chunked_launch_matches_a_single_launch(S, n_mol, monkeypatch):
    """One launch and many launches must give identical output."""
    from shepherd_score.accel.drivers import terms as T

    g = torch.Generator().manual_seed(0)
    K = n_mol * S
    A = torch.rand(n_mol, 5, 3, generator=g)
    B = torch.rand(n_mol, 5, 3, generator=g)
    q = torch.rand(K, 4, generator=g)
    t = torch.rand(K, 3, generator=g)
    N_real = torch.randint(1, 6, (n_mol,), generator=g, dtype=torch.int32)
    M_real = torch.randint(1, 6, (n_mol,), generator=g, dtype=torch.int32)
    kw = dict(alpha=0.81, N_real=N_real, M_real=M_real, NEED_GRAD=True)

    monkeypatch.setattr(T, "_CHUNK", 10 ** 9)          # one launch
    whole, _, _ = T._chunked(_fake_kernel, K, S, (A, B), (q, t), kw, {})

    for chunk in (S, 3 * S, 7 * S):
        monkeypatch.setattr(T, "_CHUNK", max(chunk, 1))
        part, _, _ = T._chunked(_fake_kernel, K, S, (A, B), (q, t), kw, {})
        assert torch.equal(whole, part), (
            f"S={S} n_mol={n_mol}: chunking at {chunk} changed {int((whole != part).sum())} of "
            f"{K} values; a per-molecule argument is not being sliced with the molecules")


def test_chunked_keeps_each_seed_group_whole(monkeypatch):
    """Chunk sizes are rounded down to a multiple of S so a seed group is never split."""
    from shepherd_score.accel.drivers import terms as T

    seen = []

    def spy(A, B, q, t, *, alpha, N_real, M_real, NEED_GRAD=True, **extra):
        seen.append((A.shape[0], q.shape[0]))
        return _fake_kernel(A, B, q, t, alpha=alpha, N_real=N_real, M_real=M_real)

    S, n_mol = 8, 50
    K = n_mol * S
    A = torch.zeros(n_mol, 4, 3); B = torch.zeros(n_mol, 4, 3)
    q = torch.zeros(K, 4); t = torch.zeros(K, 3)
    kw = dict(alpha=0.81, N_real=torch.full((n_mol,), 4, dtype=torch.int32),
              M_real=torch.full((n_mol,), 4, dtype=torch.int32), NEED_GRAD=True)
    monkeypatch.setattr(T, "_CHUNK", 30)               # not a multiple of S
    T._chunked(spy, K, S, (A, B), (q, t), kw, {})

    assert seen, "the kernel was never called"
    for n_rows, n_poses in seen:
        assert n_poses % S == 0, f"a chunk of {n_poses} poses splits a seed group of {S}"
        assert n_poses == n_rows * S, (
            f"{n_poses} poses against {n_rows} molecule rows: the two slices disagree")
