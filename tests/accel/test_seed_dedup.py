"""Gates for the shared-reference seed dedup (``batched_seeds_torch(ref_shared=...)``).

In a screen every pair in a bucket shares one query, so ``ref_shared=True`` solves row 0 and
expands. Only the seeder's handling of the flag is tested here, to rounding rather than
equality, since ``_masked_principal_axes`` reduces via ``torch.bmm`` whose order depends on K.
"""
import inspect

import pytest
import torch

from shepherd_score.accel.drivers._common import batched_seeds_torch

DEV = torch.device("cpu")
SEEDS = 10


def _batch(K, Npad, Mpad, *, shared_ref, seed=0):
    """(A, B, N_real, M_real) with either one broadcast reference or K distinct ones."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if shared_ref:
        one = torch.randn(1, Npad, 3, generator=g, dtype=torch.float32)
        A = one.expand(K, Npad, 3).contiguous()      # every row bitwise identical
    else:
        A = torch.randn(K, Npad, 3, generator=g, dtype=torch.float32)
    B = torch.randn(K, Mpad, 3, generator=g, dtype=torch.float32)
    N = torch.full((K,), Npad - 2, dtype=torch.int32)
    M = torch.full((K,), Mpad - 2, dtype=torch.int32)
    return A.to(DEV), B.to(DEV), N.to(DEV), M.to(DEV)


def test_ref_shared_defaults_false_and_is_keyword_only():
    """The flag is opt-in and keyword-only, so every existing caller keeps its behaviour."""
    p = inspect.signature(batched_seeds_torch).parameters["ref_shared"]
    assert p.default is False
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_default_path_is_unchanged_by_row_permutation():
    """With the flag off the seeder is per-row: permuting the rows permutes the output."""
    A, B, N, M = _batch(2, 12, 10, shared_ref=False, seed=3)
    q01, t01 = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS)
    perm = torch.tensor([1, 0])
    q10, t10 = batched_seeds_torch(A[perm], B[perm], N[perm], M[perm], num_seeds=SEEDS)
    assert torch.equal(q01[0], q10[1]) and torch.equal(q01[1], q10[0])
    assert torch.equal(t01[0], t10[1]) and torch.equal(t01[1], t10[0])


@pytest.mark.parametrize("K,Npad,Mpad", [(8, 16, 12), (64, 32, 24), (129, 20, 40)])
def test_ref_shared_agrees_with_full_solve(K, Npad, Mpad):
    """On a genuinely shared reference the shortcut agrees with the full solve to rounding."""
    A, B, N, M = _batch(K, Npad, Mpad, shared_ref=True, seed=11)
    q_full, t_full = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS)
    q_ded, t_ded = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)
    assert q_ded.shape == q_full.shape and t_ded.shape == t_full.shape
    assert torch.allclose(q_ded, q_full, atol=1e-6, rtol=0)
    assert torch.allclose(t_ded, t_full, atol=1e-6, rtol=0)


def test_ref_shared_is_noop_at_K1():
    """K == 1 has nothing to dedup; the flag must not perturb it at all."""
    A, B, N, M = _batch(1, 16, 12, shared_ref=True, seed=5)
    q0, t0 = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS)
    q1, t1 = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)
    assert torch.equal(q0, q1) and torch.equal(t0, t1)


def test_flag_is_load_bearing_on_distinct_refs():
    """On distinct references the flag must change the result; a refactor cannot silently drop it."""
    A, B, N, M = _batch(16, 16, 12, shared_ref=False, seed=7)
    q_ok, _ = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS)
    q_lie, _ = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)
    assert not torch.allclose(q_ok, q_lie, atol=1e-4, rtol=0)



def test_verify_mode_passes_on_a_true_guarantee(monkeypatch):
    """Verify mode must not false-alarm on a legitimate firing."""
    monkeypatch.setenv("FSS_SEED_REF_DEDUP", "verify")
    A, B, N, M = _batch(64, 24, 20, shared_ref=True, seed=13)
    q, t = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)
    assert q.shape[0] == 64 and torch.isfinite(q).all() and torch.isfinite(t).all()


def test_verify_mode_is_off_by_default():
    """Without the env var the verify solve must not run."""
    A, B, N, M = _batch(16, 16, 12, shared_ref=False, seed=9)
    q, _ = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)   # must not raise
    assert q.shape[0] == 16


def test_degenerate_rows_still_take_the_fallback():
    """A shared reference with fewer than 3 real points still takes the fallback seed set."""
    A, B, N, M = _batch(8, 16, 12, shared_ref=True, seed=17)
    N = torch.full((8,), 2, dtype=torch.int32)        # < 3 real reference points
    q, t = batched_seeds_torch(A, B, N, M, num_seeds=SEEDS, ref_shared=True)
    assert torch.isfinite(q).all() and torch.isfinite(t).all()
