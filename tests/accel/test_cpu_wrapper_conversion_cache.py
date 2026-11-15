"""``kernels/cpu.py::_np_cached`` converts invariant tensors to numpy once per tensor object.

A hit must return exactly what a fresh conversion would, and any change to the tensor
(in-place write, new object, other dtype) must convert again.
"""
import numpy as np
import pytest
import torch

from shepherd_score.accel.kernels import cpu


def test_hit_returns_the_same_copy_and_equals_a_fresh_conversion():
    x = torch.randint(0, 9, (5, 7), dtype=torch.int32)
    a = cpu._np_cached(x, np.int64)
    b = cpu._np_cached(x, np.int64)
    assert a is b, "second call must be served from the stash"
    assert a.dtype == np.int64 and np.array_equal(a, x.numpy().astype(np.int64))
    assert not np.shares_memory(a, x.numpy()), "the stash must be a copy, not a view"


def test_in_place_write_invalidates():
    x = torch.randint(0, 9, (4, 6), dtype=torch.int32)
    a = cpu._np_cached(x, np.int64)
    x[0, 0] += 1                                       # bumps the version counter
    b = cpu._np_cached(x, np.int64)
    assert b is not a and b[0, 0] == a[0, 0] + 1 and np.array_equal(b, x.numpy().astype(np.int64))


def test_new_object_over_the_same_storage_is_not_confused():
    base = torch.randint(0, 9, (3, 8), dtype=torch.int32)
    a = cpu._np_cached(base, np.int64)
    view = base[1:]                                    # different object, overlapping storage
    b = cpu._np_cached(view, np.int64)
    assert b.shape == (2, 8) and np.array_equal(b, base[1:].numpy().astype(np.int64))
    assert np.array_equal(cpu._np_cached(base, np.int64), a)


def test_dtype_is_part_of_the_key():
    x = torch.tensor([1.5, 2.5], dtype=torch.float32)
    f = cpu._np_cached(x, np.float64)
    i = cpu._np_cached(x, np.int64)
    assert f.dtype == np.float64 and i.dtype == np.int64 and f is not i


def test_pharm_wrapper_repeats_are_identical_and_track_mutation():
    """Repeated calls give identical results and an in-place type mutation reaches the kernel."""
    torch.manual_seed(0)
    P, N, M = 6, 5, 4
    q = torch.nn.functional.normalize(torch.randn(P, 4), dim=1)
    t = torch.randn(P, 3) * 0.1
    A, B = torch.randn(P, N, 3), torch.randn(P, M, 3)
    VA = torch.nn.functional.normalize(torch.randn(P, N, 3), dim=2)
    VB = torch.nn.functional.normalize(torch.randn(P, M, 3), dim=2)
    TA = torch.randint(0, 3, (P, N), dtype=torch.int32)
    TB = torch.randint(0, 3, (P, M), dtype=torch.int32)
    al = torch.full((3,), 0.81)
    Ks = torch.ones(3)
    cats = torch.arange(3, dtype=torch.int32)
    Nr = torch.full((P,), N, dtype=torch.int32)
    Mr = torch.full((P,), M, dtype=torch.int32)
    kw = dict(N_real=Nr, M_real=Mr, NEED_GRAD=True)

    first = cpu.pharm_grad_dq_se3_batch(q, t, TA, TB, A, B, VA, VB, al, Ks, cats, **kw)
    again = cpu.pharm_grad_dq_se3_batch(q, t, TA, TB, A, B, VA, VB, al, Ks, cats, **kw)
    for x, y in zip(first, again):
        assert torch.equal(x, y)

    TB[:, 0] = (TB[:, 0] + 1) % 3                     # in place, same object
    fresh = cpu.pharm_grad_dq_se3_batch(q, t, TA, TB.clone(), A, B, VA, VB, al, Ks, cats, **kw)
    mutated = cpu.pharm_grad_dq_se3_batch(q, t, TA, TB, A, B, VA, VB, al, Ks, cats, **kw)
    for x, y in zip(fresh, mutated):
        assert torch.equal(x, y)
    assert not torch.equal(first[0], mutated[0]), "the mutation must be visible to the kernel"
