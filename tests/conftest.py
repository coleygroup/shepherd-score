"""Cap the numba thread pool before any test imports numba.

The test batches are a handful of pairs, so a ``parallel=True`` kernel over every core spends its
time waking threads rather than computing; on a busy many-core node a call over every core is
~250x slower than over a few. The default is the CPUs this process may run on (its affinity
mask, as numba itself counts them), clamped to 2..8: small CI runners and laptops are not
oversubscribed, and the floor of 2 keeps the thread-cap tests in ``test_cpu_threads.py`` running
on a 1-CPU machine (2 threads on 1 CPU just take turns). On an idle 32-core node the full suite
took 2:22 at 8 threads and 9:06 at 2. Export ``NUMBA_NUM_THREADS`` to override.

Tests marked ``cuda`` are skipped when no CUDA device is visible, so the marker alone gates a
GPU test and ``pytest -m cuda`` selects every one of them.
"""
import os

import pytest


def _default_threads(cap: int = 8, floor: int = 2) -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                       # not on Linux
        n = os.cpu_count() or 1
    return max(floor, min(cap, n))


os.environ.setdefault("NUMBA_NUM_THREADS", str(_default_threads()))


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda") is None:
        return
    try:
        import torch
    except ImportError:
        pytest.skip("CUDA test: PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
