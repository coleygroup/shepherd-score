"""Scoped CPU thread caps for the numba kernels and torch (``cpu_threads=`` on
``MoleculePairBatch`` and ``screen``).

numba sizes its thread pool once, when it is first imported: ``NUMBA_NUM_THREADS`` if set,
otherwise the CPUs the process may run on (its affinity mask, e.g. ``taskset`` or a SLURM
allocation). ``numba.set_num_threads`` can then move the active count anywhere from 1 to that
pool size, down or up, any number of times. A cap is therefore applied per call and the
previous counts restored afterwards; asking for more threads than the pool holds is an error
rather than a silent clamp.

On a machine with two hardware threads per core the logical CPU count is twice the core count,
and the arithmetic-bound kernels gain little from the second thread of a core.
"""
import numbers
from contextlib import contextmanager
from typing import Optional


def check_cpu_threads(n: Optional[int]) -> Optional[int]:
    """Validate a ``cpu_threads`` value (``None`` = leave the thread counts alone)."""
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError(f"cpu_threads must be a positive integer or None, got {n!r}")
    return int(n)


@contextmanager
def cpu_threads(n: Optional[int]):
    """Cap the numba kernels and torch's intra-op pool at ``n`` threads inside the block, then
    restore both counts. ``None`` is a no-op. torch is only ever lowered here: the CPU batch
    path pins it to one thread on its own so it does not compete with the numba kernels."""
    n = check_cpu_threads(n)
    if n is None:
        yield
        return
    import torch
    try:
        import numba
    except ImportError:
        numba = None
    prev_nb = None
    if numba is not None:
        pool = int(numba.config.NUMBA_NUM_THREADS)
        if n > pool:
            raise ValueError(
                f"cpu_threads={n} exceeds numba's thread pool of {pool}. The pool is sized when "
                "numba is first imported, from NUMBA_NUM_THREADS or the process's CPU affinity; "
                "set NUMBA_NUM_THREADS (or launch under taskset / a larger allocation) before "
                "importing shepherd_score.")
        prev_nb = numba.get_num_threads()
        numba.set_num_threads(n)
    prev_t = torch.get_num_threads()
    if prev_t > n:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        if prev_nb is not None:
            numba.set_num_threads(prev_nb)
        if torch.get_num_threads() != prev_t:
            torch.set_num_threads(prev_t)
