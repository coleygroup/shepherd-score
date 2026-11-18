"""Shard-parallel CPU screening: one strided library shard per forked, single-threaded worker.

Each ``align`` call has a single-threaded prologue, so in-call thread parallelism is
Amdahl-capped; whole independent aligns in separate worker processes are not. The featurized
library is stashed as a module global before the pool forks (inherited copy-on-write), and the
pool persists across calls for the same library object, keyed by its identity and length;
molecules mutated in place after the first call are not seen by the workers. Workers pin
themselves to one physical core each. Never run numba in this process before the first call for
a library: forking a process with a live GNU-OpenMP pool aborts the child.
"""
from __future__ import annotations

import atexit
import os
from multiprocessing import get_context

# _modes is pure data (no torch), so importing it here stays fork-safe.
from ._modes import MODE_ATTRS as _MODE_ATTRS

# Parent-set global inherited by forked workers (never pickled).
_LIBRARY: list = []
_POOL = None                       # {"key": (id(library), len(library), n_workers), "pool": Pool}

_SCORE_ATTR = {m: score_attr for m, (_tf, score_attr) in _MODE_ATTRS.items()}


def _shard(task):
    """Align ``query`` against ``_LIBRARY[index_range]`` in a single-threaded forked worker."""
    query, index_range, mode, kw = task
    import torch
    import numba
    torch.set_num_threads(1)          # one thread per worker
    numba.set_num_threads(1)          # active-count mask; the pool size is capped by the parent env
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    attr = _SCORE_ATTR[mode]
    pairs = [MoleculePair(query, _LIBRARY[i], do_center=True) for i in index_range]
    MoleculePairBatch(pairs)._align_mode(mode, backend="numba", **{"return_aligned": False, **kw})
    return [(i, float(getattr(p, attr))) for i, p in zip(index_range, pairs)]


def _chunks(n, k):
    """k strided index ranges covering range(n): worker w gets w, w+k, w+2k, ... so that runs of
    same-compound conformers are spread across the workers."""
    return [range(w, n, k) for w in range(min(k, n))]


def _physical_cores(allowed):
    """One CPU per physical core among ``allowed`` (the lowest hardware-thread sibling), in CPU
    order; ``allowed`` itself where the topology is not exposed."""
    try:
        seen, cores = set(), []
        for c in sorted(allowed):
            if c in seen:
                continue
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                spec = f.read().strip()
            sib = set()
            for part in spec.split(","):
                a, _, b = part.partition("-")
                sib.update(range(int(a), int(b or a) + 1))
            sib &= set(allowed)
            seen |= sib
            cores.append(min(sib))
        return cores
    except (OSError, ValueError):
        return sorted(allowed)


def _pin_worker(cores):
    """Pool initializer: pin this worker to ``cores[worker number]`` (wrapping when there are
    more workers than cores). Worker numbers are contiguous within one pool."""
    if not cores:
        return
    from multiprocessing import current_process
    ident = getattr(current_process(), "_identity", None) or (1,)
    try:
        os.sched_setaffinity(0, {cores[(ident[0] - 1) % len(cores)]})
    except (AttributeError, OSError):      # no sched_setaffinity, or a CPU we may not use
        pass


def screen_parallel_close():
    """Shut down the persistent worker pool, if any, and drop the library reference."""
    global _POOL, _LIBRARY
    held, _POOL = _POOL, None
    if held is not None:
        held["pool"].close()
        held["pool"].join()
    _LIBRARY = []


atexit.register(screen_parallel_close)


def _pool_for(library, n_workers):
    """The pool for this library and worker count: reused when it exists, forked otherwise."""
    global _LIBRARY, _POOL
    key = (id(library), len(library), n_workers)
    if _POOL is not None and _POOL["key"] == key:
        return _POOL["pool"]
    screen_parallel_close()
    _LIBRARY = library
    # Cap each worker's thread pools to one thread before forking: numba sizes its pool from
    # NUMBA_NUM_THREADS at import, and a child's set_num_threads(1) only masks it. This works only
    # if this process has not yet imported numba; export NUMBA_NUM_THREADS=1 for a guaranteed cap.
    _cap = {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"}
    _saved = {k: os.environ.get(k) for k in _cap}
    os.environ.update(_cap)
    try:
        # Always fork (even for one worker) so the parent stays numba-clean and libgomp-safe;
        # each worker pins itself to its own physical core.
        try:
            cores = _physical_cores(os.sched_getaffinity(0))
        except AttributeError:                           # no affinity API on this platform
            cores = []
        pool = get_context("fork").Pool(n_workers, initializer=_pin_worker, initargs=(cores,))
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _POOL = {"key": key, "pool": pool}
    return pool


def screen_parallel(query, library, mode, n_workers=None, **align_kwargs):
    """Screen ``query`` against ``library`` (a list of featurized ``Molecule`` objects) with the
    numba CPU backend, sharded across ``n_workers`` forked processes (default: all CPUs).
    Returns aligned similarity scores in library order. ``align_kwargs`` are the mode's kwargs
    (e.g. ``alpha=0.81`` for surf, ``lam=0.3`` for vol_esp). Always forks, even for one worker,
    so this process never runs numba itself; the pool is kept for later calls against the same
    library (see the module docstring)."""
    if mode not in _SCORE_ATTR:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_SCORE_ATTR)}")
    n = len(library)
    n_workers = max(1, min(n_workers or os.cpu_count() or 1, n))
    chunks = _chunks(n, n_workers)
    pool = _pool_for(library, len(chunks))
    shards = pool.map(_shard, [(query, rng, mode, align_kwargs) for rng in chunks], chunksize=1)
    scores = [None] * n
    for shard in shards:
        for i, s in shard:
            scores[i] = s
    return scores
