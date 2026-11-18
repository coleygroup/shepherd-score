"""Data-parallel multi-GPU driver for batch alignment: one OS process per GPU.

Alignment is host-bound, so driving every GPU from one process serialises the host work behind
the GIL. Each worker here owns its shard end to end (build and align, data resident on its GPU)
with CPU threads capped to cores/ndev; the cap is mandatory, since uncapped workers each size an
all-core MKL/OMP pool and oversubscribe the machine. Only ``Molecule`` objects cross the process
boundary, never CUDA tensors; each worker rebuilds its ``MoleculePair`` objects on its own GPU.
:func:`align_multi_gpu` is a one-shot launcher whose spawn cost suits large batches;
:class:`MultiGPUAligner` keeps the workers and their shards resident for repeated screening.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

# Per-mode result attributes written in place by align_with_*, taken from the mode registry so
# they cannot drift; the process path supports exactly PROCESS_MODES.
from ._modes import (MODE_ATTRS as _MODE_ATTRS, PROCESS_MODES as _PROCESS_MODES,
                     LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES)
from ._procs import collect, shutdown, start_worker, stop
_TRANSFORM_ATTR = {m: _MODE_ATTRS[m][0] for m in _PROCESS_MODES}
_SCORE_ATTR = {m: _MODE_ATTRS[m][1] for m in _PROCESS_MODES}


def _cap_threads(threads):
    """Cap this worker's CPU intra-op threads at runtime. A forked worker inherits the parent's
    already-sized MKL/OMP pools (env vars set after import do not resize them), so the cap goes
    through torch for ATen and threadpoolctl, if present, for MKL/OpenBLAS/OMP."""
    if not threads:
        return
    import torch
    torch.set_num_threads(int(threads))
    try:                                     # best-effort: resizes MKL/BLAS/OMP live
        import threadpoolctl
        threadpoolctl.threadpool_limits(int(threads))
    except Exception:
        pass


def _worker(rank, mode, backend, do_center, threads, align_kwargs, shard_mols, out_conn):
    """One GPU's worker: pin to ``cuda:rank``, rebuild the pairs on that GPU, align the shard
    and return numpy results."""
    try:
        import time
        import torch
        from shepherd_score.container import MoleculePair, MoleculePairBatch

        _cap_threads(threads)
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        t0 = time.perf_counter()
        pairs = [MoleculePair(ref, fit, do_center=do_center, device=dev)
                 for (ref, fit) in shard_mols]
        torch.cuda.synchronize()
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        MoleculePairBatch(pairs)._align_mode(
            mode, backend=backend, **{"return_aligned": False, **align_kwargs})
        torch.cuda.synchronize()
        t_align = time.perf_counter() - t0

        sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
        scores = np.array([float(getattr(p, sc_attr)) for p in pairs], dtype=np.float64)
        transforms = np.stack([
            torch.as_tensor(getattr(p, tf_attr)).detach().cpu().numpy().astype(np.float64)
            for p in pairs])
        out_conn.send((rank, scores, transforms, t_build, t_align))
    except Exception:                            # noqa: BLE001 - relayed to parent
        import traceback
        out_conn.send((rank, "__ERR__", traceback.format_exc(), 0.0, 0.0))


def align_multi_gpu(pairs: Sequence,
                    mode: str,
                    *,
                    ndev: Optional[int] = None,
                    threads: Optional[int] = None,
                    backend: str = "triton",
                    do_center: bool = False,
                    write_back: bool = True,
                    return_timing: bool = False,
                    **align_kwargs):
    """Align ``pairs`` across ``ndev`` GPUs, one OS process per GPU.

    Parameters
    ----------
    pairs : list[MoleculePair]
        Pairs to align. Only their ``ref_molec`` / ``fit_molec`` (lightweight,
        picklable ``Molecule`` objects) cross the process boundary; each worker
        rebuilds the ``MoleculePair`` on its own GPU.
    mode : str
        One of ``accel._modes.PROCESS_MODES`` (legacy ``esp`` / ``esp_combo`` accepted).
    ndev : int, optional
        Number of GPUs/processes (default: all visible CUDA devices).
    threads : int, optional
        CPU intra-op threads per worker (default: cpu_cores // ndev). This cap is what
        keeps the workers from oversubscribing the cores.
    backend : str
        Alignment backend forwarded to ``align_with_*`` (default "triton").
    do_center : bool
        Forwarded to ``MoleculePair`` construction; keep it matching how the single-GPU
        pairs were built, else results differ.
    write_back : bool
        If True, write ``sim_aligned_*`` / ``transform_*`` back onto the input ``pairs``
        in order, matching the single-GPU API's in-place convention.
    return_timing : bool
        If True, also return a dict of per-rank build/align timings.
    **align_kwargs
        Forwarded verbatim to ``MoleculePairBatch.align_with_<mode>``.

    Returns
    -------
    (scores, transforms) : (np.ndarray (K,), np.ndarray (K, 4, 4))
        In input order. If ``return_timing``, a timing dict follows them.
    """
    import torch
    import torch.multiprocessing as mp

    mode = _LEGACY_MODE_ALIASES.get(mode, mode)    # accept legacy esp / esp_combo
    if mode not in _SCORE_ATTR:
        raise ValueError(f"mode must be one of {list(_SCORE_ATTR)}, got {mode!r}")
    pairs = list(pairs)
    K = len(pairs)
    if K == 0:
        empty = (np.zeros((0,)), np.zeros((0, 4, 4)))
        return (*empty, {}) if return_timing else empty

    ndev = ndev or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    ndev = max(1, min(ndev, K))
    if threads is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or ndev
        threads = max(1, cores // ndev)

    # Only picklable Molecule objects cross the process boundary, never CUDA tensors.
    specs = [(p.ref_molec, p.fit_molec) for p in pairs]
    # Contiguous balanced shards plus their original indices, for the in-order gather.
    bounds = np.linspace(0, K, ndev + 1).astype(int)
    shard_idx = [list(range(bounds[r], bounds[r + 1])) for r in range(ndev)]

    # Spawned children read the MKL/OMP env at their own numpy/torch import, so it must be set
    # in the parent before spawning; restored afterwards.
    _saved = {k: os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in _saved:
        os.environ[k] = str(threads)
    procs, pending = [], {}                  # pending: rank -> (read end, worker)
    try:
        ctx = mp.get_context("spawn")
        for r in range(ndev):
            shard = [specs[i] for i in shard_idx[r]]
            _, read, p = start_worker(ctx, _worker, (r, mode, backend, do_center, threads,
                                                     dict(align_kwargs), shard), task_queue=False)
            procs.append(p); pending[r] = (read, p)
        # Each worker runs one job and exits, so the first failure stops the others at once.
        replies = collect(pending, lambda: stop(procs, grace=0.0), what="multi-GPU worker",
                          fail_fast=True,
                          error_of=lambda m: m[2] if isinstance(m[1], str) and m[1] == "__ERR__" else None)
        results = {r: m[1:] for r, m in replies.items()}
        stop(procs, grace=30.0)              # every worker has replied and is exiting
    finally:
        stop(procs, grace=0.0)               # the rest of a failed start; else already exited
        for read, _ in pending.values():
            read.close()
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    scores = np.empty(K, dtype=np.float64)
    transforms = np.empty((K, 4, 4), dtype=np.float64)
    for r in range(ndev):
        sc, tf, _, _ = results[r]
        for j, i in enumerate(shard_idx[r]):
            scores[i] = sc[j]
            transforms[i] = tf[j]

    if write_back:
        sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
        for i, p in enumerate(pairs):
            setattr(p, sc_attr, float(scores[i]))
            setattr(p, tf_attr, torch.as_tensor(transforms[i], dtype=torch.float32))

    if return_timing:
        timing = {"ndev": ndev, "threads": threads, "K": K,
                  "build_max": max(results[r][2] for r in results),
                  "align_max": max(results[r][3] for r in results),
                  "per_rank_align": {r: results[r][3] for r in results}}
        return scores, transforms, timing
    return scores, transforms

# Persistent pool: build+retain shards once, align resident data many times.

def _pool_worker(rank, threads, do_center, shard_mols, in_q, out_conn):
    """Persistent worker: build and retain this GPU's shard once, then align it in place on every
    job. Only ``(mode, backend, kwargs)`` come in and ``(scores, transforms)`` go out per call;
    the bulk molecule data never recrosses the process boundary."""
    try:
        import time
        import numpy as _np
        import torch
        from shepherd_score.container import MoleculePair, MoleculePairBatch

        _cap_threads(threads)
        torch.cuda.set_device(rank)          # creates this worker's CUDA context
        dev = torch.device("cuda", rank)
        pairs = [MoleculePair(ref, fit, do_center=do_center, device=dev)
                 for (ref, fit) in shard_mols]
        batch = MoleculePairBatch(pairs)
        torch.cuda.synchronize()
        out_conn.send(("READY", rank))
    except Exception:                            # noqa: BLE001 - startup failed; relayed
        import traceback
        out_conn.send(("ERR", rank, traceback.format_exc()))
        return

    while True:
        job = in_q.get()
        if job is None:
            break
        try:                                     # a failed job is reported; the worker stays up
            mode, backend, kwargs = job
            t0 = time.perf_counter()
            batch._align_mode(
                mode, backend=backend, **{"return_aligned": False, **kwargs})
            torch.cuda.synchronize()
            t_align = time.perf_counter() - t0
            sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
            scores = _np.array([float(getattr(p, sc_attr)) for p in pairs], dtype=_np.float64)
            transforms = _np.stack([
                torch.as_tensor(getattr(p, tf_attr)).detach().cpu().numpy().astype(_np.float64)
                for p in pairs])
            out_conn.send(("RES", rank, scores, transforms, t_align))
        except Exception:                        # noqa: BLE001 - relayed to parent
            import traceback
            out_conn.send(("ERR", rank, traceback.format_exc()))


class MultiGPUAligner:
    """Persistent one-process-per-GPU pool that builds and retains its shard, so repeated
    :meth:`align` calls run on resident data with no per-call re-ship or rebuild. Use it for
    repeated screening (several modes over the same pairs, or a resident library against many
    queries); for a single align of a huge batch use the one-shot :func:`align_multi_gpu`.

    By default the pool forks its workers, so they inherit the parent's imported modules and the
    molecule data copy-on-write. ``fork`` is only CUDA-safe if the parent has not yet initialised
    CUDA (nor imported Open3D), so build ``pairs`` on CPU and create the pool before any GPU
    work; otherwise the pool falls back to the slower ``spawn`` start method.

    Usage::

        pairs = [MoleculePair(a, b, device="cpu") for a, b in mols]   # CPU build
        with MultiGPUAligner(pairs) as pool:                          # fork: fast
            scores, transforms = pool.align("vol", no_H=True, alpha=0.81)
            esp_scores, _      = pool.align("surf_esp", alpha=0.81, lam=0.3, num_repeats=16)
    """

    def __init__(self, pairs, *, ndev=None, threads=None, do_center=False,
                 start_method=None):
        import sys
        import warnings
        import torch
        import torch.multiprocessing as mp

        # fork lets workers inherit the imported stack and data copy-on-write, but is only safe
        # if the parent has neither initialised CUDA nor imported Open3D (its import alone
        # poisons a later fork+CUDA). Detect either and fall back to spawn.
        cuda_live = torch.cuda.is_initialized()
        o3d_live = "open3d" in sys.modules
        avail = mp.get_all_start_methods()
        unsafe = cuda_live or o3d_live
        if start_method is None:
            start_method = "fork" if ("fork" in avail and not unsafe) else "spawn"
            if unsafe and "fork" in avail:
                warnings.warn(
                    "MultiGPUAligner: "
                    + ("CUDA already initialized" if cuda_live else "Open3D already imported")
                    + " in the parent -> using slower 'spawn' startup. To get fast 'fork'"
                    " startup, create the pool before any GPU work / Open3D surface build"
                    " (e.g. from disk-cached molecules).")
        elif start_method == "fork" and unsafe:
            raise RuntimeError(
                "start_method='fork' needs the parent to have NOT initialized CUDA nor "
                "imported Open3D (both poison fork+CUDA). Create the pool earlier.")
        self.start_method = start_method

        pairs = list(pairs)
        self._K = len(pairs)
        if self._K == 0:
            raise ValueError("MultiGPUAligner needs at least one pair")
        ndev = ndev or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        ndev = max(1, min(ndev, self._K))
        if threads is None:
            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:
                cores = os.cpu_count() or ndev
            threads = max(1, cores // ndev)
        self.ndev, self.threads = ndev, threads

        specs = [(p.ref_molec, p.fit_molec) for p in pairs]
        bounds = np.linspace(0, self._K, ndev + 1).astype(int)
        self._shard_idx = [list(range(bounds[r], bounds[r + 1])) for r in range(ndev)]

        # spawn/forkserver children read the OMP/MKL env at their own import, so cap it in the
        # parent first; fork inherits live pools, which _cap_threads resizes in the worker.
        _saved = {}
        if start_method != "fork":
            _saved = {k: os.environ.get(k) for k in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
            for k in _saved:
                os.environ[k] = str(threads)
        # Closable from the first started worker on, so a startup failure stops the others.
        self._in_qs, self._out_conns, self._procs, self._closed = [], [], [], False
        try:
            ctx = mp.get_context(start_method)
            for r in range(ndev):
                shard = [specs[i] for i in self._shard_idx[r]]
                in_q, read, p = start_worker(ctx, _pool_worker, (r, threads, do_center, shard))
                self._in_qs.append(in_q); self._out_conns.append(read); self._procs.append(p)
            collect(self._pending(), self.close, what="MultiGPUAligner worker", fail_fast=True,
                    error_of=lambda m: f"failed to start:\n{m[2]}" if m[0] == "ERR" else None)
        except BaseException:
            self.close()
            raise
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def align(self, mode, *, backend="triton", return_timing=False, **align_kwargs):
        """Align the resident pairs with ``mode``; returns ``(scores, transforms)`` in the
        original input order. Only parameters go in and results come out. A job that raises in
        any worker, a worker that dies, or an interrupted call closes the pool and raises: a
        CUDA error can leave a worker alive with a broken context, so it is not reused."""
        if self._closed:
            raise RuntimeError("MultiGPUAligner is closed")
        mode = _LEGACY_MODE_ALIASES.get(mode, mode)    # accept legacy esp / esp_combo
        if mode not in _SCORE_ATTR:
            raise ValueError(f"mode must be one of {list(_SCORE_ATTR)}, got {mode!r}")

        def submit():
            for q in self._in_qs:
                q.put((mode, backend, dict(align_kwargs)))

        replies = collect(self._pending(), self.close, submit=submit, what="MultiGPUAligner worker",
                          fail_fast=True, error_of=lambda m: m[2] if m[0] == "ERR" else None)
        results = {r: m[2:] for r, m in replies.items()}
        scores = np.empty(self._K, dtype=np.float64)
        transforms = np.empty((self._K, 4, 4), dtype=np.float64)
        for r in range(self.ndev):
            sc, tf, _ = results[r]
            for j, i in enumerate(self._shard_idx[r]):
                scores[i] = sc[j]
                transforms[i] = tf[j]
        if return_timing:
            return scores, transforms, {"align_max": max(results[r][2] for r in results)}
        return scores, transforms

    def _pending(self):
        """Every worker, keyed by rank, as ``recv_any`` takes them."""
        return {r: (c, p) for r, (c, p) in enumerate(zip(self._out_conns, self._procs))}

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        shutdown(self._in_qs, self._procs, self._out_conns, grace=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
