"""Worker pools must not hang on a dead worker, leak workers on a failed start or an
interrupted call, or hand one call's replies to the next."""
import multiprocessing as mp
import signal
import time
import types

import pytest

torch = pytest.importorskip("torch")

from shepherd_score.accel._procs import collect, recv_any, result_pipe, stop   # noqa: E402


def _reply_then_exit(conn):
    conn.send("done")


def _exit_silently(conn):
    pass


def _fail_at_once(conn):
    conn.send("ERR")


def _reply_late(conn):
    time.sleep(60)
    conn.send("done")


def _ignore_sigterm(conn):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    conn.send("up")
    time.sleep(60)


def _start(target, name):
    ctx = mp.get_context("spawn")
    read, write = result_pipe(ctx)
    p = ctx.Process(target=target, args=(write,), name=name, daemon=True)
    p.start()
    write.close()
    return read, p


def test_recv_any_returns_a_reply_sent_just_before_exit():
    read, p = _start(_reply_then_exit, "replying-worker")
    p.join()
    assert recv_any({0: (read, p)}) == (0, "done")


def test_recv_any_raises_at_once_when_the_worker_dies_silently():
    read, p = _start(_exit_silently, "silent-worker")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="silent-worker died without replying"):
        recv_any({0: (read, p)}, what="test worker")
    assert time.monotonic() - t0 < 5


def test_collect_fails_fast_without_waiting_for_the_other_workers():
    workers = {0: _start(_fail_at_once, "failing-worker"), 1: _start(_reply_late, "slow-worker")}
    procs = [p for _, p in workers.values()]
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="test worker 0 failed"):
        collect(workers, lambda: stop(procs, grace=0.0), what="test worker", fail_fast=True,
                error_of=lambda m: "boom" if m == "ERR" else None)
    assert time.monotonic() - t0 < 20
    assert all(not p.is_alive() for p in procs)


def test_stop_kills_a_worker_that_ignores_sigterm():
    read, p = _start(_ignore_sigterm, "stubborn-worker")
    assert read.recv() == "up"
    stop([p], grace=0.0)
    assert not p.is_alive() and p.exitcode == -signal.SIGKILL


def test_cpu_pool_job_errors_leave_the_pool_usable():
    from shepherd_score.accel import cpu_pool
    pool = cpu_pool.get_pool(2)
    try:
        for _ in range(2):                    # the second call sees only its own replies
            with pytest.raises(RuntimeError, match="CPU align pool worker 0 failed"):
                pool.align("no-such-mode", [(), ()], {})
        assert not pool._closed and all(p.is_alive() for p in pool._procs)
    finally:
        cpu_pool._shutdown_pool()


def test_cpu_pool_closes_when_submission_is_interrupted(monkeypatch):
    from shepherd_score.accel import cpu_pool
    pool = cpu_pool.get_pool(2)
    try:
        q, real_put = pool._task_qs[1], pool._task_qs[1].put

        def interrupted(*args):               # Ctrl-C after worker 0 already has its task
            monkeypatch.setattr(q, "put", real_put)
            raise KeyboardInterrupt
        monkeypatch.setattr(q, "put", interrupted)
        with pytest.raises(KeyboardInterrupt):
            pool.align("vol", [(), ()], {})
        assert pool._closed and all(not p.is_alive() for p in pool._procs)
    finally:
        cpu_pool._shutdown_pool()


def test_cpu_pool_stops_its_workers_when_startup_fails(monkeypatch):
    from shepherd_score.accel import cpu_pool
    real, started = cpu_pool.start_worker, []

    def fail_second(*args, **kw):
        if started:
            raise OSError("no more processes")
        started.append(real(*args, **kw))
        return started[-1]
    monkeypatch.setattr(cpu_pool, "start_worker", fail_second)
    with pytest.raises(OSError, match="no more processes"):
        cpu_pool.CpuAlignPool(3)
    assert not started[0][2].is_alive()


def test_cpu_pool_closes_and_rebuilds_after_a_worker_dies():
    from shepherd_score.accel import cpu_pool
    pool = cpu_pool.get_pool(2)
    try:
        pool._procs[1].kill()
        pool._procs[1].join()
        with pytest.raises(RuntimeError, match="died without replying"):
            pool.align("vol", [(), ()], {})
        assert pool._closed
        assert all(not p.is_alive() for p in pool._procs)
        fresh = cpu_pool.get_pool(2)
        assert fresh is not pool and not fresh._closed
    finally:
        cpu_pool._shutdown_pool()


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.cuda
def test_multi_gpu_aligner_stops_its_workers_when_startup_fails():
    _require_cuda()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices: one worker starts, the other fails")
    from shepherd_score.accel.multi_gpu import MultiGPUAligner
    from .._support.legacy_fixture import make_pair
    good = make_pair(0, 1)
    bad = types.SimpleNamespace(ref_molec=None, fit_molec=None)    # MoleculePair(None, None) raises
    before = set(mp.active_children())
    with pytest.raises(RuntimeError, match="failed to start"):
        MultiGPUAligner([good, bad], ndev=2, start_method="spawn")   # rank 0 is ready, rank 1 fails
    leaked = [p for p in mp.active_children() if p not in before]
    assert leaked == [], f"workers left running: {leaked}"


@pytest.mark.cuda
def test_multi_gpu_aligner_closes_after_a_failed_job():
    """A failed job may have left a broken CUDA context, so the pool is not reused."""
    _require_cuda()
    pytest.importorskip("triton")
    from shepherd_score.accel.multi_gpu import MultiGPUAligner
    from .._support.legacy_fixture import make_pair
    pairs = [make_pair(0, 1), make_pair(2, 3)]
    with MultiGPUAligner(pairs, ndev=1, start_method="spawn") as pool:
        with pytest.raises(RuntimeError, match="MultiGPUAligner worker 0 failed"):
            pool.align("vol", not_a_keyword=1)
        assert pool._closed and all(not p.is_alive() for p in pool._procs)
        with pytest.raises(RuntimeError, match="is closed"):
            pool.align("vol")
