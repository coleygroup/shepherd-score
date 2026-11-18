"""Waiting on worker processes without hanging when one dies (shared by ``cpu_pool`` and
``multi_gpu``). Imports nothing heavy, so it is safe in fork and spawn workers alike.

Each worker replies on its own one-way pipe. :func:`recv_any` blocks on those pipes and on the
workers' process sentinels together, so it wakes as soon as a reply arrives or a worker exits.
:func:`collect` runs one job across a pool on top of it, and :func:`shutdown` stops a pool.
"""
from __future__ import annotations

import time
from multiprocessing.connection import wait


def result_pipe(ctx):
    """``(read_end, write_end)`` of a one-way pipe for one worker's replies."""
    return ctx.Pipe(duplex=False)


def start_worker(ctx, target, args, *, task_queue=True, **process_kw):
    """Start ``target(*args, task_queue, result_write_end)`` in a new process, or
    ``target(*args, result_write_end)`` without ``task_queue``. Returns
    ``(task_queue or None, result_read_end, process)``; a failed start leaves nothing open."""
    q = ctx.Queue() if task_queue else None
    read, write = result_pipe(ctx)
    try:
        p = ctx.Process(target=target, args=(*args, *([] if q is None else [q]), write), **process_kw)
        p.start()
    except BaseException:
        read.close()
        if q is not None:
            q.close()
        raise
    finally:
        write.close()                                # the worker holds the write end
    return q, read, p


def recv_any(pending: dict, *, what: str = "worker"):
    """The next reply from ``pending`` (``{key: (read_end, process)}``) as ``(key, message)``.

    Raises instead of blocking forever when a worker exits without replying. A reply sent just
    before the worker exited is still returned, since the pipe is checked before the sentinel.
    The caller removes ``key`` from ``pending`` once it has its reply.
    """
    by_conn = {conn: key for key, (conn, _) in pending.items()}
    by_sentinel = {proc.sentinel: key for key, (_, proc) in pending.items()}
    ready = wait(list(by_conn) + list(by_sentinel))
    for obj in ready:
        if obj in by_conn:
            try:
                return by_conn[obj], obj.recv()
            except EOFError:                         # closed with nothing sent: the worker died
                return _died(pending[by_conn[obj]][1], what)
    key = by_sentinel[ready[0]]
    conn, proc = pending[key]
    if conn.poll():
        return key, conn.recv()
    return _died(proc, what)


def collect(pending: dict, close, *, what: str, error_of, fail_fast: bool, submit=None) -> dict:
    """Run one job across a pool: call ``submit()`` to hand out the work, then return
    ``{key: reply}`` from every worker in ``pending`` (as :func:`recv_any` takes it).

    ``error_of(reply)`` is the traceback of a job that raised in a worker, or None for a result.
    Such errors are raised once every reply is in, leaving the pool usable; with ``fail_fast``
    the first one closes the pool and is raised at once, without waiting for the other workers.
    Any other way out (an interrupted submission or wait, a dead worker) calls ``close()``
    before re-raising, since replies still in flight would otherwise be read by the next job.
    """
    pending, replies, errs = dict(pending), {}, {}
    try:
        if submit is not None:
            submit()
        while pending:
            key, msg = recv_any(pending, what=what)
            del pending[key]
            tb = error_of(msg)
            if tb is None:
                replies[key] = msg
                continue
            errs[key] = tb
            if fail_fast:
                raise _failed(what, errs)
    except BaseException:
        close()
        raise
    if errs:
        raise _failed(what, errs)
    return replies


def shutdown(queues, procs, conns, *, grace: float) -> None:
    """Stop a pool of workers that each serve a task queue until a ``None`` sentinel: send the
    sentinels, :func:`stop` the workers and close the queues and result pipes."""
    for q in queues:
        try:
            q.put(None)
        except Exception:                            # noqa: BLE001 - a broken queue is moot
            pass
        q.cancel_join_thread()                       # never block exit flushing work to a dead worker
        q.close()
    stop(procs, grace=grace)
    for c in conns:
        c.close()


def stop(procs, *, grace: float) -> None:
    """Wait up to ``grace`` seconds in all for the workers to exit after their stop sentinels,
    then terminate (and finally kill) any that are left, waiting until each has exited."""
    _join_all(procs, grace)
    for p in procs:
        if p.is_alive():
            p.terminate()
    _join_all(procs, 5.0)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()


def _join_all(procs, timeout):
    deadline = time.monotonic() + timeout
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))


def _failed(what, errs):
    return RuntimeError("\n".join(f"{what} {k} failed:\n{tb}" for k, tb in sorted(errs.items())))


def _died(proc, what):
    proc.join(timeout=1.0)                           # settle the exit code
    raise RuntimeError(f"{what} {proc.name} died without replying (exit code {proc.exitcode})")
