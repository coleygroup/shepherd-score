"""Opt-in fine-loop schedule recorder (off by default).

Every fine loop (the eager torch loop in ``drivers/*.py`` and the CUDA-graph replay loop
in ``drivers/_graphed.py``) calls :func:`record` once as it exits, reporting how many
value+grad evaluations ran against the configured budget and which path ran, so early
stops and graph/eager fallbacks are visible where the returned scores alone are not.
Usage: ``reset()`` (clears and enables), run, ``summary()``, ``disable()``. When off,
:func:`record` reads one module-global boolean and returns. Process-local and not
thread-safe: the counters are plain module globals.
"""
from __future__ import annotations

_enabled = False
_calls = 0
_graphed = 0
_steps_total = 0
_steps_min = 0
_steps_max = 0
_early = 0
_configured: set = set()
_configured_total = 0        # sum of the budget over calls, for a call-weighted mean


def reset() -> None:
    """Clear the counters and enable recording."""
    global _enabled, _calls, _graphed, _steps_total, _steps_min, _steps_max, _early
    global _configured, _configured_total
    _enabled = True
    _configured_total = 0
    _calls = 0
    _graphed = 0
    _steps_total = 0
    _steps_min = 0
    _steps_max = 0
    _early = 0
    _configured = set()


def disable() -> None:
    """Stop recording. Counters already collected stay readable via :func:`summary`."""
    global _enabled
    _enabled = False


def record(steps_run: int, steps_configured: int, early_stopped: bool,
           graphed: bool = False) -> None:
    """Register one fine-loop invocation; no-op unless :func:`reset` enabled recording.

    ``steps_run`` is the number of value+grad evaluations executed, ``steps_configured``
    the budget (``steps_fine``), ``early_stopped`` whether the loop left before exhausting
    it, and ``graphed`` whether the CUDA-graph replay loop ran rather than the eager one.
    """
    if not _enabled:
        return
    global _calls, _graphed, _steps_total, _steps_min, _steps_max, _early, _configured_total
    steps_run = int(steps_run)
    if _calls == 0 or steps_run < _steps_min:
        _steps_min = steps_run
    if steps_run > _steps_max:
        _steps_max = steps_run
    _calls += 1
    if graphed:
        _graphed += 1
    _steps_total += steps_run
    if early_stopped:
        _early += 1
    _configured.add(int(steps_configured))
    _configured_total += int(steps_configured)


def summary() -> dict:
    """Aggregate of everything recorded since the last :func:`reset`; ``{}`` if nothing was.

    ``graphed`` counts the calls that ran the CUDA-graph replay loop (the eager count is
    ``calls - graphed``). ``steps_configured`` is the budget when every call shared one,
    otherwise the call-weighted mean, with ``steps_configured_mixed`` set True.
    """
    if not _calls:
        return {}
    mixed = len(_configured) > 1
    if not mixed:
        cfg = next(iter(_configured))
    else:
        # Call-weighted mean, so an effort fraction taken against it stays meaningful.
        cfg = _configured_total / _calls
    return {
        "calls": _calls,
        "graphed": _graphed,
        "steps_min": _steps_min,
        "steps_max": _steps_max,
        "steps_mean": _steps_total / _calls,
        "steps_configured": cfg,
        "steps_configured_mixed": mixed,
        "early_stop_frac": _early / _calls,
    }
