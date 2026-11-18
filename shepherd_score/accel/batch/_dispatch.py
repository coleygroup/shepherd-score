# shepherd_score/accel/batch/_dispatch.py
"""The per-mode tensor spec used by the CPU process pool.

Multi-GPU alignment is the explicit persistent pool
:class:`shepherd_score.accel.multi_gpu.MultiGPUAligner` (one process per GPU); the batched
aligners always run on the pairs' own device.
"""
from __future__ import annotations

import torch

from .._modes import SPECS
from ..channels import CHANNELS


# --- per-mode tensor spec (consumed by the CPU process pool, cpu_pool.py) -----
# Each mode declares how to pull its per-pair inputs off the pair as picklable numpy arrays
# (``extract``: ``(side, reader)`` pairs, side in ref_molec / fit_molec / pair), how to rebuild
# the device tensors in a worker (``tensors``, positional with ``extract``) and which
# attributes to read back (``out``). Derived from each mode's channels; ``_modes.PROCESS_MODES``
# is the same set.
_DTYPES = {"float32": torch.float32, "int64": torch.int64}


def _spec_entry(spec):
    extract, tensors = [], []
    for name in spec.all_channels():
        ch = CHANNELS[name]
        dt = _DTYPES[ch.dtype]
        if ch.is_pair:
            extract.append(("pair", ch.read))
            tensors.append((ch.ref_attr, dt))
        else:
            extract.append(("ref_molec", ch.read))
            tensors.append((ch.ref_attr, dt))
            extract.append(("fit_molec", ch.read))
            tensors.append((ch.fit_attr, dt))
    return {"extract": extract, "tensors": tensors, "out": spec.attrs}


_MODE_SPEC = {m: _spec_entry(s) for m, s in SPECS.items()}


class _ProcStandIn:
    """Minimal MoleculePair stand-in for a worker process: carries only the cached device
    tensors the batched aligner reads (no RDKit / Molecule), and receives the results."""
    def __init__(self, device):
        self.device = device
