"""Device-driven kernel dispatch for the batched coarse-to-fine aligners.

Each value+gradient kernel has two implementations with identical signatures: a Triton
kernel (:mod:`.shape_triton`, :mod:`.esp_triton`, :mod:`.pharm_triton`,
:mod:`.avoid_triton`) for CUDA tensors and a numba kernel (:mod:`.cpu`) for CPU tensors.
Each name exported here routes a call by the device of its tensor arguments. The choice
is per call, never frozen at import, so one process can run both paths (``backend="numba"``
on a machine where Triton imports). The Triton modules are imported lazily, on the first
CUDA dispatch, so a CPU-only box never touches them.
"""
from __future__ import annotations

import torch


def has_triton() -> bool:
    """True if the Triton package is importable (i.e. the GPU kernels can be loaded)."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


# Lazily-imported, cached source modules, keyed by a short tag.
_MODS: dict = {}


def _mod(tag: str):
    m = _MODS.get(tag)
    if m is None:
        if tag == "cpu":
            from . import cpu as m
        elif tag == "shape":
            from . import shape_triton as m
        elif tag == "esp":
            from . import esp_triton as m
        elif tag == "pharm":
            from . import pharm_triton as m
        elif tag == "avoid":
            from . import avoid_triton as m
        else:  # pragma: no cover - defensive
            raise KeyError(tag)
        _MODS[tag] = m
    return m


def _args_on_cuda(args, kwargs) -> bool:
    """True if the first ``torch.Tensor`` among the call's arguments lives on CUDA."""
    for a in args:
        if torch.is_tensor(a):
            return a.is_cuda
    for a in kwargs.values():
        if torch.is_tensor(a):
            return a.is_cuda
    # No tensor argument found -> fall back to machine capability. The kernels below
    # all take tensors as their leading arguments, so this branch is not expected.
    return torch.cuda.is_available()


# Cache of resolved (name, source-tag) -> concrete function, so dispatch costs one dict
# lookup after the first call for each (kernel, device) combination.
_RESOLVED: dict = {}


def _make(name: str, triton_tag: str):
    """Build a wrapper for kernel ``name`` that dispatches to the Triton module
    ``triton_tag`` for CUDA tensors and to the numba ``cpu`` module for CPU tensors."""

    def _call(*args, **kwargs):
        tag = triton_tag if _args_on_cuda(args, kwargs) else "cpu"
        key = (name, tag)
        fn = _RESOLVED.get(key)
        if fn is None:
            fn = getattr(_mod(tag), name)
            _RESOLVED[key] = fn
        return fn(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = name
    _call.__doc__ = (
        f"Device-dispatched kernel '{name}': Triton ({triton_tag}) on CUDA tensors, "
        f"numba (cpu) on CPU tensors."
    )
    return _call


# --- shape kernels (shape_triton <-> cpu) -------------------------------------
overlap_score_grad_se3_batch = _make("overlap_score_grad_se3_batch", "shape")
fused_adam_qt = _make("fused_adam_qt", "shape")
fused_adam_qt_with_tangent_proj = _make("fused_adam_qt_with_tangent_proj", "shape")
_batch_self_overlap = _make("_batch_self_overlap", "shape")

# --- ESP kernels (esp_triton <-> cpu) -----------------------------------------
overlap_score_grad_esp_se3_batch = _make("overlap_score_grad_esp_se3_batch", "esp")
_batch_self_overlap_esp = _make("_batch_self_overlap_esp", "esp")
# ShaEP surface-ESP agreement of a posed pair, value + SE(3) gradient (vol_and_surf_esp).
esp_agreement_grad_se3_batch = _make("esp_agreement_grad_se3_batch", "esp")

# --- avoid (linear hard-sphere excluded-volume) kernel (avoid_triton <-> cpu) -
overlap_score_grad_avoid_se3_batch = _make("overlap_score_grad_avoid_se3_batch", "avoid")

# --- pharmacophore kernels (pharm_triton <-> cpu) -----------------------------
# Directional pharm value + quaternion-gradient kernel (pharm mode, in-register dQ).
pharm_grad_dq_se3_batch = _make("pharm_grad_dq_se3_batch", "pharm")
# Directionless "color" value+quaternion-grad kernel (vol_color).
pharm_color_score_grad_se3_batch = _make("pharm_color_score_grad_se3_batch", "pharm")
