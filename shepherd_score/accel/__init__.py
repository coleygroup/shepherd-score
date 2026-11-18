"""GPU/CPU acceleration for batch molecular alignment.

:mod:`~shepherd_score.accel.kernels` holds the Triton GPU kernels and their numba CPU
mirrors behind a per-call device dispatcher; :mod:`~shepherd_score.accel.drivers` the
batched coarse-to-fine SE(3) optimizers; :mod:`~shepherd_score.accel.batch`,
``cpu_pool`` and ``multi_gpu`` the orchestration (size bucketing, GPU-memory
sub-batching, CPU pool, multi-GPU data parallelism). The public entry point is the
``backend=`` argument on :meth:`shepherd_score.container.MoleculePairBatch.align_with_*`
(``"jax"``, ``"triton"``, ``"numba"``); :func:`align_multi_gpu` drives multi-GPU screens.
"""
from .kernels.dispatch import has_triton

__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner", "clear_caches"]


def clear_caches() -> None:
    """Free the process-global accel caches and return their memory.

    These are the learned per-pair GPU-memory footprint table and the CUDA-graph LRU.
    The caches are keyed on tensor shapes, so a long-lived process that sees many distinct
    molecule sizes accumulates device memory; call this between independent workloads.
    """
    from .batch import _pad
    _pad._PAIR_FOOTPRINT_BYTES.clear()
    from .drivers._graphed import reset_graph_cache
    reset_graph_cache()


def __getattr__(name):
    # Lazy re-export of the multi-GPU driver, so importing the acceleration internals
    # never pulls in ``multi_gpu`` -> ``container`` eagerly.
    if name in ("align_multi_gpu", "MultiGPUAligner"):
        from . import multi_gpu
        return getattr(multi_gpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
