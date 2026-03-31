"""
Parallel (multi-device) JAX alignment via jax.shard_map.

Each public ``optimize_*_jax_*_shmap`` function accepts flat ``(total, ...)``
arrays (where ``total`` is the number of pairs padded to a multiple of
``len(jax.devices())``) and returns flat ``(total, ...)`` results.  The
leading axis is automatically distributed across devices by
``jax.shard_map`` with ``PartitionSpec('i')``; do **not** pre-reshape to
``(n_devices, B, ...)``.

``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set **before
any JAX import** so that ``len(jax.devices()) == N``.
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.sharding import Mesh, PartitionSpec as P

from shepherd_score.alignment._jax import _per_pair_optimize_vol_mask_scan

# ---------------------------------------------------------------------------
# Volumetric alignment
# ---------------------------------------------------------------------------

# Cache: (max_num_steps, n_devices) -> jit-compiled shard_map function
_shmap_vol_cache: dict = {}


def _make_shmap_vol_fn(max_num_steps: int, mesh: Mesh):
    """Build a ``jit(shard_map(vmap(...)))`` volumetric alignment function.

    Closes over ``max_num_steps`` so ``lax.scan`` sees a static Python int.
    Wraps the shard_map in ``jit`` so XLA compiles it once (shard_map alone
    does not trigger compilation the way pmap does).
    The result is stored in ``_shmap_vol_cache`` and reused for matching calls.
    """
    def _per_shard(ref_b, fit_b, mask_r_b, mask_f_b, se3_b, alpha, VAA_b, VBB_b, lr):
        """Processes one device's shard: shape (B, N/M, 3) etc."""
        def per_pair(ref, fit, mr, mf, s, VAA, VBB):
            return _per_pair_optimize_vol_mask_scan(
                ref, fit, mr, mf, s, alpha, VAA, VBB, lr, max_num_steps
            )
        return vmap(per_pair)(ref_b, fit_b, mask_r_b, mask_f_b, se3_b, VAA_b, VBB_b)

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P('i'), P('i'), P(), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_ROCS_overlay_jax_vol_shmap(
    ref_batch,
    fit_batch,
    mask_ref_batch,
    mask_fit_batch,
    VAA_batch,
    VBB_batch,
    se3_init_batch,
    alpha: float,
    lr: float,
    max_num_steps: int,
):
    """Volumetric alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    All ``*_batch`` arrays use a flat leading axis of size ``total`` (i.e. the
    number of pairs padded to a multiple of ``len(jax.devices())``).  Unlike
    ``pmap``, ``shard_map`` automatically distributes the flat leading axis
    across devices; do **not** pre-reshape to ``(n_devices, B, ...)``.
    Pre-compute self-overlaps ``VAA``/``VBB`` and SE(3) initialisations outside
    this function (they are invariant to the optimisation loop).

    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set
    **before JAX is first imported** so that ``len(jax.devices()) == N``.

    Parameters
    ----------
    ref_batch       : (total, N, 3)  padded reference positions
    fit_batch       : (total, M, 3)  padded fit positions
    mask_ref_batch  : (total, N)
    mask_fit_batch  : (total, M)
    VAA_batch       : (total,)       pre-computed ref self-overlaps
    VBB_batch       : (total,)       pre-computed fit self-overlaps
    se3_init_batch  : (total, R, 7)  pre-initialised SE(3) params
    alpha           : float
    lr              : float
    max_num_steps   : int (Python int; determines compiled kernel)

    Returns
    -------
    aligned_pts   : (total, M, 3)
    se3_transform : (total, 4, 4)
    scores        : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (max_num_steps, len(devices))
    if cache_key not in _shmap_vol_cache:
        _shmap_vol_cache[cache_key] = _make_shmap_vol_fn(max_num_steps, mesh)
    fn = _shmap_vol_cache[cache_key]
    return fn(
        ref_batch, fit_batch,
        mask_ref_batch, mask_fit_batch,
        se3_init_batch, alpha, VAA_batch, VBB_batch, lr,
    )
