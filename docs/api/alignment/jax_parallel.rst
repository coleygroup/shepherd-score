.. _jax-parallel:

JAX Parallel Alignment
======================

.. note::

   ``XLA_FLAGS`` must be set before JAX is imported so that
   ``len(jax.devices())`` equals the desired number of virtual CPU devices.
   Put these lines at the top of the script:

   .. code-block:: python

      import os
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
      os.environ['JAX_PLATFORMS'] = 'cpu'

This module provides multi-device volumetric alignment using
:func:`jax.shard_map`.  It is the backend called by
:meth:`shepherd_score.container.MoleculePairBatch.align_with_vol` when
``backend="jax", use_shmap=True``.

Overview
--------

:func:`optimize_ROCS_overlay_jax_vol_shmap` distributes a flat batch of
molecule pairs across virtual CPU devices in a single process:

* Accepts **flat** ``(total, ...)`` arrays where ``total`` is the number of
  pairs padded to a multiple of ``len(jax.devices())``.  Do not
  reshape to ``(n_devices, B, ...)``.
* Internally wraps ``_per_pair_optimize_vol_mask_scan`` (from
  :mod:`shepherd_score.alignment._jax`) in ``vmap``, then distributes via
  ``shard_map`` with ``PartitionSpec('i')`` on the leading axis.
* The compiled function is cached in ``_shmap_vol_cache`` keyed by
  ``(max_num_steps, n_devices)`` so the XLA kernel is compiled only once
  per unique ``(steps, device-count)`` combination.
* Uses ``lax.scan`` for a fixed number of steps (no convergence-based early
  stopping), which enables full ahead-of-time compilation.

Usage via the high-level API
-----------------------------

The usual entry point is
:meth:`~shepherd_score.container.MoleculePairBatch.align_with_vol`:

.. code-block:: python

   import os
   os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
   os.environ['JAX_PLATFORMS'] = 'cpu'

   from shepherd_score.container import MoleculePairBatch

   batch = MoleculePairBatch(pairs)

   # Default: single pass (n_buckets=1)
   scores, aligned = batch.align_with_vol(backend="jax", num_workers=4, use_shmap=True)

   # Bucketed: useful for >10k pairs with diverse molecule sizes
   scores, aligned = batch.align_with_vol(backend="jax", num_workers=4, use_shmap=True, num_buckets=8)

See :doc:`../container/molecule_pair_batch` for details on bucketing and masking strategy.

Direct usage
------------

Pre-compute self-overlaps and SE(3) initializations outside the function
(they do not change during optimization), then call:

.. code-block:: python

   from shepherd_score.alignment._jax_parallel import optimize_ROCS_overlay_jax_vol_shmap

   aligned_pts, se3_transform, scores = optimize_ROCS_overlay_jax_vol_shmap(
       ref_batch,       # (total, N, 3)
       fit_batch,       # (total, M, 3)
       mask_ref_batch,  # (total, N)
       mask_fit_batch,  # (total, M)
       VAA_batch,       # (total,)  pre-computed ref self-overlaps
       VBB_batch,       # (total,)  pre-computed fit self-overlaps
       se3_init_batch,  # (total, R, 7)  initial SE(3) params
       alpha=0.81,
       lr=0.1,
       max_num_steps=200,
   )

API
---

.. automodule:: shepherd_score.alignment._jax_parallel
   :members:
   :undoc-members:
   :show-inheritance:
