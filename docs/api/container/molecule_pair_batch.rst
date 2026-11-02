MoleculePairBatch
=================

``MoleculePairBatch`` aligns a list of ``MoleculePair`` objects and writes each
pair's score and transform to the same slots as
``MoleculePair.align_with_<mode>``. The Numba and Triton backends group pairs
by feature size, pad and mask the arrays, and optimize several starting poses
per pair.

.. code-block:: python

   from shepherd_score.container import MoleculePairBatch

   batch = MoleculePairBatch(pairs)
   scores, aligned = batch.align_with_vol(backend="numba")
   scores, aligned = batch.align("vol", similarity="tversky", backend="triton")

Backends
--------

- ``"numba"`` (alias ``"cpu"``): CPU kernels.
- ``"triton"`` (aliases ``"cuda"``, ``"gpu"``): CUDA kernels.
- ``None`` (default): Triton when CUDA is available, otherwise Numba.
- ``"jax"``: padded JAX path for ``vol``, ``vol_esp``, ``surf``, ``surf_esp``
  and ``pharm``. Other modes fall back to each pair's ``MoleculePair.align``,
  with a warning if the mode has no JAX optimizer.
- ``"torch"``: each pair's ``MoleculePair.align`` with PyTorch.

The Numba and Triton backends align heavy atoms only. JAX-specific options
such as ``use_shmap`` and ``num_buckets`` take effect only with
``backend="jax"``.

Common options
--------------

Every registered mode has an ``align_with_<mode>`` method, and
``align(mode, ...)`` calls it by name. ``align_with_esp`` and
``align_with_esp_combo`` are aliases for ``surf_esp`` and
``vol_and_surf_esp``.

- ``num_repeats``, ``max_num_steps`` and ``lr`` default to ``None``: the
  mode's values in ``shepherd_score/accel/_modes.py`` on Numba and Triton, and
  50 / 200 / 0.1 on JAX and PyTorch.
- ``similarity`` selects Tanimoto or Tversky normalization, as on
  ``MoleculePair``. Tversky results go to the ``<mode>_tversky`` slots.
- ``return_aligned`` (default ``False``) also returns the aligned fit coordinates.
- ``num_workers`` shards the pairs across single-threaded worker processes on
  Numba. Each shard stops early on its own, so results can differ slightly
  from a single process.
- ``verbose`` applies only to the JAX and PyTorch paths.

For JAX multi-device alignment see :doc:`../alignment/jax_parallel`. For
libraries on disk see :doc:`../screening`.

Class reference
---------------

.. autoclass:: shepherd_score.container._batch.MoleculePairBatch
   :members:
   :undoc-members:
   :show-inheritance:
