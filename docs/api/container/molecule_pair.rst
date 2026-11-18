MoleculePair
============

:class:`~shepherd_score.container.MoleculePair` holds a reference and a fit
:class:`~shepherd_score.container.Molecule` and scores or aligns them with the
PyTorch reference implementation.

- :meth:`~shepherd_score.container._core.MoleculePair.align` takes a mode name
  (or :class:`~shepherd_score.modes.Mode`) and its parameters and returns an
  :class:`~shepherd_score.container._core.AlignmentResult` without storing it.
- :meth:`~shepherd_score.container._core.MoleculePair.score` scores the current
  poses.
- ``align_with_<mode>`` aligns and stores the result on the pair
  (``transform_<mode>``, ``sim_aligned_<mode>``). A Tversky ``similarity``
  stores it in the ``<mode>_tversky`` slots.

.. code-block:: python

   res = pair.align("vol", similarity="tversky")
   res.score, res.transform

   pair.align_with_vol(similarity="tversky")
   pair.sim_aligned_vol_tversky

See :doc:`../modes` for the available modes and similarity options.

.. autoclass:: shepherd_score.container._core.AlignmentResult
   :members:
   :exclude-members: score, transform, aligned
   :undoc-members:
   :show-inheritance:

.. autoclass:: shepherd_score.container._core.MoleculePair
   :members:
   :undoc-members:
   :show-inheritance:
