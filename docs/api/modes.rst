Modes and Similarity
====================

:mod:`shepherd_score.modes` defines the reference alignment modes used by
:meth:`~shepherd_score.container.MoleculePair.align`,
:meth:`~shepherd_score.container.MoleculePair.score` and the
``MoleculePair.align_with_<mode>`` methods. A mode is a frozen dataclass that
says which arrays it reads from a :class:`~shepherd_score.container.Molecule`
and how it scores a posed fit against the reference. Seeding, the SE(3)
optimizer and result storage are shared by all modes.

Modes are usually selected by name, with their parameters as keywords:

.. code-block:: python

   pair.align("vol")
   pair.align("surf_esp", lam=0.3, similarity="tversky")

A :class:`~shepherd_score.modes.Mode` instance can be passed instead:

.. code-block:: python

   from shepherd_score.modes import Vol

   pair.align(Vol(no_H=False, similarity=(0.7, 0.3)))

To add a mode, subclass :class:`~shepherd_score.modes.Mode`, implement
``inputs`` and ``overlap`` (or ``score`` when the result is not an overlap
ratio), and list the inputs the pose moves in ``points`` and ``vectors``.
:class:`~shepherd_score.modes.Combined` blends existing modes. The batched
Numba/Triton modes are registered separately in
``shepherd_score/accel/_modes.py``.

Registered names are the keys of :data:`shepherd_score.modes.MODES`. See
:doc:`../modes` for what each one compares, what it needs and which backends
run it.

Modes
-----

.. automodule:: shepherd_score.modes
   :members: Mode, Vol, Surf, VolAvoid, Avoid, VolESP, SurfESP, VolAndSurfESP, Pharm,
             Field, AtomType, Combined, blend, vol_color, vol_lipo, vol_mr, vol_fukui,
             vol_atomtype, get_mode, MODES
   :show-inheritance:

Similarity
----------

.. automodule:: shepherd_score.score.similarity
   :members: similarity_from_overlaps, tversky_weights, Similarity, TVERSKY_WEIGHTS

Optimizer
---------

``optimize_se3(mode, ref, fit, ...)`` is the shared PyTorch optimizer that
:meth:`~shepherd_score.container.MoleculePair.align` runs for a mode without a
specialized backend. It is documented under :doc:`alignment/main`.
