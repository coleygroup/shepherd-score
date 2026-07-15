MoleculePair
============

:class:`~shepherd_score.container.MoleculePair` holds a reference and a fit
:class:`~shepherd_score.container.Molecule` and provides methods for scoring
and aligning them using shape, ESP, or pharmacophore interaction profiles.

Each alignment mode stores its result in an :class:`~shepherd_score.container._core.AlignmentResult`
(``transform_<mode>``, ``sim_aligned_<mode>``).

.. autoclass:: shepherd_score.container._core.AlignmentResult
   :members:
   :exclude-members: score, transform
   :undoc-members:
   :show-inheritance:

.. autoclass:: shepherd_score.container._core.MoleculePair
   :members:
   :undoc-members:
   :show-inheritance:
