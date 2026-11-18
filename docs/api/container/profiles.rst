Profile Containers
==================

Containers for extracted interaction profiles. :class:`~shepherd_score.container.Molecule`
stores its surface and pharmacophores in them, and the visualization functions accept them.

.. autoclass:: shepherd_score.container.profiles.Surface
   :members:
   :exclude-members: positions, esp, probe_radius
   :undoc-members:
   :show-inheritance:

``Pharmacophore`` is defined in :mod:`shepherd_score.pharm_utils.pharmacophore`
and re-exported from :mod:`shepherd_score.container`. See
:doc:`../pharmacophore` for priority-label semantics.
