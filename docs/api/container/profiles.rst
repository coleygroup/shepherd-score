Profile Containers
==================

Structured containers for extracted interaction profiles, used internally by
:class:`~shepherd_score.container.Molecule` and available for serialization and
visualization.

.. autoclass:: shepherd_score.container.profiles.Surface
   :members:
   :exclude-members: positions, esp, probe_radius
   :undoc-members:
   :show-inheritance:

``Pharmacophore`` is defined in :mod:`shepherd_score.pharm_utils.pharmacophore`
and re-exported from :mod:`shepherd_score.container` for convenience. See
:doc:`../pharmacophore` for priority-label semantics.
