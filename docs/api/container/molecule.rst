Molecule
========

:class:`~shepherd_score.container.Molecule` wraps an RDKit ``Mol`` object
together with its extracted interaction profiles.

Surface and pharmacophore data are stored in :class:`~shepherd_score.container.profiles.Surface`
and :class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore` containers
(``mol.surface``, ``mol.pharmacophore``). Legacy flat attributes
(``surf_pos``, ``surf_esp``, ``pharm_types``, etc.) remain for backwards
compatibility. Use :meth:`~shepherd_score.container._core.Molecule.get_pharmacophore`
with ``return_atom_ids`` / ``priority_atoms`` for priority pharmacophore labeling.

Helper functions
----------------

.. autofunction:: shepherd_score.container._core.update_mol_coordinates

Class reference
---------------

.. autoclass:: shepherd_score.container._core.Molecule
   :members:
   :undoc-members:
   :show-inheritance:
