Pharmacophore Utilities
=======================

Modules for pharmacophore feature extraction and manipulation.

Pharmacophore container and priority labels
-------------------------------------------

:func:`~shepherd_score.pharm_utils.pharmacophore.get_pharmacophores` returns a
:class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore` dataclass. For backwards
compatibility it still unpacks and indexes as the original ``(types, positions, vectors)``
3-tuple.

When built with ``return_atom_ids=True`` (or when ``priority_atoms`` is provided), each
pharmacophore retains the set of contributing atom indices on ``.atom_ids``. This enables
:class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore.priority_labels` to compute
0/1 priority labels lazily against any number of priority-atom sets without re-extracting
pharmacophores. When ``priority_atoms`` is passed directly to
:func:`~shepherd_score.pharm_utils.pharmacophore.get_pharmacophores`, the labels are
computed immediately and stored on ``.labels``.

Labeling semantics (:func:`~shepherd_score.pharm_utils.pharmacophore.priority_pharm_labels`):

- **Point features** (Acceptor, Donor, Halogen, Cation, Anion, ZnBinder): label ``1`` if
  any contributing atom appears in ``priority_atoms``.
- **Aromatic and ring-derived Hydrophobe**: require at least ``min_ring_priority_atoms``
  heavy atoms (default ``3``) from a shared ring to also be in ``priority_atoms``. This
  prevents a single priority atom on an aromatic ring from flagging the whole ring feature.
  Use ``min_ring_priority_atoms=1`` for looser labeling.

The same options are available on :meth:`~shepherd_score.container._core.Molecule.get_pharmacophore`.
Priority pharmacophore **array indices** (not atom indices) can be passed to
:class:`~shepherd_score.evaluations.evaluate.evals.ConditionalEval` via
``priority_pharm_indices`` for subset Tversky scoring after full-set pharmacophore alignment.

Pharmacophore Extraction
------------------------

.. autoclass:: shepherd_score.pharm_utils.pharmacophore.Pharmacophore
   :members:
   :exclude-members: types, positions, vectors, mol, atom_ids, labels
   :undoc-members:
   :show-inheritance:

.. automodule:: shepherd_score.pharm_utils.pharmacophore
   :members:
   :exclude-members: Pharmacophore
   :undoc-members:
   :show-inheritance:

Pharmacophore Vectors
---------------------

.. automodule:: shepherd_score.pharm_utils.pharmvec
   :members:
   :undoc-members:
   :show-inheritance:
