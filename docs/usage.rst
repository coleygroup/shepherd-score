Usage
=====

Molecules and profiles
----------------------

``Molecule`` wraps an RDKit molecule with a 3D conformer and the features
computed from it:

- ``num_surf_points`` sets the surface sample size; leave it ``None`` if no
  surface modes are needed.
- ``pharm_multi_vector`` turns on pharmacophore extraction. ``False`` gives one
  vector per directional pharmacophore.
- ``charge_model`` selects partial charges: ``"xtb_water"`` (default, xTB in
  implicit water), ``"xtb"`` (vacuum) or ``"mmff"``. The xTB models fall back
  to MMFF94 with a warning when xTB is unavailable. Charges can also be passed
  in directly with ``partial_charges``.

Charges are computed on first use; building a surface computes its ESP as well.
Mesh surface sampling is random, so save and reuse surface points and ESP
values when two runs must see identical features.

.. code-block:: python

   from shepherd_score.conformer_generation import embed_conformer_from_smiles
   from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

   def prepare(smiles):
       rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True)
       return Molecule(rd, num_surf_points=200, pharm_multi_vector=False,
                       charge_model="mmff")

   ref, fit = prepare("CCOc1ccccc1"), prepare("CCNc1ccccc1")
   pair = MoleculePair(ref, fit, num_surf_points=200)

Pass the same ``num_surf_points`` to ``MoleculePair`` when using surface modes; the
pair checks its own value, not the molecules'.

Aligning one pair
-----------------

``MoleculePair.align`` takes a mode name and that mode's parameters, runs the
PyTorch optimizer and returns an ``AlignmentResult`` with ``score``,
``transform`` and ``aligned``. ``MoleculePair.score`` scores the current poses
without optimizing.

.. code-block:: python

   res = pair.align("surf_esp", lam=0.3)
   res = pair.align("vol", similarity="tversky_ref", num_repeats=20)
   pair.score("vol_color", color_weight=0.3)

The ``align_with_<mode>`` methods do the same and also store the score and
transform on the pair (``sim_aligned_<mode>``, ``transform_<mode>``).

Aligning many pairs
-------------------

.. code-block:: python

   scores, aligned = MoleculePairBatch([pair]).align_with_vol(backend="numba")

``MoleculePairBatch`` runs the batched kernels: ``backend="numba"`` on CPU and
``backend="triton"`` on GPU. The default picks Triton when CUDA is available.
``backend="jax"`` and ``backend="torch"`` run the reference optimizers.
Results are written to each pair as with ``MoleculePair.align_with_<mode>``.
Pass ``return_aligned=True`` to also get the aligned coordinates.

Scoring modes
-------------

``vol`` and ``surf`` compare heavy-atom and surface point clouds. ``vol_esp``
and ``surf_esp`` weight overlaps by partial charge and surface potential.
``pharm`` compares typed, directional pharmacophores. ``vol_color``,
``vol_lipo``, ``vol_mr``, ``vol_fukui``, ``vol_atomtype`` and ``vol_pharm``
blend shape with another atomic feature. ``vol_and_surf_esp`` combines shape
with a masked surface-potential agreement term. ``vol_avoid`` subtracts an
excluded-volume penalty and needs an ``avoid_points`` array.

``similarity=`` sets how an overlap becomes a score: ``"tanimoto"`` (default),
``"tversky"``, ``"tversky_ref"``, ``"tversky_fit"`` or explicit weights
``(a, b)``. See :doc:`theory` for the formulas.

Default seeds, step budgets and early-stopping patience for the batched
kernels are in ``shepherd_score/accel/_modes.py``.

Screening
---------

.. code-block:: python

   from shepherd_score.screen import ProfileStore, screen

   with ProfileStore.create("library.fss", num_surf_points=200,
                            modes=["vol", "surf_esp"]) as store:
       store.add(fit, id="compound-1")
   hits = screen(ref, ProfileStore.open("library.fss"), mode="vol", top_k=1)

A store keeps only the arrays its declared modes need, and every molecule and
query must use the same ``num_surf_points``. Returned transforms are in the
library molecule's input frame. See :doc:`api/screening` for multi-query and
multi-GPU screening.

Evaluation
----------

``scripts/README.md`` lists the scripts that evaluate generated ShEPhERD
samples. Throughput and retrieval benchmarks are in the paper repository. The
tutorials cover profile extraction, scoring and evaluation; their xTB examples
need the xTB executable.
