Usage Guide
===========

This guide covers the basic usage of shepherd-score for interaction profile extraction, 
3D similarity scoring, and alignment.

Overview
--------

The package has base functions and convenience wrappers. Scoring can be done with either 
NumPy or Torch, but alignment requires Torch. There are also JAX implementations for both 
scoring and alignment of gaussian overlap, ESP similarity, and pharmacophore similarity.

.. note::

   Applicable xTB functions and evaluation pipeline evaluations are now parallelizable 
   through the ``num_workers`` argument in the ``.evaluate`` method.

Base Functions
--------------

Conformer Generation
~~~~~~~~~~~~~~~~~~~~

Useful conformer generation functions are found in the :mod:`shepherd_score.conformer_generation` module.

Interaction Profile Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Interaction Profile
     - Function
   * - Shape
     - :func:`shepherd_score.extract_profiles.get_molecular_surface`
   * - Electrostatics
     - :func:`shepherd_score.extract_profiles.get_electrostatic_potential`
   * - Pharmacophores
     - :func:`shepherd_score.extract_profiles.get_pharmacophores`

Scoring
~~~~~~~

:mod:`shepherd_score.score` contains the base scoring functions with separate modules for those 
dependent on PyTorch (``*.py``), NumPy (``*_np.py``), and JAX (``*_jax.py``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Similarity
     - Function
   * - Shape
     - :func:`shepherd_score.score.gaussian_overlap.get_overlap`
   * - Electrostatics
     - :func:`shepherd_score.score.electrostatic_scoring.get_overlap_esp`
   * - Pharmacophores
     - :func:`shepherd_score.score.pharmacophore_scoring.get_overlap_pharm`

Convenience Wrappers
--------------------

Molecule Class
~~~~~~~~~~~~~~

:class:`shepherd_score.container.Molecule` accepts an RDKit ``Mol`` object (with an associated 
conformer) and generates user-specified interaction profiles.

MoleculePair Class
~~~~~~~~~~~~~~~~~~

:class:`shepherd_score.container.MoleculePair` operates on ``Molecule`` objects and prepares 
them for scoring and alignment.

Extraction Example
------------------

Extraction of interaction profiles:

.. code-block:: python

   from shepherd_score.conformer_generation import embed_conformer_from_smiles
   from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
   from shepherd_score.extract_profiles import get_atomic_vdw_radii, get_molecular_surface
   from shepherd_score.extract_profiles import get_pharmacophores, get_electrostatic_potential

   # Embed conformer with RDKit and partial charges from xTB
   ref_mol = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
   partial_charges = charges_from_single_point_conformer_with_xtb(ref_mol)

   # Radii are needed for surface extraction
   radii = get_atomic_vdw_radii(ref_mol)
   # `surface` is an np.array with shape (200, 3)
   surface = get_molecular_surface(ref_mol.GetConformer().GetPositions(), radii, num_points=200)

   # Get electrostatic potential at each point on the surface
   # `esp`: np.array (200,)
   esp = get_electrostatic_potential(ref_mol, partial_charges, surface)

   # Pharmacophores as arrays with averaged vectors
   # pharm_types: np.array (P,)
   # pharm_{pos/vecs}: np.array (P,3)
   pharm_types, pharm_pos, pharm_vecs = get_pharmacophores(ref_mol, multi_vector=False)

3D Similarity Scoring Example
-----------------------------

Scoring the similarity of two different molecules using 3D surface, ESP, and pharmacophore 
similarity metrics:

.. code-block:: python

   from shepherd_score.score.constants import ALPHA
   from shepherd_score.conformer_generation import embed_conformer_from_smiles
   from shepherd_score.conformer_generation import optimize_conformer_with_xtb
   from shepherd_score.container import Molecule, MoleculePair

   # Embed a random conformer with RDKit
   ref_mol_rdkit = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
   fit_mol_rdkit = embed_conformer_from_smiles('O=CCc1ccccc1', MMFF_optimize=True)

   # Local relaxation with xTB
   ref_mol, _, ref_charges = optimize_conformer_with_xtb(ref_mol_rdkit)
   fit_mol, _, fit_charges = optimize_conformer_with_xtb(fit_mol_rdkit)

   # Extract interaction profiles
   ref_molec = Molecule(ref_mol,
                        num_surf_points=200,
                        partial_charges=ref_charges,
                        pharm_multi_vector=False)
   fit_molec = Molecule(fit_mol,
                        num_surf_points=200,
                        partial_charges=fit_charges,
                        pharm_multi_vector=False)

   # Centers the two molecules' COM's to the origin
   mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

   # Compute the similarity score for each interaction profile
   shape_score = mp.score_with_surf(ALPHA(mp.num_surf_points))
   esp_score = mp.score_with_esp(ALPHA(mp.num_surf_points), lam=0.3)
   pharm_score = mp.score_with_pharm()

Alignment Example
-----------------

Alignment using the MoleculePair class:

.. code-block:: python

   # Centers the two molecules' COM's to the origin
   mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

   # Align fit_molec to ref_molec with your preferred objective function
   # By default we use automatic differentiation via pytorch
   surf_points_aligned = mp.align_with_surf(ALPHA(mp.num_surf_points),
                                            num_repeats=50)
   surf_points_esp_aligned = mp.align_with_esp(ALPHA(mp.num_surf_points),
                                               lam=0.3,
                                               num_repeats=50)
   pharm_pos_aligned, pharm_vec_aligned = mp.align_with_pharm(num_repeats=50)

   # Optimal scores and SE(3) transformation matrices are stored as attributes
   # mp.sim_aligned_surf, mp.sim_aligned_esp, mp.sim_aligned_pharm
   # mp.transform_surf, mp.transform_esp, mp.transform_pharm

   # Get a copy of the optimally aligned fit Molecule object
   transformed_fit_molec = mp.get_transformed_molecule(
       se3_transform=mp.transform_surf  # or mp.transform_esp, mp.transform_pharm
   )

Evaluation Pipelines
--------------------

We implement three evaluations of generated 3D conformers. Evaluations can be done on an 
individual basis or in a pipeline.

* **ConfEval**: Checks validity, pre-/post-xTB relaxation, calculates 2D graph properties
* **ConsistencyEval**: Inherits from ``ConfEval`` and evaluates the consistency of the 
  molecule's jointly generated interaction profiles with the true interaction profiles
* **ConditionalEval**: Inherits from ``ConfEval`` and evaluates the 3D similarity between 
  generated molecules and the target molecule

.. note::

   Evaluations can be run from any molecule's atomic numbers and positions with explicit 
   hydrogens (i.e., straight from an xyz file).

Example:

.. code-block:: python

   from shepherd_score.evaluations.evaluate import ConfEval
   from shepherd_score.evaluations.evaluate import UnconditionalEvalPipeline

   # ConfEval evaluates the validity of a given molecule, optimizes it with xTB,
   #   and also computes various 2D graph properties
   # `atom_array` np.ndarray (N,) atomic numbers of the molecule (with explicit H)
   # `position_array` np.ndarray (N,3) atom coordinates for the molecule
   conf_eval = ConfEval(atoms=atom_array, positions=position_array)

   # Alternatively, if you have a list of molecules you want to test:
   uncond_pipe = UnconditionalEvalPipeline(
       generated_mols=[(a, p) for a, p in zip(atom_arrays, position_arrays)]
   )
   uncond_pipe.evaluate(num_workers=4)

   # Properties are stored as attributes and can be converted into pandas df's
   sample_df, global_series = uncond_pipe.to_pandas()

For more detailed examples, see the Jupyter notebooks in the ``examples/`` directory of the repository.
