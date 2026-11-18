Tutorials
=========

Jupyter notebooks covering profile extraction, scoring, evaluation and library
screening. Notebooks 1-3 use the single-pair API; notebook 4 introduces batched
alignment and screening (see also :doc:`../accelerated_api`).

.. toctree::
   :maxdepth: 2

   01_profiles
   02_scoring
   03_evaluation
   04_screening

Tutorial Overview
-----------------

1. **Interaction Profiles** (``01_profiles.ipynb``): extracting shape,
   electrostatic and pharmacophore profiles from molecules.

2. **Scoring and Alignment** (``02_scoring.ipynb``): computing 3D similarity scores
   and aligning molecules by their interaction profiles.

3. **Evaluation Pipelines** (``03_evaluation.ipynb``): evaluating generated
   conformers.

4. **Screening** (``04_screening.ipynb``): batched alignment with
   ``MoleculePairBatch`` and screening a library stored in a ``ProfileStore``.

Running the Tutorials
---------------------

The notebooks are in ``examples/`` of the repository (the docs build copies them into
``docs/tutorials/``). From a checkout, install the package and open them:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install jupyter
   cd examples
   jupyter notebook

See :doc:`../installation` for optional GPU and xTB dependencies.
