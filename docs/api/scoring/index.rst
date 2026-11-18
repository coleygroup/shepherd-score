Scoring Functions
=================

Functions for computing 3D similarity scores between molecules.

Constants
---------

.. automodule:: shepherd_score.score.constants
   :members:
   :undoc-members:
   :show-inheritance:

Atom-Identity Scoring
---------------------

Categorical Gaussian overlap in which only atoms sharing the same label (the atomic number)
contribute to the cross overlap. This is the scoring channel of the ``vol_atomtype`` alignment
mode.

.. automodule:: shepherd_score.score.atomtype_scoring
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   gaussian_overlap
   electrostatic
   pharmacophore
