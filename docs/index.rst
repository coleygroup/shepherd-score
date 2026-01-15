ShEPhERD Score Documentation
============================

.. image:: _static/logo.svg
   :width: 200
   :align: center
   :alt: ShEPhERD Logo

|

.. image:: https://img.shields.io/pypi/v/shepherd-score.svg
   :target: https://pypi.org/project/shepherd-score/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/shepherd-score.svg
   :target: https://pypi.org/project/shepherd-score/
   :alt: Python versions

``shepherd-score`` provides tools for **generating/optimizing conformers**, **extracting interaction profiles**, 
**aligning interaction profiles**, and **differentiably scoring 3D similarity**. It also contains modules to 
evaluate conformers generated with *ShEPhERD* and other generative models.

The formulation of the interaction profile representation, scoring, alignment, and evaluations are found in our 
preprint `ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design <https://arxiv.org/abs/2411.04130>`_.

*ShEPhERD*: **S**\ hape, **E**\ lectrostatics, and **Ph**\ armacophores **E**\ xplicit **R**\ epresentation **D**\ iffusion

Quick Install
-------------

.. code-block:: bash

   pip install shepherd-score

   # With optional dependencies
   pip install "shepherd-score[jax]"      # JAX support for faster scoring
   pip install "shepherd-score[docking]"  # Docking evaluation tools
   pip install "shepherd-score[all]"      # Everything

.. toctree::
   :caption: Information
   :maxdepth: 2

   installation
   usage

.. toctree::
   :caption: Tutorials
   :maxdepth: 2

   tutorials/index

.. toctree::
   :caption: Documentation
   :maxdepth: 2

   api/index

Citation
--------

If you use or adapt ``shepherd_score`` or `ShEPhERD <https://github.com/coleygroup/shepherd>`_ in your work, please cite us:

.. code-block:: bibtex

   @misc{adamsShEPhERD2024,
     title = {{ShEPhERD}: {Diffusing} Shape, Electrostatics, and Pharmacophores for Bioisosteric Drug Design},
     author = {Adams, Keir and Abeywardane, Kento and Fromer, Jenna and Coley, Connor W.},
     year = {2024},
     number = {arXiv:2411.04130},
     eprint = {2411.04130},
     publisher = {arXiv},
     doi = {10.48550/arXiv.2411.04130},
     archiveprefix = {arXiv}
   }

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
