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

``shepherd-score`` generates and optimizes conformers, extracts interaction profiles, and scores and
aligns molecules by 3D similarity. Batched alignment and library screening run on Numba (CPU) and
Triton (GPU) kernels. It also evaluates conformers generated with *ShEPhERD* and other generative models.

The formulation of the interaction profile representation, scoring, alignment, and evaluations are found in our 
ICLR 2025 paper `ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design <https://arxiv.org/abs/2411.04130>`_.

*ShEPhERD*: **S**\ hape, **E**\ lectrostatics, and **Ph**\ armacophores **E**\ xplicit **R**\ epresentation **D**\ iffusion

Quick Install
-------------

Install from the repository root::

   python -m pip install -e .

See :doc:`installation` for optional backends and external executables.

.. toctree::
   :caption: Information
   :maxdepth: 2

   installation
   modes
   usage
   theory
   accelerated_api
   scoring_mode_skills

.. toctree::
   :caption: Performance
   :maxdepth: 2

   performance/timings

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

   @inproceedings{
      adams2025shepherd,
      title={Sh{EP}h{ERD}: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design},
      author={Keir Adams and Kento Abeywardane and Jenna Fromer and Connor W. Coley},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=KSLkFYHlYg}
   }

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
