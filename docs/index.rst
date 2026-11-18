ShEPhERD-score Documentation
============================

.. image:: _static/logo.svg
   :width: 200
   :align: center
   :alt: ShEPhERD-score logo

|

.. image:: https://img.shields.io/pypi/v/shepherd-score.svg
   :target: https://pypi.org/project/shepherd-score/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/shepherd-score.svg
   :target: https://pypi.org/project/shepherd-score/
   :alt: Python versions

ShEPhERD-score generates and optimizes conformers, extracts interaction profiles, and scores and
aligns molecules by 3D similarity. Scoring modes are modular, and batched alignment and library
screening run on Numba (CPU) and Triton (GPU) kernels. It also evaluates 3D molecules sampled from
generative models such as *ShEPhERD* and *ShEPhERD-2*.

The interaction profile representation and scoring functions were introduced with the *ShEPhERD*
diffusion model (`ICLR 2025 <https://openreview.net/forum?id=KSLkFYHlYg>`_). Coming from v1? See
:doc:`migration`.

Quick Install
-------------

Install from PyPI (Python ≥3.10)::

   pip install shepherd-score            # core
   pip install "shepherd-score[cpu]"     # + Intel SVML for the fp64 Numba kernels
   pip install "shepherd-score[gpu]"     # + Triton GPU kernels

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
   migration

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

An article describing ShEPhERD-score v2.0 is forthcoming. If you use or adapt ShEPhERD-score in
your work, please cite it once available.

.. TODO: add the ShEPhERD-score v2.0 article BibTeX here

The interaction profile representation and scoring functions were introduced with
`ShEPhERD <https://github.com/coleygroup/shepherd>`_. If you use them or the ShEPhERD models,
please also cite:

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
