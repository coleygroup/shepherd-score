Installation
============

Via PyPI
--------

The simplest way to install shepherd-score is via pip:

.. code-block:: bash

   pip install shepherd-score

Install xTB
~~~~~~~~~~~

xTB will need to be installed manually since there are no PyPI bindings. This can be done in a conda 
environment, but since this approach has been reported to lead to conflicts, we suggest installing 
from `source <https://xtb-docs.readthedocs.io/en/latest/setup.html>`_ and adding it to ``PATH``.

Optional Dependencies
---------------------

JAX Support
~~~~~~~~~~~

For faster scoring and alignment using JAX:

.. code-block:: bash

   pip install "shepherd-score[jax]"

Docking Evaluation Tools
~~~~~~~~~~~~~~~~~~~~~~~~

Include docking evaluation tools:

.. code-block:: bash

   pip install "shepherd-score[docking]"

.. note::

   Installing ``shepherd-score[docking]`` will automatically install the Python bindings for 
   Autodock Vina. However, a manual installation of the executable of Autodock Vina v1.2.5 
   is required and can be found at: https://vina.scripps.edu/downloads/

All Dependencies
~~~~~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   pip install "shepherd-score[all]"

Local Development
-----------------

For local development:

.. code-block:: bash

   git clone https://github.com/coleygroup/shepherd-score.git
   cd shepherd-score
   pip install -e ".[all]"

Requirements
------------

This package works where PyTorch, Open3D, RDKit, and xTB can be installed for Python >=3.9. 
If you are coming from the *ShEPhERD* repository, you can use the same environment.

Core dependencies (installed automatically):

* ``python>=3.9``
* ``numpy``
* ``torch>=1.12``
* ``open3d>=0.18``
* ``rdkit>=2023.03``
* ``pandas>=2.0``
* ``scipy>=1.10``

.. note::

   If using ``torch<=2.4``, ensure that ``mkl==2024.0`` with conda since there is a known 
   `issue <https://github.com/pytorch/pytorch/issues/123097>`_ that prevents importing torch.
