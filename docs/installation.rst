Installation
============

ShEPhERD-score requires Python 3.10 or later::

   pip install shepherd-score

To install from source, run this from the repository root::

   pip install -e .

Numba CPU kernels are included but run slower without Intel SVML; see
`Intel SVML`_ below to install it. Optional extras (versions are in
``pyproject.toml``):

- ``cpu``: SVML-vectorized CPU kernels (``numba<0.62`` and ``icc_rt``; see below)
- ``gpu``: Triton GPU kernels (``triton>=3.6``). This needs a CUDA build of
  PyTorch >= 2.6, which you must install for your system
- ``jax``: the JAX alignment backend
- ``docking``: Vina docking and interaction analysis
- ``dev``: pytest, ruff and mypy
- ``docs``: Sphinx and the documentation theme
- ``all``: all of the above

For example, ``pip install "shepherd-score[cpu,gpu]"``.

``environment.yml`` defines a conda environment that includes SVML, xTB and Vina.

External executables
--------------------

- **xTB** must be on ``PATH`` for xTB charges, xTB relaxation and Fukui
  descriptors. ``Molecule`` uses xTB in implicit water for charges by default
  (``charge_model="xtb_water"``) and falls back to MMFF94, with a warning, when
  xTB is missing. Pass ``charge_model="mmff"`` to use MMFF94 directly.
- **Vina**: docking uses the Vina Python API from the ``docking`` extra.
- **ChemAxon**: the ChemAxon protonation path needs a licensed ``cxcalc``.
  The MolScrub path needs no extra install.

Intel SVML
----------

The CPU kernels run fastest with Intel SVML. Check whether Numba loaded it::

   python -c "import shepherd_score.accel.kernels.cpu, numba; print(numba.config.USING_SVML)"

Without SVML the kernels still run, but more slowly (``vol`` about 1.3-1.6x,
surface modes about 5x; ``pharm`` is unaffected), and results can differ within
floating-point tolerance. To get SVML, install the ``cpu`` extra
(``pip install "shepherd-score[cpu]"``; x86-64 Linux only), which pins
``numba<0.62`` because llvmlite dropped SVML at 0.62. ShEPhERD-score loads SVML
from ``icc_rt`` itself, so no ``LD_LIBRARY_PATH`` change is needed. Set
``NUMBA_DISABLE_INTEL_SVML=1`` to opt out.

Headless Linux
--------------

Surface generation (``num_surf_points``) uses Open3D, whose Linux wheels link
``libEGL.so.1`` and ``libGL.so.1``. Desktop machines and most cluster nodes
have them; minimal servers and containers may not, and fail with
``ImportError: libEGL.so.1``. Without root, install them into the conda env::

   conda install -c conda-forge libegl libgl

Otherwise install ``libegl1 libgl1`` (Debian/Ubuntu) or
``mesa-libEGL mesa-libGL`` (RHEL/Rocky). Modes without a surface do not import
Open3D.

Documentation and tests
-----------------------

From the repository root::

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests/ -m "not jax"
   python -m sphinx -b html docs docs/_build/html
