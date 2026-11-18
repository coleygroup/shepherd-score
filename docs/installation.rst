Installation
============

ShEPhERD-score requires Python 3.10 or later::

   pip install shepherd-score

To install from source, run this from the repository root::

   pip install -e .

Numba CPU kernels are included and need no extra. Optional extras (versions
are in ``pyproject.toml``):

- ``cpu``: Intel SVML for the fp64 Numba kernels (``numba<0.62`` and
  ``icc_rt``; see `Intel SVML`_ below)
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

SVML is optional, and most users do not need it.

On CPU, ``MoleculePairBatch.align*`` and ``screen`` run a fused Numba loop
with float32 kernels. These kernels compute ``exp`` as a polynomial instead of
calling the math library, so they vectorize on any Numba version, including
0.62 and later, which dropped SVML. They run equally fast and give the same
scores with or without SVML.

SVML only speeds up the float64 kernels in ``accel/kernels/cpu.py``. You hit
those in these cases:

- ``vol_esp`` **on large molecules.** The fused loop is used only while every
  padded width is at most 100 heavy atoms. A bucket wider than that runs the
  eager float64 loop.
- ``vol_and_surf_esp`` **and its Tversky variant.** These modes use the fused
  loop, but its ESP-agreement term calls a float64 kernel on every step, so
  SVML speeds up that term.
- **Calling the drivers directly with float64 tensors.** The fused loop runs
  only on float32, so a float64 run takes the eager loop. The batch and
  screening APIs pad to float32 and never take this path.
- **A fused-loop failure.** If the fused loop raises, the engine falls back
  to the eager loop.

``pharm`` and ``pharm_tversky`` also run the eager loop, but their speed does
not depend on SVML.

If you mainly screen or align large molecules with ``vol_esp``, or use
``vol_and_surf_esp`` heavily on CPU, install the extra. Otherwise you can skip
it. Check whether Numba loaded SVML::

   python -c "import shepherd_score.accel.kernels.cpu, numba; print(numba.config.USING_SVML)"

To get SVML, install the ``cpu`` extra (``pip install "shepherd-score[cpu]"``;
x86-64 Linux only), which pins ``numba<0.62`` because llvmlite dropped SVML at
0.62. ShEPhERD-score loads SVML from ``icc_rt`` itself, so no
``LD_LIBRARY_PATH`` change is needed. Set ``NUMBA_DISABLE_INTEL_SVML=1`` to opt
out.

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
