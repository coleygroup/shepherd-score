Installation
============

Install from the repository root::

   python -m pip install -e .

Numba CPU kernels are included. GPU kernels need a CUDA build of PyTorch and
Triton::

   python -m pip install -e ".[gpu]"

Other extras are ``jax``, ``docking``, ``dev`` and ``docs``. Python and
dependency versions are set in ``pyproject.toml``. Older PyPI releases do not
include the screening APIs described here.

CPU benchmark environment
-------------------------

``environment.yml`` defines a conda environment with Intel SVML (NumPy 2.2,
Numba 0.61.2 from pip, since conda-forge's llvmlite builds dropped SVML at
0.44). The published CPU timings used Numba 0.59.1. Check whether Numba loaded SVML::

   python -c "import shepherd_score.accel.kernels.cpu, numba; print(numba.config.USING_SVML)"

Without SVML the kernels still run, but more slowly, and results can differ
within floating-point tolerance.

To get SVML without conda, install the ``cpu`` extra, which pins
``numba<0.62`` (llvmlite dropped SVML at 0.62) and adds Intel's ``icc_rt``
runtime on x86-64 Linux. ``shepherd_score`` loads SVML from ``icc_rt`` itself
before compiling its CPU kernels, so no ``LD_LIBRARY_PATH`` change is needed::

   pip install "shepherd-score[cpu]"

Set ``NUMBA_DISABLE_INTEL_SVML=1`` to opt out.

External executables
--------------------

- **xTB** must be on ``PATH`` for xTB charges, xTB relaxation and Fukui
  descriptors. ``Molecule`` uses xTB in implicit water for charges by default
  and falls back to MMFF94, with a warning, when xTB is missing. Pass
  ``charge_model="mmff"`` to use MMFF94 directly.
- **Vina**: docking uses the Vina Python API from the ``docking`` extra.
- **ChemAxon**: the ChemAxon protonation path needs a licensed ``cxcalc``.
  The MolScrub path needs no extra install.

Documentation and tests
-----------------------

From the repository root::

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests/ -m "not jax"
   python -m sphinx -b html docs docs/_build/html
