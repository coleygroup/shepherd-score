"""Turn on numba's Intel SVML path when the libraries come from pip rather than conda.

numba enables SVML at import only if ``dlopen("libsvml.so")`` succeeds. Conda's llvmlite finds
it under ``$CONDA_PREFIX/lib``. The pip ``icc_rt`` wheel installs the same library into
``<env>/lib``, which the loader does not search, so pip installs get ``USING_SVML=False``
unless the user adds that directory to ``LD_LIBRARY_PATH``.

:func:`enable_svml` loads ``libintlc.so.5`` (a dependency of libsvml) and ``libsvml.so`` from
the installed ``icc_rt`` distribution, then sets the same LLVM option numba sets itself. This
works whether or not numba is already imported, but it only affects functions compiled
afterwards, so the numba kernel modules call it before defining their kernels. It does nothing
when SVML is already on, when ``NUMBA_DISABLE_INTEL_SVML`` is set, off Linux, without
``icc_rt``, or when llvmlite was built without SVML (numba >= 0.62).
"""
import ctypes
import sys

_LIBS = ("libintlc.so.5", "libsvml.so")   # load order: libsvml.so links against libintlc
_done = False


def _icc_rt_paths():
    from importlib.metadata import PackageNotFoundError, files
    try:
        found = {f.name: f.locate() for f in files("icc_rt") or () if f.name in _LIBS}
    except PackageNotFoundError:
        return None
    if set(found) != set(_LIBS) or not all(p.is_file() for p in found.values()):
        return None
    return [str(found[n]) for n in _LIBS]


def enable_svml() -> bool:
    """Enable SVML for numba code compiled from now on; return ``numba.config.USING_SVML``."""
    global _done
    try:
        import numba
        import llvmlite.binding as llb
    except ImportError:
        return False
    if _done or numba.config.USING_SVML:
        return bool(numba.config.USING_SVML)
    _done = True
    if numba.config.DISABLE_INTEL_SVML or not sys.platform.startswith("linux"):
        return False
    has_svml = getattr(llb, "has_svml", False)   # a function up to llvmlite 0.44, a bool after
    if not (has_svml() if callable(has_svml) else has_svml):
        return False
    paths = _icc_rt_paths()
    if paths is None:
        return False
    try:
        for p in paths:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        llb.load_library_permanently(paths[-1])
        llb.set_option("SVML", "-vector-library=SVML")
    except OSError:
        return False
    numba.config.USING_SVML = True
    return True
