"""The SVML loader turns SVML on for pip installs of ``shepherd-score[cpu]`` and is a no-op elsewhere."""
import subprocess
import sys

import pytest

numba = pytest.importorskip("numba")


def _llvmlite_has_svml():
    import llvmlite.binding as llb
    has = getattr(llb, "has_svml", False)
    return bool(has() if callable(has) else has)


def _icc_rt_installed():
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("icc_rt")
        return True
    except PackageNotFoundError:
        return False


def _svml_after_kernel_import(env_extra=None):
    """USING_SVML in a fresh interpreter after importing the numba kernels."""
    import os
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_INTEL_SVML"}
    env.update(env_extra or {})
    out = subprocess.run(
        [sys.executable, "-c",
         "import shepherd_score.accel.kernels.cpu, numba; print(numba.config.USING_SVML)"],
        capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip().splitlines()[-1] == "True"


def test_enable_svml_is_idempotent_and_returns_state():
    from shepherd_score.accel.kernels._svml import enable_svml
    first = enable_svml()
    assert enable_svml() is first
    assert first == bool(numba.config.USING_SVML)


@pytest.mark.skipif(not (sys.platform.startswith("linux") and _icc_rt_installed()
                         and _llvmlite_has_svml()),
                    reason="needs icc_rt and an SVML-built llvmlite (numba < 0.62) on Linux")
def test_kernels_enable_svml_without_ld_library_path():
    assert _svml_after_kernel_import({"LD_LIBRARY_PATH": ""})


def test_disable_env_var_is_respected():
    assert not _svml_after_kernel_import({"NUMBA_DISABLE_INTEL_SVML": "1"})
