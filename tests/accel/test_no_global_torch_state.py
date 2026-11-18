"""Importing the accelerated stack must not change process-wide torch settings.

``engine.py`` used to set ``torch.backends.cuda.matmul.allow_tf32 = True`` on import, which
switched every later CUDA matmul in the process to TF32, the reference optimizer's included.
``test_modes.py::test_vol_avoid_matches_the_old_optimizer`` then failed (0.467 vs 0.474)
whenever an earlier test had imported the engine.
"""
import subprocess
import sys


def test_importing_accel_leaves_tf32_alone():
    code = (
        "import torch\n"
        "before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,\n"
        "          torch.get_float32_matmul_precision())\n"
        "import shepherd_score.accel.drivers.engine, shepherd_score.accel.drivers.terms\n"
        "import shepherd_score.accel.batch.aligners, shepherd_score.accel.kernels.dispatch\n"
        "import shepherd_score.container, shepherd_score.screen\n"
        "after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,\n"
        "         torch.get_float32_matmul_precision())\n"
        "assert before == after, (before, after)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_engine_align_scopes_tf32_to_its_own_call():
    """``align`` runs its matmuls with TF32 allowed and puts the caller's setting back, also
    when it raises."""
    import torch
    import shepherd_score.accel.drivers.engine as EN

    mm, old = torch.backends.cuda.matmul, torch.backends.cuda.matmul.allow_tf32
    seen = []

    @EN._with_tf32
    def probe(fail=False):
        seen.append(mm.allow_tf32)
        if fail:
            raise RuntimeError

    try:
        mm.allow_tf32 = False
        probe()
        assert seen == [True] and mm.allow_tf32 is False
        try:
            probe(fail=True)
        except RuntimeError:
            pass
        assert seen == [True, True] and mm.allow_tf32 is False
    finally:
        mm.allow_tf32 = old
    assert EN.align.__name__ == "align"
