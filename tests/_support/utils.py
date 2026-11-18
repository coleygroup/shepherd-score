import os
import subprocess

# Conditional JAX import - if not available, JAX-specific tests will be skipped
def _configure_jax_platform():
    """
    Configure JAX platform based on GPU availability.
    Sets JAX to CPU-only mode if no GPU is detected to avoid initialization errors.

    Returns:
        bool: True if GPU detected, False otherwise
    """
    try:
        # Try to detect GPU via nvidia-smi
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=2  # Add timeout to avoid hanging
        )
        gpu_detected = result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        gpu_detected = False

    if not gpu_detected:
        # Force JAX to use CPU only if no GPU detected
        os.environ['JAX_PLATFORMS'] = 'cpu'

    return gpu_detected


def mode_loss(mode, se3_params, ref, fit):
    """``1 - mode.score`` with ``fit`` posed by ``se3_params``: the loss ``optimize_se3`` minimizes."""
    from shepherd_score.alignment.utils.se3 import get_SE3_transform
    return 1 - mode.score(ref, mode.pose(fit, get_SE3_transform(se3_params)))


def align_per_pair(pair, mode, **kw):
    """Run the per-pair ``MoleculePair`` method for an ``accel/_modes.py`` registry mode and
    return its score. A ``<base>_tversky`` mode runs ``align_with_<base>(similarity='tversky')``,
    which writes the ``<base>_tversky`` slot, except ``pharm``, whose upstream method writes
    ``sim_aligned_pharm``."""
    base = mode[:-len('_tversky')] if mode.endswith('_tversky') else mode
    if base != mode:
        kw['similarity'] = 'tversky'
    if base in ('surf', 'surf_esp', 'vol_and_surf_esp'):
        kw.setdefault('alpha', 0.81)
    getattr(pair, f'align_with_{base}')(**kw)
    slot = 'pharm' if mode == 'pharm_tversky' else mode
    return float(getattr(pair, f'sim_aligned_{slot}'))
