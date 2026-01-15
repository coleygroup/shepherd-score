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
