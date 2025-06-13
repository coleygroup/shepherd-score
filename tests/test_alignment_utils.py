"""
Unit tests for alignment and related fucntions.
"""
import os
import numpy as np
import torch
import jax.numpy as jnp

from shepherd_score.alignment_utils.se3_np import quaternions_to_rotation_matrix_np, get_SE3_transform_np, apply_SE3_transform_np
from shepherd_score.alignment_utils.se3_jax import quaternions_to_rotation_matrix_jax, get_SE3_transform_jax, apply_SE3_transform_jax
from shepherd_score.alignment_utils.se3 import quaternions_to_rotation_matrix, get_SE3_transform, apply_SE3_transform

# Configure JAX platform before import to avoid GPU errors
def _configure_jax_platform():
    """Configure JAX platform based on GPU availability."""
    try:
        # Try to detect GPU through other means first
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_detected = result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu_detected = False

    if not gpu_detected:
        # Force JAX to use CPU only if no GPU detected
        os.environ['JAX_PLATFORMS'] = 'cpu'

    return gpu_detected

# Pre-configure JAX platform
_gpu_detected_early = _configure_jax_platform()

class TestSE3:
    """
    Test SE(3) related transformations.
    Checks all versions: NumPy, Jax, Torch (batched and unbatched)
    """
    ex_quaternion = np.array([5., 0.2, -3.4, 0.4])
    # Numbers from from https://www.andre-gaschler.com/rotationconverter/
    sol_rot_matrix = np.array([
        [0.3623504, -0.1458107, -0.9205658],
        [0.0718172,  0.9891186, -0.1284004],
        [0.9292709, -0.0195865,  0.3688792]
    ])

    ex_se3_params = np.array([5., 0.2, -3.4, 0.4, 4., 5., 6.])
    sol_se3_transform = np.array([
        [0.3623504, -0.1458107, -0.9205658, 4.],
        [0.0718172,  0.9891186, -0.1284004, 5.],
        [0.9292709, -0.0195865,  0.3688792, 6.],
        [0.       , 0.        , 0.        , 1.]
    ])

    ex_set_of_points = np.array([
        [1., 2., 3.],
        [-1., 2., -3.],
        [0.4, -0.4, -0.5]
    ])
    sol_points_transformed = (sol_se3_transform[:3, :3] @ ex_set_of_points.T).T + sol_se3_transform[:3, 3]

    # NUMPY functions
    def test_quaternion_to_rotation_matrix_np(self):
        """ Test quaternions_to_rotation_matrix_np """
        out_rot_matrix = quaternions_to_rotation_matrix_np(self.ex_quaternion)
        assert np.allclose(out_rot_matrix, self.sol_rot_matrix)

    def test_se3_matrix_from_params_np(self):
        """ Test get_SE3_transform_np """
        out_se3_matrix = get_SE3_transform_np(self.ex_se3_params)
        assert np.allclose(out_se3_matrix, self.sol_se3_transform)

    def test_apply_se3_transform_np(self):
        """ Test apply_SE3_transform_np  """
        out_transformed = apply_SE3_transform_np(self.ex_set_of_points, self.sol_se3_transform)
        assert np.allclose(out_transformed, self.sol_points_transformed)

    # TORCH functions (single instance)
    def test_quaternion_to_rotation_matrix_torch(self):
        """ Test quaternions_to_rotation_matrix (Torch) """
        out_rot_matrix = quaternions_to_rotation_matrix(torch.Tensor(self.ex_quaternion))
        assert torch.allclose(out_rot_matrix, torch.Tensor(self.sol_rot_matrix))

    def test_se3_matrix_from_params_torch(self):
        """ Test get_SE3_transform (torch) """
        out_se3_matrix = get_SE3_transform(torch.Tensor(self.ex_se3_params))
        assert torch.allclose(out_se3_matrix, torch.Tensor(self.sol_se3_transform))
    
    def test_apply_se3_transform_torch(self):
        """ Test apply_SE3_transform torch  """
        out_transformed = apply_SE3_transform(torch.Tensor(self.ex_set_of_points), torch.Tensor(self.sol_se3_transform))
        assert torch.allclose(out_transformed, torch.Tensor(self.sol_points_transformed))

    # TORCH functions (batched)
    def test_quaternion_to_rotation_matrix_torch_batched(self):
        """ Test quaternions_to_rotation_matrix (Torch batched) """
        out_rot_matrix = quaternions_to_rotation_matrix(torch.Tensor(self.ex_quaternion).repeat((2, 1)))
        assert torch.allclose(out_rot_matrix, torch.Tensor(self.sol_rot_matrix).repeat((2,1,1)))

    def test_se3_matrix_from_params_torch_batched(self):
        """ Test get_SE3_transform (torch batched) """
        out_se3_matrix = get_SE3_transform(torch.Tensor(self.ex_se3_params).repeat((2,1)))
        assert torch.allclose(out_se3_matrix, torch.Tensor(self.sol_se3_transform).repeat((2, 1, 1)))
    
    def test_apply_se3_transform_torch_batched(self):
        """ Test apply_SE3_transform (torch batched)  """
        points_repeated = torch.Tensor(self.ex_set_of_points).repeat((2, 1, 1))
        transform_repeated = torch.Tensor(self.sol_se3_transform).repeat((2, 1, 1))
        out_transformed = apply_SE3_transform(points_repeated, transform_repeated)
        sol_repeated = torch.Tensor(self.sol_points_transformed).repeat((2, 1, 1))
        assert torch.allclose(out_transformed, sol_repeated)

    # Jax functions (single instance)
    def test_quaternion_to_rotation_matrix_jax(self):
        """ Test quaternions_to_rotation_matrix (Jax) """
        out_rot_matrix = quaternions_to_rotation_matrix_jax(jnp.array(self.ex_quaternion))
        assert jnp.allclose(out_rot_matrix, jnp.array(self.sol_rot_matrix))

    def test_se3_matrix_from_params_jax(self):
        """ Test get_SE3_transform (Jax) """
        out_se3_matrix = get_SE3_transform_jax(jnp.array(self.ex_se3_params))
        assert jnp.allclose(out_se3_matrix, jnp.array(self.sol_se3_transform))
    
    def test_apply_se3_transform_jax(self):
        """ Test apply_SE3_transform (Jax)  """
        out_transformed = apply_SE3_transform_jax(jnp.array(self.ex_set_of_points), jnp.array(self.sol_se3_transform))
        assert jnp.allclose(out_transformed, jnp.array(self.sol_points_transformed))
        


class TestPCA:
    """
    Test principal component analysis
    """
    