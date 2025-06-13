"""
Unit tests for the gaussian_overlap modules (PyTorch and JAX implementations).
Assumes the NumPy implementation is ground truth.

Particularly for the batching functionality that complicates the pytorch implementation.
"""

import pytest
import torch
import numpy as np
from scipy.spatial import distance
import torch.nn.functional as F
from shepherd_score.score import gaussian_overlap as go
from shepherd_score.score import gaussian_overlap_np as go_np

# Helper function for NumPy vector normalization (for 2D arrays)
def _normalize_np_vectors_2d(vectors_np):
    norm = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    return np.divide(vectors_np, norm, out=np.zeros_like(vectors_np), where=norm != 0)

# Helper function to generate random data
def _generate_data(batch_size=None, num_points1=10, num_points2=12, dim=3, fixed_seed=None):
    if fixed_seed is not None:
        np.random.seed(fixed_seed)
    if batch_size:
        centers_1_np = np.random.rand(batch_size, num_points1, dim).astype(np.float32)
        centers_2_np = np.random.rand(batch_size, num_points2, dim).astype(np.float32)
        vectors_1_np = np.random.rand(batch_size, num_points1, dim).astype(np.float32)
        vectors_2_np = np.random.rand(batch_size, num_points2, dim).astype(np.float32)
        mask_1_np = (np.random.rand(batch_size, num_points1) > 0.5).astype(np.float32)
        mask_2_np = (np.random.rand(batch_size, num_points2) > 0.5).astype(np.float32)
    else:
        centers_1_np = np.random.rand(num_points1, dim).astype(np.float32)
        centers_2_np = np.random.rand(num_points2, dim).astype(np.float32)
        vectors_1_np = np.random.rand(num_points1, dim).astype(np.float32)
        vectors_2_np = np.random.rand(num_points2, dim).astype(np.float32)
        mask_1_np = (np.random.rand(num_points1) > 0.5).astype(np.float32)
        mask_2_np = (np.random.rand(num_points2) > 0.5).astype(np.float32)

    centers_1_torch = torch.from_numpy(centers_1_np)
    centers_2_torch = torch.from_numpy(centers_2_np)
    vectors_1_torch = torch.from_numpy(vectors_1_np)
    vectors_2_torch = torch.from_numpy(vectors_2_np)
    mask_1_torch = torch.from_numpy(mask_1_np)
    mask_2_torch = torch.from_numpy(mask_2_np)
    
    return (centers_1_np, centers_2_np, vectors_1_np, vectors_2_np, mask_1_np, mask_2_np,
            centers_1_torch, centers_2_torch, vectors_1_torch, vectors_2_torch, mask_1_torch, mask_2_torch)

class TestGaussianOverlapTorch:
    def test_get_overlap_single_instance(self):
        alpha = 0.81
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data()
        expected = go_np.get_overlap_np(c1_np, c2_np, alpha=alpha)
        actual = go.get_overlap(c1_torch, c2_torch, alpha=alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_batched(self):
        alpha = 0.81
        batch_size = 4
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data(batch_size=batch_size)
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.get_overlap_np(c1_np[i], c2_np[i], alpha=alpha))
        expected = np.array(expected_batch)
        actual = go.get_overlap(c1_torch, c2_torch, alpha=alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_broadcast_c1(self):
        alpha = 0.81
        batch_size = 4
        data_full_batch = _generate_data(batch_size=batch_size, num_points1=10, num_points2=12)
        c1_np_orig_batch, c2_np_batch, _, _, _, _, c1_torch_orig_batch, c2_torch_batch, _, _, _, _ = data_full_batch
        c1_np_single = c1_np_orig_batch[0]
        c1_torch_single = c1_torch_orig_batch[0]
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.get_overlap_np(c1_np_single, c2_np_batch[i], alpha=alpha))
        expected = np.array(expected_batch)
        actual = go.get_overlap(c1_torch_single, c2_torch_batch, alpha=alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_broadcast_c2(self):
        alpha = 0.81
        batch_size = 4
        data_full_batch = _generate_data(batch_size=batch_size, num_points1=10, num_points2=12)
        c1_np_batch, c2_np_orig_batch, _, _, _, _, c1_torch_batch, c2_torch_orig_batch, _, _, _, _ = data_full_batch
        c2_np_single = c2_np_orig_batch[0]
        c2_torch_single = c2_torch_orig_batch[0]
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.get_overlap_np(c1_np_batch[i], c2_np_single, alpha=alpha))
        expected = np.array(expected_batch)
        actual = go.get_overlap(c1_torch_batch, c2_torch_single, alpha=alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_single_instance(self, allow_antiparallel):
        alpha = 0.5
        c1_np, c2_np, v1_np, v2_np, _, _, c1_torch, c2_torch, v1_torch, v2_torch, _, _ = _generate_data(fixed_seed=42)
        expected = go_np.VAB_2nd_order_cosine_np(c1_np, c2_np, v1_np, v2_np, alpha, allow_antiparallel)
        actual = go.VAB_2nd_order_cosine(c1_torch, c2_torch, v1_torch, v2_torch, alpha, allow_antiparallel)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_batched(self, allow_antiparallel):
        alpha = 0.5
        batch_size = 3
        c1_np, c2_np, v1_np, v2_np, _, _, c1_torch, c2_torch, v1_torch, v2_torch, _, _ = _generate_data(batch_size=batch_size, fixed_seed=43)
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.VAB_2nd_order_cosine_np(c1_np[i], c2_np[i], v1_np[i], v2_np[i], alpha, allow_antiparallel))
        expected = np.array(expected_batch)
        actual = go.VAB_2nd_order_cosine(c1_torch, c2_torch, v1_torch, v2_torch, alpha, allow_antiparallel)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

    def test_VAB_2nd_order_single_instance(self):
        alpha = 0.7
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data()
        expected = go_np.VAB_2nd_order_np(c1_np, c2_np, alpha)
        actual = go.VAB_2nd_order(c1_torch, c2_torch, alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_VAB_2nd_order_batched(self):
        alpha = 0.7
        batch_size = 3
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data(batch_size=batch_size)
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.VAB_2nd_order_np(c1_np[i], c2_np[i], alpha))
        expected = np.array(expected_batch)
        actual = go.VAB_2nd_order(c1_torch, c2_torch, alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_shape_tanimoto_single_instance(self):
        alpha = 0.6
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data()
        expected = go_np.shape_tanimoto_np(c1_np, c2_np, alpha)
        actual = go.shape_tanimoto(c1_torch, c2_torch, alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_shape_tanimoto_batched(self):
        alpha = 0.6
        batch_size = 3
        c1_np, c2_np, _, _, _, _, c1_torch, c2_torch, _, _, _, _ = _generate_data(batch_size=batch_size)
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(go_np.shape_tanimoto_np(c1_np[i], c2_np[i], alpha))
        expected = np.array(expected_batch)
        actual = go.shape_tanimoto(c1_torch, c2_torch, alpha)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_VAB_2nd_order_mask_single_instance(self):
        alpha = 0.75
        c1_np, c2_np, _, _, m1_np, m2_np, c1_torch, c2_torch, _, _, m1_torch, m2_torch = _generate_data()
        c1_np_filtered = c1_np[m1_np.astype(bool)]
        c2_np_filtered = c2_np[m2_np.astype(bool)]
        if c1_np_filtered.shape[0] == 0 or c2_np_filtered.shape[0] == 0:
            expected = np.array(0.0, dtype=np.float32)
        else:
            expected = go_np.VAB_2nd_order_np(c1_np_filtered, c2_np_filtered, alpha)
        actual = go.VAB_2nd_order_mask(c1_torch, c2_torch, alpha, m1_torch, m2_torch)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_VAB_2nd_order_mask_batched(self):
        alpha = 0.75
        batch_size = 3
        c1_np_batch, c2_np_batch, _, _, m1_np_batch, m2_np_batch, c1_torch_batch, c2_torch_batch, _, _, m1_torch_batch, m2_torch_batch = _generate_data(batch_size=batch_size)
        expected_batch_list = []
        for i in range(batch_size):
            c1_np, c2_np = c1_np_batch[i], c2_np_batch[i]
            m1_np, m2_np = m1_np_batch[i], m2_np_batch[i]
            c1_np_filtered = c1_np[m1_np.astype(bool)]
            c2_np_filtered = c2_np[m2_np.astype(bool)]
            if c1_np_filtered.shape[0] == 0 or c2_np_filtered.shape[0] == 0:
                expected_val = np.array(0.0, dtype=np.float32)
            else:
                expected_val = go_np.VAB_2nd_order_np(c1_np_filtered, c2_np_filtered, alpha)
            expected_batch_list.append(expected_val)
        expected = np.array(expected_batch_list)
        actual = go.VAB_2nd_order_mask(c1_torch_batch, c2_torch_batch, alpha, m1_torch_batch, m2_torch_batch)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_VAB_2nd_order_mask_batch(self):
        alpha = 0.78
        batch_size = 2
        num_points1, num_points2 = 5, 6 # N, M
        c1_np_b, c2_np_b, _, _, m1_np_b, m2_np_b, c1_torch_b, c2_torch_b, _, _, m1_torch_b, m2_torch_b = _generate_data(
            batch_size=batch_size, num_points1=num_points1, num_points2=num_points2, fixed_seed=44
        )
        cdist_21_torch = torch.cdist(c2_torch_b, c1_torch_b)**2.0
        expected_b_list = []
        term_common = (np.pi**1.5) / ((2 * alpha)**1.5)
        for i in range(batch_size):
            c1_np, c2_np = c1_np_b[i], c2_np_b[i] # c1 is N, c2 is M for points
            m1_np, m2_np = m1_np_b[i], m2_np_b[i] # m1 is N, m2 is M for masks
            # Ground truth mimics PyTorch: dense cdist, then apply mask matrix
            R2_dense_np = distance.cdist(c2_np, c1_np)**2.0  # Shape (M, N)
            mask_mat_np = m2_np[:, np.newaxis] * m1_np[np.newaxis, :] # Shape (M, N)
            exp_term_np = np.exp(-(alpha / 2) * R2_dense_np)
            expected_val = np.sum(term_common * mask_mat_np * exp_term_np)
            expected_b_list.append(expected_val)
        expected = np.array(expected_b_list, dtype=np.float32)
        actual = go.VAB_2nd_order_mask_batch(cdist_21_torch, alpha, m1_torch_b, m2_torch_b)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_mask_single_instance(self, allow_antiparallel):
        alpha = 0.55
        c1_np, c2_np, v1_np, v2_np, m1_np, m2_np, c1_torch, c2_torch, v1_torch, v2_torch, m1_torch, m2_torch = _generate_data(fixed_seed=45)
        m1_bool, m2_bool = m1_np.astype(bool), m2_np.astype(bool)
        c1_np_f, v1_np_f = c1_np[m1_bool], v1_np[m1_bool]
        c2_np_f, v2_np_f = c2_np[m2_bool], v2_np[m2_bool]
        if c1_np_f.shape[0] == 0 or c2_np_f.shape[0] == 0:
            expected = np.array(0.0, dtype=np.float32)
        else:
            expected = go_np.VAB_2nd_order_cosine_np(c1_np_f, c2_np_f, v1_np_f, v2_np_f, alpha, allow_antiparallel)
        actual = go.VAB_2nd_order_cosine_mask(c1_torch, c2_torch, v1_torch, v2_torch, alpha, allow_antiparallel, m1_torch, m2_torch)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_mask_batched(self, allow_antiparallel):
        alpha = 0.55
        batch_size = 3
        c1_np_b, c2_np_b, v1_np_b, v2_np_b, m1_np_b, m2_np_b, c1_torch_b, c2_torch_b, v1_torch_b, v2_torch_b, m1_torch_b, m2_torch_b = _generate_data(batch_size=batch_size, fixed_seed=46)
        expected_b_list = []
        for i in range(batch_size):
            c1_np, c2_np, v1_np, v2_np, m1_np, m2_np = c1_np_b[i], c2_np_b[i], v1_np_b[i], v2_np_b[i], m1_np_b[i], m2_np_b[i]
            m1_bool, m2_bool = m1_np.astype(bool), m2_np.astype(bool)
            c1_np_f, v1_np_f = c1_np[m1_bool], v1_np[m1_bool]
            c2_np_f, v2_np_f = c2_np[m2_bool], v2_np[m2_bool]
            if c1_np_f.shape[0] == 0 or c2_np_f.shape[0] == 0:
                val = np.array(0.0, dtype=np.float32)
            else:
                val = go_np.VAB_2nd_order_cosine_np(c1_np_f, c2_np_f, v1_np_f, v2_np_f, alpha, allow_antiparallel)
            expected_b_list.append(val)
        expected = np.array(expected_b_list)
        actual = go.VAB_2nd_order_cosine_mask(c1_torch_b, c2_torch_b, v1_torch_b, v2_torch_b, alpha, allow_antiparallel, m1_torch_b, m2_torch_b)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_mask_batch(self, allow_antiparallel):
        alpha = 0.81
        batch_size = 2
        num_points1, num_points2 = 5, 6 # N, M
        c1_np_b, c2_np_b, v1_np_b, v2_np_b, m1_np_b, m2_np_b, c1_torch_b, c2_torch_b, v1_torch_b, v2_torch_b, m1_torch_b, m2_torch_b = _generate_data(
            batch_size=batch_size, num_points1=num_points1, num_points2=num_points2, fixed_seed=47
        )
        
        cdist_21_torch = torch.cdist(c2_torch_b, c1_torch_b)**2.0 # (B,M,N)
        v1_torch_norm_b = F.normalize(v1_torch_b, p=2, dim=v1_torch_b.dim()-1)
        v2_torch_norm_b = F.normalize(v2_torch_b, p=2, dim=v2_torch_b.dim()-1)
        vmm_21_torch = torch.matmul(v2_torch_norm_b, v1_torch_norm_b.permute(0,2,1)) # (B,M,N)

        expected_b_list = []
        term_common = (np.pi**1.5) / ((2 * alpha)**1.5)
        for i in range(batch_size):
            c1_np, c2_np = c1_np_b[i], c2_np_b[i]
            v1_np, v2_np = v1_np_b[i], v2_np_b[i]
            m1_np, m2_np = m1_np_b[i], m2_np_b[i]

            R2_dense_np = distance.cdist(c2_np, c1_np)**2.0  # (M,N)
            
            v1n_np = _normalize_np_vectors_2d(v1_np) # (N,D)
            v2n_np = _normalize_np_vectors_2d(v2_np) # (M,D)
            VMM_dense_np = np.matmul(v2n_np, v1n_np.T) # (M,N)

            if allow_antiparallel:
                VMM_dense_np = np.abs(VMM_dense_np)
            else:
                VMM_dense_np = np.clip(VMM_dense_np, 0., 1.)
            VMM_weighted_np = (VMM_dense_np + 2.) / 3.
            
            mask_mat_np = m2_np[:, np.newaxis] * m1_np[np.newaxis, :] # (M,N)
            exp_term_np = np.exp(-(alpha / 2) * R2_dense_np)
            
            expected_val = np.sum(term_common * mask_mat_np * VMM_weighted_np * exp_term_np)
            expected_b_list.append(expected_val)
        expected = np.array(expected_b_list, dtype=np.float32)
        
        actual = go.VAB_2nd_order_cosine_mask_batch(cdist_21_torch, vmm_21_torch, alpha, allow_antiparallel, m1_torch_b, m2_torch_b)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)

# Try to import JAX and skip tests if not available
import os
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

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
go_jax = pytest.importorskip("shepherd_score.score.gaussian_overlap_jax")

@pytest.mark.jax
class TestGaussianOverlapJAX:
    def test_get_overlap_jax(self):
        alpha = 0.81
        # Using fixed_seed for reproducibility with JAX tests as well
        c1_np, c2_np, _, _, _, _, _, _, _, _, _, _ = _generate_data(fixed_seed=101)
        
        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)

        expected_np = go_np.get_overlap_np(c1_np, c2_np, alpha=alpha)
        actual_jax = go_jax.get_overlap_jax(c1_jnp, c2_jnp, alpha=alpha)
        
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    def test_VAB_2nd_order_jax(self):
        alpha = 0.7
        c1_np, c2_np, _, _, _, _, _, _, _, _, _, _ = _generate_data(fixed_seed=102)

        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)

        expected_np = go_np.VAB_2nd_order_np(c1_np, c2_np, alpha)
        actual_jax = go_jax.VAB_2nd_order_jax(c1_jnp, c2_jnp, alpha)

        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    def test_shape_tanimoto_jax(self):
        alpha = 0.6
        c1_np, c2_np, _, _, _, _, _, _, _, _, _, _ = _generate_data(fixed_seed=103)

        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)

        expected_np = go_np.shape_tanimoto_np(c1_np, c2_np, alpha)
        actual_jax = go_jax.shape_tanimoto_jax(c1_jnp, c2_jnp, alpha)

        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    def test_get_overlap_jax_mask(self):
        alpha = 0.75
        c1_np, c2_np, _, _, m1_np, m2_np, _, _, _, _, _, _ = _generate_data(fixed_seed=104)

        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)
        m1_jnp = jnp.array(m1_np)
        m2_jnp = jnp.array(m2_np)

        # Ground truth: filter NumPy arrays then compute
        c1_np_filtered = c1_np[m1_np.astype(bool)]
        c2_np_filtered = c2_np[m2_np.astype(bool)]

        if c1_np_filtered.shape[0] == 0 or c2_np_filtered.shape[0] == 0:
            expected_np = np.array(0.0, dtype=np.float32)
        else:
            expected_np = go_np.get_overlap_np(c1_np_filtered, c2_np_filtered, alpha=alpha)
        
        actual_jax = go_jax.get_overlap_jax_mask(c1_jnp, c2_jnp, m1_jnp, m2_jnp, alpha=alpha)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    def test_VAB_2nd_order_jax_mask(self):
        alpha = 0.65 # Different alpha to vary tests slightly
        c1_np, c2_np, _, _, m1_np, m2_np, _, _, _, _, _, _ = _generate_data(fixed_seed=105)

        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)
        m1_jnp = jnp.array(m1_np)
        m2_jnp = jnp.array(m2_np)

        c1_np_filtered = c1_np[m1_np.astype(bool)]
        c2_np_filtered = c2_np[m2_np.astype(bool)]

        if c1_np_filtered.shape[0] == 0 or c2_np_filtered.shape[0] == 0:
            expected_np = np.array(0.0, dtype=np.float32)
        else:
            expected_np = go_np.VAB_2nd_order_np(c1_np_filtered, c2_np_filtered, alpha)
        
        actual_jax = go_jax.VAB_2nd_order_jax_mask(c1_jnp, c2_jnp, m1_jnp, m2_jnp, alpha)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    def test_shape_tanimoto_jax_mask(self):
        alpha = 0.55
        c1_np, c2_np, _, _, m1_np, m2_np, _, _, _, _, _, _ = _generate_data(fixed_seed=106)

        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)
        m1_jnp = jnp.array(m1_np)
        m2_jnp = jnp.array(m2_np)

        # Ground truth for VAA_mask
        c1_np_filtered_m1 = c1_np[m1_np.astype(bool)]
        if c1_np_filtered_m1.shape[0] == 0:
            VAA_np_mask = 0.0 # Or handle as per shape_tanimoto_np if it expects non-empty
        else:
            VAA_np_mask = go_np.VAB_2nd_order_np(c1_np_filtered_m1, c1_np_filtered_m1, alpha)

        # Ground truth for VBB_mask
        c2_np_filtered_m2 = c2_np[m2_np.astype(bool)]
        if c2_np_filtered_m2.shape[0] == 0:
            VBB_np_mask = 0.0
        else:
            VBB_np_mask = go_np.VAB_2nd_order_np(c2_np_filtered_m2, c2_np_filtered_m2, alpha)

        # Ground truth for VAB_mask
        c1_np_filtered_m1_for_VAB = c1_np[m1_np.astype(bool)] # Redundant but clear
        c2_np_filtered_m2_for_VAB = c2_np[m2_np.astype(bool)]
        if c1_np_filtered_m1_for_VAB.shape[0] == 0 or c2_np_filtered_m2_for_VAB.shape[0] == 0:
            VAB_np_mask = 0.0
        else:
            VAB_np_mask = go_np.VAB_2nd_order_np(c1_np_filtered_m1_for_VAB, c2_np_filtered_m2_for_VAB, alpha)
        
        if (VAA_np_mask + VBB_np_mask - VAB_np_mask) == 0:
             # Avoid division by zero; Tanimoto is undefined or 0 depending on convention.
             # If VAB is also 0, 0/0 -> 0 is a common practical outcome. If VAB >0 and denom is 0, it's an issue.
             # The JAX code will likely produce nan or inf here. The test might need adjustment for this edge case.
            expected_np = np.array(0.0, dtype=np.float32) if VAB_np_mask == 0 else np.nan
        else:
            expected_np = VAB_np_mask / (VAA_np_mask + VBB_np_mask - VAB_np_mask)

        actual_jax = go_jax.shape_tanimoto_jax_mask(c1_jnp, c2_jnp, m1_jnp, m2_jnp, alpha)
        if np.isnan(expected_np):
            assert np.isnan(np.array(actual_jax)), "Expected NaN for Tanimoto due to zero denominator but VAB non-zero"
        else:
            np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

    # Test cases for JAX VAB_2nd_order_cosine_jax
    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_jax(self, allow_antiparallel):
        alpha = 0.5
        c1_np, c2_np, v1_np, v2_np, _, _, _, _, _, _, _, _ = _generate_data(fixed_seed=107) # New fixed seed
        
        c1_jnp = jnp.array(c1_np)
        c2_jnp = jnp.array(c2_np)
        v1_jnp = jnp.array(v1_np)
        v2_jnp = jnp.array(v2_np)

        expected_np = go_np.VAB_2nd_order_cosine_np(c1_np, c2_np, v1_np, v2_np, alpha, allow_antiparallel)
        actual_jax = go_jax.VAB_2nd_order_cosine_jax(c1_jnp, c2_jnp, v1_jnp, v2_jnp, alpha, allow_antiparallel)
        
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5, atol=1e-7)

#############################################################################
####################### JAX VMAP Batch Tests ##############################
#############################################################################

@pytest.mark.jax
class TestGaussianOverlapJAXBatched:
    def test_get_overlap_jax_vmap_batch(self):
        alpha = 0.81
        batch_size = 3
        c1_np_b, c2_np_b, _, _, _, _, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=201)
        
        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)

        # Vmap the JAX function: in_axes specifies batching for c1, c2; alpha is scalar (None)
        vmapped_get_overlap_jax = jax.vmap(go_jax.get_overlap_jax, in_axes=(0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(go_np.get_overlap_np(c1_np_b[i], c2_np_b[i], alpha=alpha))
        expected_np_array = np.array(expected_np_b)
        
        actual_jax_b = vmapped_get_overlap_jax(c1_jnp_b, c2_jnp_b, alpha)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)

    def test_VAB_2nd_order_jax_vmap_batch(self):
        alpha = 0.7
        batch_size = 3
        c1_np_b, c2_np_b, _, _, _, _, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=202)

        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)

        vmapped_VAB_jax = jax.vmap(go_jax.VAB_2nd_order_jax, in_axes=(0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(go_np.VAB_2nd_order_np(c1_np_b[i], c2_np_b[i], alpha))
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_VAB_jax(c1_jnp_b, c2_jnp_b, alpha)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)

    def test_shape_tanimoto_jax_vmap_batch(self):
        alpha = 0.6
        batch_size = 3
        c1_np_b, c2_np_b, _, _, _, _, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=203)

        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)

        vmapped_tanimoto_jax = jax.vmap(go_jax.shape_tanimoto_jax, in_axes=(0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(go_np.shape_tanimoto_np(c1_np_b[i], c2_np_b[i], alpha))
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_tanimoto_jax(c1_jnp_b, c2_jnp_b, alpha)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)

    def test_get_overlap_jax_mask_vmap_batch(self):
        alpha = 0.75
        batch_size = 3
        c1_np_b, c2_np_b, _, _, m1_np_b, m2_np_b, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=204)

        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)
        m1_jnp_b = jnp.array(m1_np_b)
        m2_jnp_b = jnp.array(m2_np_b)

        # Vmap: centers and masks are batched (axis 0), alpha is scalar (None)
        vmapped_get_overlap_mask_jax = jax.vmap(go_jax.get_overlap_jax_mask, in_axes=(0, 0, 0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            c1_f = c1_np_b[i][m1_np_b[i].astype(bool)]
            c2_f = c2_np_b[i][m2_np_b[i].astype(bool)]
            if c1_f.shape[0] == 0 or c2_f.shape[0] == 0:
                expected_val = np.array(0.0, dtype=np.float32)
            else:
                expected_val = go_np.get_overlap_np(c1_f, c2_f, alpha=alpha)
            expected_np_b.append(expected_val)
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_get_overlap_mask_jax(c1_jnp_b, c2_jnp_b, m1_jnp_b, m2_jnp_b, alpha)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)

    def test_VAB_2nd_order_jax_mask_vmap_batch(self):
        alpha = 0.65
        batch_size = 3
        c1_np_b, c2_np_b, _, _, m1_np_b, m2_np_b, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=205)

        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)
        m1_jnp_b = jnp.array(m1_np_b)
        m2_jnp_b = jnp.array(m2_np_b)

        vmapped_VAB_mask_jax = jax.vmap(go_jax.VAB_2nd_order_jax_mask, in_axes=(0, 0, 0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            c1_f = c1_np_b[i][m1_np_b[i].astype(bool)]
            c2_f = c2_np_b[i][m2_np_b[i].astype(bool)]
            if c1_f.shape[0] == 0 or c2_f.shape[0] == 0:
                expected_val = np.array(0.0, dtype=np.float32)
            else:
                expected_val = go_np.VAB_2nd_order_np(c1_f, c2_f, alpha)
            expected_np_b.append(expected_val)
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_VAB_mask_jax(c1_jnp_b, c2_jnp_b, m1_jnp_b, m2_jnp_b, alpha)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)

    def test_shape_tanimoto_jax_mask_vmap_batch(self):
        alpha = 0.55
        batch_size = 2 # Using a smaller batch size for this more complex ground truth
        c1_np_b, c2_np_b, _, _, m1_np_b, m2_np_b, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=206)

        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)
        m1_jnp_b = jnp.array(m1_np_b)
        m2_jnp_b = jnp.array(m2_np_b)

        vmapped_tanimoto_mask_jax = jax.vmap(go_jax.shape_tanimoto_jax_mask, in_axes=(0, 0, 0, 0, None))

        expected_np_b = []
        for i in range(batch_size):
            # Ground truth for VAA_mask
            c1_f_m1 = c1_np_b[i][m1_np_b[i].astype(bool)]
            VAA_np_mask = go_np.VAB_2nd_order_np(c1_f_m1, c1_f_m1, alpha) if c1_f_m1.shape[0] > 0 else 0.0
            # Ground truth for VBB_mask
            c2_f_m2 = c2_np_b[i][m2_np_b[i].astype(bool)]
            VBB_np_mask = go_np.VAB_2nd_order_np(c2_f_m2, c2_f_m2, alpha) if c2_f_m2.shape[0] > 0 else 0.0
            # Ground truth for VAB_mask
            VAB_np_mask = go_np.VAB_2nd_order_np(c1_f_m1, c2_f_m2, alpha) if (c1_f_m1.shape[0] > 0 and c2_f_m2.shape[0] > 0) else 0.0
            
            denominator = VAA_np_mask + VBB_np_mask - VAB_np_mask
            if denominator == 0:
                expected_val = np.array(0.0, dtype=np.float32) if VAB_np_mask == 0 else np.nan
            else:
                expected_val = VAB_np_mask / denominator
            expected_np_b.append(expected_val)
        expected_np_array = np.array(expected_np_b, dtype=np.float32) # Ensure dtype for comparison

        actual_jax_b = vmapped_tanimoto_mask_jax(c1_jnp_b, c2_jnp_b, m1_jnp_b, m2_jnp_b, alpha)
        
        # Handle NaN comparisons for Tanimoto
        nan_mask_expected = np.isnan(expected_np_array)
        nan_mask_actual = np.isnan(np.array(actual_jax_b))
        assert np.array_equal(nan_mask_expected, nan_mask_actual), "NaN presence differs"
        np.testing.assert_allclose(
            np.array(actual_jax_b)[~nan_mask_actual],
            expected_np_array[~nan_mask_expected],
            rtol=1e-5, atol=1e-7
        )

    @pytest.mark.parametrize("allow_antiparallel", [False, True])
    def test_VAB_2nd_order_cosine_jax_vmap_batch(self, allow_antiparallel):
        alpha = 0.5
        batch_size = 3
        c1_np_b, c2_np_b, v1_np_b, v2_np_b, _, _, _, _, _, _, _, _ = _generate_data(batch_size=batch_size, fixed_seed=207)
        
        c1_jnp_b = jnp.array(c1_np_b)
        c2_jnp_b = jnp.array(c2_np_b)
        v1_jnp_b = jnp.array(v1_np_b)
        v2_jnp_b = jnp.array(v2_np_b)

        # allow_antiparallel is static so it's not in in_axes, but passed to the vmapped function
        # The JAX function _VAB_2nd_order_cosine_jax_impl itself is JITted with static_argnames
        # So vmap will operate on these specialized JITted functions.
        vmapped_VAB_cosine_jax = jax.vmap(go_jax.VAB_2nd_order_cosine_jax, 
                                          in_axes=(0, 0, 0, 0, None, None) # c1, c2, v1, v2, alpha, allow_antiparallel
                                         )
        # Simpler: JAX's vmap is smart enough about static args from the underlying JITted function.
        # The main thing is passing allow_antiparallel as a concrete value to the vmapped call.
        # jit on VAB_2nd_order_cosine_jax already handles specializing for allow_antiparallel.
        # vmap will then map over these specialized versions if they differ per batch (which they don't here as it's one value for the whole batch).
        # Let's try direct approach for in_axes first.
        # vmapped_VAB_cosine_jax = jax.vmap(go_jax.VAB_2nd_order_cosine_jax, in_axes=(0, 0, 0, 0, None, None)) # c1,c2,v1,v2,alpha,allow_antiparallel

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(go_np.VAB_2nd_order_cosine_np(c1_np_b[i], c2_np_b[i], v1_np_b[i], v2_np_b[i], alpha, allow_antiparallel))
        expected_np_array = np.array(expected_np_b)
        
        # Pass allow_antiparallel as a normal argument. JAX handles its static nature due to original @jit.
        actual_jax_b = vmapped_VAB_cosine_jax(c1_jnp_b, c2_jnp_b, v1_jnp_b, v2_jnp_b, alpha, allow_antiparallel)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5, atol=1e-7)
