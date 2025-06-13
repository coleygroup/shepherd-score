"""
Unit tests for the electrostatic_scoring module (PyTorch and JAX implementations).
Assumes the NumPy implementation is ground truth.

Particularly for the batching functionality that complicates the PyTorch implementation.
"""

import pytest
import torch
import numpy as np
from shepherd_score.score import electrostatic_scoring as es
from shepherd_score.score import electrostatic_scoring_np as es_np
from shepherd_score.score.constants import LAM_SCALING # For default lam values
from shepherd_score.score.gaussian_overlap_np import get_overlap_np # For esp_combo_score ground truth

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
es_jax = pytest.importorskip("shepherd_score.score.electrostatic_scoring_jax")


# Helper function to generate random electrostatic data
def _generate_electrostatic_data(batch_size=None, 
                                 num_centers1=10, num_centers2=12, 
                                 num_atoms_w_H1=15, num_atoms_w_H2=18,
                                 num_surf_points1=20, num_surf_points2=22,
                                 dim=3, fixed_seed=None):
    if fixed_seed is not None:
        np.random.seed(fixed_seed)

    def _generate_coords(count, bs, d):
        if bs:
            return np.random.rand(bs, count, d).astype(np.float32)
        return np.random.rand(count, d).astype(np.float32)

    def _generate_charges_or_radii(count, bs):
        if bs:
            return np.random.rand(bs, count).astype(np.float32)
        return np.random.rand(count).astype(np.float32)

    # NumPy arrays
    centers_1_np = _generate_coords(num_centers1, batch_size, dim)
    centers_2_np = _generate_coords(num_centers2, batch_size, dim)
    charges_overlap_1_np = _generate_charges_or_radii(num_centers1, batch_size) 
    charges_overlap_2_np = _generate_charges_or_radii(num_centers2, batch_size)
    centers_w_H_1_np = _generate_coords(num_atoms_w_H1, batch_size, dim)
    centers_w_H_2_np = _generate_coords(num_atoms_w_H2, batch_size, dim)
    partial_charges_1_np = _generate_charges_or_radii(num_atoms_w_H1, batch_size)
    partial_charges_2_np = _generate_charges_or_radii(num_atoms_w_H2, batch_size)
    points_1_np = _generate_coords(num_surf_points1, batch_size, dim)
    points_2_np = _generate_coords(num_surf_points2, batch_size, dim)
    point_charges_1_np = _generate_charges_or_radii(num_surf_points1, batch_size)
    point_charges_2_np = _generate_charges_or_radii(num_surf_points2, batch_size)
    radii_1_np = _generate_charges_or_radii(num_atoms_w_H1, batch_size)
    radii_2_np = _generate_charges_or_radii(num_atoms_w_H2, batch_size)

    # PyTorch tensors
    centers_1_torch = torch.from_numpy(centers_1_np)
    centers_2_torch = torch.from_numpy(centers_2_np)
    charges_overlap_1_torch = torch.from_numpy(charges_overlap_1_np)
    charges_overlap_2_torch = torch.from_numpy(charges_overlap_2_np)
    centers_w_H_1_torch = torch.from_numpy(centers_w_H_1_np)
    centers_w_H_2_torch = torch.from_numpy(centers_w_H_2_np)
    partial_charges_1_torch = torch.from_numpy(partial_charges_1_np)
    partial_charges_2_torch = torch.from_numpy(partial_charges_2_np)
    points_1_torch = torch.from_numpy(points_1_np)
    points_2_torch = torch.from_numpy(points_2_np)
    point_charges_1_torch = torch.from_numpy(point_charges_1_np)
    point_charges_2_torch = torch.from_numpy(point_charges_2_np)
    radii_1_torch = torch.from_numpy(radii_1_np)
    radii_2_torch = torch.from_numpy(radii_2_np)

    # JAX arrays
    centers_1_jnp = jnp.array(centers_1_np)
    centers_2_jnp = jnp.array(centers_2_np)
    charges_overlap_1_jnp = jnp.array(charges_overlap_1_np)
    charges_overlap_2_jnp = jnp.array(charges_overlap_2_np)
    centers_w_H_1_jnp = jnp.array(centers_w_H_1_np)
    centers_w_H_2_jnp = jnp.array(centers_w_H_2_np)
    partial_charges_1_jnp = jnp.array(partial_charges_1_np)
    partial_charges_2_jnp = jnp.array(partial_charges_2_np)
    points_1_jnp = jnp.array(points_1_np)
    points_2_jnp = jnp.array(points_2_np)
    point_charges_1_jnp = jnp.array(point_charges_1_np)
    point_charges_2_jnp = jnp.array(point_charges_2_np)
    radii_1_jnp = jnp.array(radii_1_np)
    radii_2_jnp = jnp.array(radii_2_np)

    return (centers_1_np, centers_2_np, charges_overlap_1_np, charges_overlap_2_np,
            centers_w_H_1_np, centers_w_H_2_np, partial_charges_1_np, partial_charges_2_np,
            points_1_np, points_2_np, point_charges_1_np, point_charges_2_np,
            radii_1_np, radii_2_np,
            centers_1_torch, centers_2_torch, charges_overlap_1_torch, charges_overlap_2_torch,
            centers_w_H_1_torch, centers_w_H_2_torch, partial_charges_1_torch, partial_charges_2_torch,
            points_1_torch, points_2_torch, point_charges_1_torch, point_charges_2_torch,
            radii_1_torch, radii_2_torch,
            centers_1_jnp, centers_2_jnp, charges_overlap_1_jnp, charges_overlap_2_jnp,
            centers_w_H_1_jnp, centers_w_H_2_jnp, partial_charges_1_jnp, partial_charges_2_jnp,
            points_1_jnp, points_2_jnp, point_charges_1_jnp, point_charges_2_jnp,
            radii_1_jnp, radii_2_jnp)


# Fixed parameters for tests
ALPHA_ESP = 0.81
LAM_ESP_OVERLAP = 0.3 * LAM_SCALING # Default for get_overlap_esp
LAM_ESP_COMPARISON = 0.001 # Default for _esp_comparison
PROBE_RADIUS = 1.0
ESP_WEIGHT = 0.5

class TestElectrostaticScoringTorch:
    def test_VAB_2nd_order_esp_single_instance(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         c1_torch, c2_torch, ch1_torch, ch2_torch, *_) = _generate_electrostatic_data(fixed_seed=1)

        expected = es_np.VAB_2nd_order_esp_np(c1_np, c2_np, ch1_np.reshape(-1,1), ch2_np.reshape(-1,1), alpha, lam)
        actual = es.VAB_2nd_order_esp(c1_torch, c2_torch, ch1_torch.reshape(-1,1), ch2_torch.reshape(-1,1), alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_VAB_2nd_order_esp_batched(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 3
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         c1_torch_b, c2_torch_b, ch1_torch_b, ch2_torch_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=2)

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np.VAB_2nd_order_esp_np(c1_np_b[i], c2_np_b[i], 
                                                           ch1_np_b[i].reshape(-1,1), ch2_np_b[i].reshape(-1,1), 
                                                           alpha, lam))
        expected = np.array(expected_batch)
        actual = es.VAB_2nd_order_esp(c1_torch_b, c2_torch_b, ch1_torch_b.unsqueeze(-1), ch2_torch_b.unsqueeze(-1), alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_shape_tanimoto_esp_single_instance(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         c1_torch, c2_torch, ch1_torch, ch2_torch, *_) = _generate_electrostatic_data(fixed_seed=3)
        
        expected = es_np.shape_tanimoto_esp_np(c1_np, c2_np, ch1_np.reshape(-1,1), ch2_np.reshape(-1,1), alpha, lam)
        actual = es.shape_tanimoto_esp(c1_torch, c2_torch, ch1_torch.reshape(-1,1), ch2_torch.reshape(-1,1), alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_shape_tanimoto_esp_batched(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 3
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         c1_torch_b, c2_torch_b, ch1_torch_b, ch2_torch_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=4)

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np.shape_tanimoto_esp_np(c1_np_b[i], c2_np_b[i], 
                                                            ch1_np_b[i].reshape(-1,1), ch2_np_b[i].reshape(-1,1), 
                                                            alpha, lam))
        expected = np.array(expected_batch)
        actual = es.shape_tanimoto_esp(c1_torch_b, c2_torch_b, ch1_torch_b.unsqueeze(-1), ch2_torch_b.unsqueeze(-1), alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_esp_single_instance(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         c1_torch, c2_torch, ch1_torch, ch2_torch, *_) = _generate_electrostatic_data(fixed_seed=5)

        expected = es_np.get_overlap_esp_np(c1_np, c2_np, ch1_np, ch2_np, alpha, lam)
        actual = es.get_overlap_esp(c1_torch, c2_torch, ch1_torch, ch2_torch, alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_esp_batched(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 3
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         c1_torch_b, c2_torch_b, ch1_torch_b, ch2_torch_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=6)

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np.get_overlap_esp_np(c1_np_b[i], c2_np_b[i], ch1_np_b[i], ch2_np_b[i], alpha, lam))
        expected = np.array(expected_batch)
        actual = es.get_overlap_esp(c1_torch_b, c2_torch_b, ch1_torch_b, ch2_torch_b, alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_esp_broadcast_c1_ch1(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 3
        num_centers1 = 10
        num_centers2 = 12
        data_full_batch = _generate_electrostatic_data(batch_size=batch_size, num_centers1=num_centers1, num_centers2=num_centers2, fixed_seed=7)
        (c1_np_orig_b, c2_np_b, ch1_np_orig_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         c1_torch_orig_b, c2_torch_b, ch1_torch_orig_b, ch2_torch_b, *_) = data_full_batch
        
        c1_np_single = c1_np_orig_b[0]
        ch1_np_single = ch1_np_orig_b[0]
        c1_torch_single = c1_torch_orig_b[0]
        ch1_torch_single = ch1_torch_orig_b[0]

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np.get_overlap_esp_np(c1_np_single, c2_np_b[i], ch1_np_single, ch2_np_b[i], alpha, lam))
        expected = np.array(expected_batch)
        actual = es.get_overlap_esp(c1_torch_single, c2_torch_b, ch1_torch_single, ch2_torch_b, alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_get_overlap_esp_broadcast_c2_ch2(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 3
        num_centers1 = 10
        num_centers2 = 12
        data_full_batch = _generate_electrostatic_data(batch_size=batch_size, num_centers1=num_centers1, num_centers2=num_centers2, fixed_seed=8)
        (c1_np_b, c2_np_orig_b, ch1_np_b, ch2_np_orig_b, _, _, _, _, _, _, _, _, _, _,
         c1_torch_b, c2_torch_orig_b, ch1_torch_b, ch2_torch_orig_b, *_) = data_full_batch

        c2_np_single = c2_np_orig_b[0]
        ch2_np_single = ch2_np_orig_b[0]
        c2_torch_single = c2_torch_orig_b[0]
        ch2_torch_single = ch2_torch_orig_b[0]

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np.get_overlap_esp_np(c1_np_b[i], c2_np_single, ch1_np_b[i], ch2_np_single, alpha, lam))
        expected = np.array(expected_batch)
        actual = es.get_overlap_esp(c1_torch_b, c2_torch_single, ch1_torch_b, ch2_torch_single, alpha, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_comparison_single_instance_all(self):
        """Test _esp_comparison with all inputs as single instances."""
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        data = _generate_electrostatic_data(fixed_seed=9)
        
        points_1_np = data[8]
        point_charges_1_np = data[10]
        centers_w_H_2_np = data[5]
        partial_charges_2_np = data[7]
        radii_2_np = data[13]

        points_1_torch = data[22]
        point_charges_1_torch = data[24]
        centers_w_H_2_torch = data[19]
        partial_charges_2_torch = data[21]
        radii_2_torch = data[27]

        expected = es_np._esp_comparison_np(points_1_np, centers_w_H_2_np, partial_charges_2_np, 
                                            point_charges_1_np, radii_2_np, probe_radius, lam)
        actual = es._esp_comparison(points_1_torch, centers_w_H_2_torch, partial_charges_2_torch, 
                                    point_charges_1_torch, radii_2_torch, probe_radius, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_comparison_points1_single_mol2_batched(self):
        """Test _esp_comparison with points_1 single, and molecule 2 data batched."""
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        batch_size = 2
        num_surf_points1 = 20 
        num_atoms_w_H2 = 18   

        data = _generate_electrostatic_data(batch_size=batch_size, 
                                            num_surf_points1=num_surf_points1, 
                                            num_atoms_w_H2=num_atoms_w_H2, 
                                            fixed_seed=10)
        points_1_np = data[8][0] 
        point_charges_1_np = data[10][0]
        points_1_torch = torch.from_numpy(points_1_np)
        point_charges_1_torch = torch.from_numpy(point_charges_1_np)

        centers_w_H_2_np_b = data[5]
        partial_charges_2_np_b = data[7]
        radii_2_np_b = data[13]
        centers_w_H_2_torch_b = data[19]
        partial_charges_2_torch_b = data[21]
        radii_2_torch_b = data[27]

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np._esp_comparison_np(points_1_np, centers_w_H_2_np_b[i], 
                                                           partial_charges_2_np_b[i], point_charges_1_np, 
                                                           radii_2_np_b[i], probe_radius, lam))
        expected = np.array(expected_batch)
        actual = es._esp_comparison(points_1_torch, centers_w_H_2_torch_b, partial_charges_2_torch_b, 
                                    point_charges_1_torch, radii_2_torch_b, probe_radius, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_comparison_points1_batched_mol2_single(self):
        """Test _esp_comparison with points_1 batched, and molecule 2 data single (broadcast)."""
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        batch_size = 2
        num_surf_points1 = 20 
        num_atoms_w_H2 = 18   

        data = _generate_electrostatic_data(batch_size=batch_size, 
                                            num_surf_points1=num_surf_points1, 
                                            num_atoms_w_H2=num_atoms_w_H2, 
                                            fixed_seed=11)
        points_1_np_b = data[8]
        point_charges_1_np_b = data[10]
        points_1_torch_b = data[22]
        point_charges_1_torch_b = data[24]

        centers_w_H_2_np_s = data[5][0] 
        partial_charges_2_np_s = data[7][0]
        radii_2_np_s = data[13][0]
        centers_w_H_2_torch_s = torch.from_numpy(centers_w_H_2_np_s)
        partial_charges_2_torch_s = torch.from_numpy(partial_charges_2_np_s)
        radii_2_torch_s = torch.from_numpy(radii_2_np_s)

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np._esp_comparison_np(points_1_np_b[i], centers_w_H_2_np_s, 
                                                           partial_charges_2_np_s, point_charges_1_np_b[i], 
                                                           radii_2_np_s, probe_radius, lam))
        expected = np.array(expected_batch)
        actual = es._esp_comparison(points_1_torch_b, centers_w_H_2_torch_s, partial_charges_2_torch_s, 
                                    point_charges_1_torch_b, radii_2_torch_s, probe_radius, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_comparison_all_batched(self):
        """Test _esp_comparison with all inputs batched."""
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        batch_size = 2
        data = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=12)

        points_1_np_b = data[8]
        point_charges_1_np_b = data[10]
        centers_w_H_2_np_b = data[5]
        partial_charges_2_np_b = data[7]
        radii_2_np_b = data[13]

        points_1_torch_b = data[22]
        point_charges_1_torch_b = data[24]
        centers_w_H_2_torch_b = data[19]
        partial_charges_2_torch_b = data[21]
        radii_2_torch_b = data[27]
        
        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(es_np._esp_comparison_np(points_1_np_b[i], centers_w_H_2_np_b[i], 
                                                           partial_charges_2_np_b[i], point_charges_1_np_b[i], 
                                                           radii_2_np_b[i], probe_radius, lam))
        expected = np.array(expected_batch)
        actual = es._esp_comparison(points_1_torch_b, centers_w_H_2_torch_b, partial_charges_2_torch_b, 
                                    point_charges_1_torch_b, radii_2_torch_b, probe_radius, lam)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_combo_score_single_instance(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_COMPARISON 
        probe_radius = PROBE_RADIUS
        esp_weight = ESP_WEIGHT

        (centers_overlap_1_np, centers_overlap_2_np, _, _, 
        centers_w_H_1_np, centers_w_H_2_np, partial_charges_1_np, partial_charges_2_np,
        points_1_np, points_2_np, point_charges_1_np, point_charges_2_np,
        radii_1_np, radii_2_np,
        centers_overlap_1_torch, centers_overlap_2_torch, _, _,
        centers_w_H_1_torch, centers_w_H_2_torch, partial_charges_1_torch, partial_charges_2_torch,
        points_1_torch, points_2_torch, point_charges_1_torch, point_charges_2_torch,
        radii_1_torch, radii_2_torch, *_) = _generate_electrostatic_data(fixed_seed=13, num_centers1=5, num_centers2=6)

        expected = es_np.esp_combo_score_np(centers_w_H_1_np, centers_w_H_2_np, 
                                              centers_overlap_1_np, centers_overlap_2_np,
                                              points_1_np, points_2_np, 
                                              partial_charges_1_np, partial_charges_2_np,
                                              point_charges_1_np, point_charges_2_np,
                                              radii_1_np, radii_2_np,
                                              alpha, lam, probe_radius, esp_weight)
        actual = es.esp_combo_score(centers_w_H_1_torch, centers_w_H_2_torch, 
                                    centers_overlap_1_torch, centers_overlap_2_torch,
                                    points_1_torch, points_2_torch, 
                                    partial_charges_1_torch, partial_charges_2_torch,
                                    point_charges_1_torch, point_charges_2_torch,
                                    radii_1_torch, radii_2_torch,
                                    alpha, lam, probe_radius, esp_weight)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

    def test_esp_combo_score_batched(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        esp_weight = ESP_WEIGHT
        batch_size = 2

        data_b = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=14, num_centers1=5, num_centers2=6)
        (centers_overlap_1_np_b, centers_overlap_2_np_b, _, _, 
        centers_w_H_1_np_b, centers_w_H_2_np_b, partial_charges_1_np_b, partial_charges_2_np_b,
        points_1_np_b, points_2_np_b, point_charges_1_np_b, point_charges_2_np_b,
        radii_1_np_b, radii_2_np_b,
        centers_overlap_1_torch_b, centers_overlap_2_torch_b, _, _,
        centers_w_H_1_torch_b, centers_w_H_2_torch_b, partial_charges_1_torch_b, partial_charges_2_torch_b,
        points_1_torch_b, points_2_torch_b, point_charges_1_torch_b, point_charges_2_torch_b,
        radii_1_torch_b, radii_2_torch_b, *_) = data_b

        expected_batch = []
        for i in range(batch_size):
            expected_batch.append(
                es_np.esp_combo_score_np(centers_w_H_1_np_b[i], centers_w_H_2_np_b[i], 
                                         centers_overlap_1_np_b[i], centers_overlap_2_np_b[i],
                                         points_1_np_b[i], points_2_np_b[i], 
                                         partial_charges_1_np_b[i], partial_charges_2_np_b[i],
                                         point_charges_1_np_b[i], point_charges_2_np_b[i],
                                         radii_1_np_b[i], radii_2_np_b[i],
                                         alpha, lam, probe_radius, esp_weight)
            )
        expected = np.array(expected_batch)
        actual = es.esp_combo_score(centers_w_H_1_torch_b, centers_w_H_2_torch_b, 
                                    centers_overlap_1_torch_b, centers_overlap_2_torch_b,
                                    points_1_torch_b, points_2_torch_b, 
                                    partial_charges_1_torch_b, partial_charges_2_torch_b,
                                    point_charges_1_torch_b, point_charges_2_torch_b,
                                    radii_1_torch_b, radii_2_torch_b,
                                    alpha, lam, probe_radius, esp_weight)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)

# Placeholder for JAX tests
@pytest.mark.jax
class TestElectrostaticScoringJAX:
    def test_VAB_2nd_order_esp_jax(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp, c2_jnp, ch1_jnp, ch2_jnp, *_) = _generate_electrostatic_data(fixed_seed=101)

        expected_np = es_np.VAB_2nd_order_esp_np(c1_np, c2_np, ch1_np.reshape(-1,1), ch2_np.reshape(-1,1), alpha, lam)
        # JAX charges are (N,) and need to be (N,1) for cdist as in numpy version
        actual_jax = es_jax.VAB_2nd_order_esp_jax(c1_jnp, c2_jnp, ch1_jnp.reshape(-1,1), ch2_jnp.reshape(-1,1), alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5)

    def test_shape_tanimoto_esp_jax(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp, c2_jnp, ch1_jnp, ch2_jnp, *_) = _generate_electrostatic_data(fixed_seed=102)
        
        expected_np = es_np.shape_tanimoto_esp_np(c1_np, c2_np, ch1_np.reshape(-1,1), ch2_np.reshape(-1,1), alpha, lam)
        # JAX charges are (N,) and need to be (N,1) for cdist as in numpy version for underlying VAB calls
        actual_jax = es_jax.shape_tanimoto_esp_jax(c1_jnp, c2_jnp, ch1_jnp.reshape(-1,1), ch2_jnp.reshape(-1,1), alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5)

    def test_get_overlap_esp_jax(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        (c1_np, c2_np, ch1_np, ch2_np, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp, c2_jnp, ch1_jnp, ch2_jnp, *_) = _generate_electrostatic_data(fixed_seed=103)

        expected_np = es_np.get_overlap_esp_np(c1_np, c2_np, ch1_np, ch2_np, alpha, lam)
        # get_overlap_esp_jax is expected to handle reshaping of charges if necessary, like PyTorch version
        actual_jax = es_jax.get_overlap_esp_jax(c1_jnp, c2_jnp, ch1_jnp, ch2_jnp, alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5)

    def test_esp_comparison_jax(self):
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        data = _generate_electrostatic_data(fixed_seed=104)
        
        points_1_np = data[8]
        point_charges_1_np = data[10]
        centers_w_H_2_np = data[5]
        partial_charges_2_np = data[7]
        radii_2_np = data[13]

        points_1_jnp = data[36]
        point_charges_1_jnp = data[38]
        centers_w_H_2_jnp = data[33]
        partial_charges_2_jnp = data[35]
        radii_2_jnp = data[41]

        expected_np = es_np._esp_comparison_np(points_1_np, centers_w_H_2_np, partial_charges_2_np, 
                                               point_charges_1_np, radii_2_np, probe_radius, lam)
        actual_jax = es_jax._esp_comparison_jax(points_1_jnp, centers_w_H_2_jnp, partial_charges_2_jnp, 
                                              point_charges_1_jnp, radii_2_jnp, probe_radius, lam)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5)

    def test_esp_combo_score_jax(self):
        alpha = ALPHA_ESP
        lam_comp = LAM_ESP_COMPARISON 
        probe_radius = PROBE_RADIUS
        esp_weight = ESP_WEIGHT

        data = _generate_electrostatic_data(fixed_seed=105, num_centers1=5, num_centers2=6)
        (centers_overlap_1_np, centers_overlap_2_np, _, _, 
        centers_w_H_1_np, centers_w_H_2_np, partial_charges_1_np, partial_charges_2_np,
        points_1_np, points_2_np, point_charges_1_np, point_charges_2_np,
        radii_1_np, radii_2_np,
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,
        centers_overlap_1_jnp, centers_overlap_2_jnp, _, _, 
        centers_w_H_1_jnp, centers_w_H_2_jnp, partial_charges_1_jnp, partial_charges_2_jnp,
        points_1_jnp, points_2_jnp, point_charges_1_jnp, point_charges_2_jnp,
        radii_1_jnp, radii_2_jnp) = data

        expected_np = es_np.esp_combo_score_np(centers_w_H_1_np, centers_w_H_2_np, 
                                             centers_overlap_1_np, centers_overlap_2_np,
                                             points_1_np, points_2_np, 
                                             partial_charges_1_np, partial_charges_2_np,
                                             point_charges_1_np, point_charges_2_np,
                                             radii_1_np, radii_2_np,
                                             alpha, lam_comp, probe_radius, esp_weight)
        actual_jax = es_jax.esp_combo_score_jax(centers_w_H_1_jnp, centers_w_H_2_jnp, 
                                           centers_overlap_1_jnp, centers_overlap_2_jnp,
                                           points_1_jnp, points_2_jnp, 
                                           partial_charges_1_jnp, partial_charges_2_jnp,
                                           point_charges_1_jnp, point_charges_2_jnp,
                                           radii_1_jnp, radii_2_jnp,
                                           alpha, lam_comp, probe_radius, esp_weight)
        np.testing.assert_allclose(np.array(actual_jax), expected_np, rtol=1e-5)

@pytest.mark.jax
class TestElectrostaticScoringJAXBatched:
    def test_VAB_2nd_order_esp_jax_vmap_batch(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 2
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp_b, c2_jnp_b, ch1_jnp_b, ch2_jnp_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=201)

        # Vmap: centers and charges are batched (axis 0), alpha and lam are scalar (None)
        # Charges need to be (B, N, 1) for the cdist within the JAX function if it mirrors NumPy version
        vmapped_VAB_jax = jax.vmap(es_jax.VAB_2nd_order_esp_jax, 
                                   in_axes=(0, 0, 0, 0, None, None)) # c1, c2, ch1, ch2, alpha, lam

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(es_np.VAB_2nd_order_esp_np(c1_np_b[i], c2_np_b[i], 
                                                            ch1_np_b[i].reshape(-1,1), ch2_np_b[i].reshape(-1,1), 
                                                            alpha, lam))
        expected_np_array = np.array(expected_np_b)
        
        actual_jax_b = vmapped_VAB_jax(c1_jnp_b, c2_jnp_b, ch1_jnp_b.reshape(batch_size,-1,1), ch2_jnp_b.reshape(batch_size,-1,1), alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5)

    def test_shape_tanimoto_esp_jax_vmap_batch(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 2
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp_b, c2_jnp_b, ch1_jnp_b, ch2_jnp_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=202)

        vmapped_tanimoto_jax = jax.vmap(es_jax.shape_tanimoto_esp_jax, 
                                        in_axes=(0, 0, 0, 0, None, None)) # c1, c2, ch1, ch2, alpha, lam

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(es_np.shape_tanimoto_esp_np(c1_np_b[i], c2_np_b[i], 
                                                             ch1_np_b[i].reshape(-1,1), ch2_np_b[i].reshape(-1,1), 
                                                             alpha, lam))
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_tanimoto_jax(c1_jnp_b, c2_jnp_b, ch1_jnp_b.reshape(batch_size,-1,1), ch2_jnp_b.reshape(batch_size,-1,1), alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5)

    def test_get_overlap_esp_jax_vmap_batch(self):
        alpha = ALPHA_ESP
        lam = LAM_ESP_OVERLAP
        batch_size = 2
        (c1_np_b, c2_np_b, ch1_np_b, ch2_np_b, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         c1_jnp_b, c2_jnp_b, ch1_jnp_b, ch2_jnp_b, *_) = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=203)
        
        vmapped_get_overlap_jax = jax.vmap(es_jax.get_overlap_esp_jax, 
                                           in_axes=(0, 0, 0, 0, None, None)) # c1, c2, ch1, ch2, alpha, lam

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(es_np.get_overlap_esp_np(c1_np_b[i], c2_np_b[i], ch1_np_b[i], ch2_np_b[i], alpha, lam))
        expected_np_array = np.array(expected_np_b)
        
        actual_jax_b = vmapped_get_overlap_jax(c1_jnp_b, c2_jnp_b, ch1_jnp_b, ch2_jnp_b, alpha, lam)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5)

    def test_esp_comparison_jax_vmap_batch(self):
        lam = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        batch_size = 2
        data = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=204)
        
        points_1_np_b = data[8]
        centers_w_H_2_np_b = data[5]
        partial_charges_2_np_b = data[7]
        point_charges_1_np_b = data[10]
        radii_2_np_b = data[13]

        points_1_jnp_b = data[36]
        centers_w_H_2_jnp_b = data[33]
        partial_charges_2_jnp_b = data[35]
        point_charges_1_jnp_b = data[38]
        radii_2_jnp_b = data[41]

        vmapped_esp_comp_jax = jax.vmap(es_jax._esp_comparison_jax, 
                                        in_axes=(0, 0, 0, 0, 0, None, None)) # p1, c_w_H2, pc2, pc1, r2, probe, lam

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(es_np._esp_comparison_np(points_1_np_b[i], centers_w_H_2_np_b[i], 
                                                           partial_charges_2_np_b[i], point_charges_1_np_b[i], 
                                                           radii_2_np_b[i], probe_radius, lam))
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_esp_comp_jax(points_1_jnp_b, centers_w_H_2_jnp_b, partial_charges_2_jnp_b, 
                                            point_charges_1_jnp_b, radii_2_jnp_b, probe_radius, lam)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5)

    def test_esp_combo_score_jax_vmap_batch(self):
        alpha = ALPHA_ESP
        lam_comp = LAM_ESP_COMPARISON
        probe_radius = PROBE_RADIUS
        esp_weight = ESP_WEIGHT
        batch_size = 2

        data_b = _generate_electrostatic_data(batch_size=batch_size, fixed_seed=205, num_centers1=5, num_centers2=6)
        (centers_overlap_1_np_b, centers_overlap_2_np_b, _, _, 
        centers_w_H_1_np_b, centers_w_H_2_np_b, partial_charges_1_np_b, partial_charges_2_np_b,
        points_1_np_b, points_2_np_b, point_charges_1_np_b, point_charges_2_np_b,
        radii_1_np_b, radii_2_np_b,
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,
        centers_overlap_1_jnp_b, centers_overlap_2_jnp_b, _, _, 
        centers_w_H_1_jnp_b, centers_w_H_2_jnp_b, partial_charges_1_jnp_b, partial_charges_2_jnp_b,
        points_1_jnp_b, points_2_jnp_b, point_charges_1_jnp_b, point_charges_2_jnp_b,
        radii_1_jnp_b, radii_2_jnp_b) = data_b
        
        vmapped_combo_jax = jax.vmap(es_jax.esp_combo_score_jax,
                                     in_axes=(0,0,0,0,0,0,0,0,0,0,0,0, None, None, None, None))

        expected_np_b = []
        for i in range(batch_size):
            expected_np_b.append(
                es_np.esp_combo_score_np(centers_w_H_1_np_b[i], centers_w_H_2_np_b[i], 
                                         centers_overlap_1_np_b[i], centers_overlap_2_np_b[i],
                                         points_1_np_b[i], points_2_np_b[i], 
                                         partial_charges_1_np_b[i], partial_charges_2_np_b[i],
                                         point_charges_1_np_b[i], point_charges_2_np_b[i],
                                         radii_1_np_b[i], radii_2_np_b[i],
                                         alpha, lam_comp, probe_radius, esp_weight)
            )
        expected_np_array = np.array(expected_np_b)

        actual_jax_b = vmapped_combo_jax(centers_w_H_1_jnp_b, centers_w_H_2_jnp_b, 
                                           centers_overlap_1_jnp_b, centers_overlap_2_jnp_b,
                                           points_1_jnp_b, points_2_jnp_b, 
                                           partial_charges_1_jnp_b, partial_charges_2_jnp_b,
                                           point_charges_1_jnp_b, point_charges_2_jnp_b,
                                           radii_1_jnp_b, radii_2_jnp_b,
                                           alpha, lam_comp, probe_radius, esp_weight)
        np.testing.assert_allclose(np.array(actual_jax_b), expected_np_array, rtol=1e-5)
