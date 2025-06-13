"""
Tests for alignment function consistency between PyTorch and Jax.
"""
import os
import pytest
import torch
import numpy as np
import jax.numpy as jnp

from shepherd_score.alignment import (
    optimize_ROCS_overlay,
    optimize_ROCS_esp_overlay,
    optimize_esp_combo_score_overlay,
    optimize_pharm_overlay,
)
from shepherd_score.alignment_jax import (
    optimize_ROCS_overlay_jax,
    optimize_ROCS_esp_overlay_jax,
    optimize_esp_combo_score_overlay_jax,
    optimize_pharm_overlay_jax,
)
from shepherd_score.alignment_jax import convert_to_jnp_array
from shepherd_score.score.constants import P_TYPES

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

# Helper to convert JAX output to torch for comparison
def jax_outputs_to_torch(aligned_points_jax, transform_jax, score_jax):
    aligned_points_torch_equiv = torch.from_numpy(np.array(aligned_points_jax))
    transform_torch_equiv = torch.from_numpy(np.array(transform_jax))
    score_torch_equiv = torch.from_numpy(np.array(score_jax))
    return aligned_points_torch_equiv, transform_torch_equiv, score_torch_equiv

def jax_pharm_outputs_to_torch(aligned_anchors_jax, aligned_vectors_jax, transform_jax, score_jax):
    aligned_anchors_torch_equiv = torch.from_numpy(np.array(aligned_anchors_jax))
    aligned_vectors_torch_equiv = torch.from_numpy(np.array(aligned_vectors_jax))
    transform_torch_equiv = torch.from_numpy(np.array(transform_jax))
    score_torch_equiv = torch.from_numpy(np.array(score_jax))
    return aligned_anchors_torch_equiv, aligned_vectors_torch_equiv, transform_torch_equiv, score_torch_equiv

# Define default parameters for tests
DEFAULT_ALPHA = 1.0
DEFAULT_LAM = 0.1 # For ESP
DEFAULT_LR = 0.05 # Smaller LR might give more stable/comparable results in tests
DEFAULT_MAX_STEPS = 50 # Fewer steps for faster tests and to limit divergence due to small numerical diffs
DEFAULT_VERBOSE = False
DEFAULT_DTYPE = torch.float32

# Tolerances - These may need careful tuning!
# Optimization paths can diverge. Start with moderate tolerances.
COORD_ATOL = 1e-3
COORD_RTOL = 1e-2
TRANSFORM_ATOL = 1e-3
TRANSFORM_RTOL = 1e-2
SCORE_ATOL = 1e-3
SCORE_RTOL = 1e-2


# --- Fixtures for Data Generation ---
@pytest.fixture(scope="module") # Use module scope if data generation is heavy and doesn't need per-test change
def common_points_data():
    torch.manual_seed(0)
    np.random.seed(0) # JAX uses numpy for array conversion from torch
    ref_points = torch.rand(20, 3, dtype=DEFAULT_DTYPE) * 10
    fit_points = torch.rand(15, 3, dtype=DEFAULT_DTYPE) * 10
    
    ref_points_jax = jnp.array(ref_points.numpy())
    fit_points_jax = jnp.array(fit_points.numpy())
    return ref_points, fit_points, ref_points_jax, fit_points_jax

@pytest.fixture(scope="module")
def common_charges_data(common_points_data):
    ref_points_torch, fit_points_torch, _, _ = common_points_data
    ref_charges = torch.randn(ref_points_torch.shape[0], dtype=DEFAULT_DTYPE)
    fit_charges = torch.randn(fit_points_torch.shape[0], dtype=DEFAULT_DTYPE)

    ref_charges_jax = jnp.array(ref_charges.numpy())
    fit_charges_jax = jnp.array(fit_charges.numpy())
    return ref_charges, fit_charges, ref_charges_jax, fit_charges_jax

@pytest.fixture(scope="module")
def common_trans_centers_data():
    trans_centers = torch.rand(2, 3, dtype=DEFAULT_DTYPE) * 5
    trans_centers_jax = jnp.array(trans_centers.numpy())
    return trans_centers, trans_centers_jax

@pytest.fixture(scope="module")
def common_esp_combo_data(common_points_data, common_charges_data):
    ref_points_torch, fit_points_torch, ref_points_jax, fit_points_jax = common_points_data
    # For esp_combo, 'points' usually means surface points, 'centers' means atom centers
    
    # Simulate additional points for hydrogens
    num_extra_ref_h = 5
    num_extra_fit_h = 4

    ref_centers_w_H_torch = torch.cat([
        ref_points_torch, 
        torch.rand(num_extra_ref_h, 3, dtype=DEFAULT_DTYPE) * 10 + 0.1 # Additional H atoms
    ], dim=0)
    fit_centers_w_H_torch = torch.cat([
        fit_points_torch,
        torch.rand(num_extra_fit_h, 3, dtype=DEFAULT_DTYPE) * 10 + 0.1 # Additional H atoms
    ], dim=0)

    # Simplified placeholder data - this needs to be realistic for meaningful tests
    data = {
        "ref_centers_w_H": ref_centers_w_H_torch, 
        "fit_centers_w_H": fit_centers_w_H_torch,
        "ref_centers": ref_points_torch, "fit_centers": fit_points_torch, # Heavy atoms as centers
        "ref_points": ref_points_torch, "fit_points": fit_points_torch, # Surface points (could be same as heavy atom centers or different)
        "ref_partial_charges": torch.randn(ref_centers_w_H_torch.shape[0], dtype=DEFAULT_DTYPE), # Match shape of centers_w_H
        "fit_partial_charges": torch.randn(fit_centers_w_H_torch.shape[0], dtype=DEFAULT_DTYPE),
        "ref_surf_esp": common_charges_data[0], "fit_surf_esp": common_charges_data[1], # ESP on surface points
        "ref_radii": torch.rand(ref_centers_w_H_torch.shape[0], dtype=DEFAULT_DTYPE) + 1.2, # Radii in [1.2, 2.2)
        "fit_radii": torch.rand(fit_centers_w_H_torch.shape[0], dtype=DEFAULT_DTYPE) + 1.2, # Radii in [1.2, 2.2)
        "alpha": DEFAULT_ALPHA, "lam": DEFAULT_LAM, 
        "probe_radius": 1.0, "esp_weight": 0.5
    }

    # Create JAX equivalents. Note: optimize_esp_combo_score_overlay_jax handles internal conversion.
    data_jax = {key: convert_to_jnp_array(val) if isinstance(val, torch.Tensor) else val for key, val in data.items()}
    
    # Return both PyTorch and JAX versions of the data dictionary
    return data, data_jax

@pytest.fixture(scope="module")
def common_pharm_data():
    torch.manual_seed(0)
    np.random.seed(0)
    
    num_pharm_ref = 10
    num_pharm_fit = 8
    
    ptype_1_torch = torch.randint(0, len(P_TYPES), (num_pharm_ref,), dtype=torch.long)
    ptype_2_torch = torch.randint(0, len(P_TYPES), (num_pharm_fit,), dtype=torch.long)

    anchors_1_torch = torch.rand(num_pharm_ref, 3, dtype=DEFAULT_DTYPE) * 10
    anchors_2_torch = torch.rand(num_pharm_fit, 3, dtype=DEFAULT_DTYPE) * 10
    
    vectors_1_torch = torch.rand(num_pharm_ref, 3, dtype=DEFAULT_DTYPE) - 0.5
    vectors_1_torch = torch.nn.functional.normalize(vectors_1_torch, p=2, dim=1)
    vectors_2_torch = torch.rand(num_pharm_fit, 3, dtype=DEFAULT_DTYPE) - 0.5
    vectors_2_torch = torch.nn.functional.normalize(vectors_2_torch, p=2, dim=1)

    ptype_1_jax = jnp.array(ptype_1_torch.numpy())
    ptype_2_jax = jnp.array(ptype_2_torch.numpy())
    anchors_1_jax = jnp.array(anchors_1_torch.numpy())
    anchors_2_jax = jnp.array(anchors_2_torch.numpy())
    vectors_1_jax = jnp.array(vectors_1_torch.numpy())
    vectors_2_jax = jnp.array(vectors_2_torch.numpy())

    return (ptype_1_torch, ptype_2_torch, anchors_1_torch, anchors_2_torch, vectors_1_torch, vectors_2_torch,
            ptype_1_jax, ptype_2_jax, anchors_1_jax, anchors_2_jax, vectors_1_jax, vectors_2_jax)


class TestOptimizeROCSOverlayConsistency:
    def _run_and_compare(self, ref_torch, fit_torch, ref_jax, fit_jax, num_repeats, trans_torch, trans_jax):
        # PyTorch run
        aligned_torch, transform_torch, score_torch = optimize_ROCS_overlay(
            ref_torch, fit_torch, DEFAULT_ALPHA,
            num_repeats=num_repeats, trans_centers=trans_torch,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )

        # JAX run
        aligned_jax, transform_jax, score_jax = optimize_ROCS_overlay_jax(
            ref_jax, fit_jax, DEFAULT_ALPHA,
            num_repeats=num_repeats, trans_centers=trans_jax,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )
        aligned_jax_t, transform_jax_t, score_jax_t = jax_outputs_to_torch(aligned_jax, transform_jax, score_jax)
        
        assert torch.allclose(aligned_torch, aligned_jax_t, atol=COORD_ATOL, rtol=COORD_RTOL), "Aligned points mismatch"
        assert torch.allclose(transform_torch, transform_jax_t, atol=TRANSFORM_ATOL, rtol=TRANSFORM_RTOL), "Transform mismatch"
        assert torch.allclose(score_torch, score_jax_t, atol=SCORE_ATOL, rtol=SCORE_RTOL), "Score mismatch"

    def test_rocs_overlay_single(self, common_points_data):
        r_t, f_t, r_j, f_j = common_points_data
        self._run_and_compare(r_t, f_t, r_j, f_j, num_repeats=1, trans_torch=None, trans_jax=None)

    def test_rocs_overlay_num_repeats_batch(self, common_points_data):
        r_t, f_t, r_j, f_j = common_points_data
        self._run_and_compare(r_t, f_t, r_j, f_j, num_repeats=5, trans_torch=None, trans_jax=None)

    def test_rocs_overlay_trans_centers_batch(self, common_points_data, common_trans_centers_data):
        r_t, f_t, r_j, f_j = common_points_data
        tc_t, tc_j = common_trans_centers_data
        # When trans_centers is provided, num_repeats in optimize_ROCS_overlay is used by _initialize_se3_params,
        # but _initialize_se3_params_with_translations has its own num_repeats_per_trans.
        # Let's pass a num_repeats that would be used if trans_centers was None, to ensure it's handled/ignored correctly.
        self._run_and_compare(r_t, f_t, r_j, f_j, num_repeats=50, trans_torch=tc_t, trans_jax=tc_j)


class TestOptimizeROCSEspOverlayConsistency:
    def _run_and_compare(self, r_pts_t, f_pts_t, r_chg_t, f_chg_t, 
                         r_pts_j, f_pts_j, r_chg_j, f_chg_j,
                         num_repeats, trans_t, trans_j):
        # PyTorch run
        aligned_torch, transform_torch, score_torch = optimize_ROCS_esp_overlay(
            r_pts_t, f_pts_t, r_chg_t, f_chg_t, DEFAULT_ALPHA, DEFAULT_LAM,
            num_repeats=num_repeats, trans_centers=trans_t,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )

        # JAX run
        aligned_jax, transform_jax, score_jax = optimize_ROCS_esp_overlay_jax(
            r_pts_j, f_pts_j, r_chg_j, f_chg_j, DEFAULT_ALPHA, DEFAULT_LAM,
            num_repeats=num_repeats, trans_centers=trans_j,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )
        aligned_jax_t, transform_jax_t, score_jax_t = jax_outputs_to_torch(aligned_jax, transform_jax, score_jax)

        assert torch.allclose(aligned_torch, aligned_jax_t, atol=COORD_ATOL, rtol=COORD_RTOL), "Aligned points mismatch"
        assert torch.allclose(transform_torch, transform_jax_t, atol=TRANSFORM_ATOL, rtol=TRANSFORM_RTOL), "Transform mismatch"
        assert torch.allclose(score_torch, score_jax_t, atol=SCORE_ATOL, rtol=SCORE_RTOL), "Score mismatch"

    def test_rocs_esp_overlay_single(self, common_points_data, common_charges_data):
        r_pts_t, f_pts_t, r_pts_j, f_pts_j = common_points_data
        r_chg_t, f_chg_t, r_chg_j, f_chg_j = common_charges_data
        self._run_and_compare(r_pts_t, f_pts_t, r_chg_t, f_chg_t, r_pts_j, f_pts_j, r_chg_j, f_chg_j,
                              num_repeats=1, trans_t=None, trans_j=None)

    def test_rocs_esp_overlay_num_repeats_batch(self, common_points_data, common_charges_data):
        r_pts_t, f_pts_t, r_pts_j, f_pts_j = common_points_data
        r_chg_t, f_chg_t, r_chg_j, f_chg_j = common_charges_data
        self._run_and_compare(r_pts_t, f_pts_t, r_chg_t, f_chg_t, r_pts_j, f_pts_j, r_chg_j, f_chg_j,
                              num_repeats=5, trans_t=None, trans_j=None)

    def test_rocs_esp_overlay_trans_centers_batch(self, common_points_data, common_charges_data, common_trans_centers_data):
        r_pts_t, f_pts_t, r_pts_j, f_pts_j = common_points_data
        r_chg_t, f_chg_t, r_chg_j, f_chg_j = common_charges_data
        tc_t, tc_j = common_trans_centers_data
        self._run_and_compare(r_pts_t, f_pts_t, r_chg_t, f_chg_t, r_pts_j, f_pts_j, r_chg_j, f_chg_j,
                              num_repeats=50, trans_t=tc_t, trans_j=tc_j)


class TestOptimizeEspComboScoreOverlayConsistency:
    def _run_and_compare(self, data_torch, data_jax, num_repeats, trans_torch, trans_jax):
        # PyTorch run
        aligned_torch, transform_torch, score_torch = optimize_esp_combo_score_overlay(
            ref_centers_w_H=data_torch["ref_centers_w_H"], fit_centers_w_H=data_torch["fit_centers_w_H"],
            ref_centers=data_torch["ref_centers"], fit_centers=data_torch["fit_centers"],
            ref_points=data_torch["ref_points"], fit_points=data_torch["fit_points"],
            ref_partial_charges=data_torch["ref_partial_charges"], fit_partial_charges=data_torch["fit_partial_charges"],
            ref_surf_esp=data_torch["ref_surf_esp"], fit_surf_esp=data_torch["fit_surf_esp"],
            ref_radii=data_torch["ref_radii"], fit_radii=data_torch["fit_radii"],
            alpha=data_torch["alpha"], lam=data_torch["lam"],
            probe_radius=data_torch["probe_radius"], esp_weight=data_torch["esp_weight"],
            num_repeats=num_repeats, trans_centers=trans_torch,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )

        # JAX run
        aligned_jax, transform_jax, score_jax = optimize_esp_combo_score_overlay_jax(
            ref_centers_w_H=data_jax["ref_centers_w_H"], fit_centers_w_H=data_jax["fit_centers_w_H"],
            ref_centers=data_jax["ref_centers"], fit_centers=data_jax["fit_centers"],
            ref_points=data_jax["ref_points"], fit_points=data_jax["fit_points"],
            ref_partial_charges=data_jax["ref_partial_charges"], fit_partial_charges=data_jax["fit_partial_charges"],
            ref_surf_esp=data_jax["ref_surf_esp"], fit_surf_esp=data_jax["fit_surf_esp"],
            ref_radii=data_jax["ref_radii"], fit_radii=data_jax["fit_radii"],
            alpha=data_jax["alpha"], lam=data_jax["lam"],
            probe_radius=data_jax["probe_radius"], esp_weight=data_jax["esp_weight"],
            num_repeats=num_repeats, trans_centers=trans_jax,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )
        aligned_jax_t, transform_jax_t, score_jax_t = jax_outputs_to_torch(aligned_jax, transform_jax, score_jax)

        assert torch.allclose(aligned_torch, aligned_jax_t, atol=COORD_ATOL, rtol=COORD_RTOL), "Aligned points mismatch"
        assert torch.allclose(transform_torch, transform_jax_t, atol=TRANSFORM_ATOL, rtol=TRANSFORM_RTOL), "Transform mismatch"
        assert torch.allclose(score_torch, score_jax_t, atol=SCORE_ATOL, rtol=SCORE_RTOL), "Score mismatch"

    def test_esp_combo_single(self, common_esp_combo_data):
        data_t, data_j = common_esp_combo_data
        self._run_and_compare(data_t, data_j, num_repeats=1, trans_torch=None, trans_jax=None)

    def test_esp_combo_num_repeats_batch(self, common_esp_combo_data):
        data_t, data_j = common_esp_combo_data
        self._run_and_compare(data_t, data_j, num_repeats=5, trans_torch=None, trans_jax=None) # Reduced repeats for faster test

    def test_esp_combo_trans_centers_batch(self, common_esp_combo_data, common_trans_centers_data):
        data_t, data_j = common_esp_combo_data
        tc_t, tc_j = common_trans_centers_data
        self._run_and_compare(data_t, data_j, num_repeats=50, trans_torch=tc_t, trans_jax=tc_j)


class TestOptimizePharmOverlayConsistency:
    def _run_and_compare(self, pharm_data_torch, pharm_data_jax, num_repeats, trans_torch, trans_jax, similarity='tanimoto', extended_points=False, only_extended=False):
        p1_t, p2_t, a1_t, a2_t, v1_t, v2_t = pharm_data_torch
        p1_j, p2_j, a1_j, a2_j, v1_j, v2_j = pharm_data_jax

        # PyTorch run
        aligned_anchors_t, aligned_vectors_t, transform_t, score_t = optimize_pharm_overlay(
            p1_t, p2_t, a1_t, a2_t, v1_t, v2_t,
            similarity=similarity, extended_points=extended_points, only_extended=only_extended,
            num_repeats=num_repeats, trans_centers=trans_torch,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )

        # JAX run
        aligned_anchors_j, aligned_vectors_j, transform_j, score_j = optimize_pharm_overlay_jax(
            p1_j, p2_j, a1_j, a2_j, v1_j, v2_j,
            similarity=similarity, extended_points=extended_points, only_extended=only_extended,
            num_repeats=num_repeats, trans_centers=trans_jax,
            lr=DEFAULT_LR, max_num_steps=DEFAULT_MAX_STEPS, verbose=DEFAULT_VERBOSE
        )
        
        aligned_anchors_j_t, aligned_vectors_j_t, transform_j_t, score_j_t = jax_pharm_outputs_to_torch(
            aligned_anchors_j, aligned_vectors_j, transform_j, score_j
        )
        
        assert torch.allclose(aligned_anchors_t, aligned_anchors_j_t, atol=COORD_ATOL, rtol=COORD_RTOL), "Aligned anchors mismatch"
        assert torch.allclose(aligned_vectors_t, aligned_vectors_j_t, atol=COORD_ATOL, rtol=COORD_RTOL), "Aligned vectors mismatch"
        assert torch.allclose(transform_t, transform_j_t, atol=TRANSFORM_ATOL, rtol=TRANSFORM_RTOL), "Transform mismatch"
        assert torch.allclose(score_t, score_j_t, atol=SCORE_ATOL, rtol=SCORE_RTOL), "Score mismatch"

    def test_pharm_overlay_single(self, common_pharm_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        self._run_and_compare(data_torch, data_jax, num_repeats=1, trans_torch=None, trans_jax=None)

    def test_pharm_overlay_num_repeats_batch(self, common_pharm_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        self._run_and_compare(data_torch, data_jax, num_repeats=5, trans_torch=None, trans_jax=None)

    def test_pharm_overlay_trans_centers_batch(self, common_pharm_data, common_trans_centers_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        tc_t, tc_j = common_trans_centers_data
        self._run_and_compare(data_torch, data_jax, num_repeats=50, trans_torch=tc_t, trans_jax=tc_j)

    def test_pharm_overlay_extended_points(self, common_pharm_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        self._run_and_compare(data_torch, data_jax, num_repeats=5, trans_torch=None, trans_jax=None, extended_points=True)

    def test_pharm_overlay_only_extended(self, common_pharm_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        self._run_and_compare(data_torch, data_jax, num_repeats=5, trans_torch=None, trans_jax=None, extended_points=True, only_extended=True)

    def test_pharm_overlay_tversky_ref(self, common_pharm_data):
        data_torch = common_pharm_data[:6]
        data_jax = common_pharm_data[6:]
        self._run_and_compare(data_torch, data_jax, num_repeats=5, trans_torch=None, trans_jax=None, similarity='tversky_ref')
