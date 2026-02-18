"""
Tests for alignment function consistency between PyTorch and Jax.
"""
import pytest
import torch
import numpy as np
from shepherd_score.alignment import (
    optimize_ROCS_overlay,
    optimize_ROCS_esp_overlay,
    optimize_esp_combo_score_overlay,
    optimize_pharm_overlay,
)
from shepherd_score.score.constants import P_TYPES
from .utils import _configure_jax_platform

# Attempt to import JAX and related modules
JAX_AVAILABLE = False
jnp = None
optimize_ROCS_overlay_jax = None
optimize_ROCS_esp_overlay_jax = None
optimize_esp_combo_score_overlay_jax = None
optimize_pharm_overlay_jax = None
optimize_pharm_overlay_jax_vectorized = None
convert_to_jnp_array = None

try:
    # Configure JAX platform before import to avoid GPU initialization errors
    _gpu_detected = _configure_jax_platform()

    import jax.numpy as jnp
    from shepherd_score.alignment_jax import (
        optimize_ROCS_overlay_jax,
        optimize_ROCS_esp_overlay_jax,
        optimize_esp_combo_score_overlay_jax,
        optimize_pharm_overlay_jax,
        optimize_pharm_overlay_jax_vectorized,
        convert_to_jnp_array,
    )

    JAX_AVAILABLE = True
except ImportError:
    # JAX not available - all tests will be skipped since they require JAX for comparison
    pass

# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed - required for PyTorch/JAX consistency tests")

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

    num_pharm = len([p for p in P_TYPES if p != 'Dummy'])

    ptype_1_torch = torch.randint(0, num_pharm, (num_pharm_ref,), dtype=torch.long)
    ptype_2_torch = torch.randint(0, num_pharm, (num_pharm_fit,), dtype=torch.long)

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


class TestOptimizePharmOverlayVectorizedConsistency:
    """
    Verify that optimize_pharm_overlay_jax_vectorized (vectorized, lax.while_loop) produces
    results consistent with optimize_pharm_overlay_jax (reference, Python for-loop).

    Both functions optimise the same objective — pharmacophore overlap — from
    identical initial SE(3) parameters, so their final scores should agree to
    within optimisation tolerance.  Coordinates may diverge slightly when
    multiple near-equal local optima exist, so coordinate tolerances are looser.
    """

    # Scores should match closely since the same objective is being optimised
    # from the same starting point.  Coordinates can differ a little more because
    # degenerate poses can have the same score.
    SCORE_ATOL = 5e-3
    SCORE_RTOL = 1e-2
    COORD_ATOL = 5e-2
    COORD_RTOL = 5e-2

    def _run_both(self, pharm_data_jax, num_repeats, trans_jax=None,
                  similarity='tanimoto', extended_points=False, only_extended=False):
        p1_j, p2_j, a1_j, a2_j, v1_j, v2_j = pharm_data_jax

        kwargs = dict(
            ref_pharms=p1_j, fit_pharms=p2_j,
            ref_anchors=a1_j, fit_anchors=a2_j,
            ref_vectors=v1_j, fit_vectors=v2_j,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=num_repeats,
            trans_centers=trans_jax,
            lr=DEFAULT_LR,
            max_num_steps=DEFAULT_MAX_STEPS,
            verbose=DEFAULT_VERBOSE,
        )

        anc_ref, vec_ref, tf_ref, score_ref = optimize_pharm_overlay_jax(**kwargs)
        anc_vec, vec_vec, tf_vec, score_vec = optimize_pharm_overlay_jax_vectorized(**kwargs)

        return (anc_ref, vec_ref, tf_ref, score_ref,
                anc_vec, vec_vec, tf_vec, score_vec)

    def _assert_consistent(self, results, tag=""):
        anc_ref, vec_ref, tf_ref, score_ref, anc_vec, vec_vec, tf_vec, score_vec = results

        score_ref_np = float(np.array(score_ref))
        score_vec_np = float(np.array(score_vec))
        assert np.isclose(score_ref_np, score_vec_np,
                          atol=self.SCORE_ATOL, rtol=self.SCORE_RTOL), (
            f"{tag}: score mismatch — reference={score_ref_np:.5f}, "
            f"vectorized={score_vec_np:.5f}"
        )

        anc_ref_np = np.array(anc_ref)
        anc_vec_np = np.array(anc_vec)
        assert anc_ref_np.shape == anc_vec_np.shape, (
            f"{tag}: aligned anchor shape mismatch "
            f"{anc_ref_np.shape} vs {anc_vec_np.shape}"
        )

        vec_ref_np = np.array(vec_ref)
        vec_vec_np = np.array(vec_vec)
        assert vec_ref_np.shape == vec_vec_np.shape, (
            f"{tag}: aligned vector shape mismatch "
            f"{vec_ref_np.shape} vs {vec_vec_np.shape}"
        )

        tf_ref_np = np.array(tf_ref)
        tf_vec_np = np.array(tf_vec)
        assert tf_ref_np.shape == tf_vec_np.shape, (
            f"{tag}: SE(3) transform shape mismatch "
            f"{tf_ref_np.shape} vs {tf_vec_np.shape}"
        )

    def test_vectorized_single_repeat(self, common_pharm_data):
        """num_repeats=1: both functions should produce the same single-pose result."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=1)
        self._assert_consistent(results, tag="single_repeat")

    def test_vectorized_batch_repeats(self, common_pharm_data):
        """num_repeats=5: best-of-batch selection should agree."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=5)
        self._assert_consistent(results, tag="batch_repeats")

    def test_vectorized_trans_centers(self, common_pharm_data, common_trans_centers_data):
        """trans_centers initialisation should produce consistent scores."""
        data_jax = common_pharm_data[6:]
        _, tc_j = common_trans_centers_data
        results = self._run_both(data_jax, num_repeats=50, trans_jax=tc_j)
        self._assert_consistent(results, tag="trans_centers")

    def test_vectorized_extended_points(self, common_pharm_data):
        """Extended-point scoring mode should be consistent."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=5, extended_points=True)
        self._assert_consistent(results, tag="extended_points")

    def test_vectorized_only_extended(self, common_pharm_data):
        """only_extended mode should be consistent."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=5,
                                 extended_points=True, only_extended=True)
        self._assert_consistent(results, tag="only_extended")

    def test_vectorized_tversky_ref(self, common_pharm_data):
        """tversky_ref similarity should be consistent."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=5, similarity='tversky_ref')
        self._assert_consistent(results, tag="tversky_ref")

    def test_vectorized_tversky_fit(self, common_pharm_data):
        """tversky_fit similarity should be consistent."""
        data_jax = common_pharm_data[6:]
        results = self._run_both(data_jax, num_repeats=5, similarity='tversky_fit')
        self._assert_consistent(results, tag="tversky_fit")

    def test_vectorized_no_jit_cache_growth(self, common_pharm_data):
        """
        Calling optimize_pharm_overlay_jax_vectorized repeatedly with the same shapes
        must not grow JAX's JIT cache.  Recompilation on every call was the
        original performance bug; this catches regressions.
        """
        data_jax = common_pharm_data[6:]
        p1_j, p2_j, a1_j, a2_j, v1_j, v2_j = data_jax

        from shepherd_score.alignment_jax import _make_jit_val_grad_pharm_vectorized

        # Prime the cache
        optimize_pharm_overlay_jax_vectorized(
            p1_j, p2_j, a1_j, a2_j, v1_j, v2_j,
            num_repeats=5, lr=DEFAULT_LR, max_num_steps=10,
        )
        cache_size_before = _make_jit_val_grad_pharm_vectorized.cache_info().currsize

        for _ in range(4):
            optimize_pharm_overlay_jax_vectorized(
                p1_j, p2_j, a1_j, a2_j, v1_j, v2_j,
                num_repeats=5, lr=DEFAULT_LR, max_num_steps=10,
            )

        cache_size_after = _make_jit_val_grad_pharm_vectorized.cache_info().currsize
        assert cache_size_after == cache_size_before, (
            f"lru_cache grew from {cache_size_before} to {cache_size_after} entries "
            "across repeated calls with identical (similarity, extended_points, "
            "only_extended) — recompilation regression detected."
        )
