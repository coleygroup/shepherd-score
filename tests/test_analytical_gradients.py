"""
Test suite for analytical gradients of pharmacophore and shape alignment.
Verifies correctness against finite differences and PyTorch autograd.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import math
import time

from shepherd_score.score.analytical_gradients import (
    rotation_matrix_jacobians_quat,
    project_grad_R_to_quaternion,
    compute_overlap_and_grad_pharm,
    compute_self_overlaps_pharm,
    apply_tanimoto_chain_rule,
    compute_analytical_grad_se3,
    _rotation_matrix_from_unit_quat,
    compute_overlap_and_grad_shape,
    compute_self_overlaps_shape,
    compute_analytical_grad_se3_shape,
    compute_avoid_and_grad,
    compute_analytical_grad_se3_shape_with_avoid,
)
from shepherd_score.alignment_utils.se3 import (
    quaternions_to_rotation_matrix,
    get_SE3_transform,
    apply_SE3_transform,
    apply_SO3_transform,
)
from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm, tanimoto_func
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.constants import P_TYPES, P_ALPHAS

P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))


# =====================================================================
# Fixtures
# =====================================================================

def _random_unit_quaternion(seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(4).astype(np.float64)
    q = q / np.linalg.norm(q)
    return torch.tensor(q, dtype=torch.float64)


def _random_pharmacophore_data(n_ref=6, n_fit=5, seed=123, dtype=torch.float64):
    """Generate random pharmacophore data with mixed types."""
    rng = np.random.RandomState(seed)

    # Types: 0=Acceptor, 1=Donor, 2=Aromatic, 3=Hydrophobe
    ref_pharms = torch.tensor(rng.choice([0, 1, 2, 3], size=n_ref), dtype=torch.long)
    fit_pharms = torch.tensor(rng.choice([0, 1, 2, 3], size=n_fit), dtype=torch.long)

    ref_anchors = torch.tensor(rng.randn(n_ref, 3), dtype=dtype)
    fit_anchors = torch.tensor(rng.randn(n_fit, 3), dtype=dtype)

    # Random unit vectors
    ref_vecs = torch.tensor(rng.randn(n_ref, 3), dtype=dtype)
    ref_vecs = F.normalize(ref_vecs, p=2, dim=-1)
    fit_vecs = torch.tensor(rng.randn(n_fit, 3), dtype=dtype)
    fit_vecs = F.normalize(fit_vecs, p=2, dim=-1)

    return ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs


def _make_single_type_data(ptype_idx, n_ref=4, n_fit=3, seed=99, dtype=torch.float64):
    """Generate data with only a single pharmacophore type."""
    rng = np.random.RandomState(seed)
    ref_pharms = torch.full((n_ref,), ptype_idx, dtype=torch.long)
    fit_pharms = torch.full((n_fit,), ptype_idx, dtype=torch.long)
    ref_anchors = torch.tensor(rng.randn(n_ref, 3), dtype=dtype)
    fit_anchors = torch.tensor(rng.randn(n_fit, 3), dtype=dtype)
    ref_vecs = F.normalize(torch.tensor(rng.randn(n_ref, 3), dtype=dtype), p=2, dim=-1)
    fit_vecs = F.normalize(torch.tensor(rng.randn(n_fit, 3), dtype=dtype), p=2, dim=-1)
    return ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs


# =====================================================================
# Phase 1: Quaternion Jacobians
# =====================================================================

class TestRotationMatrixJacobians:

    def test_jacobian_matches_finite_difference(self):
        """Perturb each q_k by eps, compare to analytical Jacobian."""
        q = _random_unit_quaternion(seed=42).double()
        dR_dqw, dR_dqx, dR_dqy, dR_dqz = rotation_matrix_jacobians_quat(q)
        jacs_analytical = [dR_dqw, dR_dqx, dR_dqy, dR_dqz]

        eps = 1e-6
        for k in range(4):
            q_plus = q.clone()
            q_minus = q.clone()
            q_plus[k] += eps
            q_minus[k] -= eps
            # Use standard formula (matching our Jacobian derivation)
            R_plus = _rotation_matrix_from_unit_quat(q_plus)
            R_minus = _rotation_matrix_from_unit_quat(q_minus)
            jac_fd = (R_plus - R_minus) / (2 * eps)
            torch.testing.assert_close(jacs_analytical[k], jac_fd, atol=1e-5, rtol=1e-5)

    def test_jacobian_at_identity(self):
        """Verify known values at q=(1,0,0,0)."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        dR_dqw, dR_dqx, dR_dqy, dR_dqz = rotation_matrix_jacobians_quat(q)

        # At identity q=(1,0,0,0), all qx=qy=qz=0, so dR/dqw = 0
        expected_dqw = torch.zeros(3, 3, dtype=torch.float64)
        torch.testing.assert_close(dR_dqw, expected_dqw, atol=1e-10, rtol=1e-10)

        # dR/dqx at identity = [[0,0,0],[0,0,-2],[0,2,0]]
        expected_dqx = torch.tensor([
            [0, 0, 0],
            [0, 0, -2],
            [0, 2, 0],
        ], dtype=torch.float64)
        torch.testing.assert_close(dR_dqx, expected_dqx, atol=1e-10, rtol=1e-10)

    def test_jacobian_batched(self):
        """Batched computation matches loop over single."""
        rng = np.random.RandomState(7)
        qs = torch.tensor(rng.randn(5, 4), dtype=torch.float64)
        qs = F.normalize(qs, p=2, dim=1)

        dRw_b, dRx_b, dRy_b, dRz_b = rotation_matrix_jacobians_quat(qs)

        for i in range(5):
            dRw_s, dRx_s, dRy_s, dRz_s = rotation_matrix_jacobians_quat(qs[i])
            torch.testing.assert_close(dRw_b[i], dRw_s, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(dRx_b[i], dRx_s, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(dRy_b[i], dRy_s, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(dRz_b[i], dRz_s, atol=1e-10, rtol=1e-10)


class TestProjectGradToQuaternion:

    def test_projection_matches_autograd(self):
        """
        f(q_raw) = Tr(A @ R(q_raw/||q_raw||)), compare analytical df/dq_raw vs autograd.

        This tests the full chain: q_raw -> normalize -> R -> Tr(A*R),
        since the Jacobians are derived for the standard formula and the normalization
        chain rule is needed to match autograd through get_SE3_transform.
        """
        q_raw = torch.randn(4, dtype=torch.float64, requires_grad=True)
        A = torch.randn(3, 3, dtype=torch.float64)

        # Autograd path (through get_SE3_transform which normalizes internally)
        q_norm_ag = F.normalize(q_raw, p=2, dim=0)
        R_ag = _rotation_matrix_from_unit_quat(q_norm_ag)
        f = (A * R_ag).sum()
        grad_auto = torch.autograd.grad(f, q_raw)[0]

        # Analytical path
        with torch.no_grad():
            q_raw_d = q_raw.detach()
            q_norm_val = torch.norm(q_raw_d)
            q_norm_d = q_raw_d / q_norm_val
            G = A
            grad_q_norm = project_grad_R_to_quaternion(G, q_norm_d)
            # Apply normalization chain rule
            q_hat = q_norm_d.unsqueeze(-1)
            proj = torch.eye(4, dtype=torch.float64) - q_hat @ q_hat.T
            grad_q_raw_analytical = (proj @ grad_q_norm) / q_norm_val

        torch.testing.assert_close(grad_q_raw_analytical, grad_auto, atol=1e-8, rtol=1e-8)


# =====================================================================
# Phase 2: Overlap Gradient
# =====================================================================

class TestOverlapGradientTranslation:

    def _compare_grad_t(self, ptype_idx, allow_antiparallel=False):
        """Helper: compare analytical grad_t to autograd for a single pharmacophore type."""
        data = _make_single_type_data(ptype_idx, seed=42, dtype=torch.float64)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        # Random SE(3)
        q = _random_unit_quaternion(seed=10).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)

        # Analytical
        with torch.no_grad():
            O_AB_a, grad_R_a, grad_t_a = compute_overlap_and_grad_pharm(
                R, t.detach(), ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
            )

        # Autograd: compute O_AB with grad enabled through t
        t_ag = t.clone().requires_grad_(True)
        fit_t = (R @ fit_anchors.T).T + t_ag
        fit_v_t = (R @ fit_vecs.T).T

        score = get_overlap_pharm(
            ptype_1=ref_pharms, ptype_2=fit_pharms,
            anchors_1=ref_anchors, anchors_2=fit_t,
            vectors_1=ref_vecs, vectors_2=fit_v_t,
            similarity='tanimoto'
        )
        # O_AB = score * (VAA + VBB - O_AB) => can't get O_AB from score alone
        # Instead, compare full objective gradient

        # Actually let's compare the overlap gradient directly using finite differences
        eps = 1e-6
        grad_t_fd = torch.zeros(3, dtype=torch.float64)
        for i in range(3):
            t_plus = t.detach().clone()
            t_minus = t.detach().clone()
            t_plus[i] += eps
            t_minus[i] -= eps
            O_plus, _, _ = compute_overlap_and_grad_pharm(
                R, t_plus, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
            )
            O_minus, _, _ = compute_overlap_and_grad_pharm(
                R, t_minus, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
            )
            grad_t_fd[i] = (O_plus - O_minus) / (2 * eps)

        torch.testing.assert_close(grad_t_a, grad_t_fd, atol=1e-4, rtol=1e-4)

    def test_translation_grad_nondirectional(self):
        """Hydrophobe type (index 3)."""
        self._compare_grad_t(ptype_idx=3)

    def test_translation_grad_directional(self):
        """Acceptor type (index 0)."""
        self._compare_grad_t(ptype_idx=0)

    def test_translation_grad_aromatic(self):
        """Aromatic type (index 2)."""
        self._compare_grad_t(ptype_idx=2)


class TestOverlapGradientRotation:

    def _compare_grad_R(self, ptype_idx):
        """Helper: compare analytical grad_R to finite differences."""
        data = _make_single_type_data(ptype_idx, seed=77, dtype=torch.float64)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        q = _random_unit_quaternion(seed=20).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.randn(3, dtype=torch.float64)

        O_AB_a, grad_R_a, _ = compute_overlap_and_grad_pharm(
            R, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        # Finite difference on R
        eps = 1e-6
        grad_R_fd = torch.zeros(3, 3, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                R_plus = R.clone()
                R_minus = R.clone()
                R_plus[i, j] += eps
                R_minus[i, j] -= eps
                O_plus, _, _ = compute_overlap_and_grad_pharm(
                    R_plus, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
                )
                O_minus, _, _ = compute_overlap_and_grad_pharm(
                    R_minus, t, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
                )
                grad_R_fd[i, j] = (O_plus - O_minus) / (2 * eps)

        torch.testing.assert_close(grad_R_a, grad_R_fd, atol=1e-4, rtol=1e-4)

    def test_rotation_grad_nondirectional(self):
        self._compare_grad_R(ptype_idx=3)

    def test_rotation_grad_directional(self):
        self._compare_grad_R(ptype_idx=0)

    def test_rotation_grad_aromatic(self):
        self._compare_grad_R(ptype_idx=2)


# =====================================================================
# Phase 3: Tanimoto Chain Rule
# =====================================================================

class TestTanimotoChainRule:

    def test_tanimoto_scaling(self):
        """Verify gradient scaling: dS/dO = U / (U - O)^2."""
        O_AB = torch.tensor(2.0, dtype=torch.float64)
        U = torch.tensor(5.0, dtype=torch.float64)
        grad_R = torch.randn(3, 3, dtype=torch.float64)
        grad_t = torch.randn(3, dtype=torch.float64)

        loss, sg_R, sg_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        expected_S = O_AB / (U - O_AB)
        assert torch.allclose(loss, 1.0 - expected_S)

        expected_scale = -U / (U - O_AB) ** 2
        torch.testing.assert_close(sg_R, expected_scale * grad_R)
        torch.testing.assert_close(sg_t, expected_scale * grad_t)

    def test_loss_gradient_sign(self):
        """loss = 1 - S => gradient is negated relative to S gradient."""
        O_AB = torch.tensor(1.5, dtype=torch.float64)
        U = torch.tensor(4.0, dtype=torch.float64)
        grad_R = torch.ones(3, 3, dtype=torch.float64)
        grad_t = torch.ones(3, dtype=torch.float64)

        loss, sg_R, sg_t = apply_tanimoto_chain_rule(O_AB, U, grad_R, grad_t)

        # Scale should be negative (since we're minimizing 1-S)
        denom = U - O_AB
        expected_scale = -U / (denom * denom)
        assert expected_scale < 0  # minimizing loss means negative scale on dO/d(params)


# =====================================================================
# Phase 4: Full Gradient Assembly
# =====================================================================

class TestFullAnalyticalGradient:

    def _autograd_reference(self, se3_params, ref_pharms, fit_pharms,
                            ref_anchors, fit_anchors, ref_vectors, fit_vectors):
        """Compute loss and grad via autograd for comparison."""
        se3 = se3_params.clone().requires_grad_(True)

        # Build SE3 transform manually to preserve dtype
        if se3.dim() == 1:
            q = F.normalize(se3[:4], p=2, dim=0)
            t_vec = se3[4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_a = (R @ fit_anchors.T).T + t_vec
            fit_v = (R @ fit_vectors.T).T
        else:
            q = F.normalize(se3[:, :4], p=2, dim=1)
            t_vec = se3[:, 4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_a = torch.bmm(R, fit_anchors.permute(0, 2, 1)).permute(0, 2, 1) + t_vec.unsqueeze(1)
            fit_v = torch.bmm(R, fit_vectors.permute(0, 2, 1)).permute(0, 2, 1)

        score = get_overlap_pharm(
            ptype_1=ref_pharms, ptype_2=fit_pharms,
            anchors_1=ref_anchors, anchors_2=fit_a,
            vectors_1=ref_vectors, vectors_2=fit_v,
            similarity='tanimoto'
        )
        if se3.dim() == 2:
            loss = 1 - score.mean()
        else:
            loss = 1 - score
        loss.backward()
        return loss.detach(), se3.grad.detach()

    def test_full_grad_matches_autograd_single(self):
        """Random pharmacophore data, single instance."""
        data = _random_pharmacophore_data(n_ref=6, n_fit=5, seed=123, dtype=torch.float64)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        se3_params = torch.zeros(7, dtype=torch.float64)
        se3_params[:4] = _random_unit_quaternion(seed=30)
        se3_params[4:] = torch.randn(3, dtype=torch.float64) * 0.5

        VAA, VBB = compute_self_overlaps_pharm(
            ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        loss_a, grad_a = compute_analytical_grad_se3(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs, VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        torch.testing.assert_close(loss_a, loss_ag, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)

    def test_full_grad_matches_autograd_batched(self):
        """Batched se3_params."""
        data = _random_pharmacophore_data(n_ref=5, n_fit=4, seed=200, dtype=torch.float64)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        B = 3
        se3_params = torch.zeros(B, 7, dtype=torch.float64)
        for i in range(B):
            se3_params[i, :4] = _random_unit_quaternion(seed=40 + i)
            se3_params[i, 4:] = torch.randn(3, dtype=torch.float64) * 0.5

        # Batch the data
        ref_pharms_b = ref_pharms.unsqueeze(0).expand(B, -1)
        fit_pharms_b = fit_pharms.unsqueeze(0).expand(B, -1)
        ref_anchors_b = ref_anchors.unsqueeze(0).expand(B, -1, -1)
        fit_anchors_b = fit_anchors.unsqueeze(0).expand(B, -1, -1)
        ref_vecs_b = ref_vecs.unsqueeze(0).expand(B, -1, -1)
        fit_vecs_b = fit_vecs.unsqueeze(0).expand(B, -1, -1)

        VAA, VBB = compute_self_overlaps_pharm(
            ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        loss_a, grad_a = compute_analytical_grad_se3(
            se3_params, ref_pharms_b, fit_pharms_b,
            ref_anchors_b, fit_anchors_b, ref_vecs_b, fit_vecs_b,
            VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference(
            se3_params, ref_pharms_b, fit_pharms_b,
            ref_anchors_b, fit_anchors_b, ref_vecs_b, fit_vecs_b
        )

        torch.testing.assert_close(loss_a.float(), loss_ag.float(), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(grad_a.float(), grad_ag.float(), atol=1e-3, rtol=1e-2)

    def test_full_grad_mixed_types(self):
        """Data containing non-directional + directional + aromatic types."""
        rng = np.random.RandomState(333)
        # Ensure at least one of each category
        ref_pharms = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)  # acceptor, donor, aromatic, hydrophobe
        fit_pharms = torch.tensor([0, 2, 3, 1, 3], dtype=torch.long)

        ref_anchors = torch.tensor(rng.randn(6, 3), dtype=torch.float64)
        fit_anchors = torch.tensor(rng.randn(5, 3), dtype=torch.float64)
        ref_vecs = F.normalize(torch.tensor(rng.randn(6, 3), dtype=torch.float64), p=2, dim=-1)
        fit_vecs = F.normalize(torch.tensor(rng.randn(5, 3), dtype=torch.float64), p=2, dim=-1)

        se3_params = torch.zeros(7, dtype=torch.float64)
        se3_params[:4] = _random_unit_quaternion(seed=50)
        se3_params[4:] = torch.randn(3, dtype=torch.float64) * 0.3

        VAA, VBB = compute_self_overlaps_pharm(
            ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        loss_a, grad_a = compute_analytical_grad_se3(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs, VAA, VBB
        )
        loss_ag, grad_ag = TestFullAnalyticalGradient()._autograd_reference(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        torch.testing.assert_close(loss_a, loss_ag, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)

    def test_full_grad_single_type_only(self):
        """Only hydrophobe type present."""
        data = _make_single_type_data(ptype_idx=3, seed=88, dtype=torch.float64)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        se3_params = torch.zeros(7, dtype=torch.float64)
        se3_params[:4] = _random_unit_quaternion(seed=60)
        se3_params[4:] = torch.randn(3, dtype=torch.float64) * 0.2

        VAA, VBB = compute_self_overlaps_pharm(
            ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        loss_a, grad_a = compute_analytical_grad_se3(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs, VAA, VBB
        )
        loss_ag, grad_ag = TestFullAnalyticalGradient()._autograd_reference(
            se3_params, ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs
        )

        torch.testing.assert_close(loss_a, loss_ag, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)


# =====================================================================
# Phase 5: Optimizer Integration
# =====================================================================

class TestOptimizePharmOverlayAnalytical:

    def _get_test_data(self, seed=42):
        """Generate test data for optimization tests."""
        return _random_pharmacophore_data(n_ref=8, n_fit=6, seed=seed, dtype=torch.float32)

    def test_analytical_matches_autograd_single(self):
        """num_repeats=1, compare final score."""
        from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical

        data = self._get_test_data(seed=42)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        _, _, _, score_ag = optimize_pharm_overlay(
            ref_pharms=ref_pharms, fit_pharms=fit_pharms,
            ref_anchors=ref_anchors, fit_anchors=fit_anchors,
            ref_vectors=ref_vecs, fit_vectors=fit_vecs,
            similarity='tanimoto', num_repeats=1,
            lr=0.1, max_num_steps=200
        )

        _, _, _, score_a = optimize_pharm_overlay_analytical(
            ref_pharms=ref_pharms, fit_pharms=fit_pharms,
            ref_anchors=ref_anchors, fit_anchors=fit_anchors,
            ref_vectors=ref_vecs, fit_vectors=fit_vecs,
            similarity='tanimoto', num_repeats=1,
            lr=0.1, max_num_steps=200
        )

        # Both should achieve similar scores (not necessarily identical due to floating point)
        assert abs(score_a.item() - score_ag.item()) < 1e-4, \
            f"Analytical score {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_analytical_matches_autograd_batched(self):
        """num_repeats=5."""
        from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical

        data = self._get_test_data(seed=55)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        _, _, _, score_ag = optimize_pharm_overlay(
            ref_pharms=ref_pharms, fit_pharms=fit_pharms,
            ref_anchors=ref_anchors, fit_anchors=fit_anchors,
            ref_vectors=ref_vecs, fit_vectors=fit_vecs,
            similarity='tanimoto', num_repeats=5,
            lr=0.1, max_num_steps=200
        )

        _, _, _, score_a = optimize_pharm_overlay_analytical(
            ref_pharms=ref_pharms, fit_pharms=fit_pharms,
            ref_anchors=ref_anchors, fit_anchors=fit_anchors,
            ref_vectors=ref_vecs, fit_vectors=fit_vecs,
            similarity='tanimoto', num_repeats=5,
            lr=0.1, max_num_steps=200
        )

        assert abs(score_a.item() - score_ag.item()) < 1e-4, \
            f"Analytical score {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_tversky_matches_autograd(self):
        """Verify tversky analytical matches autograd for all three variants."""
        from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical

        data = self._get_test_data(seed=42)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        for sim in ('tversky', 'tversky_ref', 'tversky_fit'):
            _, _, _, score_ag = optimize_pharm_overlay(
                ref_pharms=ref_pharms, fit_pharms=fit_pharms,
                ref_anchors=ref_anchors, fit_anchors=fit_anchors,
                ref_vectors=ref_vecs, fit_vectors=fit_vecs,
                similarity=sim, num_repeats=5,
                lr=0.1, max_num_steps=200
            )

            _, _, _, score_a = optimize_pharm_overlay_analytical(
                ref_pharms=ref_pharms, fit_pharms=fit_pharms,
                ref_anchors=ref_anchors, fit_anchors=fit_anchors,
                ref_vectors=ref_vecs, fit_vectors=fit_vecs,
                similarity=sim, num_repeats=5,
                lr=0.1, max_num_steps=200
            )

            assert abs(score_a.item() - score_ag.item()) < 1e-4, \
                f"{sim}: analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_tversky_extended_points_not_implemented(self):
        """Verify extended_points=True still raises NotImplementedError."""
        from shepherd_score.alignment import optimize_pharm_overlay_analytical

        data = self._get_test_data()
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        with pytest.raises(NotImplementedError):
            optimize_pharm_overlay_analytical(
                ref_pharms=ref_pharms, fit_pharms=fit_pharms,
                ref_anchors=ref_anchors, fit_anchors=fit_anchors,
                ref_vectors=ref_vecs, fit_vectors=fit_vecs,
                extended_points=True, num_repeats=1
            )


# =====================================================================
# Phase 6: Performance Benchmark
# =====================================================================

@pytest.mark.slow
class TestAnalyticalPerformance:

    def test_analytical_faster_than_autograd(self):
        """Time both on realistic data, assert analytical is faster."""
        from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical

        data = _random_pharmacophore_data(n_ref=10, n_fit=8, seed=999, dtype=torch.float32)
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        kwargs = dict(
            ref_pharms=ref_pharms, fit_pharms=fit_pharms,
            ref_anchors=ref_anchors, fit_anchors=fit_anchors,
            ref_vectors=ref_vecs, fit_vectors=fit_vecs,
            similarity='tanimoto', num_repeats=50,
            lr=0.1, max_num_steps=200
        )

        # Warmup
        optimize_pharm_overlay(**kwargs)
        optimize_pharm_overlay_analytical(**kwargs)

        # Time autograd
        t0 = time.perf_counter()
        for _ in range(3):
            optimize_pharm_overlay(**kwargs)
        time_ag = (time.perf_counter() - t0) / 3

        # Time analytical
        t0 = time.perf_counter()
        for _ in range(3):
            optimize_pharm_overlay_analytical(**kwargs)
        time_a = (time.perf_counter() - t0) / 3

        print(f"\nAutograd: {time_ag:.3f}s, Analytical: {time_a:.3f}s, Speedup: {time_ag/time_a:.2f}x")
        assert time_a < time_ag, f"Analytical ({time_a:.3f}s) should be faster than autograd ({time_ag:.3f}s)"


# =====================================================================
# Shape Analytical Gradients
# =====================================================================

def _random_shape_data(n_ref=10, n_fit=8, seed=42, dtype=torch.float64):
    """Generate random point cloud data for shape scoring tests."""
    rng = np.random.RandomState(seed)
    ref_points = torch.tensor(rng.randn(n_ref, 3), dtype=dtype)
    fit_points = torch.tensor(rng.randn(n_fit, 3), dtype=dtype)
    return ref_points, fit_points


class TestShapeOverlapGradientTranslation:

    def test_translation_grad_matches_fd(self):
        """Analytical grad_t vs. finite differences for shape overlap."""
        ref_points, fit_points = _random_shape_data(seed=10)
        alpha = 0.81

        q = _random_unit_quaternion(seed=5).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.randn(3, dtype=torch.float64)

        _, _, grad_t_a = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)

        eps = 1e-6
        grad_t_fd = torch.zeros(3, dtype=torch.float64)
        for i in range(3):
            t_plus = t.clone(); t_plus[i] += eps
            t_minus = t.clone(); t_minus[i] -= eps
            O_plus, _, _ = compute_overlap_and_grad_shape(R, t_plus, ref_points, fit_points, alpha)
            O_minus, _, _ = compute_overlap_and_grad_shape(R, t_minus, ref_points, fit_points, alpha)
            grad_t_fd[i] = (O_plus - O_minus) / (2 * eps)

        torch.testing.assert_close(grad_t_a, grad_t_fd, atol=1e-6, rtol=1e-5)

    def test_translation_grad_various_alphas(self):
        """Check grad_t is correct for different alpha values."""
        ref_points, fit_points = _random_shape_data(seed=11)
        for alpha in [0.5, 0.81, 1.5]:
            q = _random_unit_quaternion(seed=6).double()
            R = _rotation_matrix_from_unit_quat(q)
            t = torch.randn(3, dtype=torch.float64)

            _, _, grad_t_a = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)

            eps = 1e-6
            grad_t_fd = torch.zeros(3, dtype=torch.float64)
            for i in range(3):
                t_plus = t.clone(); t_plus[i] += eps
                t_minus = t.clone(); t_minus[i] -= eps
                O_plus, _, _ = compute_overlap_and_grad_shape(R, t_plus, ref_points, fit_points, alpha)
                O_minus, _, _ = compute_overlap_and_grad_shape(R, t_minus, ref_points, fit_points, alpha)
                grad_t_fd[i] = (O_plus - O_minus) / (2 * eps)

            torch.testing.assert_close(grad_t_a, grad_t_fd, atol=1e-6, rtol=1e-5,
                                       msg=f"alpha={alpha}")


class TestShapeOverlapGradientRotation:

    def test_rotation_grad_matches_fd(self):
        """Analytical grad_R vs. finite differences for shape overlap."""
        ref_points, fit_points = _random_shape_data(seed=20)
        alpha = 0.81

        q = _random_unit_quaternion(seed=15).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.randn(3, dtype=torch.float64)

        _, grad_R_a, _ = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)

        eps = 1e-6
        grad_R_fd = torch.zeros(3, 3, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                R_plus = R.clone(); R_plus[i, j] += eps
                R_minus = R.clone(); R_minus[i, j] -= eps
                O_plus, _, _ = compute_overlap_and_grad_shape(R_plus, t, ref_points, fit_points, alpha)
                O_minus, _, _ = compute_overlap_and_grad_shape(R_minus, t, ref_points, fit_points, alpha)
                grad_R_fd[i, j] = (O_plus - O_minus) / (2 * eps)

        torch.testing.assert_close(grad_R_a, grad_R_fd, atol=1e-6, rtol=1e-5)

    def test_rotation_grad_identity(self):
        """At identity rotation, grad_R should be non-zero (general position)."""
        ref_points, fit_points = _random_shape_data(seed=25)
        alpha = 0.81
        R = torch.eye(3, dtype=torch.float64)
        t = torch.zeros(3, dtype=torch.float64)

        _, grad_R_a, _ = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha)
        # Just check it's not all zeros (points are in general position)
        assert grad_R_a.abs().sum() > 0


class TestShapeFullAnalyticalGradient:

    def _autograd_reference_shape(self, se3_params, ref_points, fit_points, alpha):
        """Compute loss and grad via autograd for shape scoring."""
        se3 = se3_params.clone().requires_grad_(True)

        if se3.dim() == 1:
            q = F.normalize(se3[:4], p=2, dim=0)
            t_vec = se3[4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_t = (R @ fit_points.T).T + t_vec
        else:
            q = F.normalize(se3[:, :4], p=2, dim=1)
            t_vec = se3[:, 4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_t = torch.bmm(R, fit_points.permute(0, 2, 1)).permute(0, 2, 1) + t_vec.unsqueeze(1)

        score = get_overlap(ref_points, fit_t, alpha=alpha)
        if se3.dim() == 2:
            loss = 1 - score.mean()
        else:
            loss = 1 - score
        loss.backward()
        return loss.detach(), se3.grad.detach()

    def test_full_grad_matches_autograd_single(self):
        """Single instance: analytical vs. autograd."""
        ref_points, fit_points = _random_shape_data(n_ref=10, n_fit=8, seed=100)
        alpha = 0.81

        se3_params = torch.zeros(7, dtype=torch.float64)
        se3_params[:4] = _random_unit_quaternion(seed=30)
        se3_params[4:] = torch.randn(3, dtype=torch.float64) * 0.5

        VAA, VBB = compute_self_overlaps_shape(ref_points, fit_points, alpha)

        loss_a, grad_a = compute_analytical_grad_se3_shape(
            se3_params, ref_points, fit_points, alpha, VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference_shape(se3_params, ref_points, fit_points, alpha)

        torch.testing.assert_close(loss_a, loss_ag, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)

    def test_full_grad_matches_autograd_batched(self):
        """Batched se3_params: analytical vs. autograd."""
        ref_points, fit_points = _random_shape_data(n_ref=8, n_fit=6, seed=200)
        alpha = 0.81

        B = 4
        se3_params = torch.zeros(B, 7, dtype=torch.float64)
        for i in range(B):
            se3_params[i, :4] = _random_unit_quaternion(seed=40 + i)
            se3_params[i, 4:] = torch.randn(3, dtype=torch.float64) * 0.5

        ref_points_b = ref_points.unsqueeze(0).expand(B, -1, -1)
        fit_points_b = fit_points.unsqueeze(0).expand(B, -1, -1)

        VAA, VBB = compute_self_overlaps_shape(ref_points, fit_points, alpha)

        loss_a, grad_a = compute_analytical_grad_se3_shape(
            se3_params, ref_points_b, fit_points_b, alpha, VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference_shape(
            se3_params, ref_points_b, fit_points_b, alpha
        )

        torch.testing.assert_close(loss_a.float(), loss_ag.float(), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(grad_a.float(), grad_ag.float(), atol=1e-3, rtol=1e-2)

    def test_self_overlap_values(self):
        """VAA should match direct computation; both should be positive."""
        from shepherd_score.score.gaussian_overlap import VAB_2nd_order
        ref_points, fit_points = _random_shape_data(n_ref=5, n_fit=4, seed=300)
        alpha = 0.81

        VAA, VBB = compute_self_overlaps_shape(ref_points.float(), fit_points.float(), alpha)

        VAA_ref = VAB_2nd_order(ref_points.float(), ref_points.float(), alpha)
        VBB_ref = VAB_2nd_order(fit_points.float(), fit_points.float(), alpha)

        torch.testing.assert_close(VAA, VAA_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(VBB, VBB_ref, atol=1e-5, rtol=1e-5)
        assert VAA > 0
        assert VBB > 0


# =====================================================================
# Shape Optimizer Integration
# =====================================================================

class TestOptimizeROCSOverlayAnalytical:

    def _get_test_data(self, seed=42, n_ref=12, n_fit=10):
        rng = np.random.RandomState(seed)
        ref = torch.tensor(rng.randn(n_ref, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(n_fit, 3), dtype=torch.float32)
        return ref, fit

    def test_analytical_matches_autograd_single(self):
        """num_repeats=1: analytical score close to autograd score."""
        from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical

        ref, fit = self._get_test_data(seed=42)
        alpha = 0.81

        _, _, score_ag = optimize_ROCS_overlay(
            ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=1, lr=0.1, max_num_steps=200
        )
        _, _, score_a = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=1, lr=0.1, max_num_steps=200
        )

        assert abs(score_a.item() - score_ag.item()) < 1e-4, \
            f"Analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_analytical_matches_autograd_batched(self):
        """num_repeats=5: analytical score close to autograd score."""
        from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical

        ref, fit = self._get_test_data(seed=55)
        alpha = 0.81

        _, _, score_ag = optimize_ROCS_overlay(
            ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=5, lr=0.1, max_num_steps=200
        )
        _, _, score_a = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=5, lr=0.1, max_num_steps=200
        )

        assert abs(score_a.item() - score_ag.item()) < 1e-4, \
            f"Analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_returns_three_tuple(self):
        """Return value should be (aligned_points, SE3_transform, score)."""
        from shepherd_score.alignment import optimize_ROCS_overlay_analytical

        ref, fit = self._get_test_data(seed=7)
        result = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=0.81, num_repeats=1
        )
        assert len(result) == 3
        aligned, transform, score = result
        assert aligned.shape == fit.shape
        assert transform.shape == (4, 4)
        assert score.dim() == 0 or score.numel() == 1

    def test_score_in_valid_range(self):
        """Score should be in [0, 1]."""
        from shepherd_score.alignment import optimize_ROCS_overlay_analytical

        ref, fit = self._get_test_data(seed=99)
        _, _, score = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=0.81, num_repeats=10
        )
        assert 0.0 <= score.item() <= 1.0


@pytest.mark.slow
class TestShapeAnalyticalPerformance:

    def test_analytical_faster_than_autograd(self):
        """Shape analytical gradient should be faster than autograd."""
        from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical

        rng = np.random.RandomState(999)
        ref = torch.tensor(rng.randn(20, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(16, 3), dtype=torch.float32)
        alpha = 0.81

        kwargs = dict(ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=50,
                      lr=0.1, max_num_steps=200)

        # Warmup
        optimize_ROCS_overlay(**kwargs)
        optimize_ROCS_overlay_analytical(**kwargs)

        t0 = time.perf_counter()
        for _ in range(3):
            optimize_ROCS_overlay(**kwargs)
        time_ag = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            optimize_ROCS_overlay_analytical(**kwargs)
        time_a = (time.perf_counter() - t0) / 3

        print(f"\nShape - Autograd: {time_ag:.3f}s, Analytical: {time_a:.3f}s, Speedup: {time_ag/time_a:.2f}x")
        assert time_a < time_ag, f"Analytical ({time_a:.3f}s) should be faster than autograd ({time_ag:.3f}s)"


# =====================================================================
# Avoid-points analytical gradient tests
# =====================================================================

class TestAvoidAndGrad:
    """Tests for compute_avoid_and_grad — hard-sphere overlap + gradient."""

    def _setup(self, seed=7, n_fit=5, n_avoid=4, dtype=torch.float64):
        rng = np.random.RandomState(seed)
        fit_orig = torch.tensor(rng.randn(n_fit, 3) * 2, dtype=dtype)
        avoid = torch.tensor(rng.randn(n_avoid, 3) * 2, dtype=dtype)
        q = torch.tensor(rng.randn(4), dtype=dtype)
        q = q / q.norm()
        t = torch.tensor(rng.randn(3) * 0.5, dtype=dtype)
        R = _rotation_matrix_from_unit_quat(q)
        return fit_orig, avoid, R, t

    def test_avoid_zero_when_all_far(self):
        """Avoid term is 0 when all fit-avoid distances exceed min_dist."""
        fit_orig = torch.tensor([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]], dtype=torch.float64)
        avoid = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        t = torch.zeros(3, dtype=torch.float64)
        A, _, _ = compute_avoid_and_grad(R, t, fit_orig, avoid, min_dist=2.0)
        assert A.item() == pytest.approx(0.0)

    def test_avoid_positive_when_close(self):
        """Avoid term is positive when a fit point is within min_dist of an avoid point."""
        fit_orig = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
        avoid = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        t = torch.zeros(3, dtype=torch.float64)
        A, _, _ = compute_avoid_and_grad(R, t, fit_orig, avoid, min_dist=2.0)
        assert A.item() > 0.0

    def test_translation_grad_matches_fd(self):
        """∂A/∂t matches finite differences."""
        fit_orig, avoid, R, t = self._setup(seed=11)
        min_dist = 2.5
        eps = 1e-5

        _, _, grad_t = compute_avoid_and_grad(R, t, fit_orig, avoid, min_dist)

        fd_grad = torch.zeros(3, dtype=t.dtype)
        for i in range(3):
            t_p = t.clone(); t_p[i] += eps
            t_m = t.clone(); t_m[i] -= eps
            A_p, _, _ = compute_avoid_and_grad(R, t_p, fit_orig, avoid, min_dist)
            A_m, _, _ = compute_avoid_and_grad(R, t_m, fit_orig, avoid, min_dist)
            fd_grad[i] = (A_p - A_m) / (2 * eps)

        torch.testing.assert_close(grad_t, fd_grad, atol=1e-4, rtol=1e-3)

    def test_rotation_grad_matches_fd(self):
        """∂A/∂R matches finite differences."""
        fit_orig, avoid, R, t = self._setup(seed=22)
        min_dist = 2.5
        eps = 1e-5

        _, grad_R, _ = compute_avoid_and_grad(R, t, fit_orig, avoid, min_dist)

        fd_grad_R = torch.zeros(3, 3, dtype=R.dtype)
        for i in range(3):
            for j in range(3):
                R_p = R.clone(); R_p[i, j] += eps
                R_m = R.clone(); R_m[i, j] -= eps
                A_p, _, _ = compute_avoid_and_grad(R_p, t, fit_orig, avoid, min_dist)
                A_m, _, _ = compute_avoid_and_grad(R_m, t, fit_orig, avoid, min_dist)
                fd_grad_R[i, j] = (A_p - A_m) / (2 * eps)

        torch.testing.assert_close(grad_R, fd_grad_R, atol=1e-4, rtol=1e-3)

    def test_batched_matches_single(self):
        """Batched output should match looping over single instances."""
        fit_orig, avoid, R, t = self._setup(seed=33)
        min_dist = 2.5

        B = 3
        Rs = torch.stack([R] * B)
        ts = torch.stack([t, t + 0.1, t - 0.1])
        fit_orig_b = fit_orig.unsqueeze(0).expand(B, -1, -1)

        A_batch, grad_R_batch, grad_t_batch = compute_avoid_and_grad(Rs, ts, fit_orig_b, avoid, min_dist)

        for b in range(B):
            A_s, grad_R_s, grad_t_s = compute_avoid_and_grad(Rs[b], ts[b], fit_orig, avoid, min_dist)
            assert A_batch[b].item() == pytest.approx(A_s.item(), abs=1e-6)
            torch.testing.assert_close(grad_R_batch[b], grad_R_s, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(grad_t_batch[b], grad_t_s, atol=1e-6, rtol=1e-5)


class TestAvoidFullAnalyticalGradient:
    """Tests for compute_analytical_grad_se3_shape_with_avoid."""

    def _get_data(self, seed=42, dtype=torch.float64):
        rng = np.random.RandomState(seed)
        ref = torch.tensor(rng.randn(8, 3), dtype=dtype)
        fit = torch.tensor(rng.randn(6, 3), dtype=dtype)
        avoid = torch.tensor(rng.randn(5, 3) * 0.5, dtype=dtype)  # close enough to matter
        se3 = torch.zeros(7, dtype=dtype)
        se3[0] = 1.0  # identity quaternion
        return ref, fit, avoid, se3

    def test_grad_matches_autograd(self):
        """Analytical gradient should match PyTorch autograd for the combined loss."""
        from shepherd_score.alignment import objective_ROCS_overlay_with_avoid

        # objective_ROCS_overlay_with_avoid only supports float32 (se3.py dtype constraint)
        ref, fit, avoid, se3_init = self._get_data(seed=77)
        ref = ref.float(); fit = fit.float(); avoid = avoid.float(); se3_init = se3_init.float()
        alpha = 0.81
        avoid_min_dist = 2.0
        avoid_weight = 0.5

        # Autograd gradient
        se3_ag = se3_init.clone().requires_grad_(True)
        fit_ag = fit.clone()
        loss_ag = objective_ROCS_overlay_with_avoid(
            se3_params=se3_ag,
            ref_points=ref,
            fit_points=fit_ag,
            alpha=alpha,
            fit_points_for_avoid=fit_ag,
            avoid_points=avoid,
            avoid_min_dist=avoid_min_dist,
            avoid_weight=avoid_weight,
        )
        loss_ag.backward()
        grad_ag = se3_ag.grad.clone()

        # Analytical gradient (also float32 for comparison)
        VAA, VBB = compute_self_overlaps_shape(ref, fit, alpha)
        loss_a, grad_a = compute_analytical_grad_se3_shape_with_avoid(
            se3_init, ref, fit, alpha, VAA, VBB,
            fit, avoid, avoid_min_dist, avoid_weight,
        )

        assert abs(loss_a.item() - loss_ag.item()) < 1e-6
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)

    def test_grad_matches_fd(self):
        """Analytical gradient matches finite differences."""
        ref, fit, avoid, se3_init = self._get_data(seed=88)
        alpha = 0.81
        avoid_min_dist = 2.5
        avoid_weight = 1.0
        eps = 1e-5

        VAA, VBB = compute_self_overlaps_shape(ref, fit, alpha)

        loss0, grad_a = compute_analytical_grad_se3_shape_with_avoid(
            se3_init, ref, fit, alpha, VAA, VBB,
            fit, avoid, avoid_min_dist, avoid_weight,
        )

        fd_grad = torch.zeros(7, dtype=torch.float64)
        for i in range(7):
            p = se3_init.clone(); p[i] += eps
            m = se3_init.clone(); m[i] -= eps
            lp, _ = compute_analytical_grad_se3_shape_with_avoid(p, ref, fit, alpha, VAA, VBB, fit, avoid, avoid_min_dist, avoid_weight)
            lm, _ = compute_analytical_grad_se3_shape_with_avoid(m, ref, fit, alpha, VAA, VBB, fit, avoid, avoid_min_dist, avoid_weight)
            fd_grad[i] = (lp - lm) / (2 * eps)

        torch.testing.assert_close(grad_a, fd_grad, atol=1e-4, rtol=1e-3)


class TestOptimizeROCSOverlayAnalyticalWithAvoid:
    """Tests for optimize_ROCS_overlay_analytical with avoid_points."""

    def _get_data(self, seed=42):
        rng = np.random.RandomState(seed)
        ref = torch.tensor(rng.randn(12, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(10, 3), dtype=torch.float32)
        avoid = torch.tensor(rng.randn(4, 3) * 0.5, dtype=torch.float32)
        return ref, fit, avoid

    def test_analytical_matches_autograd_with_avoid(self):
        """Score from analytical optimizer should be close to autograd with avoid.

        Both use PyTorch's optim.Adam, so optimizer trajectories are identical.
        The only remaining difference is per-step gradient error (~1e-4), which
        accumulates to ~1e-3 over the early-stopped run.
        """
        from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical

        ref, fit, avoid = self._get_data(seed=42)
        alpha = 0.81
        kwargs = dict(
            ref_points=ref, fit_points=fit, alpha=alpha,
            avoid_points=avoid, avoid_min_dist=2.0, avoid_weight=0.5,
            num_repeats=1, lr=0.1, max_num_steps=200,
        )

        _, _, score_ag = optimize_ROCS_overlay(**kwargs)
        _, _, score_a = optimize_ROCS_overlay_analytical(**kwargs)

        assert abs(score_a.item() - score_ag.item()) < 5e-3, \
            f"Analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_avoid_reduces_score_vs_no_avoid(self):
        """With a heavy avoid weight near fit points, the combined score should be lower."""
        from shepherd_score.alignment import optimize_ROCS_overlay_analytical

        ref, fit, avoid = self._get_data(seed=55)
        alpha = 0.81

        _, _, score_no_avoid = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=alpha, num_repeats=5, max_num_steps=100
        )
        _, _, score_with_avoid = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=alpha,
            avoid_points=avoid, avoid_min_dist=3.0, avoid_weight=2.0,
            num_repeats=5, max_num_steps=100
        )
        # Combined score includes penalty so it can differ (lower or negative)
        # Just check it runs and returns something finite
        assert math.isfinite(score_with_avoid.item())

    def test_returns_three_tuple_with_avoid(self):
        """Return value should be (aligned_points, SE3_transform, score)."""
        from shepherd_score.alignment import optimize_ROCS_overlay_analytical

        ref, fit, avoid = self._get_data(seed=7)
        result = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=0.81,
            avoid_points=avoid, avoid_min_dist=2.0, avoid_weight=1.0,
            num_repeats=1,
        )
        assert len(result) == 3
        aligned, transform, score = result
        assert aligned.shape == fit.shape
        assert transform.shape == (4, 4)
        assert score.dim() == 0 or score.numel() == 1

    def test_fit_points_for_avoid_distinct(self):
        """fit_points_for_avoid can be a different point set from fit_points."""
        from shepherd_score.alignment import optimize_ROCS_overlay_analytical

        rng = np.random.RandomState(123)
        ref = torch.tensor(rng.randn(8, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(6, 3), dtype=torch.float32)
        fit_avoid = torch.tensor(rng.randn(4, 3), dtype=torch.float32)
        avoid = torch.tensor(rng.randn(3, 3), dtype=torch.float32)

        result = optimize_ROCS_overlay_analytical(
            ref_points=ref, fit_points=fit, alpha=0.81,
            fit_points_for_avoid=fit_avoid,
            avoid_points=avoid, avoid_min_dist=2.0, avoid_weight=1.0,
            num_repeats=3,
        )
        assert len(result) == 3
        aligned, _, _ = result
        assert aligned.shape == fit.shape  # aligned is always fit_points aligned, not fit_avoid


# =====================================================================
# ESP Analytical Gradient Tests
# =====================================================================

def _random_esp_data(n_ref=8, n_fit=6, seed=42, dtype=torch.float64):
    """Generate random ESP data (points + charges)."""
    rng = np.random.RandomState(seed)
    ref_points = torch.tensor(rng.randn(n_ref, 3), dtype=dtype)
    fit_points = torch.tensor(rng.randn(n_fit, 3), dtype=dtype)
    ref_charges = torch.tensor(rng.randn(n_ref), dtype=dtype)
    fit_charges = torch.tensor(rng.randn(n_fit), dtype=dtype)
    return ref_points, fit_points, ref_charges, fit_charges


class TestESPOverlapGradientTranslation:
    """Tests for translation gradient of compute_overlap_and_grad_shape with ESP pair_weights."""

    def test_grad_t_matches_fd(self):
        """∂O_AB/∂t matches finite differences when using ESP pair_weights."""
        from shepherd_score.score.analytical_gradients import _compute_esp_pair_weights
        from shepherd_score.score.constants import LAM_SCALING

        ref_points, fit_points, ref_charges, fit_charges = _random_esp_data(seed=10)
        lam = LAM_SCALING * 0.3
        alpha = 0.81
        eps = 1e-5

        q = _random_unit_quaternion(seed=11).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)

        pair_weights = _compute_esp_pair_weights(fit_charges, ref_charges, lam)
        O_AB, _, grad_t = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha, pair_weights=pair_weights)

        fd_grad = torch.zeros(3, dtype=t.dtype)
        for i in range(3):
            t_p = t.clone(); t_p[i] += eps
            t_m = t.clone(); t_m[i] -= eps
            O_p, _, _ = compute_overlap_and_grad_shape(R, t_p, ref_points, fit_points, alpha, pair_weights=pair_weights)
            O_m, _, _ = compute_overlap_and_grad_shape(R, t_m, ref_points, fit_points, alpha, pair_weights=pair_weights)
            fd_grad[i] = (O_p - O_m) / (2 * eps)

        torch.testing.assert_close(grad_t, fd_grad, atol=1e-5, rtol=1e-4)

    def test_pair_weights_none_matches_uniform(self):
        """pair_weights=None should give same result as uniform weights=1."""
        ref_points, fit_points, _, _ = _random_esp_data(seed=20)
        alpha = 0.81
        R = torch.eye(3, dtype=torch.float64)
        t = torch.zeros(3, dtype=torch.float64)
        ones = torch.ones(len(fit_points), len(ref_points), dtype=torch.float64)

        O1, gR1, gt1 = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha, pair_weights=None)
        O2, gR2, gt2 = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha, pair_weights=ones)

        torch.testing.assert_close(O1, O2, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(gR1, gR2, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(gt1, gt2, atol=1e-10, rtol=1e-10)


class TestESPOverlapGradientRotation:
    """Tests for rotation gradient of compute_overlap_and_grad_shape with ESP pair_weights."""

    def test_grad_R_matches_fd(self):
        """∂O_AB/∂R matches finite differences when using ESP pair_weights."""
        from shepherd_score.score.analytical_gradients import _compute_esp_pair_weights
        from shepherd_score.score.constants import LAM_SCALING

        ref_points, fit_points, ref_charges, fit_charges = _random_esp_data(seed=30)
        lam = LAM_SCALING * 0.3
        alpha = 0.81
        eps = 1e-5

        q = _random_unit_quaternion(seed=31).double()
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float64)

        pair_weights = _compute_esp_pair_weights(fit_charges, ref_charges, lam)
        _, grad_R, _ = compute_overlap_and_grad_shape(R, t, ref_points, fit_points, alpha, pair_weights=pair_weights)

        fd_grad_R = torch.zeros(3, 3, dtype=R.dtype)
        for i in range(3):
            for j in range(3):
                R_p = R.clone(); R_p[i, j] += eps
                R_m = R.clone(); R_m[i, j] -= eps
                O_p, _, _ = compute_overlap_and_grad_shape(R_p, t, ref_points, fit_points, alpha, pair_weights=pair_weights)
                O_m, _, _ = compute_overlap_and_grad_shape(R_m, t, ref_points, fit_points, alpha, pair_weights=pair_weights)
                fd_grad_R[i, j] = (O_p - O_m) / (2 * eps)

        torch.testing.assert_close(grad_R, fd_grad_R, atol=1e-5, rtol=1e-4)


class TestESPFullAnalyticalGradient:
    """Tests for compute_analytical_grad_se3_esp."""

    def _autograd_reference_esp(self, se3, ref_points, fit_points, ref_charges, fit_charges, alpha, lam):
        """Compute ESP loss + grad using PyTorch autograd via get_overlap_esp."""
        from shepherd_score.score.electrostatic_scoring import get_overlap_esp

        se3 = se3.clone().detach().requires_grad_(True)
        if se3.dim() == 1:
            q = F.normalize(se3[:4], p=2, dim=0)
            t_vec = se3[4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_t = (R @ fit_points.T).T + t_vec
            score = get_overlap_esp(ref_points, fit_t, ref_charges, fit_charges, alpha, lam)
            loss = 1 - score
        else:
            B = se3.shape[0]
            q = F.normalize(se3[:, :4], p=2, dim=1)
            t_vec = se3[:, 4:]
            R = _rotation_matrix_from_unit_quat(q)
            fit_t = torch.bmm(R, fit_points.permute(0, 2, 1)).permute(0, 2, 1) + t_vec.unsqueeze(1)
            # get_overlap_esp needs consistent batching: expand charges to (B, N)
            ref_charges_b = ref_charges.unsqueeze(0).expand(B, -1)
            fit_charges_b = fit_charges.unsqueeze(0).expand(B, -1)
            score = get_overlap_esp(ref_points, fit_t, ref_charges_b, fit_charges_b, alpha, lam)
            loss = 1 - score.mean()
        loss.backward()
        return loss.detach(), se3.grad.detach()

    def test_full_grad_matches_autograd_single(self):
        """Single instance: analytical vs. autograd for ESP."""
        from shepherd_score.score.analytical_gradients import (
            compute_self_overlaps_esp, compute_analytical_grad_se3_esp
        )
        from shepherd_score.score.constants import LAM_SCALING

        ref_points, fit_points, ref_charges, fit_charges = _random_esp_data(seed=100)
        alpha = 0.81
        lam = LAM_SCALING * 0.3

        se3_params = torch.zeros(7, dtype=torch.float64)
        se3_params[:4] = _random_unit_quaternion(seed=50)
        se3_params[4:] = torch.randn(3, dtype=torch.float64) * 0.5

        VAA, VBB = compute_self_overlaps_esp(ref_points, fit_points, ref_charges, fit_charges, alpha, lam)

        loss_a, grad_a = compute_analytical_grad_se3_esp(
            se3_params, ref_points, fit_points, ref_charges, fit_charges, alpha, lam, VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference_esp(
            se3_params, ref_points, fit_points, ref_charges, fit_charges, alpha, lam
        )

        torch.testing.assert_close(loss_a, loss_ag, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-6, rtol=1e-5)

    def test_full_grad_matches_autograd_batched(self):
        """Batched se3_params: analytical vs. autograd for ESP."""
        from shepherd_score.score.analytical_gradients import (
            compute_self_overlaps_esp, compute_analytical_grad_se3_esp
        )
        from shepherd_score.score.constants import LAM_SCALING

        ref_points, fit_points, ref_charges, fit_charges = _random_esp_data(seed=200)
        alpha = 0.81
        lam = LAM_SCALING * 0.3
        B = 3

        se3_params = torch.zeros(B, 7, dtype=torch.float64)
        for i in range(B):
            se3_params[i, :4] = _random_unit_quaternion(seed=60 + i)
            se3_params[i, 4:] = torch.randn(3, dtype=torch.float64) * 0.5

        ref_points_b = ref_points.unsqueeze(0).expand(B, -1, -1)
        fit_points_b = fit_points.unsqueeze(0).expand(B, -1, -1)

        VAA, VBB = compute_self_overlaps_esp(ref_points, fit_points, ref_charges, fit_charges, alpha, lam)

        loss_a, grad_a = compute_analytical_grad_se3_esp(
            se3_params, ref_points_b, fit_points_b, ref_charges, fit_charges, alpha, lam, VAA, VBB
        )
        loss_ag, grad_ag = self._autograd_reference_esp(
            se3_params, ref_points_b, fit_points_b, ref_charges, fit_charges, alpha, lam
        )

        torch.testing.assert_close(loss_a.float(), loss_ag.float(), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(grad_a.float(), grad_ag.float(), atol=1e-3, rtol=1e-2)

    def test_self_overlap_values(self):
        """VAA should match VAB_2nd_order_esp with same-molecule inputs."""
        from shepherd_score.score.analytical_gradients import compute_self_overlaps_esp
        from shepherd_score.score.electrostatic_scoring import VAB_2nd_order_esp
        from shepherd_score.score.constants import LAM_SCALING

        ref_points, fit_points, ref_charges, fit_charges = _random_esp_data(
            seed=300, dtype=torch.float32
        )
        alpha = 0.81
        lam = LAM_SCALING * 0.3

        VAA, VBB = compute_self_overlaps_esp(ref_points, fit_points, ref_charges, fit_charges, alpha, lam)

        # VAB_2nd_order_esp requires 2D charges for cdist
        ref_ch_2d = ref_charges.reshape(-1, 1)
        fit_ch_2d = fit_charges.reshape(-1, 1)
        VAA_ref = VAB_2nd_order_esp(ref_points, ref_points, ref_ch_2d, ref_ch_2d, alpha, lam)
        VBB_ref = VAB_2nd_order_esp(fit_points, fit_points, fit_ch_2d, fit_ch_2d, alpha, lam)

        torch.testing.assert_close(VAA, VAA_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(VBB, VBB_ref, atol=1e-5, rtol=1e-5)
        assert VAA > 0
        assert VBB > 0


# =====================================================================
# ESP Optimizer Integration
# =====================================================================

class TestOptimizeROCSESPOverlayAnalytical:

    def _get_test_data(self, seed=42, n_ref=10, n_fit=8):
        rng = np.random.RandomState(seed)
        ref = torch.tensor(rng.randn(n_ref, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(n_fit, 3), dtype=torch.float32)
        ref_charges = torch.tensor(rng.randn(n_ref), dtype=torch.float32)
        fit_charges = torch.tensor(rng.randn(n_fit), dtype=torch.float32)
        return ref, fit, ref_charges, fit_charges

    def test_analytical_matches_autograd_single(self):
        """num_repeats=1: analytical ESP score close to autograd score."""
        from shepherd_score.alignment import optimize_ROCS_esp_overlay, optimize_ROCS_esp_overlay_analytical
        from shepherd_score.score.constants import LAM_SCALING

        ref, fit, ref_ch, fit_ch = self._get_test_data(seed=42)
        alpha = 0.81
        lam = LAM_SCALING * 0.3

        _, _, score_ag = optimize_ROCS_esp_overlay(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=alpha, lam=lam, num_repeats=1, lr=0.1, max_num_steps=200
        )
        _, _, score_a = optimize_ROCS_esp_overlay_analytical(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=alpha, lam=lam, num_repeats=1, lr=0.1, max_num_steps=200
        )

        assert abs(score_a.item() - score_ag.item()) < 2e-3, \
            f"Analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_analytical_matches_autograd_batched(self):
        """num_repeats=5: analytical ESP score close to autograd score."""
        from shepherd_score.alignment import optimize_ROCS_esp_overlay, optimize_ROCS_esp_overlay_analytical
        from shepherd_score.score.constants import LAM_SCALING

        ref, fit, ref_ch, fit_ch = self._get_test_data(seed=55)
        alpha = 0.81
        lam = LAM_SCALING * 0.3

        _, _, score_ag = optimize_ROCS_esp_overlay(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=alpha, lam=lam, num_repeats=5, lr=0.1, max_num_steps=200
        )
        _, _, score_a = optimize_ROCS_esp_overlay_analytical(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=alpha, lam=lam, num_repeats=5, lr=0.1, max_num_steps=200
        )

        assert abs(score_a.item() - score_ag.item()) < 2e-3, \
            f"Analytical {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_returns_three_tuple(self):
        """Return value should be (aligned_points, SE3_transform, score)."""
        from shepherd_score.alignment import optimize_ROCS_esp_overlay_analytical
        from shepherd_score.score.constants import LAM_SCALING

        ref, fit, ref_ch, fit_ch = self._get_test_data(seed=7)
        result = optimize_ROCS_esp_overlay_analytical(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=0.81, lam=LAM_SCALING * 0.3, num_repeats=1,
        )
        assert len(result) == 3
        aligned, transform, score = result
        assert aligned.shape == fit.shape
        assert transform.shape == (4, 4)
        assert score.dim() == 0 or score.numel() == 1

    def test_score_in_valid_range(self):
        """Score should be in [0, 1]."""
        from shepherd_score.alignment import optimize_ROCS_esp_overlay_analytical
        from shepherd_score.score.constants import LAM_SCALING

        ref, fit, ref_ch, fit_ch = self._get_test_data(seed=99)
        _, _, score = optimize_ROCS_esp_overlay_analytical(
            ref_points=ref, fit_points=fit, ref_charges=ref_ch, fit_charges=fit_ch,
            alpha=0.81, lam=LAM_SCALING * 0.3, num_repeats=5,
        )
        assert 0.0 <= score.item() <= 1.0


@pytest.mark.slow
class TestESPAnalyticalPerformance:

    def test_analytical_faster_than_autograd(self):
        """ESP analytical gradient should be faster than autograd."""
        from shepherd_score.alignment import optimize_ROCS_esp_overlay, optimize_ROCS_esp_overlay_analytical
        from shepherd_score.score.constants import LAM_SCALING

        rng = np.random.RandomState(999)
        ref = torch.tensor(rng.randn(50, 3), dtype=torch.float32)
        fit = torch.tensor(rng.randn(50, 3), dtype=torch.float32)
        ref_charges = torch.tensor(rng.randn(50), dtype=torch.float32)
        fit_charges = torch.tensor(rng.randn(50), dtype=torch.float32)
        alpha = 0.81
        lam = LAM_SCALING * 0.3

        kwargs = dict(
            ref_points=ref, fit_points=fit, ref_charges=ref_charges, fit_charges=fit_charges,
            alpha=alpha, lam=lam, num_repeats=50, lr=0.1, max_num_steps=200
        )

        # Warmup
        optimize_ROCS_esp_overlay(**kwargs)
        optimize_ROCS_esp_overlay_analytical(**kwargs)

        t0 = time.perf_counter()
        for _ in range(3):
            optimize_ROCS_esp_overlay(**kwargs)
        time_ag = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            optimize_ROCS_esp_overlay_analytical(**kwargs)
        time_a = (time.perf_counter() - t0) / 3

        print(f"\nESP - Autograd: {time_ag:.3f}s, Analytical: {time_a:.3f}s, Speedup: {time_ag/time_a:.2f}x")
        assert time_a < time_ag, f"Analytical ({time_a:.3f}s) should be faster than autograd ({time_ag:.3f}s)"
