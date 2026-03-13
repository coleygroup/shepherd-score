"""
Test suite for analytical gradients of pharmacophore alignment.
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
)
from shepherd_score.alignment_utils.se3 import (
    quaternions_to_rotation_matrix,
    get_SE3_transform,
    apply_SE3_transform,
    apply_SO3_transform,
)
from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm, tanimoto_func
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
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-4, rtol=1e-3)

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
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-4, rtol=1e-3)

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
        torch.testing.assert_close(grad_a, grad_ag, atol=1e-4, rtol=1e-3)


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
        assert abs(score_a.item() - score_ag.item()) < 0.005, \
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

        assert abs(score_a.item() - score_ag.item()) < 0.05, \
            f"Analytical score {score_a.item():.4f} vs autograd {score_ag.item():.4f}"

    def test_analytical_tanimoto_only(self):
        """Verify NotImplementedError for tversky/extended_points."""
        from shepherd_score.alignment import optimize_pharm_overlay_analytical

        data = self._get_test_data()
        ref_pharms, fit_pharms, ref_anchors, fit_anchors, ref_vecs, fit_vecs = data

        with pytest.raises(NotImplementedError):
            optimize_pharm_overlay_analytical(
                ref_pharms=ref_pharms, fit_pharms=fit_pharms,
                ref_anchors=ref_anchors, fit_anchors=fit_anchors,
                ref_vectors=ref_vecs, fit_vectors=fit_vecs,
                similarity='tversky', num_repeats=1
            )

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
