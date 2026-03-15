"""
Analytical gradients package for shape and pharmacophore alignment scoring.

Re-exports all PyTorch implementations from _torch.py for backwards compatibility.
JAX implementations are in _jax.py (optional dependency).
"""
from shepherd_score.score.analytical_gradients._torch import (
    _rotation_matrix_from_unit_quat,
    rotation_matrix_jacobians_quat,
    project_grad_R_to_quaternion,
    build_lookup_tables_cached,
    build_lookup_tables,
    compute_overlap_and_grad_pharm,
    compute_self_overlaps_pharm,
    apply_tanimoto_chain_rule,
    apply_tversky_chain_rule,
    compute_overlap_and_grad_shape,
    compute_self_overlaps_shape,
    _compute_esp_pair_weights,
    compute_self_overlaps_esp,
    compute_analytical_grad_se3_esp,
    compute_analytical_grad_se3_shape,
    compute_avoid_and_grad,
    compute_analytical_grad_se3_shape_with_avoid,
    compute_analytical_grad_se3,
)

__all__ = [
    "_rotation_matrix_from_unit_quat",
    "rotation_matrix_jacobians_quat",
    "project_grad_R_to_quaternion",
    "build_lookup_tables_cached",
    "build_lookup_tables",
    "compute_overlap_and_grad_pharm",
    "compute_self_overlaps_pharm",
    "apply_tanimoto_chain_rule",
    "apply_tversky_chain_rule",
    "compute_overlap_and_grad_shape",
    "compute_self_overlaps_shape",
    "_compute_esp_pair_weights",
    "compute_self_overlaps_esp",
    "compute_analytical_grad_se3_esp",
    "compute_analytical_grad_se3_shape",
    "compute_avoid_and_grad",
    "compute_analytical_grad_se3_shape_with_avoid",
    "compute_analytical_grad_se3",
]
