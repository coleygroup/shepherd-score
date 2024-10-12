"""
Jax Implementations for principal component alignment (pca).

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
"""

import jax.numpy as jnp
from jax import vmap, Array
from shepherd_score.alignment_utils.se3_jax import get_SE3_transform_jax, apply_SE3_transform_jax

def compute_moment_of_inertia_jax(points: Array) -> Array:
    """
    Computes the moment of inertia tensor for a set of points. Jax implementation.
    A = x^2 + y^2 + z^2
    B = X^T X
    """
    points = points - jnp.mean(points, axis=0)
    A = jnp.sum(points**2)
    B = points.T @ points
    eye = jnp.eye(3)
    return (A * eye - B) / points.shape[0]


def compute_principal_moments_of_interia_jax(points: Array) -> Array:
    """
    Calculate principal moment of inertia. Jax implementation.
    """
    momint = compute_moment_of_inertia_jax(points)
    eigvals, eigvecs = jnp.linalg.eigh(momint)
    indices = jnp.argsort(-eigvals) #sorting it returns the 'long' axis in index 0.
    # Return transposed which is more intuitive format
    return eigvecs[:, indices].T


def angle_between_vecs_jax(v1: Array, v2: Array) -> Array:
    """ Compute the angle in radians between two vectors. Jax implementation. """
    v1_u = v1 / jnp.linalg.norm(v1)
    v2_u = v2 / jnp.linalg.norm(v2)
    return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

vmap_angle_between_vecs_jax = vmap(angle_between_vecs_jax, (0, 0))

vmap_allclose = vmap(jnp.allclose, 0)
def rotation_axis_jax(v1: Array, v2: Array) -> Array:
    """
    Calculate the vector about which to order to rotate `a` to align with `b` (cross product).
    Jax implementation.
    """
    if len(v1.shape) == 2:
        all_close = vmap_allclose(v1, v2)
        same_vectors_idx = jnp.where(all_close)[0]
        v3 = jnp.zeros(v1.shape)
        if same_vectors_idx.size > 0:
            v3 = v3.at[same_vectors_idx].set(jnp.tile(jnp.array([1., 0., 0.]), (len(same_vectors_idx), 1)))
        diff_vectors_idx = jnp.where(all_close == False)[0]
        if diff_vectors_idx.size > 0:
            v3 = v3.at[diff_vectors_idx].set(jnp.cross(v1.at[diff_vectors_idx].get(),
                                                       v2.at[diff_vectors_idx].get(), axis=-1))
            v3 = v3.at[diff_vectors_idx].get() / jnp.linalg.norm(v3.at[diff_vectors_idx].get(), axis=1).reshape((-1,1))
    else:
        if jnp.allclose(v1, v2):
            return jnp.array([1., 0., 0.])
        v3 = jnp.cross(v1, v2)
        v3 = v3 / jnp.linalg.norm(v3)
    return v3

def quaternion_from_axis_angle_jax(axis: Array, angle: Array) -> Array:
    """
    Create a Quaternion from a rotation axis and an angle in radians.
    Jax implementation.
    
    Parameters
    ----------
    axis : Array (3,)
        Axis to rotate about.
    angle: Array (1,)
        Angle in radians.
    
    Returns
    -------
    quaternion : Array (4,)
    """
    theta = angle / 2.0
    r = jnp.cos(theta)
    i = axis * jnp.sin(theta)
    return jnp.array([r, i[0], i[1], i[2]])

vmap_quaternion_from_axis_angle_jax = vmap(quaternion_from_axis_angle_jax, (0, 0))


def quaternion_mult_jax(p: Array, q: Array) -> Array:
    """
    Mulitplication of quaternions p and q. Jax implementation
    https://academicflight.com/articles/kinematics/rotation-formalisms/quaternions/
    General use case:
        The consequtive rotations of q_1 then q_2 is equivalent to q_3 = q_2*q_1. (order matters)
    Parameters
    ----------
    p : Array (4,)
        The first quaternion.
    q : Array (4,)
        The second quaternion.
    Returns
    -------
    pq : Array (4,)
        The product of the two quaternions.
    """
    mat1 = jnp.array([[p[0], -p[1], -p[2], -p[3]],
                      [p[1],  p[0], -p[3],  p[2]],
                      [p[2],  p[3],  p[0], -p[1]],
                      [p[3], -p[2],  p[1],  p[0]]])
    pq = mat1 @ q
    return pq

vmap_quaternion_mult_jax = vmap(quaternion_mult_jax, (0, 0))


def quaternions_for_principal_component_alignment_jax(ref_points: Array,
                                                      fit_points: Array
                                                      ) -> Array:
    """
    Computes the 4 quaternions required for alignment of the fit mol along the
    principal components of the reference mol.
    NumPy implementation.
    
    The computed quaternions assumes that fit_points will be rotated after being centered at COM.

    Parameters
    ----------
    ref_points : Array (N, 3)
        Set of reference points that fit_points will be aligned to.
    fit_points : Array (M, 3)
        Set of points that will be aligned to ref_points.

    Returns
    -------
    quaternions : Array (4, 4)
        Set of four quaternions corresponding to the alignment of fit_points to ref_points in the
        four possible principal component combinations.
    """
    pmi_ref = compute_principal_moments_of_interia_jax(ref_points)
    
    quaternions = jnp.zeros((4,4))
    for q_index in range(4):
        if q_index == 1:
            # flip orientation of longest axis
            pmi_ref = pmi_ref.at[0].set(-pmi_ref[0])
        elif q_index == 2:
            # unflip orientation of longest axis
            pmi_ref = pmi_ref.at[0].set(-pmi_ref[0])
            # flip orientation of 2nd longest axis
            pmi_ref = pmi_ref.at[1].set(-pmi_ref[1])
        elif q_index == 3:
            # flip orientation of both axes
            pmi_ref = pmi_ref.at[0].set(-pmi_ref[0])

        quat_order = jnp.zeros((2,4))
        # Initially center to COM
        fit_points_adjust = fit_points - jnp.mean(fit_points, axis=0)
        for ax_idx in range(2):
            pmi_fit = compute_principal_moments_of_interia_jax(fit_points_adjust)
            # Angle between principal axis of fit mol and referencne mol
            angle = angle_between_vecs_jax(pmi_fit[ax_idx], pmi_ref[ax_idx])
            # Axis that we are rotating about
            ax = rotation_axis_jax(pmi_fit[ax_idx], pmi_ref[ax_idx])
            # Quaternion
            quat_order = quat_order.at[ax_idx].set(quaternion_from_axis_angle_jax(ax, angle))
            # get SE(3) transformation matrix
            se3_params = jnp.concatenate((quat_order[ax_idx], jnp.zeros(3)))
            # get transformed matrix
            fit_points_adjust = apply_SE3_transform_jax(fit_points_adjust, get_SE3_transform_jax(se3_params))
        quaternions = quaternions.at[q_index].set(quaternion_mult_jax(quat_order[1], quat_order[0]))
    return quaternions
