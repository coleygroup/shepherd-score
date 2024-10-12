"""
Numpy Implementations for principal component alignment (pca).
Using the numpy version of quaternions_for_principal_component_alignment is faster than
the torch implementation (~2.5ms vs ~5ms).

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
"""
import numpy as np
from shepherd_score.alignment_utils.se3_np import get_SE3_transform_np, apply_SE3_transform_np

def compute_moment_of_inertia_np(points: np.ndarray) -> np.ndarray:
    """
    Computes the moment of inertia tensor for a set of points. Numpy implementation.
    A = x^2 + y^2 + z^2
    B = X^T X
    """
    points = points - np.mean(points, axis=0)
    A = np.sum(points**2)
    B = points.T @ points
    eye = np.eye(3)
    return (A * eye - B) / points.shape[0]


def compute_principal_moments_of_interia_np(points: np.ndarray) -> np.ndarray:
    """
    Calculate principal moment of inertia. Numpy implementation.
    """
    momint = compute_moment_of_inertia_np(points)
    eigvals, eigvecs = np.linalg.eigh(momint)
    indices = np.argsort(-eigvals) #sorting it returns the 'long' axis in index 0.
    # Return transposed which is more intuitive format
    return eigvecs[:, indices].T


def angle_between_vecs_np(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """ Compute the angle in radians between two vectors. Numpy implementation. """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_axis_np(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the vector about which to order to rotate `a` to align with `b` (cross product).
    Numpy implementation.
    """
    if np.allclose(v1, v2):
        return np.array([1, 0, 0])
    v3 = np.cross(v1, v2)
    return v3 / np.linalg.norm(v3)


def quaternion_from_axis_angle_np(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Create a Quaternion from a rotation axis and an angle in radians.
    Numpy implementation.
    
    Parameters
    ----------
    axis : np.ndarray (3,)
        Axis to rotate about.
    angle: np.ndarray (1,)
        Angle in radians.
    
    Returns
    -------
    quaternion : np.ndarray (4,)
    """
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    # Ensure axis is in unit vector form
    if (abs(1.0 - mag_sq) > 1e-12):
        axis = axis / np.sqrt(mag_sq)
    theta = angle / 2.0
    r = np.cos(theta)
    i = axis * np.sin(theta)
    return np.array([r, i[0], i[1], i[2]])


def quaternion_mult_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Mulitplication of quaternions p and q.
    https://academicflight.com/articles/kinematics/rotation-formalisms/quaternions/
    General use case:
        The consequtive rotations of q_1 then q_2 is equivalent to q_3 = q_2*q_1. (order matters)
    Parameters
    ----------
    p : np.ndarray (4,)
        The first quaternion.
    q : np.ndarray (4,)
        The second quaternion.
    Returns
    -------
    pq : np.ndarray (4,)
        The product of the two quaternions.
    """
    mat1 = np.array([[p[0], -p[1], -p[2], -p[3]],
                     [p[1],  p[0], -p[3],  p[2]],
                     [p[2],  p[3],  p[0], -p[1]],
                     [p[3], -p[2],  p[1],  p[0]]])
    pq = mat1 @ q
    return pq


def quaternions_for_principal_component_alignment_np(ref_points: np.ndarray,
                                                     fit_points: np.ndarray
                                                     ) -> np.ndarray:
    """ 
    Computes the 4 quaternions required for alignment of the fit mol along the
    principal components of the reference mol.
    NumPy implementation.
    
    The computed quaternions assumes that fit_points will be rotated after being centered at COM.

    Parameters
    ----------
    ref_points : np.ndarray (N, 3)
        Set of reference points that fit_points will be aligned to.
    fit_points : np.ndarray (M, 3)
        Set of points that will be aligned to ref_points.

    Returns
    -------
    quaternions : np.ndarray (4, 4)
        Set of four quaternions corresponding to the alignment of fit_points to ref_points in the
        four possible principal component combinations.
    """
    pmi_ref = compute_principal_moments_of_interia_np(ref_points)
    
    quaternions = np.zeros((4,4))
    for q_index in range(4):
        if q_index == 1:
            # flip orientation of longest axis
            pmi_ref[0] = -pmi_ref[0]
        elif q_index == 2:
            # unflip orientation of longest axis
            pmi_ref[0] = -pmi_ref[0]
            # flip orientation of 2nd longest axis
            pmi_ref[1] = -pmi_ref[1]
        elif q_index == 3:
            # flip orientation of both axes
            pmi_ref[0] = -pmi_ref[0]

        quat_order = np.zeros((2,4))
        # Initially center to COM
        fit_points_adjust = fit_points - np.mean(fit_points, axis=0)
        for ax_idx in range(2):
            pmi_fit = compute_principal_moments_of_interia_np(fit_points_adjust)
            # Angle between principal axis of fit mol and referencne mol
            angle = angle_between_vecs_np(pmi_fit[ax_idx], pmi_ref[ax_idx])
            # Axis that we are rotating about
            ax = rotation_axis_np(pmi_fit[ax_idx], pmi_ref[ax_idx])
            # Quaternion
            quat_order[ax_idx] = quaternion_from_axis_angle_np(ax, angle)
            # get SE(3) transformation matrix
            se3_params = np.concatenate((quat_order[ax_idx], np.zeros(3)))
            # get transformed matrix
            fit_points_adjust = apply_SE3_transform_np(fit_points_adjust, get_SE3_transform_np(se3_params))
        quaternions[q_index] = quaternion_mult_np(quat_order[1], quat_order[0])
    return quaternions
