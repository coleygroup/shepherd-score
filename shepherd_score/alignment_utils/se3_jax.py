"""
Functions used for SE(3) transformations. (Jax implementation).

Namely, converting quaternions to rotation matrices, getting an SE(3) transform from SE(3)
parameters, and applying the SE(3) transformation on a set of points.

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
and PyTorch's implementations.
"""
import jax.numpy as jnp
from jax import Array

def quaternions_to_rotation_matrix_jax(quaternions: Array) -> Array:
    """
    Converts quaternion to a rotation matrix. Jax implementation
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
    
    Parameters
    ----------
    quaternions : Array (4,)
        Quaternion parameters in (r,i,j,k) order.
        set.
    
    Returns
    -------
    rotation_matrix : Array (3,3)
        Rotation matrix converted from quaternion.
    """
    # Single instance
    r, i, j, k = quaternions
    two_s = 2. / (quaternions * quaternions).sum()

    rotation_matrix = jnp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rotation_matrix.reshape((3, 3))


def get_SE3_transform_jax(se3_params: Array) -> Array:
    """
    Constructs an SE(3) transformtion matrix from parameters. Jax implementation
    
    Parameters
    ----------
    se3_params : Array (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    
    Returns
    -------
    se3_matrix : Array (4, 4)
        se3_params converted to a 4x4 SE(3) transformation matrix.
    """
    # Extract quaternion and translation parameters
    quaternion_params = se3_params[:4]
    translation_params = se3_params[4:]

    # Normalize quaternion to ensure unit length
    quaternion_params = quaternion_params / jnp.linalg.norm(quaternion_params)
    rotation_matrix = quaternions_to_rotation_matrix_jax(quaternion_params)

    # Construct SE(3) transformation matrix
    se3_matrix = jnp.eye(4)
    se3_matrix = se3_matrix.at[:3, :3].set(rotation_matrix)
    se3_matrix = se3_matrix.at[:3, 3].set(translation_params)
    return se3_matrix


def apply_SE3_transform_jax(points: Array,
                           SE3_transform: Array
                           ) -> Array:
    """
    Takes a point cloud and transforms it according to the provided SE3 transformation matrix.
    Jax implementation.
    
    Parameters
    ----------
    points : Array (N, 3)
        Set of coordinates representing a point cloud.
    SE3_transform : Array (4, 4)
        SE(3) transformation matrix.
    
    Returns
    -------
    transformed_points : Array (N, 3)
        Set of coordinates transformed by the corresponding SE(3) transformation.
    """
    # Single instance
    transformed_points = (SE3_transform[:3,:3] @ points.T).T + SE3_transform[:3,3]
    return transformed_points
