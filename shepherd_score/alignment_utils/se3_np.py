"""
Functions used for SE(3) transformations. (NumPy implementation).

Namely, converting quaternions to rotation matrices, getting an SE(3) transform from SE(3)
parameters, and applying the SE(3) transformation on a set of points.

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
and PyTorch's implementations.
"""
import numpy as np

def quaternions_to_rotation_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    """
    Converts quaternion to a rotation matrix.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
    
    Parameters
    ----------
    quaternions : np.ndarray (4,)
        Quaternion parameters in (r,i,j,k) order.
        set.
    
    Returns
    -------
    rotation_matrix : np.ndarray (3,3)
        Rotation matrix converted from quaternion.
    """
    # Single instance
    if quaternions.shape == (4,):
        r, i, j, k = quaternions
        two_s = 2. / (quaternions * quaternions).sum()

        rotation_matrix = np.stack(
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
    else:
        raise ValueError(f'Input "quaternions" must be a 1D Tensor of length 4. Instead the shape given was: {quaternions.shape}')
    return rotation_matrix.reshape((3, 3))


def get_SE3_transform_np(se3_params: np.ndarray
                         ) -> np.ndarray:
    """
    Constructs an SE(3) transformtion matrix from parameters. NumPy implementation
    
    Parameters
    ----------
    se3_params : np.ndarray (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    
    Returns
    -------
    se3_matrix : np.ndarray (4, 4)
        se3_params converted to a 4x4 SE(3) transformation matrix.
    """
    if se3_params.shape == (7,):
        # Extract quaternion and translation parameters
        quaternion_params = se3_params[:4]
        translation_params = se3_params[4:]

        # Normalize quaternion to ensure unit length
        quaternion_params = quaternion_params / np.linalg.norm(quaternion_params)
        rotation_matrix = quaternions_to_rotation_matrix_np(quaternion_params)

        # Construct SE(3) transformation matrix
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = rotation_matrix
        se3_matrix[:3, 3] = translation_params
    else:
        raise ValueError(f'Input "se3_params" must be a 1D Tensor of length 7. Instead the shape given was: {se3_params.shape}')
    return se3_matrix


def apply_SE3_transform_np(points: np.ndarray,
                           SE3_transform: np.ndarray
                           ) -> np.ndarray:
    """
    Takes a point cloud and transforms it according to the provided SE3 transformation matrix.
    NumPy implementation.
    
    Parameters
    ----------
    points : np.ndarray (N, 3)
        Set of coordinates representing a point cloud.
    SE3_transform : np.ndarray (4, 4)
        SE(3) transformation matrix.
    
    Returns
    -------
    transformed_points : np.ndarray (N, 3)
        Set of coordinates transformed by the corresponding SE(3) transformation.
    """
    if points.shape[-1] != 3:
        raise ValueError(f'"points" should have shape (N_points, 3). Instead the shape given was: {points.shape}')
    if SE3_transform.shape[-2:] != (4,4):
        raise ValueError(f'"SE3_transform" should have shape (4, 4). Instead the shape given was: {SE3_transform.shape}')
    if len(SE3_transform.shape) != len(points.shape):
        raise ValueError(f'Shapes of points and SE3_transform should be the same length. Instead {len(SE3_transform.shape)} and {len(points.shape)} were given.')

    # Single instance
    if len(SE3_transform.shape) == 2:
        transformed_points = (SE3_transform[:3,:3] @ points.T).T + SE3_transform[:3,3]
    else:
        raise ValueError(f'The expected length of shape for "points" and "SE3_transform" must be 2 but {len(SE3_transform)} was given.')
    return transformed_points


def apply_SO3_transform_np(points: np.ndarray,
                           SE3_transform: np.ndarray
                           ) -> np.ndarray:
    """
    Takes a point cloud and ONLY ROTATES it according to the provided SE3 transformation matrix.
    Supports batched and non-batched inputs.
    
    Parameters
    ----------
    points : np.array (N, 3)
        Set of coordinates representing a point cloud.
    SE3_transform : (4, 4)
        SE(3) transformation matrix. If 'points' argument is batched, this one should be too.
    
    Returns
    -------
    rotated_points : torch.Tensor (batch, N, 3) or (N, 3)
        Set of coordinates rotated by the rotation component of the SE(3) transformation.
    """
    if points.shape[-1] != 3:
        raise ValueError(f'"points" should have shape (N_points, 3). Instead the shape given was: {points.shape}')
    if SE3_transform.shape[-2:] != (4,4):
        raise ValueError(f'"SE3_transform" should have shape (4, 4). Instead the shape given was: {SE3_transform.shape}')
    if len(SE3_transform.shape) != len(points.shape):
        raise ValueError(f'Shapes of points and SE3_transform should be the same length. Instead {len(SE3_transform.shape)} and {len(points.shape)} were given.')

    # Single instance
    if len(SE3_transform.shape) == 2:
        rotated_points = (SE3_transform[:3,:3] @ points.T).T
    else:
        raise ValueError(f'"points" and "SE3_transform" must be a single instance. \
        The expected length of shape for both should be 2 single instance but {len(SE3_transform)} was given.')

    return rotated_points
