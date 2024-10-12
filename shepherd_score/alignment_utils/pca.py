"""
Torch implementations for principal component alignment (pca). Written to handle batching.

IT IS RECOMMENDED TO USE THE NUMPY VERSION.
Using the numpy version of quaternions_for_principal_component_alignment is faster
(~2.5ms vs ~5ms). Further, the parallel GPU version is slightly slower (50us) than the serial,
CPU version.

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
"""
import torch
import torch.nn.functional as F
from shepherd_score.alignment_utils.se3 import get_SE3_transform, apply_SE3_transform

def compute_moment_of_inertia(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the moment of inertia of a set of points.
    A = x^2 + y^2 + z^2
    B = X^T X
    """
    # Single instance
    if len(points.shape) == 2:
        # Translate points to center of mass
        translated_points = points - torch.mean(points, dim=0)

        A = torch.sum(translated_points**2)
        B = translated_points.T @ translated_points
        return (A * torch.eye(3).to(points.device) - B) / points.shape[0]

    # Batched
    elif len(points.shape) == 3:
        batch_size = points.shape[0]
        # center to COM
        translated_points = points - torch.mean(points, dim=1).unsqueeze(1)

        A = torch.sum(translated_points**2, dim=1).sum(dim=1)
        B = torch.bmm(translated_points.permute(0,2,1), translated_points)
        A_eye = (A.unsqueeze(1)
                 * torch.eye(3).flatten().repeat((batch_size,1)).to(points.device)
                ).reshape((batch_size, 3, 3))
        return (A_eye - B) / points.shape[1]

    else:
        raise ValueError(f'Expected "points" to have shape (batch, N, 3), or (N, 3), but {points.shape} was passed.')


def compute_principal_moments_of_interia(points: torch.Tensor) -> torch.Tensor:
    """ Compute the principal moments of inertia of a set of points. """
    moment_of_inertia_tensor = compute_moment_of_inertia(points)
    # Eigvals are sorted in ascending order
    _, eigvecs = torch.linalg.eigh(moment_of_inertia_tensor)
    # Single Instance
    if len(points.shape) == 2:
        return torch.flip(eigvecs, (1,)).T
    # Batched
    elif len(points.shape) == 3:
        return torch.flip(eigvecs, (2,)).permute(0,2,1)
    else:
        raise ValueError(f'Expected "points" to have shape (batch, N, 3), or (N, 3), but {points.shape} was passed.')


def angle_between_vecs(v1, v2):
    """ Compute the angle in radians between two vectors (already normalized). """
    # Single Instance
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        return torch.acos(torch.clamp(torch.dot(v1, v2), min=-1., max=1.)) # radians
    # Batched
    elif len(v1.shape) == 2 and len(v2.shape) == 2:
        return torch.acos(torch.clamp(torch.sum(v1 * v2, dim=1), min=-1., max=1.)).unsqueeze(1)
    else:
        raise ValueError(f'Expected "v1" and "v2" to have shape (batch, 3), or (3,), but {v1.shape} and {v2.shape} was passed.')


def rotation_axis(v1, v2):
    """
    Calculate the vector about which to order to rotate `a` to align with `b` (cross product).
    """
    # Single Instance
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        if torch.allclose(v1, v2):
            return torch.Tensor([1, 0, 0]).to(v1.device)
        v3 = torch.linalg.cross(v1, v2, dim=0)

    # Batched
    elif len(v1.shape) == 2 and len(v2.shape) == 2:
        idx_not_same = torch.where(torch.isclose(v1, v2).sum(1) != 3)[0]
        v3 = torch.zeros((v1.shape[0], 3)).to(v1.device)
        v3[:, 0] = 1.
        v3[idx_not_same] = torch.linalg.cross(v1, v2, dim=1)
    else:
        raise ValueError(f'Expected "v1" and "v2" to have shape (batch, 3), or (3,), but {v1.shape} and {v2.shape} was passed.')

    return F.normalize(v3, p=2, dim=len(v1.shape)-1)


def quaternion_from_axis_angle(axis, angle):
    """
    Create a Quaternion from a rotation axis and an angle in radians.
    
    Parameters
    ----------
    axis : torch.Tensor (3,)
        Axis to rotate about.
    angle: torch.Tensor (1,)
        Angle in radians.
    
    Returns
    -------
    quaternion : torch.Tensor (4,)
    """
    # Single Instance
    if len(axis.shape) == 1:
        mag_sq = torch.dot(axis, axis)
        if not torch.is_nonzero(mag_sq):
            raise ZeroDivisionError("Provided rotation axis has no length")
        theta = angle / 2.0
        r = torch.cos(theta)
        i = axis * torch.sin(theta)
        return torch.Tensor([r, i[0], i[1], i[2]]).to(axis.device)

    # Batched
    elif len(axis.shape) == 2 and len(angle) == axis.shape[0]:
        mag_sq = torch.sum(axis ** 2, dim=1, keepdim=True)
        if torch.any(mag_sq == 0):
            raise ZeroDivisionError("Provided rotation axis has no length")
        theta = angle / 2.0
        r = torch.cos(theta)
        i = axis * torch.sin(theta)
        return torch.cat((r, i), dim=1).to(axis.device)
    else:
        raise ValueError(f'Expected "axis" and "angle" to have corresponding shapes (batch, 3)+(batch,1), or (3,)+(1,), but {axis.shape} and {angle.shape} was passed.')


def quaternion_mult(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Mulitplication of quaternions p and q.
    https://academicflight.com/articles/kinematics/rotation-formalisms/quaternions/
    General use case:
        The consequtive rotations of q_1 then q_2 is equivalent to q_3 = q_2*q_1. (order matters)
    """
    if len(p.shape) == 1 and len(q.shape) == 1:
        mat1 = torch.Tensor([[p[0], -p[1], -p[2], -p[3]],
                             [p[1],  p[0], -p[3],  p[2]],
                             [p[2],  p[3],  p[0], -p[1]],
                             [p[3], -p[2],  p[1],  p[0]]]).to(p.device)
        pq = mat1 @ q
    elif len(p.shape) == 2 and len(q.shape)==2:
        pq = torch.empty_like(p).to(p.device)
        pq[:, 0] = p[:, 0] * q[:, 0] - p[:, 1] * q[:, 1] - p[:, 2] * q[:, 2] - p[:, 3] * q[:, 3]
        pq[:, 1] = p[:, 0] * q[:, 1] + p[:, 1] * q[:, 0] + p[:, 2] * q[:, 3] - p[:, 3] * q[:, 2]
        pq[:, 2] = p[:, 0] * q[:, 2] - p[:, 1] * q[:, 3] + p[:, 2] * q[:, 0] + p[:, 3] * q[:, 1]
        pq[:, 3] = p[:, 0] * q[:, 3] + p[:, 1] * q[:, 2] - p[:, 2] * q[:, 1] + p[:, 3] * q[:, 0]
    else:
        raise ValueError(f'Expected "p" and "q" to have the same shape (batch, 4), or (4,), but {p.shape} and {q.shape} was passed.')
    return pq


def quaternions_for_principal_component_alignment(ref_points: torch.Tensor, fit_points: torch.Tensor) -> torch.Tensor:
    """ 
    Computes the 4 quaternions required for alignment of the fit mol along the
    principal components of the reference mol.
    
    The computed quaternions assumes that fit_points will be rotated after being centered at COM. 
    """
    pmi_ref = compute_principal_moments_of_interia(ref_points)

    # If CPU compute with for-loops
    if ref_points.get_device() == -1:
        quaternions = torch.zeros((4,4)).to(ref_points.device)
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

            quat_order = torch.zeros((2,4))
            # Initially center to COM
            fit_points_adjust = fit_points - torch.mean(fit_points, dim=0)
            for ax_idx in range(2):
                pmi_fit = compute_principal_moments_of_interia(fit_points_adjust)
                # Angle between principal axis of fit mol and referencne mol
                angle = angle_between_vecs(pmi_fit[ax_idx], pmi_ref[ax_idx])
                # Axis that we are rotating about
                ax = rotation_axis(pmi_fit[ax_idx], pmi_ref[ax_idx])
                # Quaternion
                quat_order[ax_idx] = quaternion_from_axis_angle(ax, angle)
                # get SE(3) transformation matrix
                se3_params = torch.concatenate((quat_order[ax_idx], torch.zeros(3)))
                # get transformed matrix
                fit_points_adjust = apply_SE3_transform(fit_points_adjust, get_SE3_transform(se3_params))
            quaternions[q_index] = quaternion_mult(quat_order[1], quat_order[0])

    else: # GPU
        pmi_refs = pmi_ref.repeat((4, 1, 1))
        pmi_refs[1][0] = -pmi_refs[1][0] # flip orientation of longest axis
        pmi_refs[2][1] = -pmi_refs[2][1] # flip orientation of 2nd longest axis
        # flip orientation of longest and 2nd longest axes
        pmi_refs[3][0] = -pmi_refs[3][0] 
        pmi_refs[3][1] = -pmi_refs[3][1]

        fit_points_adjust = fit_points.repeat((4,1,1))
        # Initially center to COM
        fit_points_adjust = fit_points_adjust - torch.mean(fit_points_adjust, dim=1).unsqueeze(1)
        quat_order = torch.zeros((8,4))
        for ax_idx in range(2):
            # Principal moment of inertia of molecule getting aligned
            pmi_fit = compute_principal_moments_of_interia(fit_points_adjust)
            # Angle between principal axis of fit mol and referencne mol
            angle = angle_between_vecs(pmi_fit[:, ax_idx], pmi_refs[:, ax_idx])
            # Axis that we are rotating about
            ax = rotation_axis(pmi_fit[:, ax_idx], pmi_refs[:, ax_idx])
            # Quaternion
            quat_order[ax_idx*4:(ax_idx+1)*4] = quaternion_from_axis_angle(ax, angle)
            # get SE(3) transformation matrix
            se3_params = torch.concatenate((quat_order[ax_idx*4:(ax_idx+1)*4], torch.zeros((4,3))), axis=1).to(ref_points.device)
            # get transformed matrix
            fit_points_adjust = apply_SE3_transform(fit_points_adjust, get_SE3_transform(se3_params))
        quaternions = quaternion_mult(quat_order[4:], quat_order[:4])
    return quaternions
