"""
Rotation representations and conversions for 6-DoF pose estimation.

Supports: rotation matrices, quaternions, axis-angle, Euler angles,
and the continuous 6D representation (Zhou et al., CVPR 2019).

References:
    - Zhou et al. (2019) "On the Continuity of Rotation Representations
      in Neural Networks" — 6D continuous representation
    - Huynh (2009) "Metrics for 3D Rotations: Comparison and Analysis"
"""

import numpy as np
import torch
from typing import Union


# ============================================================
# Quaternion operations (w, x, y, z convention)
# ============================================================

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to rotation matrix.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        (3, 3) rotation matrix
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to unit quaternion (Shepperd's method).

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (4,) quaternion [w, x, y, z], w >= 0
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Enforce w >= 0 (canonical form)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions.

    Args:
        q1, q2: (4,) quaternions [w, x, y, z]

    Returns:
        (4,) product quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance between two unit quaternions.

    Returns the geodesic distance on the quaternion hypersphere.

    Args:
        q1, q2: (4,) unit quaternions

    Returns:
        Angular distance in radians [0, π]
    """
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


# ============================================================
# Axis-angle
# ============================================================

def axisangle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to rotation matrix (Rodrigues).

    Args:
        axis: (3,) unit axis
        angle: rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    from .lie_groups import so3_exp
    return so3_exp(axis * angle)


def matrix_to_axisangle(R: np.ndarray):
    """Convert rotation matrix to axis-angle.

    Returns:
        axis: (3,) unit vector
        angle: scalar in [0, π]
    """
    from .lie_groups import so3_log
    omega = so3_log(R)
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.array([0., 0., 1.]), 0.0
    return omega / angle, angle


# ============================================================
# Euler angles (ZYX convention, common in robotics)
# ============================================================

def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (ZYX) to rotation matrix.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        roll, pitch, yaw: angles in radians

    Returns:
        (3, 3) rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])


def matrix_to_euler(R: np.ndarray):
    """Extract Euler angles (ZYX) from rotation matrix.

    Note: Gimbal lock occurs when pitch ≈ ±π/2.

    Returns:
        roll, pitch, yaw: angles in radians
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))

    if np.abs(np.cos(pitch)) < 1e-6:
        # Gimbal lock
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[1, 1])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    return roll, pitch, yaw


# ============================================================
# 6D Continuous Representation (Zhou et al., CVPR 2019)
# ============================================================

def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D continuous representation.

    Takes the first two columns of the rotation matrix.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (6,) continuous representation [r1; r2]
    """
    return np.concatenate([R[:, 0], R[:, 1]])


def sixd_to_matrix(rep: np.ndarray) -> np.ndarray:
    """Convert 6D continuous representation to rotation matrix.

    Recovers orthonormal R via Gram-Schmidt on the two input vectors.

    Args:
        rep: (6,) continuous representation [a1; a2]

    Returns:
        (3, 3) rotation matrix
    """
    a1 = rep[:3]
    a2 = rep[3:]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    return np.column_stack([b1, b2, b3])


# ============================================================
# PyTorch versions (for differentiable pipeline)
# ============================================================

def sixd_to_matrix_torch(rep: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to rotation matrix (differentiable).

    Args:
        rep: (..., 6) continuous representation

    Returns:
        (..., 3, 3) rotation matrices
    """
    a1 = rep[..., :3]
    a2 = rep[..., 3:]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_6d_torch(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation (differentiable).

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        (..., 6) continuous representation
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
