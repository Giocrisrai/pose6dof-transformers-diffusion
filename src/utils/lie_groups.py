"""
Lie Groups: SE(3) and SO(3) operations for 6-DoF pose estimation.

Implements exponential/logarithmic maps, adjoint representation,
and composition operations on the Special Euclidean and Special
Orthogonal groups.

References:
    - Sola et al. (2018) "A micro Lie theory for state estimation in robotics"
    - Lynch & Park (2017) "Modern Robotics", Ch. 3
"""

import numpy as np
from typing import Tuple


# ============================================================
# SO(3) — Special Orthogonal Group (rotaciones 3D)
# ============================================================

def so3_hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix (hat operator) from R^3 → so(3).

    Args:
        omega: (3,) rotation vector

    Returns:
        (3, 3) skew-symmetric matrix [omega]_×
    """
    assert omega.shape == (3,), f"Expected shape (3,), got {omega.shape}"
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])


def so3_vee(Omega: np.ndarray) -> np.ndarray:
    """Inverse of hat operator: so(3) → R^3.

    Args:
        Omega: (3, 3) skew-symmetric matrix

    Returns:
        (3,) rotation vector
    """
    return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map: so(3) → SO(3) via Rodrigues' formula.

    Maps a rotation vector (axis * angle) to a rotation matrix.

    Args:
        omega: (3,) rotation vector (axis × angle)

    Returns:
        (3, 3) rotation matrix R ∈ SO(3)
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + so3_hat(omega)

    axis = omega / theta
    K = so3_hat(axis)
    # Rodrigues: R = I + sin(θ)K + (1 - cos(θ))K²
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map: SO(3) → so(3).

    Maps a rotation matrix to its rotation vector.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (3,) rotation vector (axis × angle)
    """
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity: use first-order approximation
        return so3_vee(R - R.T) / 2

    if np.abs(theta - np.pi) < 1e-6:
        # Near π: special case
        # Find the column of R + I with largest norm
        M = R + np.eye(3)
        col = np.argmax(np.sum(M ** 2, axis=0))
        v = M[:, col]
        v = v / np.linalg.norm(v)
        return v * theta

    return theta / (2 * np.sin(theta)) * so3_vee(R - R.T)


# ============================================================
# SE(3) — Special Euclidean Group (rigid transformations)
# ============================================================

def se3_hat(xi: np.ndarray) -> np.ndarray:
    """Hat operator: R^6 → se(3).

    Args:
        xi: (6,) twist vector [v; omega] (translational; rotational)

    Returns:
        (4, 4) twist matrix in se(3)
    """
    assert xi.shape == (6,), f"Expected shape (6,), got {xi.shape}"
    v, omega = xi[:3], xi[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = so3_hat(omega)
    Xi[:3, 3] = v
    return Xi


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map: se(3) → SE(3).

    Maps a twist vector to a homogeneous transformation matrix.

    Args:
        xi: (6,) twist vector [v; omega]

    Returns:
        (4, 4) transformation matrix T ∈ SE(3)
    """
    v, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)

    T = np.eye(4)

    if theta < 1e-10:
        T[:3, :3] = np.eye(3)
        T[:3, 3] = v
        return T

    R = so3_exp(omega)
    axis = omega / theta
    K = so3_hat(axis)

    # V matrix (left Jacobian of SO(3))
    V = (np.eye(3)
         + ((1 - np.cos(theta)) / theta) * K
         + ((theta - np.sin(theta)) / theta) * K @ K)

    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithmic map: SE(3) → se(3).

    Maps a transformation matrix to its twist vector.

    Args:
        T: (4, 4) transformation matrix

    Returns:
        (6,) twist vector [v; omega]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        return np.concatenate([t, omega])

    axis = omega / theta
    K = so3_hat(axis)

    # Inverse of V (left Jacobian)
    half_theta = theta / 2
    V_inv = (np.eye(3)
             - 0.5 * theta * K
             + (1 - (half_theta / np.tan(half_theta))) * K @ K)

    v = V_inv @ t
    return np.concatenate([v, omega])


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transformations: T1 ∘ T2.

    Args:
        T1: (4, 4) first transformation
        T2: (4, 4) second transformation

    Returns:
        (4, 4) composed transformation
    """
    return T1 @ T2


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of SE(3) transformation.

    Uses the structure of SE(3) for efficient inversion:
    T^{-1} = [R^T, -R^T @ t; 0, 1]

    Args:
        T: (4, 4) transformation matrix

    Returns:
        (4, 4) inverse transformation
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint representation of SE(3): Ad_T.

    Maps twists from one frame to another: xi' = Ad_T @ xi

    Args:
        T: (4, 4) transformation matrix

    Returns:
        (6, 6) adjoint matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = so3_hat(t) @ R
    Ad[3:, 3:] = R
    return Ad


# ============================================================
# Pose utilities
# ============================================================

def pose_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build SE(3) matrix from rotation and translation.

    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        (4, 4) homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_to_Rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation and translation from SE(3) matrix.

    Args:
        T: (4, 4) homogeneous transformation matrix

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    return T[:3, :3].copy(), T[:3, 3].copy()


def geodesic_distance_SO3(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two rotations on SO(3).

    d(R1, R2) = ||log(R1^T R2)||

    Args:
        R1, R2: (3, 3) rotation matrices

    Returns:
        Angular distance in radians
    """
    R_diff = R1.T @ R2
    return np.linalg.norm(so3_log(R_diff))


def geodesic_distance_SE3(T1: np.ndarray, T2: np.ndarray) -> Tuple[float, float]:
    """Geodesic distance between two poses on SE(3).

    Returns separate angular and translational distances.

    Args:
        T1, T2: (4, 4) transformation matrices

    Returns:
        rot_dist: angular distance in radians
        trans_dist: translational distance in meters
    """
    R1, t1 = pose_to_Rt(T1)
    R2, t2 = pose_to_Rt(T2)
    rot_dist = geodesic_distance_SO3(R1, R2)
    trans_dist = np.linalg.norm(t1 - t2)
    return rot_dist, trans_dist
