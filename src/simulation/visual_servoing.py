"""
Visual Servoing — IBVS and PBVS for robotic bin picking.

Implements Image-Based (IBVS) and Position-Based (PBVS) visual servoing
controllers for guiding a robot manipulator to a target pose estimated
by FoundationPose.

References:
    - Chaumette & Hutchinson (2006) "Visual Servo Control Part I: Basic Approaches"
    - Corke (2017) "Robotics, Vision and Control" Ch. 15-16
"""

import numpy as np
from typing import Tuple, Optional
import logging

from src.utils.lie_groups import (
    so3_exp, so3_log, se3_log,
    pose_from_Rt, pose_to_Rt,
    se3_inverse, se3_compose,
)

logger = logging.getLogger(__name__)


class PBVSController:
    """Position-Based Visual Servoing (PBVS).

    Uses the 3D pose error directly to compute velocity commands.
    Requires accurate pose estimation (from FoundationPose).

    Control law:
        v = -λ * se3_log(T_current^{-1} @ T_target)

    where v is the spatial velocity (twist) in the camera frame.
    """

    def __init__(
        self,
        gain: float = 0.5,
        max_linear_vel: float = 0.1,    # m/s
        max_angular_vel: float = 0.5,   # rad/s
        position_threshold: float = 0.002,  # m (2mm)
        rotation_threshold: float = 0.02,   # rad (~1 deg)
    ):
        """
        Args:
            gain: proportional gain λ
            max_linear_vel: velocity saturation (m/s)
            max_angular_vel: angular velocity saturation (rad/s)
            position_threshold: convergence threshold for translation
            rotation_threshold: convergence threshold for rotation
        """
        self.gain = gain
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold

    def compute_velocity(
        self,
        T_current: np.ndarray,
        T_target: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Compute velocity command for one control step.

        Args:
            T_current: (4, 4) current end-effector pose
            T_target: (4, 4) desired target pose

        Returns:
            velocity: (6,) twist vector [v_x, v_y, v_z, ω_x, ω_y, ω_z]
            converged: True if within threshold
        """
        # Pose error in se(3)
        T_error = se3_compose(se3_inverse(T_current), T_target)
        xi_error = se3_log(T_error)

        # Check convergence
        pos_error = np.linalg.norm(xi_error[:3])
        rot_error = np.linalg.norm(xi_error[3:])

        converged = (pos_error < self.position_threshold and
                     rot_error < self.rotation_threshold)

        if converged:
            return np.zeros(6), True

        # Proportional control
        velocity = self.gain * xi_error

        # Velocity saturation
        lin_norm = np.linalg.norm(velocity[:3])
        if lin_norm > self.max_linear_vel:
            velocity[:3] *= self.max_linear_vel / lin_norm

        ang_norm = np.linalg.norm(velocity[3:])
        if ang_norm > self.max_angular_vel:
            velocity[3:] *= self.max_angular_vel / ang_norm

        return velocity, converged

    def servo_trajectory(
        self,
        T_start: np.ndarray,
        T_target: np.ndarray,
        dt: float = 0.01,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, bool]:
        """Simulate PBVS trajectory (open-loop, for visualization).

        Args:
            T_start: (4, 4) starting pose
            T_target: (4, 4) target pose
            dt: simulation timestep
            max_steps: maximum number of steps

        Returns:
            trajectory: (N, 4, 4) sequence of poses
            success: whether convergence was reached
        """
        trajectory = [T_start.copy()]
        T_current = T_start.copy()

        for step in range(max_steps):
            velocity, converged = self.compute_velocity(T_current, T_target)

            if converged:
                logger.info(f"PBVS converged in {step} steps")
                return np.array(trajectory), True

            # Integrate velocity (first-order Euler on SE(3))
            from src.utils.lie_groups import se3_exp
            delta_T = se3_exp(velocity * dt)
            T_current = se3_compose(T_current, delta_T)
            trajectory.append(T_current.copy())

        logger.warning(f"PBVS did not converge in {max_steps} steps")
        return np.array(trajectory), False


class IBVSController:
    """Image-Based Visual Servoing (IBVS).

    Uses 2D image feature errors to compute velocity commands.
    Does not require explicit 3D pose estimation.

    Control law:
        v = -λ * L_e^+ * (s - s*)

    where L_e is the interaction matrix and s are image features.
    """

    def __init__(
        self,
        gain: float = 0.5,
        max_velocity: float = 0.2,
        n_points: int = 4,
    ):
        self.gain = gain
        self.max_velocity = max_velocity
        self.n_points = n_points

    @staticmethod
    def interaction_matrix(points_2d: np.ndarray, depths: np.ndarray,
                           K: np.ndarray) -> np.ndarray:
        """Compute the image interaction matrix L_s.

        For each 2D point (u, v) with known depth Z:
        L_s = [[-fx/Z,  0,     u/Z,   u*v/fx,       -(fx^2+u^2)/fx, v],
               [0,      -fy/Z, v/Z,   (fy^2+v^2)/fy, -u*v/fy,       -u]]

        Args:
            points_2d: (N, 2) image points in pixels
            depths: (N,) depth values in meters
            K: (3, 3) camera intrinsics

        Returns:
            (2N, 6) interaction matrix
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        L = np.zeros((2 * len(points_2d), 6))

        for i, (pt, Z) in enumerate(zip(points_2d, depths)):
            # Normalized coordinates
            x = (pt[0] - cx) / fx
            y = (pt[1] - cy) / fy

            L[2*i] = [-1/Z, 0, x/Z, x*y, -(1 + x**2), y]
            L[2*i+1] = [0, -1/Z, y/Z, 1 + y**2, -x*y, -x]

        # Scale by focal lengths
        L[0::2] *= fx
        L[1::2] *= fy

        return L

    def compute_velocity(
        self,
        features_current: np.ndarray,
        features_target: np.ndarray,
        depths: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute velocity from image feature error.

        Args:
            features_current: (N, 2) current 2D feature positions
            features_target: (N, 2) desired 2D feature positions
            depths: (N,) depth of each feature point in meters
            K: (3, 3) camera intrinsics

        Returns:
            velocity: (6,) twist vector
            error_norm: feature error magnitude
        """
        error = (features_current - features_target).flatten()
        error_norm = np.linalg.norm(error)

        L = self.interaction_matrix(features_current, depths, K)

        # Pseudo-inverse of interaction matrix
        L_pinv = np.linalg.pinv(L)

        velocity = -self.gain * L_pinv @ error

        # Saturation
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.max_velocity:
            velocity *= self.max_velocity / vel_norm

        return velocity, error_norm


class HybridServoController:
    """Hybrid PBVS/IBVS controller for bin picking.

    Strategy:
    1. PBVS for approach phase (far from object)
    2. Switch to IBVS for final alignment (close to object)

    The switch happens when the position error drops below a threshold.
    """

    def __init__(
        self,
        pbvs_gain: float = 0.5,
        ibvs_gain: float = 0.3,
        switch_distance: float = 0.05,  # 5cm
    ):
        self.pbvs = PBVSController(gain=pbvs_gain)
        self.ibvs = IBVSController(gain=ibvs_gain)
        self.switch_distance = switch_distance
        self.mode = "pbvs"

    def compute_velocity(
        self,
        T_current: np.ndarray,
        T_target: np.ndarray,
        features_current: Optional[np.ndarray] = None,
        features_target: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, str]:
        """Compute velocity with automatic mode switching.

        Returns:
            velocity: (6,) twist
            converged: bool
            mode: "pbvs" or "ibvs"
        """
        _, t_current = pose_to_Rt(T_current)
        _, t_target = pose_to_Rt(T_target)
        distance = np.linalg.norm(t_current - t_target)

        # Mode switching
        if distance < self.switch_distance and features_current is not None:
            self.mode = "ibvs"
            vel, error = self.ibvs.compute_velocity(
                features_current, features_target, depths, K
            )
            converged = error < 2.0  # pixels
        else:
            self.mode = "pbvs"
            vel, converged = self.pbvs.compute_velocity(T_current, T_target)

        return vel, converged, self.mode
