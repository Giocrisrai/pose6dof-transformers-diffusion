"""Tests for SE(3) and SO(3) Lie group operations."""

import numpy as np
import pytest
from src.utils.lie_groups import (
    so3_exp, so3_log, so3_hat, so3_vee,
    se3_exp, se3_log,
    se3_compose, se3_inverse,
    pose_from_Rt, pose_to_Rt,
    geodesic_distance_SO3, geodesic_distance_SE3,
)

TOL = 1e-10


class TestSO3:
    """Tests for SO(3) operations."""

    def test_hat_vee_roundtrip(self):
        omega = np.array([0.1, -0.5, 0.3])
        assert np.allclose(so3_vee(so3_hat(omega)), omega)

    def test_exp_identity(self):
        R = so3_exp(np.zeros(3))
        assert np.allclose(R, np.eye(3), atol=TOL)

    def test_exp_log_roundtrip(self):
        for _ in range(50):
            omega = np.random.randn(3) * 0.5
            R = so3_exp(omega)
            omega_rec = so3_log(R)
            assert np.allclose(omega, omega_rec, atol=TOL), \
                f"Failed: {omega} vs {omega_rec}"

    def test_exp_produces_valid_rotation(self):
        omega = np.array([0.3, -0.7, 1.2])
        R = so3_exp(omega)
        # Orthogonality: R^T R = I
        assert np.allclose(R.T @ R, np.eye(3), atol=TOL)
        # det(R) = 1
        assert np.isclose(np.linalg.det(R), 1.0, atol=TOL)

    def test_geodesic_distance_identity(self):
        R = so3_exp(np.array([0.1, 0.2, 0.3]))
        assert np.isclose(geodesic_distance_SO3(R, R), 0.0, atol=TOL)

    def test_geodesic_distance_symmetric(self):
        R1 = so3_exp(np.array([0.1, 0.2, 0.3]))
        R2 = so3_exp(np.array([-0.3, 0.1, 0.5]))
        d1 = geodesic_distance_SO3(R1, R2)
        d2 = geodesic_distance_SO3(R2, R1)
        assert np.isclose(d1, d2, atol=TOL)

    def test_exp_log_near_pi(self):
        """Test near-π rotation (edge case)."""
        omega = np.array([np.pi - 0.01, 0.0, 0.0])
        R = so3_exp(omega)
        omega_rec = so3_log(R)
        R_rec = so3_exp(omega_rec)
        assert np.allclose(R, R_rec, atol=1e-6)


class TestSE3:
    """Tests for SE(3) operations."""

    def test_exp_identity(self):
        T = se3_exp(np.zeros(6))
        assert np.allclose(T, np.eye(4), atol=TOL)

    def test_exp_log_roundtrip(self):
        for _ in range(50):
            xi = np.random.randn(6) * 0.5
            T = se3_exp(xi)
            xi_rec = se3_log(T)
            assert np.allclose(xi, xi_rec, atol=TOL)

    def test_compose_with_inverse(self):
        xi = np.array([1.0, -0.5, 0.3, 0.1, 0.2, -0.1])
        T = se3_exp(xi)
        T_inv = se3_inverse(T)
        result = se3_compose(T, T_inv)
        assert np.allclose(result, np.eye(4), atol=TOL)

    def test_pose_from_to_Rt(self):
        R = so3_exp(np.array([0.1, 0.2, 0.3]))
        t = np.array([1.0, 2.0, 3.0])
        T = pose_from_Rt(R, t)
        R_rec, t_rec = pose_to_Rt(T)
        assert np.allclose(R, R_rec, atol=TOL)
        assert np.allclose(t, t_rec, atol=TOL)

    def test_geodesic_distance_identity(self):
        T = se3_exp(np.random.randn(6) * 0.3)
        rot_d, trans_d = geodesic_distance_SE3(T, T)
        assert np.isclose(rot_d, 0.0, atol=TOL)
        assert np.isclose(trans_d, 0.0, atol=TOL)
