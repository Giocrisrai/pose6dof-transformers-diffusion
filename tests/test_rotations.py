"""Tests for rotation representation conversions."""

import numpy as np
import torch
import pytest
from src.utils.rotations import (
    quat_to_matrix, matrix_to_quat,
    quat_multiply, quat_conjugate, quat_angular_distance,
    axisangle_to_matrix, matrix_to_axisangle,
    euler_to_matrix, matrix_to_euler,
    matrix_to_6d, sixd_to_matrix,
    sixd_to_matrix_torch, matrix_to_6d_torch,
)
from src.utils.lie_groups import so3_exp

TOL = 1e-10


class TestQuaternions:
    def test_roundtrip(self):
        for _ in range(50):
            R = so3_exp(np.random.randn(3) * 0.5)
            q = matrix_to_quat(R)
            R_rec = quat_to_matrix(q)
            assert np.allclose(R, R_rec, atol=1e-8)

    def test_canonical_form(self):
        """w component should be >= 0."""
        R = so3_exp(np.array([2.0, 1.0, 0.5]))
        q = matrix_to_quat(R)
        assert q[0] >= 0

    def test_multiply_inverse(self):
        q = matrix_to_quat(so3_exp(np.random.randn(3)))
        q_inv = quat_conjugate(q)
        result = quat_multiply(q, q_inv)
        # Should be identity quaternion [1, 0, 0, 0]
        assert np.allclose(np.abs(result[0]), 1.0, atol=TOL)

    def test_angular_distance_zero(self):
        q = matrix_to_quat(so3_exp(np.random.randn(3)))
        assert np.isclose(quat_angular_distance(q, q), 0.0, atol=TOL)

    def test_angular_distance_antipodal(self):
        """q and -q represent the same rotation."""
        q = matrix_to_quat(so3_exp(np.random.randn(3)))
        assert np.isclose(quat_angular_distance(q, -q), 0.0, atol=TOL)


class TestAxisAngle:
    def test_roundtrip(self):
        R = so3_exp(np.array([0.3, -0.5, 0.7]))
        axis, angle = matrix_to_axisangle(R)
        R_rec = axisangle_to_matrix(axis, angle)
        assert np.allclose(R, R_rec, atol=1e-8)

    def test_identity(self):
        axis, angle = matrix_to_axisangle(np.eye(3))
        assert np.isclose(angle, 0.0, atol=TOL)


class TestEuler:
    def test_roundtrip(self):
        for _ in range(50):
            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi/3, np.pi/3)  # Avoid gimbal lock
            yaw = np.random.uniform(-np.pi, np.pi)
            R = euler_to_matrix(roll, pitch, yaw)
            r_rec, p_rec, y_rec = matrix_to_euler(R)
            R_rec = euler_to_matrix(r_rec, p_rec, y_rec)
            assert np.allclose(R, R_rec, atol=1e-8)


class TestSixD:
    def test_roundtrip_numpy(self):
        for _ in range(50):
            R = so3_exp(np.random.randn(3) * 0.5)
            rep = matrix_to_6d(R)
            R_rec = sixd_to_matrix(rep)
            assert np.allclose(R, R_rec, atol=1e-8)

    def test_roundtrip_torch(self):
        R_np = so3_exp(np.random.randn(3) * 0.5)
        rep = matrix_to_6d(R_np)
        rep_t = torch.tensor(rep, dtype=torch.float64)
        R_t = sixd_to_matrix_torch(rep_t)
        assert np.allclose(R_np, R_t.detach().numpy(), atol=1e-8)

    def test_differentiable(self):
        """Gradient should flow through 6D → matrix conversion."""
        rep = torch.randn(6, requires_grad=True)
        R = sixd_to_matrix_torch(rep)
        loss = R.sum()
        loss.backward()
        assert rep.grad is not None
        assert not torch.all(rep.grad == 0)

    def test_batch(self):
        """Test batched 6D conversion."""
        batch = torch.randn(10, 6)
        R = sixd_to_matrix_torch(batch)
        assert R.shape == (10, 3, 3)
        # All should be valid rotations
        for i in range(10):
            det = torch.det(R[i])
            assert torch.isclose(det, torch.tensor(1.0), atol=1e-5)
