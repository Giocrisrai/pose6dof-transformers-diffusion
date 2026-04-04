"""Tests for BOP metrics module."""

import numpy as np
import pytest
from src.utils.metrics import (
    add_metric, add_s_metric, compute_add, compute_adds,
    vsd, mssd, mspd,
    compute_recall, compute_auc,
)


class TestADD:
    """Tests for ADD and ADD-S metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.points = np.random.randn(500, 3) * 50  # mm scale

    def test_identity_pose_zero_error(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, 500.0])
        err = add_metric(R, t, R, t, self.points)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_translation_only_error(self):
        R = np.eye(3)
        t_gt = np.array([0.0, 0.0, 500.0])
        t_est = np.array([10.0, 0.0, 500.0])  # 10mm off in x
        err = add_metric(R, t_est, R, t_gt, self.points)
        assert err == pytest.approx(10.0, abs=1e-6)

    def test_adds_leq_add(self):
        """ADD-S should always be <= ADD (closest point is tighter)."""
        R_gt = np.eye(3)
        t_gt = np.zeros(3)
        # Small rotation
        theta = 0.1
        R_est = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        t_est = np.array([5.0, 0.0, 0.0])

        err_add = add_metric(R_est, t_est, R_gt, t_gt, self.points)
        err_adds = add_s_metric(R_est, t_est, R_gt, t_gt, self.points)
        assert err_adds <= err_add + 1e-6

    def test_symmetric_error(self):
        """ADD-S should be low for symmetric objects even with 180 deg rotation."""
        # Create symmetric points (sphere-like)
        pts = np.random.randn(200, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 50

        R_gt = np.eye(3)
        t = np.zeros(3)
        # 180 degree rotation around z
        R_est = np.diag([-1.0, -1.0, 1.0])

        err_adds = add_s_metric(R_est, t, R_gt, t, pts)
        # For a sphere, ADD-S should be very small regardless of rotation
        assert err_adds < 20.0  # reasonable for unit sphere * 50mm

    def test_aliases_match(self):
        """compute_add/compute_adds should be aliases for add_metric/add_s_metric."""
        R = np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([5.0, 3.0, 1.0])

        assert compute_add(R, t1, R, t2, self.points) == add_metric(R, t1, R, t2, self.points)
        assert compute_adds(R, t1, R, t2, self.points) == add_s_metric(R, t1, R, t2, self.points)


class TestMSSD:
    """Tests for MSSD metric."""

    def setup_method(self):
        np.random.seed(42)
        self.points = np.random.randn(100, 3) * 30

    def test_identity_zero(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, 500.0])
        err = mssd(R, t, R, t, self.points)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_no_symmetry_vs_with_symmetry(self):
        """Adding symmetries should reduce or maintain MSSD."""
        R_gt = np.eye(3)
        t = np.zeros(3)
        R_est = np.diag([-1.0, -1.0, 1.0])  # 180 deg around z

        err_no_sym = mssd(R_est, t, R_gt, t, self.points)

        # Add 180 deg z-rotation as a symmetry
        S = np.eye(4)
        S[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        err_with_sym = mssd(R_est, t, R_gt, t, self.points, symmetries=[np.eye(4), S])

        assert err_with_sym <= err_no_sym + 1e-6


class TestMSPD:
    """Tests for MSPD metric."""

    def setup_method(self):
        np.random.seed(42)
        self.points = np.random.randn(100, 3) * 30
        self.points[:, 2] += 500  # move to z=500mm
        self.K = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=float)

    def test_identity_zero(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, 500.0])
        err = mspd(R, t, R, t, self.points, self.K)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_different_poses(self):
        R_gt = np.eye(3)
        t_gt = np.array([0.0, 0.0, 500.0])
        t_est = np.array([10.0, 0.0, 500.0])
        err = mspd(R_gt, t_est, R_gt, t_gt, self.points, self.K)
        assert err > 0


class TestVSD:
    """Tests for VSD metric."""

    def test_identical_depths_zero(self):
        depth = np.random.rand(100, 100) * 1000 + 100
        err = vsd(depth, depth, depth)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_very_different_depths(self):
        depth_gt = np.ones((50, 50)) * 500
        depth_est = np.ones((50, 50)) * 800
        depth_test = np.ones((50, 50)) * 500
        err = vsd(depth_est, depth_gt, depth_test, tau=20.0)
        assert err > 0.5  # Should be high error

    def test_empty_visibility(self):
        """If nothing is visible, VSD should be 0."""
        depth_est = np.zeros((50, 50))
        depth_gt = np.zeros((50, 50))
        depth_test = np.ones((50, 50)) * 500
        err = vsd(depth_est, depth_gt, depth_test)
        assert err == 0.0


class TestRecallAUC:
    """Tests for recall and AUC computation."""

    def test_perfect_recall(self):
        errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert compute_recall(errors, threshold=1.0) == 1.0

    def test_zero_recall(self):
        errors = [10.0, 20.0, 30.0]
        assert compute_recall(errors, threshold=1.0) == 0.0

    def test_partial_recall(self):
        errors = [1.0, 2.0, 3.0, 4.0]
        assert compute_recall(errors, threshold=2.5) == 0.5

    def test_auc_perfect(self):
        errors = [0.0, 0.0, 0.0]
        auc = compute_auc(errors, max_threshold=10.0)
        assert auc == pytest.approx(1.0, abs=0.02)

    def test_auc_between_zero_and_one(self):
        errors = [5.0, 15.0, 25.0, 35.0]
        auc = compute_auc(errors, max_threshold=50.0)
        assert 0.0 < auc < 1.0

    def test_empty_errors(self):
        assert compute_recall([], threshold=1.0) == 0.0
        assert compute_auc([], max_threshold=10.0) == 0.0
