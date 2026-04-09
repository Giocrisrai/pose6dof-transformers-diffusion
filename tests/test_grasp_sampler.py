"""Tests for grasp sampling strategies."""

import numpy as np
import pytest
from src.planning.grasp_sampler import GraspSampler, GraspCandidate
from src.utils.lie_groups import so3_exp, pose_from_Rt


@pytest.fixture
def sampler():
    return GraspSampler(gripper_width=0.08, standoff_distance=0.10)


@pytest.fixture
def object_pose():
    R = so3_exp(np.array([0.1, 0.0, 0.0]))
    t = np.array([0.3, 0.0, 0.15])
    return pose_from_Rt(R, t)


@pytest.fixture
def point_cloud_with_normals():
    """Synthetic box point cloud with normals."""
    np.random.seed(42)
    n = 200
    # Simple box: points on surfaces
    points = []
    normals = []
    for _ in range(n // 2):
        # Top surface
        p = np.array([np.random.uniform(-0.02, 0.02),
                       np.random.uniform(-0.02, 0.02), 0.03])
        points.append(p)
        normals.append(np.array([0, 0, 1]))
    for _ in range(n // 4):
        # Left surface
        p = np.array([-0.02, np.random.uniform(-0.02, 0.02),
                       np.random.uniform(0, 0.03)])
        points.append(p)
        normals.append(np.array([-1, 0, 0]))
    for _ in range(n // 4):
        # Right surface
        p = np.array([0.02, np.random.uniform(-0.02, 0.02),
                       np.random.uniform(0, 0.03)])
        points.append(p)
        normals.append(np.array([1, 0, 0]))

    return np.array(points) + np.array([0.3, 0.0, 0.15]), np.array(normals)


class TestGraspCandidate:
    def test_creation(self):
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.2, 0.3]
        g = GraspCandidate(pose=T, score=0.8, method="test")
        assert g.score == 0.8
        assert g.method == "test"
        np.testing.assert_array_equal(g.position(), [0.1, 0.2, 0.3])

    def test_approach_vector(self):
        T = np.eye(4)
        g = GraspCandidate(pose=T)
        # Z-axis of identity = [0, 0, 1]
        np.testing.assert_array_equal(g.approach_vector(), [0, 0, 1])

    def test_closing_vector(self):
        T = np.eye(4)
        g = GraspCandidate(pose=T)
        np.testing.assert_array_equal(g.closing_vector(), [1, 0, 0])


class TestGraspSampler:
    def test_topdown_sampling(self, sampler, object_pose):
        candidates = sampler.sample(
            object_pose=object_pose,
            n_candidates=20,
            methods=["topdown"],
        )
        assert len(candidates) > 0
        assert all(isinstance(g, GraspCandidate) for g in candidates)
        assert all(g.method == "topdown" for g in candidates)

    def test_scores_are_valid(self, sampler, object_pose):
        candidates = sampler.sample(
            object_pose=object_pose,
            n_candidates=20,
            methods=["topdown"],
        )
        for g in candidates:
            assert 0.0 <= g.score <= 1.0

    def test_sorted_by_score(self, sampler, object_pose):
        candidates = sampler.sample(
            object_pose=object_pose,
            n_candidates=20,
        )
        scores = [g.score for g in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_antipodal_sampling(self, sampler, object_pose, point_cloud_with_normals):
        pts, normals = point_cloud_with_normals
        candidates = sampler.sample(
            object_pose=object_pose,
            point_cloud=pts,
            normals=normals,
            n_candidates=20,
            methods=["antipodal"],
        )
        # Antipodal might not always find valid pairs, but should return list
        assert isinstance(candidates, list)

    def test_surface_sampling(self, sampler, object_pose, point_cloud_with_normals):
        pts, _ = point_cloud_with_normals
        candidates = sampler.sample(
            object_pose=object_pose,
            point_cloud=pts,
            n_candidates=20,
            methods=["surface"],
        )
        assert len(candidates) > 0

    def test_multi_method(self, sampler, object_pose, point_cloud_with_normals):
        pts, normals = point_cloud_with_normals
        candidates = sampler.sample(
            object_pose=object_pose,
            point_cloud=pts,
            normals=normals,
            n_candidates=50,
        )
        methods_used = set(g.method for g in candidates)
        assert len(methods_used) >= 1

    def test_approach_trajectory(self, sampler, object_pose):
        candidates = sampler.sample(
            object_pose=object_pose, n_candidates=5, methods=["topdown"]
        )
        if candidates:
            traj = sampler.generate_approach_trajectory(candidates[0], n_waypoints=5)
            assert traj.shape == (5, 4, 4)
            # Last waypoint should be at the grasp pose
            np.testing.assert_array_almost_equal(
                traj[-1], candidates[0].pose, decimal=5
            )
            # First waypoint should be further away (standoff)
            first_dist = np.linalg.norm(traj[0, :3, 3] - candidates[0].position())
            assert first_dist > 0.05  # should be standoff_distance away

    def test_side_approach(self, sampler, object_pose):
        candidates = sampler.sample(
            object_pose=object_pose,
            n_candidates=10,
            methods=["side"],
        )
        assert isinstance(candidates, list)
        for g in candidates:
            assert g.method == "side"


class TestCoppeliaBridge:
    def test_camera_config(self):
        from src.simulation.coppeliasim_bridge import CameraConfig
        cam = CameraConfig(resolution=(640, 480), fov=1.047)
        K = cam.K
        assert K.shape == (3, 3)
        assert K[0, 2] == 320.0  # cx = w/2
        assert K[1, 2] == 240.0  # cy = h/2
        assert K[0, 0] > 0  # fx > 0

    def test_robot_config(self):
        from src.simulation.coppeliasim_bridge import RobotConfig
        cfg = RobotConfig()
        assert cfg.n_joints == 6
        assert len(cfg.joint_names) == 6
        assert cfg.name == "UR5e"
