"""Tests for the end-to-end pipeline module."""

import numpy as np
import pytest
from src.pipeline import PipelineConfig, PoseResult, GraspResult, PipelineResult


class TestDataClasses:
    """Tests for pipeline data structures."""

    def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert config.pose_method == "foundationpose"
        assert config.grasp_method == "diffusion"
        assert config.n_grasp_samples == 4
        assert config.grasp_horizon == 16
        assert config.device == "cpu"

    def test_pose_result(self):
        result = PoseResult(
            obj_id=1,
            R=np.eye(3),
            t=np.zeros(3),
            score=0.95,
            T=np.eye(4),
        )
        assert result.obj_id == 1
        assert result.score == 0.95
        assert result.bbox is None
        assert result.mask is None

    def test_grasp_result(self):
        pose = PoseResult(
            obj_id=0, R=np.eye(3), t=np.zeros(3),
            score=0.9, T=np.eye(4),
        )
        grasp = GraspResult(
            trajectory=np.random.randn(16, 7),
            score=0.85,
            target_pose=pose,
        )
        assert grasp.trajectory.shape == (16, 7)
        assert grasp.score == 0.85

    def test_pipeline_result(self):
        result = PipelineResult(
            poses=[],
            grasps=[],
            best_grasp=None,
            timing={"pose_estimation": 0.1, "total": 0.1},
        )
        assert len(result.poses) == 0
        assert result.best_grasp is None
        assert result.timing["total"] == 0.1
