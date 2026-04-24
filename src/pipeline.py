"""
End-to-End Pipeline — Pose Estimation + Grasp Planning for Bin Picking.

Orchestrates the full perception-to-action pipeline:
    1. RGB-D capture
    2. Object detection + segmentation
    3. 6-DoF pose estimation (FoundationPose / GDR-Net)
    4. Grasp trajectory generation (Diffusion Policy)
    5. Execution command generation (for CoppeliaSim / ROS 2)

Usage:
    pipeline = BinPickingPipeline(config)
    pipeline.initialize()
    result = pipeline.run(rgb, depth, K)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the bin picking pipeline."""
    # Pose estimation
    pose_method: str = "foundationpose"  # "foundationpose" or "gdrnet"
    pose_weights_dir: str = "weights/"
    cad_model_path: str = ""

    # Grasp planning
    grasp_method: str = "diffusion"  # "diffusion" or "heuristic"
    n_grasp_samples: int = 4
    grasp_horizon: int = 16

    # Camera
    depth_scale: float = 1.0  # depth to meters conversion

    # Thresholds
    min_detection_score: float = 0.5
    min_pose_score: float = 0.3

    # Device
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass
class PoseResult:
    """Result of pose estimation for a single object."""
    obj_id: int
    R: np.ndarray           # (3, 3) rotation
    t: np.ndarray            # (3,) translation in meters
    score: float
    T: np.ndarray            # (4, 4) SE(3) matrix
    bbox: Optional[np.ndarray] = None  # (4,) [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None  # (H, W) binary mask


@dataclass
class GraspResult:
    """Result of grasp planning."""
    trajectory: np.ndarray   # (horizon, 7) [x, y, z, rx, ry, rz, gripper]
    score: float
    target_pose: PoseResult


@dataclass
class PipelineResult:
    """Full pipeline output."""
    poses: List[PoseResult]
    grasps: List[GraspResult]
    best_grasp: Optional[GraspResult] = None
    timing: Dict[str, float] = field(default_factory=dict)


class BinPickingPipeline:
    """Main pipeline orchestrating pose estimation and grasp planning."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pose_estimator = None
        self._grasp_planner = None
        self._initialized = False

    def initialize(self):
        """Load all models. Call once before running."""
        logger.info("Initializing Bin Picking Pipeline...")
        logger.info(f"  Pose method: {self.config.pose_method}")
        logger.info(f"  Grasp method: {self.config.grasp_method}")

        # Initialize pose estimator
        if self.config.pose_method == "foundationpose":
            from src.perception.foundation_pose import FoundationPoseEstimator
            self._pose_estimator = FoundationPoseEstimator(
                weights_dir=self.config.pose_weights_dir,
                device=self.config.device,
            )
        elif self.config.pose_method == "gdrnet":
            from src.perception.gdrnet import GDRNetEstimator
            self._pose_estimator = GDRNetEstimator(
                device=self.config.device,
            )
        else:
            raise ValueError(f"Unknown pose method: {self.config.pose_method}")

        # Load CAD model if provided
        if self.config.cad_model_path:
            self._pose_estimator.load_cad_model(self.config.cad_model_path)

        # Initialize grasp planner
        from src.planning.diffusion_policy import DiffusionGraspPlanner
        self._grasp_planner = DiffusionGraspPlanner(
            horizon=self.config.grasp_horizon,
            device=self.config.device,
        )

        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def estimate_poses(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[PoseResult]:
        """Estimate 6-DoF poses for all detected objects.

        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth map (raw sensor values)
            K: (3, 3) camera intrinsics
            masks: optional list of binary masks (one per object)

        Returns:
            List of PoseResult
        """
        depth_m = depth.astype(np.float32) * self.config.depth_scale

        if masks is None:
            # TODO: Run detection + segmentation (CNOS/SAM)
            logger.warning("No masks provided — using full image")
            masks = [np.ones(depth.shape[:2], dtype=bool)]

        results = []
        for i, mask in enumerate(masks):
            try:
                pred = self._pose_estimator.estimate_pose(
                    rgb=rgb, depth=depth_m, mask=mask, K=K
                )
                result = PoseResult(
                    obj_id=i,
                    R=pred["R"],
                    t=pred["t"],
                    score=pred["score"],
                    T=pred["T"],
                    mask=mask,
                )
                if result.score >= self.config.min_pose_score:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Pose estimation failed for object {i}: {e}")

        return results

    def plan_grasps(
        self,
        poses: List[PoseResult],
    ) -> List[GraspResult]:
        """Plan grasp trajectories for estimated poses.

        Args:
            poses: list of PoseResult from estimate_poses

        Returns:
            List of GraspResult, sorted by score (best first)
        """
        results = []

        for pose in poses:
            if self.config.grasp_method == "diffusion":
                trajectories = self._grasp_planner.plan_grasp(
                    object_pose=pose.T,
                    n_samples=self.config.n_grasp_samples,
                )
            else:
                trajectories = self._grasp_planner.plan_grasp_heuristic(
                    object_pose=pose.T,
                )

            # Score each trajectory (simple: use pose confidence)
            for traj in trajectories:
                results.append(GraspResult(
                    trajectory=traj,
                    score=pose.score,
                    target_pose=pose,
                ))

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def run(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth map
            K: (3, 3) camera intrinsics
            masks: optional segmentation masks

        Returns:
            PipelineResult with poses, grasps, and timing
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        timing = {}

        # Step 1: Pose estimation
        t0 = time.time()
        poses = self.estimate_poses(rgb, depth, K, masks)
        timing["pose_estimation"] = time.time() - t0
        logger.info(f"Estimated {len(poses)} poses in {timing['pose_estimation']:.3f}s")

        # Step 2: Grasp planning
        t0 = time.time()
        grasps = self.plan_grasps(poses)
        timing["grasp_planning"] = time.time() - t0
        logger.info(f"Planned {len(grasps)} grasps in {timing['grasp_planning']:.3f}s")

        # Select best grasp
        best = grasps[0] if grasps else None

        timing["total"] = sum(timing.values())

        return PipelineResult(
            poses=poses,
            grasps=grasps,
            best_grasp=best,
            timing=timing,
        )

    def run_on_dataset(
        self,
        dataset_root: str,
        split: str = "test",
        max_scenes: Optional[int] = None,
    ) -> Dict:
        """Run pipeline on a BOP dataset for evaluation.

        Args:
            dataset_root: path to BOP dataset
            split: "test" or "train"
            max_scenes: limit number of scenes (for debugging)

        Returns:
            dict with predictions in BOP format
        """
        from collections import defaultdict
        from src.utils.dataset_loader import BOPDataset

        dataset = BOPDataset(dataset_root, split)
        predictions = {}

        # BOP-19 test subset: un dataset de video (YCB-V) tiene miles de frames pero solo
        # los listados en test_targets_bop19.json son evaluables contra el leaderboard.
        bop_targets = dataset.load_bop_test_targets()
        targets_by_scene: Dict[str, set] = defaultdict(set)
        for t in bop_targets:
            targets_by_scene[f"{t['scene_id']:06d}"].add(int(t['im_id']))

        scenes = [s for s in dataset.get_scene_ids() if s in targets_by_scene] \
            if targets_by_scene else dataset.get_scene_ids()
        if max_scenes:
            scenes = scenes[:max_scenes]

        for scene_id in scenes:
            # Solo imagenes del BOP subset (si existe). Sin subset, fallback a todos los frames.
            if targets_by_scene:
                image_ids = sorted(targets_by_scene[scene_id])
            else:
                image_ids = dataset.get_image_ids(scene_id)
            logger.info(f"Processing scene {scene_id} ({len(image_ids)} images)")

            for img_id in image_ids:
                try:
                    sample = dataset.load_sample(scene_id, img_id)
                    result = self.run(
                        rgb=sample["rgb"],
                        depth=sample["depth"],
                        K=sample["cam_K"],
                    )

                    # Key format debe coincidir con `evaluator.evaluate_method` que usa
                    # `img_id_str` de scene_gt.json (entero como string, NO zero-padded).
                    for pose in result.poses:
                        key = f"{scene_id}/{img_id}"
                        predictions[key] = {
                            "obj_id": pose.obj_id,
                            "R": pose.R.tolist(),
                            "t": pose.t.tolist(),
                            "score": pose.score,
                        }
                except Exception as e:
                    logger.warning(f"Failed on {scene_id}/{img_id}: {e}")

        return predictions
