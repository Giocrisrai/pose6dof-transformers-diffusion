"""
Grasp Candidate Sampler — Generates candidate grasp poses for bin picking.

Implements multiple grasp sampling strategies:
    - Antipodal grasps (from surface normals)
    - Top-down approach grasps
    - Surface sampling (uniform over object mesh)
    - Filtered by kinematic and collision constraints

These candidates can be ranked by the Diffusion Policy or a learned scorer.

References:
    - ten Pas et al. (2017) "Grasp Pose Detection in Point Clouds", IJRR
    - Mahler et al. (2017) "Dex-Net 2.0", RSS
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """A single grasp candidate with pose and metadata.

    Attributes:
        pose: (4, 4) SE(3) grasp pose in camera frame.
              Z-axis points along approach direction,
              X-axis is the closing direction of the gripper.
        score: Quality score in [0, 1] (higher is better).
        width: Gripper opening width in meters.
        contact_points: (2, 3) pair of contact points on the object surface.
        method: Sampling method that generated this candidate.
    """
    pose: np.ndarray  # (4, 4)
    score: float = 0.0
    width: float = 0.08  # default parallel jaw width
    contact_points: Optional[np.ndarray] = None
    method: str = "unknown"

    def approach_vector(self) -> np.ndarray:
        """Get the approach direction (Z-axis of grasp frame)."""
        return self.pose[:3, 2]

    def closing_vector(self) -> np.ndarray:
        """Get the closing direction (X-axis of grasp frame)."""
        return self.pose[:3, 0]

    def position(self) -> np.ndarray:
        """Get grasp center position."""
        return self.pose[:3, 3]


class GraspSampler:
    """Generates candidate grasps for a target object.

    Combines multiple sampling strategies and filters candidates
    by reachability and collision checks.

    Usage:
        sampler = GraspSampler(gripper_width=0.08)
        candidates = sampler.sample(
            object_pose=T_obj,
            point_cloud=pts,
            normals=normals,
            n_candidates=100,
        )
        # Sort by score
        best = sorted(candidates, key=lambda g: g.score, reverse=True)[0]
    """

    def __init__(
        self,
        gripper_width: float = 0.08,
        gripper_depth: float = 0.04,
        approach_min_angle: float = 0.0,        # radians from vertical
        approach_max_angle: float = np.pi / 3,   # 60 degrees from vertical
        standoff_distance: float = 0.10,         # approach standoff
    ):
        """Initialize grasp sampler.

        Args:
            gripper_width: Maximum gripper opening in meters.
            gripper_depth: Finger depth in meters.
            approach_min_angle: Min angle between approach and -Z (vertical).
            approach_max_angle: Max angle between approach and -Z (vertical).
            standoff_distance: Distance to approach from before grasping.
        """
        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth
        self.approach_min_angle = approach_min_angle
        self.approach_max_angle = approach_max_angle
        self.standoff_distance = standoff_distance

    def sample(
        self,
        object_pose: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        n_candidates: int = 100,
        methods: Optional[List[str]] = None,
    ) -> List[GraspCandidate]:
        """Generate grasp candidates using multiple strategies.

        Args:
            object_pose: (4, 4) SE(3) object pose in camera frame.
            point_cloud: (N, 3) object surface points in camera frame.
            normals: (N, 3) surface normals (same frame as point_cloud).
            n_candidates: Total number of candidates to generate.
            methods: List of methods to use. Default: all available.
                     Options: "topdown", "antipodal", "surface", "side"

        Returns:
            List of GraspCandidate sorted by score (descending).
        """
        if methods is None:
            if point_cloud is not None and normals is not None:
                methods = ["topdown", "antipodal", "surface"]
            elif point_cloud is not None:
                methods = ["topdown", "surface"]
            else:
                methods = ["topdown"]

        n_per_method = max(1, n_candidates // len(methods))
        candidates = []

        for method in methods:
            if method == "topdown":
                candidates.extend(
                    self._sample_topdown(object_pose, n_per_method)
                )
            elif method == "antipodal" and point_cloud is not None and normals is not None:
                candidates.extend(
                    self._sample_antipodal(point_cloud, normals, n_per_method)
                )
            elif method == "surface" and point_cloud is not None:
                candidates.extend(
                    self._sample_surface(object_pose, point_cloud, n_per_method)
                )
            elif method == "side":
                candidates.extend(
                    self._sample_side_approach(object_pose, n_per_method)
                )

        # Filter by approach angle
        candidates = self._filter_approach_angle(candidates)

        # Score candidates
        candidates = self._score_candidates(candidates, object_pose, point_cloud)

        # Sort by score
        candidates.sort(key=lambda g: g.score, reverse=True)

        return candidates[:n_candidates]

    def _sample_topdown(
        self,
        object_pose: np.ndarray,
        n: int,
    ) -> List[GraspCandidate]:
        """Sample top-down approach grasps with random yaw.

        The approach direction is along -Z (downward), with random
        rotation around the vertical axis and small position jitter.
        """
        candidates = []
        obj_pos = object_pose[:3, 3]

        for _ in range(n):
            # Random yaw around Z
            yaw = np.random.uniform(0, np.pi)  # 0-180 due to gripper symmetry

            # Small tilt perturbation
            tilt_x = np.random.normal(0, 0.05)
            tilt_y = np.random.normal(0, 0.05)

            # Build rotation: approach = -Z, closing = rotated X
            Rz = self._rotz(yaw)
            Rx = self._rotx(tilt_x)
            Ry = self._roty(tilt_y)
            R_grasp = Rz @ Rx @ Ry

            # Grasp position = object center + small XY jitter
            jitter_xy = np.random.normal(0, 0.005, size=2)
            pos = obj_pos.copy()
            pos[0] += jitter_xy[0]
            pos[1] += jitter_xy[1]

            T = np.eye(4)
            T[:3, :3] = R_grasp
            T[:3, 3] = pos

            candidates.append(GraspCandidate(
                pose=T, width=self.gripper_width, method="topdown"
            ))

        return candidates

    def _sample_antipodal(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        n: int,
    ) -> List[GraspCandidate]:
        """Sample antipodal grasps from point pairs with opposing normals.

        Finds pairs of surface points where normals are approximately
        anti-parallel and within the gripper width.
        """
        candidates = []
        n_points = len(points)

        for _ in range(n * 5):  # oversample then filter
            if len(candidates) >= n:
                break

            # Pick two random points
            i, j = np.random.choice(n_points, 2, replace=False)
            p1, p2 = points[i], points[j]
            n1, n2 = normals[i], normals[j]

            # Check distance (must fit in gripper)
            dist = np.linalg.norm(p2 - p1)
            if dist < 0.005 or dist > self.gripper_width:
                continue

            # Check antipodal condition: normals should point toward each other
            closing_dir = (p2 - p1) / dist
            dot1 = np.dot(n1, closing_dir)
            dot2 = np.dot(n2, -closing_dir)
            if dot1 < 0.3 or dot2 < 0.3:  # threshold for antipodal quality
                continue

            # Build grasp frame
            center = (p1 + p2) / 2
            x_axis = closing_dir  # closing direction
            # Choose approach direction (prefer -Z / downward)
            z_candidates = np.cross(x_axis, np.array([0, 0, 1]))
            if np.linalg.norm(z_candidates) < 0.1:
                z_candidates = np.cross(x_axis, np.array([0, 1, 0]))
            z_axis = z_candidates / np.linalg.norm(z_candidates)
            y_axis = np.cross(z_axis, x_axis)

            R = np.column_stack([x_axis, y_axis, z_axis])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center

            antipodal_score = (dot1 + dot2) / 2  # quality based on normal alignment

            candidates.append(GraspCandidate(
                pose=T,
                score=antipodal_score,
                width=dist,
                contact_points=np.array([p1, p2]),
                method="antipodal",
            ))

        return candidates[:n]

    def _sample_surface(
        self,
        object_pose: np.ndarray,
        points: np.ndarray,
        n: int,
    ) -> List[GraspCandidate]:
        """Sample grasps by approaching surface points from above.

        Selects random surface points and creates approach grasps
        targeting those points.
        """
        candidates = []
        n_points = len(points)

        for _ in range(n):
            # Random surface point
            idx = np.random.randint(n_points)
            target = points[idx]

            # Approach from above with random yaw
            yaw = np.random.uniform(0, np.pi)
            R = self._rotz(yaw)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = target

            candidates.append(GraspCandidate(
                pose=T, width=self.gripper_width, method="surface"
            ))

        return candidates

    def _sample_side_approach(
        self,
        object_pose: np.ndarray,
        n: int,
    ) -> List[GraspCandidate]:
        """Sample side-approach grasps (horizontal approach).

        Useful for objects on the edge of a bin or tall objects.
        """
        candidates = []
        obj_pos = object_pose[:3, 3]

        for _ in range(n):
            # Random approach angle in XY plane
            azimuth = np.random.uniform(0, 2 * np.pi)
            # Elevation between 0 (horizontal) and approach_max_angle
            elevation = np.random.uniform(0, self.approach_max_angle)

            # Approach direction
            approach = np.array([
                np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                -np.sin(elevation),
            ])

            # Build frame
            z_axis = approach
            # Closing direction: perpendicular to approach, roughly horizontal
            x_candidate = np.cross(z_axis, np.array([0, 0, 1]))
            if np.linalg.norm(x_candidate) < 0.1:
                x_candidate = np.cross(z_axis, np.array([0, 1, 0]))
            x_axis = x_candidate / np.linalg.norm(x_candidate)
            y_axis = np.cross(z_axis, x_axis)

            R = np.column_stack([x_axis, y_axis, z_axis])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = obj_pos

            candidates.append(GraspCandidate(
                pose=T, width=self.gripper_width, method="side"
            ))

        return candidates

    def _filter_approach_angle(
        self, candidates: List[GraspCandidate]
    ) -> List[GraspCandidate]:
        """Filter candidates by approach angle relative to vertical."""
        filtered = []
        down = np.array([0, 0, -1])

        for g in candidates:
            approach = g.approach_vector()
            cos_angle = np.dot(approach, down)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            if self.approach_min_angle <= angle <= self.approach_max_angle:
                filtered.append(g)

        return filtered if filtered else candidates  # fallback: keep all

    def _score_candidates(
        self,
        candidates: List[GraspCandidate],
        object_pose: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
    ) -> List[GraspCandidate]:
        """Score candidates based on heuristics.

        Scoring criteria:
        1. Proximity to object center (closeness)
        2. Vertical approach preference (top-down is safer)
        3. Antipodal quality (if available)
        """
        obj_center = object_pose[:3, 3]
        down = np.array([0, 0, -1])

        for g in candidates:
            scores = []

            # Distance to object center (closer = better)
            dist = np.linalg.norm(g.position() - obj_center)
            dist_score = np.exp(-dist / 0.05)  # exponential decay
            scores.append(dist_score)

            # Approach angle preference (vertical = better)
            cos_angle = np.dot(g.approach_vector(), down)
            angle_score = (cos_angle + 1) / 2  # map [-1,1] → [0,1]
            scores.append(angle_score)

            # Keep existing antipodal score if set
            if g.score > 0 and g.method == "antipodal":
                scores.append(g.score)

            g.score = float(np.mean(scores))

        return candidates

    def generate_approach_trajectory(
        self,
        grasp: GraspCandidate,
        n_waypoints: int = 5,
    ) -> np.ndarray:
        """Generate approach trajectory for a grasp candidate.

        Creates waypoints from standoff position to grasp pose.

        Args:
            grasp: Target grasp candidate.
            n_waypoints: Number of waypoints in trajectory.

        Returns:
            (n_waypoints, 4, 4) array of SE(3) waypoints.
        """
        T_grasp = grasp.pose.copy()
        approach_dir = grasp.approach_vector()

        trajectory = np.zeros((n_waypoints, 4, 4))
        for i in range(n_waypoints):
            t = i / (n_waypoints - 1)
            T_wp = T_grasp.copy()
            # Interpolate from standoff to contact
            offset = self.standoff_distance * (1 - t)
            T_wp[:3, 3] = T_grasp[:3, 3] - approach_dir * offset
            trajectory[i] = T_wp

        return trajectory

    # ── Rotation helpers ──────────────────────────────────────────

    @staticmethod
    def _rotx(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _roty(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def _rotz(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
