"""
BOP Challenge evaluation metrics for 6-DoF pose estimation.

Implements the three official BOP metrics:
    - VSD:  Visible Surface Discrepancy
    - MSSD: Maximum Symmetry-Aware Surface Distance
    - MSPD: Maximum Symmetry-Aware Projection Distance

References:
    - Hodaň et al. (2020) "BOP: Benchmark for 6D Object Pose Estimation"
    - Hodaň et al. (2025) "BOP Challenge 2024 on Detection, Segmentation
      and Pose Estimation of Seen and Unseen Rigid Objects"
"""

import numpy as np
from typing import List, Optional, Tuple


def add_metric(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    points: np.ndarray
) -> float:
    """Average Distance of model points (ADD).

    Classic pose error metric: average L2 distance between
    model points transformed by estimated and ground-truth poses.

    ADD = (1/m) Σ ||R_est @ p_i + t_est - R_gt @ p_i - t_gt||

    Args:
        R_est: (3, 3) estimated rotation
        t_est: (3,) estimated translation
        R_gt: (3, 3) ground-truth rotation
        t_gt: (3,) ground-truth translation
        points: (N, 3) model point cloud

    Returns:
        ADD score (lower is better)
    """
    est_pts = (R_est @ points.T).T + t_est
    gt_pts = (R_gt @ points.T).T + t_gt
    return np.mean(np.linalg.norm(est_pts - gt_pts, axis=1))


def add_s_metric(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    points: np.ndarray
) -> float:
    """ADD-S: Symmetry-aware average distance.

    Uses closest point distance instead of corresponding points,
    making it suitable for symmetric objects.

    ADD-S = (1/m) Σ min_j ||R_est @ p_i + t_est - R_gt @ p_j - t_gt||

    Args:
        R_est, t_est, R_gt, t_gt: estimated and GT pose
        points: (N, 3) model point cloud

    Returns:
        ADD-S score (lower is better)
    """
    est_pts = (R_est @ points.T).T + t_est
    gt_pts = (R_gt @ points.T).T + t_gt

    distances = []
    for p_est in est_pts:
        dists = np.linalg.norm(gt_pts - p_est, axis=1)
        distances.append(np.min(dists))

    return np.mean(distances)


def vsd(
    depth_est: np.ndarray,
    depth_gt: np.ndarray,
    depth_test: np.ndarray,
    delta: float = 15.0,
    tau: float = 20.0,
    cost_type: str = "step"
) -> float:
    """Visible Surface Discrepancy (VSD).

    Compares rendered depth maps of estimated and ground-truth poses,
    considering only visible surfaces. Robust to symmetries.

    Args:
        depth_est: (H, W) rendered depth from estimated pose
        depth_gt: (H, W) rendered depth from ground-truth pose
        depth_test: (H, W) observed depth from sensor
        delta: visibility threshold in mm
        tau: distance threshold in mm for the step cost
        cost_type: "step" or "tlinear"

    Returns:
        VSD score in [0, 1] (lower is better)
    """
    # Visibility masks
    V_est = (depth_est > 0) & (np.abs(depth_est - depth_test) < delta)
    V_gt = (depth_gt > 0) & (np.abs(depth_gt - depth_test) < delta)

    # Union of visible regions
    V_union = V_est | V_gt

    if np.sum(V_union) == 0:
        return 0.0

    # Distance between rendered depth maps
    diff = np.abs(depth_est - depth_gt)

    if cost_type == "step":
        cost = np.where(diff <= tau, 0.0, 1.0)
    elif cost_type == "tlinear":
        cost = np.clip(diff / tau, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

    # Average cost over visible union
    return np.sum(cost[V_union]) / np.sum(V_union)


def mssd(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    points: np.ndarray,
    symmetries: Optional[List[np.ndarray]] = None
) -> float:
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    Finds the best-matching symmetry transformation and computes
    the maximum point-wise distance.

    Args:
        R_est, t_est: estimated pose
        R_gt, t_gt: ground-truth pose
        points: (N, 3) model surface points
        symmetries: list of (4, 4) symmetry transformations of the object.
                    If None, uses identity only (no symmetries).

    Returns:
        MSSD score in mm (lower is better)
    """
    if symmetries is None:
        symmetries = [np.eye(4)]

    est_pts = (R_est @ points.T).T + t_est

    best_mssd = float("inf")
    for S in symmetries:
        R_s = S[:3, :3]
        t_s = S[:3, 3]
        # Apply symmetry to ground-truth
        R_gt_s = R_gt @ R_s
        t_gt_s = R_gt @ t_s + t_gt
        gt_pts_s = (R_gt_s @ points.T).T + t_gt_s

        max_dist = np.max(np.linalg.norm(est_pts - gt_pts_s, axis=1))
        best_mssd = min(best_mssd, max_dist)

    return best_mssd


def mspd(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    points: np.ndarray,
    K: np.ndarray,
    symmetries: Optional[List[np.ndarray]] = None
) -> float:
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    Like MSSD but in 2D projection space — measures the maximum
    reprojection error considering symmetries.

    Args:
        R_est, t_est: estimated pose
        R_gt, t_gt: ground-truth pose
        points: (N, 3) model surface points
        K: (3, 3) camera intrinsic matrix
        symmetries: list of (4, 4) symmetry transformations

    Returns:
        MSPD score in pixels (lower is better)
    """
    if symmetries is None:
        symmetries = [np.eye(4)]

    def project(R, t, pts):
        """Project 3D points to 2D using pinhole model."""
        pts_cam = (R @ pts.T).T + t
        pts_2d = (K @ pts_cam.T).T
        return pts_2d[:, :2] / pts_2d[:, 2:3]

    proj_est = project(R_est, t_est, points)

    best_mspd = float("inf")
    for S in symmetries:
        R_s = S[:3, :3]
        t_s = S[:3, 3]
        R_gt_s = R_gt @ R_s
        t_gt_s = R_gt @ t_s + t_gt
        proj_gt_s = project(R_gt_s, t_gt_s, points)

        max_dist = np.max(np.linalg.norm(proj_est - proj_gt_s, axis=1))
        best_mspd = min(best_mspd, max_dist)

    return best_mspd


def compute_recall(
    errors: List[float],
    threshold: float
) -> float:
    """Compute recall at a given threshold.

    Args:
        errors: list of error values (VSD, MSSD, or MSPD)
        threshold: error threshold

    Returns:
        Recall rate in [0, 1]
    """
    correct = sum(1 for e in errors if e <= threshold)
    return correct / len(errors) if errors else 0.0


def compute_auc(
    errors: List[float],
    max_threshold: float,
    num_steps: int = 100
) -> float:
    """Compute Area Under the Recall Curve.

    Args:
        errors: list of error values
        max_threshold: maximum threshold for the curve
        num_steps: number of evaluation points

    Returns:
        AUC in [0, 1]
    """
    thresholds = np.linspace(0, max_threshold, num_steps)
    recalls = [compute_recall(errors, t) for t in thresholds]
    return np.trapz(recalls, thresholds) / max_threshold


# ── Convenience aliases (used by notebooks and Colab) ──────────────────

def compute_add(
    R_est: np.ndarray, t_est: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    points: np.ndarray,
) -> float:
    """Alias for add_metric — used in notebooks."""
    return add_metric(R_est, t_est, R_gt, t_gt, points)


def compute_adds(
    R_est: np.ndarray, t_est: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    points: np.ndarray,
) -> float:
    """Alias for add_s_metric — used in notebooks."""
    return add_s_metric(R_est, t_est, R_gt, t_gt, points)
