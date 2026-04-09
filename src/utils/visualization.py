"""
Visualization utilities for 6-DoF pose estimation.

Provides functions to overlay estimated poses on RGB images,
visualize 3D point clouds, and plot comparative results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def draw_pose_axes(
    img: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    axis_length: float = 0.1,
    line_width: int = 3
) -> np.ndarray:
    """Draw 3D coordinate axes on image at the estimated pose.

    RGB convention: X=Red, Y=Green, Z=Blue

    Args:
        img: (H, W, 3) BGR image
        R: (3, 3) rotation matrix
        t: (3,) translation vector (meters)
        K: (3, 3) camera intrinsic matrix
        axis_length: length of axes in meters
        line_width: thickness of drawn lines

    Returns:
        Image with axes drawn
    """
    img_out = img.copy()

    # 3D axis endpoints
    origin = t.reshape(3, 1)
    axes_3d = np.array([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ]).T  # (3, 3)

    axes_3d = R @ axes_3d + origin  # (3, 3)

    # Project to 2D
    def proj(pt3d):
        pt2d = K @ pt3d.reshape(3, 1)
        return int(pt2d[0, 0] / pt2d[2, 0]), int(pt2d[1, 0] / pt2d[2, 0])

    center = proj(origin)
    x_end = proj(axes_3d[:, 0])
    y_end = proj(axes_3d[:, 1])
    z_end = proj(axes_3d[:, 2])

    # Draw (BGR colors)
    cv2.line(img_out, center, x_end, (0, 0, 255), line_width)   # X = Red
    cv2.line(img_out, center, y_end, (0, 255, 0), line_width)   # Y = Green
    cv2.line(img_out, center, z_end, (255, 0, 0), line_width)   # Z = Blue

    return img_out


def draw_projected_points(
    img: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1
) -> np.ndarray:
    """Project 3D model points onto image and draw them.

    Args:
        img: (H, W, 3) BGR image
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        K: (3, 3) camera intrinsics
        points: (N, 3) model points
        color: BGR color tuple
        radius: point radius

    Returns:
        Image with projected points
    """
    img_out = img.copy()
    pts_cam = (R @ points.T).T + t
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

    for pt in pts_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_out, (x, y), radius, color, -1)

    return img_out


def draw_bbox_3d(
    img: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    bbox_3d: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2
) -> np.ndarray:
    """Draw 3D bounding box projected onto image.

    Args:
        img: (H, W, 3) BGR image
        R, t: pose
        K: camera intrinsics
        bbox_3d: (8, 3) corners of 3D bounding box
        color: BGR color
        line_width: line thickness

    Returns:
        Image with 3D bbox
    """
    img_out = img.copy()

    # Project corners
    corners_cam = (R @ bbox_3d.T).T + t
    corners_2d = (K @ corners_cam.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
    corners_2d = corners_2d.astype(int)

    # 12 edges of a cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    for i, j in edges:
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[j])
        cv2.line(img_out, pt1, pt2, color, line_width)

    return img_out


def plot_pose_comparison(
    img: np.ndarray,
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    K: np.ndarray,
    points: Optional[np.ndarray] = None,
    title: str = ""
) -> plt.Figure:
    """Side-by-side comparison of estimated vs ground-truth pose.

    Args:
        img: (H, W, 3) RGB image
        R_est, t_est: estimated pose
        R_gt, t_gt: ground-truth pose
        K: camera intrinsics
        points: (N, 3) model points (optional, for point overlay)
        title: plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Estimated
    img_est = draw_pose_axes(img_bgr.copy(), R_est, t_est, K)
    if points is not None:
        img_est = draw_projected_points(img_est, R_est, t_est, K, points,
                                         color=(0, 255, 0))

    # Ground truth
    img_gt = draw_pose_axes(img_bgr.copy(), R_gt, t_gt, K)
    if points is not None:
        img_gt = draw_projected_points(img_gt, R_gt, t_gt, K, points,
                                        color=(255, 0, 0))

    axes[0].imshow(cv2.cvtColor(img_est, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Estimated Pose")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    methods: List[str],
    vsd_scores: List[float],
    mssd_scores: List[float],
    mspd_scores: List[float],
    title: str = "BOP Metrics Comparison"
) -> plt.Figure:
    """Bar chart comparing BOP metrics across methods.

    Args:
        methods: list of method names
        vsd_scores: VSD recall per method
        mssd_scores: MSSD recall per method
        mspd_scores: MSPD recall per method
        title: plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, vsd_scores, width, label="VSD",
                   color="#0098CD")
    bars2 = ax.bar(x, mssd_scores, width, label="MSSD",
                   color="#006C8F")
    bars3 = ax.bar(x + width, mspd_scores, width, label="MSPD",
                   color="#333333")

    ax.set_ylabel("Recall (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig
