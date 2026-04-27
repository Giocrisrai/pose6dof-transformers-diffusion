"""
2D Object Detector — Detection and Segmentation for Pose Estimation.

Provides bounding boxes and masks as input to FoundationPose / GDR-Net++.
Supports multiple detection backends:
    - CNOS (Cutie + SAM): zero-shot segmentation
    - YOLOX: trained detector (used by GDR-Net++)
    - Ground truth masks (for ablation studies)

References:
    - Nguyen et al. (2023) "CNOS: A Strong Baseline for CAD-based
      Novel Object Segmentation"
    - Ge et al. (2021) "YOLOX: Exceeding YOLO Series in 2021"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def _cv2():
    """Importa cv2 perezosamente. Mensaje claro si falta."""
    try:
        import cv2 as _cv
        return _cv
    except ImportError as e:
        raise ImportError(
            "Esta función requiere opencv-python. Instala con "
            "`pip install opencv-python` o `uv sync`."
        ) from e


@dataclass
class Detection:
    """A single object detection result."""
    bbox: np.ndarray       # (4,) [x1, y1, x2, y2]
    mask: np.ndarray       # (H, W) binary mask
    score: float           # confidence score
    class_id: int          # object class ID
    class_name: str = ""   # optional label


class GTDetector:
    """Ground truth detector — uses BOP annotations directly.

    Useful for isolating pose estimation errors from detection errors
    in ablation studies.
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: BOPDataset instance with GT annotations
        """
        self.dataset = dataset

    def detect(self, scene_id: str, img_id: int) -> List[Detection]:
        """Return GT detections for an image.

        Args:
            scene_id: BOP scene identifier
            img_id: image index

        Returns:
            list of Detection objects with GT bboxes and masks
        """
        gt_poses = self.dataset.load_scene_gt(scene_id)
        gt_list = gt_poses.get(str(img_id), [])

        detections = []
        for obj_idx, gt in enumerate(gt_list):
            # Load visible mask
            mask = self.dataset.load_mask(
                scene_id, img_id, obj_idx, visible_only=True
            )
            if mask is None:
                continue

            # Compute bounding box from mask
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue

            detections.append(Detection(
                bbox=bbox,
                mask=mask,
                score=1.0,
                class_id=gt["obj_id"],
            ))

        return detections


class SimpleSegmentor:
    """Simple depth-based segmentation for bin picking scenarios.

    Uses depth discontinuities and connected components to segment
    objects on a planar surface (table/bin). No neural network needed.

    Suitable for structured environments with known workspace geometry.
    """

    def __init__(
        self,
        depth_threshold: float = 10.0,   # mm
        min_area: int = 500,              # pixels
        max_area: int = 100000,           # pixels
        plane_distance: float = 50.0,     # mm from plane
    ):
        self.depth_threshold = depth_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.plane_distance = plane_distance

    def detect(
        self,
        depth: np.ndarray,
        plane_height: Optional[float] = None,
    ) -> List[Detection]:
        """Segment objects from depth image.

        Args:
            depth: (H, W) depth map in mm
            plane_height: known table plane depth in mm.
                If None, estimated as the mode of the depth histogram.

        Returns:
            list of Detection objects
        """
        # Estimate plane if not provided
        if plane_height is None:
            valid_depth = depth[depth > 0]
            if len(valid_depth) == 0:
                return []
            # Use histogram mode as plane estimate
            hist, edges = np.histogram(valid_depth, bins=100)
            plane_height = edges[np.argmax(hist)]

        cv2 = _cv2()
        # Objects are above the plane (closer to camera = smaller depth)
        object_mask = (depth > 0) & (depth < plane_height - self.plane_distance)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask = object_mask.astype(np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            object_mask, connectivity=8
        )

        detections = []
        for i in range(1, n_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue

            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            mask_i = (labels == i).astype(bool)

            detections.append(Detection(
                bbox=np.array([x1, y1, x1 + w, y1 + h]),
                mask=mask_i,
                score=float(area) / self.max_area,  # pseudo-score
                class_id=-1,  # unknown class
            ))

        # Sort by score (area) descending
        detections.sort(key=lambda d: d.score, reverse=True)
        logger.info(f"SimpleSegmentor: found {len(detections)} objects")
        return detections


# ── Utility functions ──

def mask_to_bbox(mask: np.ndarray, margin: int = 0) -> Optional[np.ndarray]:
    """Extract tight bounding box from binary mask.

    Args:
        mask: (H, W) binary mask
        margin: pixels to add around the bbox

    Returns:
        (4,) array [x1, y1, x2, y2] or None if mask is empty
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    y1 = max(0, rows[0] - margin)
    y2 = min(mask.shape[0], rows[-1] + 1 + margin)
    x1 = max(0, cols[0] - margin)
    x2 = min(mask.shape[1], cols[-1] + 1 + margin)

    return np.array([x1, y1, x2, y2])


def bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        bbox_a, bbox_b: (4,) arrays [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1]
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def draw_detections(
    img: np.ndarray,
    detections: List[Detection],
    show_masks: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Visualize detections on an image.

    Args:
        img: (H, W, 3) RGB image
        detections: list of Detection objects
        show_masks: overlay semi-transparent masks
        show_labels: show class ID and score

    Returns:
        annotated image
    """
    cv2 = _cv2()
    img_out = img.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]

        # Mask overlay
        if show_masks and det.mask is not None:
            mask_rgb = np.zeros_like(img_out)
            mask_rgb[det.mask] = color
            img_out = cv2.addWeighted(img_out, 1.0, mask_rgb, 0.3, 0)

        # Bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # Label
        if show_labels:
            label = f"ID:{det.class_id} ({det.score:.2f})"
            cv2.putText(img_out, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_out
