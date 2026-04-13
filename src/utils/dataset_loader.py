"""
BOP Dataset Loader for T-LESS and YCB-Video.

Loads RGB-D images, ground-truth poses, camera intrinsics, and 3D models
in the standard BOP format.

References:
    - BOP format: https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class BOPDataset:
    """Loader for datasets in BOP format.

    Expected structure:
        dataset_root/
        ├── models/           # 3D object models (.ply)
        │   ├── models_info.json
        │   ├── obj_000001.ply
        │   └── ...
        ├── test/             # Test scenes
        │   ├── 000001/
        │   │   ├── rgb/
        │   │   ├── depth/
        │   │   ├── mask/        (optional)
        │   │   ├── mask_visib/  (optional)
        │   │   ├── scene_camera.json
        │   │   └── scene_gt.json
        │   └── ...
        ├── camera.json       # Default camera intrinsics
        └── test_targets_bop19.json  # Evaluation targets
    """

    def __init__(self, dataset_root: str, split: str = "test"):
        """Initialize BOP dataset loader.

        Args:
            dataset_root: Path to the dataset root (e.g., data/datasets/tless)
            split: "test" or "train"
        """
        self.root = Path(dataset_root)
        self.split = split
        self.split_dir = self.root / split

        # Load default camera
        camera_path = self.root / "camera.json"
        if camera_path.exists():
            with open(camera_path) as f:
                cam = json.load(f)
            self.default_K = np.array([
                [cam["fx"], 0, cam["cx"]],
                [0, cam["fy"], cam["cy"]],
                [0, 0, 1]
            ])
            self.depth_scale = cam.get("depth_scale", 1.0)
        else:
            self.default_K = None
            self.depth_scale = 1.0

        # Discover scenes
        self.scenes = sorted([
            d.name for d in self.split_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]) if self.split_dir.exists() else []

        # JSON cache for scene_camera and scene_gt (avoid re-reading per image)
        self._camera_cache: Dict[str, Dict] = {}
        self._gt_cache: Dict[str, Dict] = {}

        # Load models info
        models_info_path = self.root / "models" / "models_info.json"
        if models_info_path.exists():
            with open(models_info_path) as f:
                self.models_info = json.load(f)
        else:
            self.models_info = {}

    def __repr__(self) -> str:
        return (
            f"BOPDataset(root='{self.root}', split='{self.split}', "
            f"scenes={len(self.scenes)}, objects={len(self.models_info)})"
        )

    def get_scene_ids(self) -> List[str]:
        """List all scene IDs in this split."""
        return self.scenes

    def get_image_ids(self, scene_id: str) -> List[int]:
        """List available image IDs in a scene (sorted).

        BOP image IDs may not start from 0 (e.g., YCB-Video starts from 1).
        """
        rgb_dir = self.split_dir / scene_id / "rgb"
        if not rgb_dir.exists():
            return []
        files = list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))
        ids = sorted(int(f.stem) for f in files)
        return ids

    def get_num_images(self, scene_id: str) -> int:
        """Number of images in a scene."""
        return len(self.get_image_ids(scene_id))

    def load_scene_camera(self, scene_id: str) -> Dict:
        """Load per-image camera parameters for a scene (cached).

        Returns:
            dict mapping image_id (str) → {"cam_K": (3,3), "depth_scale": float}
        """
        if scene_id in self._camera_cache:
            return self._camera_cache[scene_id]

        path = self.split_dir / scene_id / "scene_camera.json"
        with open(path) as f:
            raw = json.load(f)

        cameras = {}
        for img_id, cam in raw.items():
            K = np.array(cam["cam_K"]).reshape(3, 3)
            ds = cam.get("depth_scale", self.depth_scale)
            cameras[img_id] = {"cam_K": K, "depth_scale": ds}

        self._camera_cache[scene_id] = cameras
        return cameras

    def load_scene_gt(self, scene_id: str) -> Dict:
        """Load ground-truth annotations for a scene (cached).

        Returns:
            dict mapping image_id (str) → list of {
                "obj_id": int,
                "cam_R_m2c": (3,3) rotation,
                "cam_t_m2c": (3,) translation in mm
            }
        """
        if scene_id in self._gt_cache:
            return self._gt_cache[scene_id]

        path = self.split_dir / scene_id / "scene_gt.json"
        with open(path) as f:
            raw = json.load(f)

        gts = {}
        for img_id, annotations in raw.items():
            gt_list = []
            for ann in annotations:
                R = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                t = np.array(ann["cam_t_m2c"]).reshape(3)
                gt_list.append({
                    "obj_id": ann["obj_id"],
                    "cam_R_m2c": R,
                    "cam_t_m2c": t,
                })
            gts[img_id] = gt_list

        self._gt_cache[scene_id] = gts
        return gts

    def load_rgb(self, scene_id: str, img_id: int) -> np.ndarray:
        """Load RGB image.

        Args:
            scene_id: Scene identifier (e.g., "000001")
            img_id: Image index (0-based)

        Returns:
            (H, W, 3) RGB image (uint8)
        """
        rgb_dir = self.split_dir / scene_id / "rgb"
        filename = f"{img_id:06d}.png"
        path = rgb_dir / filename
        if not path.exists():
            filename = f"{img_id:06d}.jpg"
            path = rgb_dir / filename

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_depth(self, scene_id: str, img_id: int) -> np.ndarray:
        """Load depth image.

        Args:
            scene_id: Scene identifier
            img_id: Image index

        Returns:
            (H, W) depth map in mm (uint16 or float)
        """
        depth_dir = self.split_dir / scene_id / "depth"
        path = depth_dir / f"{img_id:06d}.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {path}")
        return depth

    def load_mask(self, scene_id: str, img_id: int, obj_idx: int,
                  visible_only: bool = True) -> Optional[np.ndarray]:
        """Load object mask.

        Args:
            scene_id: Scene identifier
            img_id: Image index
            obj_idx: Object index in the GT list (0-based)
            visible_only: If True, use mask_visib (visible part only)

        Returns:
            (H, W) binary mask (bool), or None if not available
        """
        mask_type = "mask_visib" if visible_only else "mask"
        mask_dir = self.split_dir / scene_id / mask_type
        path = mask_dir / f"{img_id:06d}_{obj_idx:06d}.png"
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return mask > 0

    def load_sample(self, scene_id: str, img_id: int) -> Dict:
        """Load a complete sample: RGB, depth, camera, GT, and first-object mask.

        Returns:
            dict with keys: rgb, depth, cam_K, depth_scale, gt_poses, mask
            - depth: depth in METERS (float32), ready for FoundationPose
                     Converted from BOP raw uint16 via: raw * depth_scale * 0.001
        """
        rgb = self.load_rgb(scene_id, img_id)
        depth_raw = self.load_depth(scene_id, img_id)

        cameras = self.load_scene_camera(scene_id)
        cam = cameras.get(str(img_id), {
            "cam_K": self.default_K,
            "depth_scale": self.depth_scale
        })

        gts = self.load_scene_gt(scene_id)
        gt_poses = gts.get(str(img_id), [])

        # Load visible mask for the first annotated object (if available)
        mask = self.load_mask(scene_id, img_id, obj_idx=0, visible_only=True)

        # Convert depth to METERS: raw_uint16 * depth_scale * 0.001
        # BOP depth_scale converts raw sensor ticks to mm, then *0.001 -> meters
        # FoundationPose expects depth in meters with zfar=100
        depth_scale = cam["depth_scale"]
        depth = depth_raw.astype(np.float32) * depth_scale * 1e-3

        return {
            "rgb": rgb,
            "depth": depth,
            "cam_K": cam["cam_K"],
            "depth_scale": depth_scale,
            "gt_poses": gt_poses,
            "mask": mask,
        }

    def get_model_path(self, obj_id: int) -> Path:
        """Get path to 3D model PLY file."""
        return self.root / "models" / f"obj_{obj_id:06d}.ply"

    def get_object_ids(self) -> List[int]:
        """List all object IDs from models_info."""
        return sorted([int(k) for k in self.models_info.keys()])

    def get_object_diameter(self, obj_id: int) -> float:
        """Get object diameter in mm (used for threshold in ADD metric)."""
        info = self.models_info.get(str(obj_id), {})
        return info.get("diameter", 0.0)

    def get_symmetries(self, obj_id: int) -> Dict:
        """Get object symmetry information.

        Returns:
            dict with "symmetries_discrete" and/or "symmetries_continuous"
        """
        info = self.models_info.get(str(obj_id), {})
        return {
            "symmetries_discrete": info.get("symmetries_discrete", []),
            "symmetries_continuous": info.get("symmetries_continuous", []),
        }


def verify_dataset(dataset_root: str, split: str = "test") -> Dict:
    """Verify dataset integrity and print summary.

    Args:
        dataset_root: Path to dataset
        split: "test" or "train"

    Returns:
        dict with verification results
    """
    ds = BOPDataset(dataset_root, split)
    results = {
        "root": str(ds.root),
        "split": split,
        "scenes": len(ds.scenes),
        "objects": len(ds.models_info),
        "has_camera": ds.default_K is not None,
        "errors": [],
    }

    # Check models
    models_dir = ds.root / "models"
    if models_dir.exists():
        ply_files = list(models_dir.glob("*.ply"))
        results["model_files"] = len(ply_files)
    else:
        results["model_files"] = 0
        results["errors"].append("models/ directory not found")

    # Check scenes
    total_images = 0
    for scene_id in ds.scenes:
        n = ds.get_num_images(scene_id)
        total_images += n

        # Check required files
        scene_dir = ds.split_dir / scene_id
        for required in ["scene_camera.json", "scene_gt.json"]:
            if not (scene_dir / required).exists():
                results["errors"].append(f"Scene {scene_id}: missing {required}")

    results["total_images"] = total_images

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset: {ds.root.name}")
    print(f"{'='*50}")
    print(f"  Split:        {split}")
    print(f"  Scenes:       {results['scenes']}")
    print(f"  Total images: {results['total_images']}")
    print(f"  3D models:    {results['model_files']}")
    print(f"  Objects:      {results['objects']}")
    print(f"  Camera K:     {'OK' if results['has_camera'] else 'MISSING'}")
    if results["errors"]:
        print(f"  Errors:       {len(results['errors'])}")
        for e in results["errors"][:5]:
            print(f"    - {e}")
    else:
        print(f"  Status:       ✓ ALL OK")
    print()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.dataset_loader <dataset_path> [split]")
        print("Example: python -m src.utils.dataset_loader data/datasets/tless test")
        sys.exit(1)

    dataset_path = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    verify_dataset(dataset_path, split)
