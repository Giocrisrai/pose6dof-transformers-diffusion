#!/usr/bin/env python3
"""Verify BOP datasets integrity and print summary.

Usage:
    python scripts/verify_datasets.py
    python scripts/verify_datasets.py --dataset ycbv
    python scripts/verify_datasets.py --data-root /path/to/datasets
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def check_dataset(root: Path, name: str) -> dict:
    """Check a single BOP dataset and return status."""
    status = {
        "name": name,
        "root": str(root),
        "exists": root.exists(),
        "models": 0,
        "models_info": False,
        "camera_json": False,
        "splits": {},
        "issues": [],
    }

    if not root.exists():
        status["issues"].append(f"Directory not found: {root}")
        return status

    # Check models
    models_dir = root / "models"
    if models_dir.exists():
        plys = list(models_dir.glob("obj_*.ply"))
        status["models"] = len(plys)
        if len(plys) == 0:
            status["issues"].append("No .ply model files found")
    else:
        status["issues"].append("models/ directory missing")

    # Check models_info.json
    models_info = root / "models" / "models_info.json"
    if models_info.exists():
        status["models_info"] = True
        with open(models_info) as f:
            info = json.load(f)
        # Verify all models have diameters
        missing_diameter = [k for k, v in info.items() if "diameter" not in v]
        if missing_diameter:
            status["issues"].append(
                f"models_info: {len(missing_diameter)} objects missing diameter"
            )
    else:
        status["issues"].append("models/models_info.json missing")

    # Check camera.json
    camera_json = root / "camera.json"
    if camera_json.exists():
        status["camera_json"] = True
    else:
        # Some datasets use camera_*.json variants
        camera_variants = list(root.glob("camera*.json"))
        if camera_variants:
            status["camera_json"] = True
        else:
            status["issues"].append("camera.json missing")

    # Check splits (test, train, test_primesense, etc.)
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        # Check if it looks like a BOP split (contains numbered scene dirs)
        scenes = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if not scenes:
            continue

        split_name = split_dir.name
        split_status = {
            "n_scenes": len(scenes),
            "scenes_with_rgb": 0,
            "scenes_with_depth": 0,
            "scenes_with_gt": 0,
            "scenes_with_camera": 0,
            "total_images": 0,
        }

        for scene in scenes:
            rgb_dir = scene / "rgb"
            depth_dir = scene / "depth"
            gt_file = scene / "scene_gt.json"
            cam_file = scene / "scene_camera.json"

            if rgb_dir.exists():
                split_status["scenes_with_rgb"] += 1
                n_imgs = len(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
                split_status["total_images"] += n_imgs

            if depth_dir.exists():
                split_status["scenes_with_depth"] += 1

            if gt_file.exists():
                split_status["scenes_with_gt"] += 1

            if cam_file.exists():
                split_status["scenes_with_camera"] += 1

        status["splits"][split_name] = split_status

    return status


def print_status(status: dict):
    """Pretty-print dataset status."""
    name = status["name"].upper()
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not status["exists"]:
        print(f"  NOT FOUND: {status['root']}")
        return

    print(f"  Path: {status['root']}")
    print(f"  3D Models: {status['models']}")
    print(f"  models_info.json: {'OK' if status['models_info'] else 'MISSING'}")
    print(f"  camera.json: {'OK' if status['camera_json'] else 'MISSING'}")

    if status["splits"]:
        print(f"\n  Splits:")
        for split_name, split in status["splits"].items():
            n = split["n_scenes"]
            imgs = split["total_images"]
            rgb = split["scenes_with_rgb"]
            depth = split["scenes_with_depth"]
            gt = split["scenes_with_gt"]
            cam = split["scenes_with_camera"]
            print(f"    {split_name}: {n} scenes, {imgs} images")
            print(f"      rgb: {rgb}/{n}, depth: {depth}/{n}, "
                  f"gt: {gt}/{n}, camera: {cam}/{n}")
            if rgb < n:
                print(f"      WARNING: {n - rgb} scenes missing RGB images")
    else:
        print(f"\n  No test/train splits found (images not downloaded yet)")

    if status["issues"]:
        print(f"\n  Issues:")
        for issue in status["issues"]:
            print(f"    - {issue}")
    else:
        print(f"\n  Status: ALL OK")


def main():
    parser = argparse.ArgumentParser(description="Verify BOP datasets")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "datasets"),
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--dataset",
        choices=["ycbv", "tless", "all"],
        default="all",
        help="Which dataset to verify",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print(f"Data root: {data_root}")

    datasets = ["ycbv", "tless"] if args.dataset == "all" else [args.dataset]
    all_ok = True

    for ds_name in datasets:
        status = check_dataset(data_root / ds_name, ds_name)
        print_status(status)
        if status["issues"]:
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL DATASETS OK")
    else:
        print("  SOME ISSUES FOUND (see above)")
    print(f"{'='*60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
