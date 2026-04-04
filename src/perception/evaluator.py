"""
BOP Evaluation Pipeline — Comparative evaluation of pose estimation methods.

Runs FoundationPose and GDR-Net++ on BOP datasets and generates
comparison tables and visualizations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.utils.metrics import (
    add_metric, add_s_metric, mssd, mspd,
    compute_recall, compute_auc,
)
from src.utils.dataset_loader import BOPDataset

logger = logging.getLogger(__name__)


# BOP standard thresholds
BOP_VSD_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BOP_MSSD_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # fraction of diameter
BOP_MSPD_THRESHOLDS = [5, 10, 15, 20, 25, 30]  # pixels


def load_predictions(pred_path: str) -> Dict:
    """Load pose predictions from JSON file.

    Expected format:
        {
            "scene_id/image_id": {
                "obj_id": int,
                "R": [[...], [...], [...]],
                "t": [x, y, z],
                "score": float
            },
            ...
        }

    Returns:
        dict of predictions
    """
    with open(pred_path) as f:
        preds = json.load(f)
    return preds


def evaluate_method(
    dataset: BOPDataset,
    predictions: Dict,
    method_name: str = "method",
) -> Dict:
    """Evaluate a method's predictions against ground truth.

    Args:
        dataset: BOPDataset instance
        predictions: dict of predictions (from load_predictions)
        method_name: name for logging

    Returns:
        dict with per-metric results
    """
    add_errors = []
    adds_errors = []
    mssd_errors = []
    mspd_errors = []

    n_evaluated = 0
    n_skipped = 0

    for scene_id in dataset.get_scene_ids():
        gt_poses = dataset.load_scene_gt(scene_id)
        cameras = dataset.load_scene_camera(scene_id)

        for img_id_str, gt_list in gt_poses.items():
            key = f"{scene_id}/{img_id_str}"
            if key not in predictions:
                n_skipped += 1
                continue

            pred = predictions[key]
            cam = cameras.get(img_id_str, {})
            K = cam.get("cam_K", dataset.default_K)

            for gt in gt_list:
                obj_id = gt["obj_id"]
                R_gt = gt["cam_R_m2c"]
                t_gt = gt["cam_t_m2c"]

                R_est = np.array(pred["R"])
                t_est = np.array(pred["t"])

                # Get model points for this object
                model_path = dataset.get_model_path(obj_id)
                if not model_path.exists():
                    continue

                # Load model points (subsample for speed)
                try:
                    import trimesh
                    mesh = trimesh.load(str(model_path))
                    points = np.array(mesh.vertices)
                    # Subsample to max 1000 points
                    if len(points) > 1000:
                        idx = np.random.choice(len(points), 1000, replace=False)
                        points = points[idx]
                except ImportError:
                    logger.warning("trimesh not available, skipping mesh-based metrics")
                    continue

                # Compute metrics
                add_err = add_metric(R_est, t_est, R_gt, t_gt, points)
                adds_err = add_s_metric(R_est, t_est, R_gt, t_gt, points)

                # MSSD (uses object diameter as normalization)
                diameter = dataset.get_object_diameter(obj_id)
                symmetries_info = dataset.get_symmetries(obj_id)

                # Build symmetry transforms
                sym_transforms = [np.eye(4)]
                for sym in symmetries_info.get("symmetries_discrete", []):
                    if isinstance(sym, list) and len(sym) == 16:
                        sym_transforms.append(np.array(sym).reshape(4, 4))

                mssd_err = mssd(R_est, t_est, R_gt, t_gt, points, sym_transforms)
                mspd_err = mspd(R_est, t_est, R_gt, t_gt, points, K, sym_transforms)

                add_errors.append(add_err)
                adds_errors.append(adds_err)
                mssd_errors.append(mssd_err)
                mspd_errors.append(mspd_err)
                n_evaluated += 1

    logger.info(f"[{method_name}] Evaluated: {n_evaluated}, Skipped: {n_skipped}")

    # Compute recalls at standard thresholds
    results = {
        "method": method_name,
        "n_evaluated": n_evaluated,
        "ADD": {
            "mean": float(np.mean(add_errors)) if add_errors else 0.0,
            "median": float(np.median(add_errors)) if add_errors else 0.0,
        },
        "ADD-S": {
            "mean": float(np.mean(adds_errors)) if adds_errors else 0.0,
            "median": float(np.median(adds_errors)) if adds_errors else 0.0,
        },
        "MSSD": {
            "mean": float(np.mean(mssd_errors)) if mssd_errors else 0.0,
            "auc": compute_auc(mssd_errors, max_threshold=50.0) if mssd_errors else 0.0,
        },
        "MSPD": {
            "mean": float(np.mean(mspd_errors)) if mspd_errors else 0.0,
            "auc": compute_auc(mspd_errors, max_threshold=50.0) if mspd_errors else 0.0,
        },
    }

    return results


def compare_methods(
    dataset_name: str,
    dataset_root: str,
    prediction_files: Dict[str, str],
    output_dir: str = "experiments/results",
) -> Dict:
    """Compare multiple methods on a BOP dataset.

    Args:
        dataset_name: "tless" or "ycbv"
        dataset_root: Path to BOP dataset
        prediction_files: {method_name: path_to_predictions.json}
        output_dir: Where to save comparison results

    Returns:
        dict with comparative results
    """
    dataset = BOPDataset(dataset_root, split="test")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method_name, pred_path in prediction_files.items():
        logger.info(f"Evaluating {method_name} on {dataset_name}...")
        preds = load_predictions(pred_path)
        results = evaluate_method(dataset, preds, method_name)
        all_results[method_name] = results

    # Save comparison
    comparison_path = output_path / f"comparison_{dataset_name}.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Comparison saved to {comparison_path}")

    # Print table
    print(f"\n{'='*70}")
    print(f"  Comparison on {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'ADD↓':>8} {'ADD-S↓':>8} {'MSSD-AUC↑':>10} {'MSPD-AUC↑':>10}")
    print(f"  {'-'*56}")
    for name, res in all_results.items():
        print(f"  {name:<20} "
              f"{res['ADD']['mean']:>8.2f} "
              f"{res['ADD-S']['mean']:>8.2f} "
              f"{res['MSSD']['auc']:>10.4f} "
              f"{res['MSPD']['auc']:>10.4f}")
    print()

    return all_results
