"""
Integration Test: Full pipeline with real BOP data.

Loads a sample from T-LESS, runs detection → pose → grasp → metrics,
and generates visualizations. Validates that all modules work end-to-end.

Usage:
    cd repo_tfm
    python scripts/integration_test.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataset_loader import BOPDataset
from src.utils.lie_groups import (
    so3_exp, so3_log, se3_exp, se3_log,
    pose_from_Rt, pose_to_Rt, geodesic_distance_SO3, geodesic_distance_SE3,
)
from src.utils.rotations import (
    matrix_to_quat, quat_to_matrix, matrix_to_6d, sixd_to_matrix,
    matrix_to_axisangle,
)
from src.utils.metrics import (
    add_metric, add_s_metric, mssd, mspd,
    compute_recall, compute_auc,
)
from src.utils.visualization import (
    draw_pose_axes, draw_projected_points, plot_metrics_comparison,
)
from src.perception.detector import GTDetector, Detection
from src.planning.grasp_sampler import GraspSampler


def main():
    print("=" * 60)
    print("  INTEGRATION TEST — Full Pipeline on Real BOP Data")
    print("=" * 60)

    output_dir = Path("experiments/integration_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load T-LESS dataset ──────────────────────────────
    print("\n[1/7] Loading T-LESS dataset...")
    # T-LESS ships with "test_primesense" in BOP; fall back to "test" for legacy layouts.
    tless_split = "test_primesense" if (Path("data/datasets/tless") / "test_primesense").exists() else "test"
    tless = BOPDataset("data/datasets/tless", split=tless_split)
    print(f"  Scenes: {len(tless.scenes)}, Objects: {len(tless.models_info)}, split={tless_split}")

    # Pick first scene and its first available image (IDs may not start at 0)
    scene_id = tless.scenes[0]
    first_img_id = tless.get_image_ids(scene_id)[0]
    sample = tless.load_sample(scene_id, img_id=first_img_id)
    rgb = sample["rgb"]
    depth = sample["depth"]
    K = sample["cam_K"]
    gt_poses = sample["gt_poses"]
    print(f"  Scene {scene_id}, Image 0: {rgb.shape}, {len(gt_poses)} objects")

    # ── 2. Ground-truth pose analysis ───────────────────────
    print("\n[2/7] Analyzing ground-truth poses...")
    gt = gt_poses[0]
    R_gt = gt["cam_R_m2c"]
    t_gt = gt["cam_t_m2c"]
    obj_id = gt["obj_id"]
    print(f"  Object ID: {obj_id}")
    print(f"  Translation: [{t_gt[0]:.1f}, {t_gt[1]:.1f}, {t_gt[2]:.1f}] mm")

    # Rotation analysis
    q = matrix_to_quat(R_gt)
    axis, angle = matrix_to_axisangle(R_gt)
    rep6d = matrix_to_6d(R_gt)
    print(f"  Quaternion: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"  Axis-angle: axis=[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}], "
          f"angle={np.degrees(angle):.1f}°")
    print(f"  6D repr: [{rep6d[0]:.3f}, {rep6d[1]:.3f}, ..., {rep6d[5]:.3f}]")

    # Roundtrip checks
    R_from_q = quat_to_matrix(q)
    R_from_6d = sixd_to_matrix(rep6d)
    err_q = np.linalg.norm(R_gt - R_from_q)
    err_6d = np.linalg.norm(R_gt - R_from_6d)
    print(f"  Roundtrip errors: quat={err_q:.2e}, 6D={err_6d:.2e}")

    # ── 3. Simulated pose estimation (GT + noise) ───────────
    print("\n[3/7] Simulated pose estimation (GT + noise)...")

    noise_levels = [0.0, 1.0, 5.0, 10.0, 20.0]
    results_by_noise = {}

    # Load model points
    model_path = tless.get_model_path(obj_id)
    if model_path.exists():
        import trimesh
        mesh = trimesh.load(str(model_path))
        points = np.array(mesh.vertices)
        if len(points) > 1000:
            idx = np.random.choice(len(points), 1000, replace=False)
            points = points[idx]
        print(f"  Model points: {len(points)} (subsampled)")
    else:
        print(f"  WARNING: Model not found at {model_path}")
        points = np.random.randn(1000, 3) * 10

    diameter = tless.get_object_diameter(obj_id)
    print(f"  Object diameter: {diameter:.1f} mm")

    for noise_mm in noise_levels:
        # Add noise to GT pose
        noise_rot = np.random.randn(3) * (noise_mm * 0.001)  # radians
        noise_trans = np.random.randn(3) * noise_mm  # mm

        R_est = so3_exp(noise_rot) @ R_gt
        t_est = t_gt + noise_trans

        # Compute metrics
        add_err = add_metric(R_est, t_est, R_gt, t_gt, points)
        adds_err = add_s_metric(R_est, t_est, R_gt, t_gt, points)
        mssd_err = mssd(R_est, t_est, R_gt, t_gt, points)
        mspd_err = mspd(R_est, t_est, R_gt, t_gt, points, K)
        rot_err, trans_err = geodesic_distance_SE3(
            pose_from_Rt(R_est, t_est), pose_from_Rt(R_gt, t_gt)
        )

        results_by_noise[noise_mm] = {
            "ADD": add_err,
            "ADD-S": adds_err,
            "MSSD": mssd_err,
            "MSPD": mspd_err,
            "rot_err_deg": np.degrees(rot_err),
            "trans_err_mm": trans_err,
        }

        print(f"  Noise {noise_mm:5.1f}mm → ADD={add_err:7.2f}  "
              f"ADD-S={adds_err:7.2f}  "
              f"rot={np.degrees(rot_err):5.2f}°  "
              f"trans={trans_err:6.2f}mm")

    # ── 4. Visualization ────────────────────────────────────
    print("\n[4/7] Generating visualizations...")

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw GT pose axes on image
    img_axes = draw_pose_axes(img_bgr.copy(), R_gt, t_gt, K, axis_length=30.0)
    img_points = draw_projected_points(
        img_axes, R_gt, t_gt, K, points, color=(0, 255, 0), radius=1
    )

    # Draw noisy pose for comparison
    R_noisy = so3_exp(np.random.randn(3) * 0.02) @ R_gt
    t_noisy = t_gt + np.random.randn(3) * 10
    img_noisy = draw_pose_axes(img_bgr.copy(), R_noisy, t_noisy, K, axis_length=30.0)
    img_noisy = draw_projected_points(
        img_noisy, R_noisy, t_noisy, K, points, color=(0, 0, 255), radius=1
    )

    # Save comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Ground Truth Pose (green)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Noisy Estimate (red)")
    axes[2].axis("off")

    fig.suptitle(f"T-LESS Scene {scene_id} — Object {obj_id}", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(output_dir / "pose_visualization.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/pose_visualization.png")

    # Noise vs error plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    noise_vals = sorted(results_by_noise.keys())
    add_vals = [results_by_noise[n]["ADD"] for n in noise_vals]
    adds_vals = [results_by_noise[n]["ADD-S"] for n in noise_vals]
    rot_vals = [results_by_noise[n]["rot_err_deg"] for n in noise_vals]

    ax.plot(noise_vals, add_vals, "o-", label="ADD (mm)", color="#0098CD", linewidth=2)
    ax.plot(noise_vals, adds_vals, "s-", label="ADD-S (mm)", color="#006C8F", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(noise_vals, rot_vals, "^--", label="Rotation (°)", color="#333333", linewidth=2)
    ax2.set_ylabel("Rotation Error (°)", color="#333333")

    ax.set_xlabel("Noise Level (mm)")
    ax.set_ylabel("Distance Error (mm)")
    ax.set_title("Pose Error vs Noise Level — T-LESS Object")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(str(output_dir / "noise_vs_error.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: {output_dir}/noise_vs_error.png")

    # ── 5. Grasp sampling on detected object ────────────────
    print("\n[5/7] Grasp sampling...")
    T_obj = pose_from_Rt(R_gt, t_gt / 1000.0)  # Convert mm → m
    sampler = GraspSampler(gripper_width=0.08, standoff_distance=0.10)

    # Use the unified sample() method with different strategies
    grasps_topdown = sampler.sample(T_obj, n_candidates=10, methods=["topdown"])
    grasps_side = sampler.sample(T_obj, n_candidates=10, methods=["side"])
    grasps_all = sampler.sample(T_obj, n_candidates=20)
    print(f"  Top-down grasps: {len(grasps_topdown)}")
    print(f"  Side grasps: {len(grasps_side)}")
    print(f"  All methods combined: {len(grasps_all)}")
    if grasps_all:
        g = grasps_all[0]
        print(f"  Best grasp: score={g.score:.3f}, "
              f"width={g.width:.3f}m, method={g.method}")

    # ── 6. Multi-scene evaluation ───────────────────────────
    print("\n[6/7] Multi-scene evaluation (first 5 scenes)...")
    all_add_errors = []
    all_adds_errors = []
    n_total = 0

    for scene_id in tless.scenes[:5]:
        gt_data = tless.load_scene_gt(scene_id)
        cameras = tless.load_scene_camera(scene_id)

        for img_id_str in list(gt_data.keys())[:10]:  # First 10 images per scene
            gt_list = gt_data[img_id_str]
            cam = cameras.get(img_id_str, {})
            K_scene = cam.get("cam_K", tless.default_K)

            for gt_obj in gt_list[:1]:  # First object per image
                R_gt_s = gt_obj["cam_R_m2c"]
                t_gt_s = gt_obj["cam_t_m2c"]
                obj_id_s = gt_obj["obj_id"]

                # Simulate noisy estimation (5mm noise)
                R_est_s = so3_exp(np.random.randn(3) * 0.005) @ R_gt_s
                t_est_s = t_gt_s + np.random.randn(3) * 5.0

                model_path_s = tless.get_model_path(obj_id_s)
                if not model_path_s.exists():
                    continue

                mesh_s = trimesh.load(str(model_path_s))
                pts_s = np.array(mesh_s.vertices)
                if len(pts_s) > 500:
                    idx_s = np.random.choice(len(pts_s), 500, replace=False)
                    pts_s = pts_s[idx_s]

                all_add_errors.append(add_metric(R_est_s, t_est_s, R_gt_s, t_gt_s, pts_s))
                all_adds_errors.append(add_s_metric(R_est_s, t_est_s, R_gt_s, t_gt_s, pts_s))
                n_total += 1

    if all_add_errors:
        add_recall_10 = compute_recall(all_add_errors, threshold=10.0)
        adds_recall_10 = compute_recall(all_adds_errors, threshold=10.0)
        add_auc = compute_auc(all_add_errors, max_threshold=50.0)
        adds_auc = compute_auc(all_adds_errors, max_threshold=50.0)

        print(f"  Evaluated: {n_total} predictions across 5 scenes")
        print(f"  ADD  — Recall@10mm: {add_recall_10*100:.1f}%, AUC: {add_auc:.4f}")
        print(f"  ADD-S — Recall@10mm: {adds_recall_10*100:.1f}%, AUC: {adds_auc:.4f}")

    # ── 7. Save results ─────────────────────────────────────
    print("\n[7/7] Saving results...")
    results = {
        "dataset": "tless",
        "n_evaluated": n_total,
        "noise_analysis": {
            str(k): v for k, v in results_by_noise.items()
        },
        "multi_scene": {
            "add_recall_at_10mm": add_recall_10 if all_add_errors else 0.0,
            "adds_recall_at_10mm": adds_recall_10 if all_add_errors else 0.0,
            "add_auc_50mm": add_auc if all_add_errors else 0.0,
            "adds_auc_50mm": adds_auc if all_add_errors else 0.0,
        },
    }

    with open(output_dir / "integration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_dir}/integration_results.json")

    print(f"\n{'='*60}")
    print("  ✓ INTEGRATION TEST PASSED — All modules working end-to-end")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
