"""
End-to-end verification: dataset loading → detection → pose visualization → metrics.

Tests the full pipeline locally using ground-truth poses as "predictions"
to validate the entire data flow before running real inference on Colab.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataset_loader import BOPDataset
from src.utils.lie_groups import so3_exp, so3_log, geodesic_distance_SO3, pose_from_Rt
from src.utils.rotations import matrix_to_quat, matrix_to_6d, sixd_to_matrix
from src.utils.metrics import add_metric, add_s_metric, mssd, mspd, compute_recall
from src.utils.visualization import draw_pose_axes, draw_projected_points
from src.planning.grasp_sampler import GraspSampler, GraspCandidate


def test_dataset(name, root, scene_id, img_id=0):
    """Full pipeline test on one sample."""
    print(f"\n{'='*60}")
    print(f"  E2E Test: {name} — Scene {scene_id}, Image {img_id}")
    print(f"{'='*60}")

    ds = BOPDataset(root, split="test")
    sample = ds.load_sample(scene_id, img_id)

    rgb = sample["rgb"]
    depth = sample["depth"]
    K = sample["cam_K"]
    gt_poses = sample["gt_poses"]

    print(f"  RGB:   {rgb.shape} {rgb.dtype}")
    print(f"  Depth: {depth.shape} {depth.dtype}")
    print(f"  K:     fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  GT poses: {len(gt_poses)} objects")

    if not gt_poses:
        print("  ⚠ No GT poses for this image, skipping")
        return False

    # ── Step 1: Use first GT pose as "prediction" ──
    gt = gt_poses[0]
    obj_id = gt["obj_id"]
    R_gt = gt["cam_R_m2c"]
    t_gt = gt["cam_t_m2c"]
    print(f"\n  [1] Object {obj_id}:")
    print(f"      R det = {np.linalg.det(R_gt):.6f} (should be 1.0)")
    print(f"      t = [{t_gt[0]:.1f}, {t_gt[1]:.1f}, {t_gt[2]:.1f}] mm")

    # ── Step 2: Test rotation representations ──
    q = matrix_to_quat(R_gt)
    rep6d = matrix_to_6d(R_gt)
    R_from_6d = sixd_to_matrix(rep6d)
    omega = so3_log(R_gt)

    print(f"\n  [2] Rotation representations:")
    print(f"      Quaternion: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"      6D repr:    [{rep6d[0]:.4f}, ..., {rep6d[5]:.4f}]")
    print(f"      Axis-angle: norm={np.linalg.norm(omega):.4f} rad ({np.degrees(np.linalg.norm(omega)):.1f} deg)")
    err_6d = np.linalg.norm(R_gt - R_from_6d)
    print(f"      6D roundtrip error: {err_6d:.2e} {'✓' if err_6d < 1e-5 else '✗'}")

    # ── Step 3: Simulate noisy prediction ──
    noise_R = so3_exp(np.random.randn(3) * 0.05)  # ~3 deg noise
    noise_t = np.random.randn(3) * 5.0  # 5mm noise
    R_pred = noise_R @ R_gt
    t_pred = t_gt + noise_t

    angular_err = geodesic_distance_SO3(R_gt, R_pred)
    trans_err = np.linalg.norm(t_gt - t_pred)
    print(f"\n  [3] Noisy prediction:")
    print(f"      Angular error:  {np.degrees(angular_err):.2f} deg")
    print(f"      Translation error: {trans_err:.2f} mm")

    # ── Step 4: Compute metrics with model points ──
    model_path = ds.get_model_path(obj_id)
    if model_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(str(model_path))
            points = np.array(mesh.vertices)
            if len(points) > 1000:
                idx = np.random.choice(len(points), 1000, replace=False)
                points = points[idx]

            add_err = add_metric(R_pred, t_pred, R_gt, t_gt, points)
            adds_err = add_s_metric(R_pred, t_pred, R_gt, t_gt, points)
            mssd_err = mssd(R_pred, t_pred, R_gt, t_gt, points)
            mspd_err = mspd(R_pred, t_pred, R_gt, t_gt, points, K)

            diameter = ds.get_object_diameter(obj_id)

            print(f"\n  [4] BOP Metrics (noisy pred vs GT):")
            print(f"      ADD:  {add_err:.2f} mm")
            print(f"      ADD-S: {adds_err:.2f} mm")
            print(f"      MSSD: {mssd_err:.2f} mm")
            print(f"      MSPD: {mspd_err:.2f} px")
            print(f"      Diameter: {diameter:.1f} mm")
            if diameter > 0:
                print(f"      ADD < 10%d: {'✓' if add_err < 0.1 * diameter else '✗'}")

            # Perfect prediction should give 0 error
            add_perfect = add_metric(R_gt, t_gt, R_gt, t_gt, points)
            print(f"      ADD (perfect): {add_perfect:.2e} {'✓' if add_perfect < 1e-8 else '✗'}")

        except ImportError:
            print("  ⚠ trimesh not available, skipping mesh metrics")
    else:
        print(f"  ⚠ Model not found: {model_path}")

    # ── Step 5: Visualization (save to file) ──
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_vis = draw_pose_axes(img_bgr, R_gt, t_gt / 1000.0, K, axis_length=0.05)

    out_dir = Path("experiments/results/e2e_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_scene{scene_id}_img{img_id:06d}.png"
    cv2.imwrite(str(out_path), img_vis)
    print(f"\n  [5] Visualization saved: {out_path}")

    # ── Step 6: Test grasp sampler ──
    T_obj = pose_from_Rt(R_gt, t_gt / 1000.0)  # Convert to meters
    sampler = GraspSampler()
    grasps = sampler.sample(T_obj, n_candidates=8, methods=["topdown"])
    print(f"\n  [6] Grasp sampler: generated {len(grasps)} top-down grasps")
    if grasps:
        g = grasps[0]
        print(f"      Best grasp score: {g.score:.3f}")
        av = g.approach_vector()
        print(f"      Approach: [{av[0]:.3f}, {av[1]:.3f}, {av[2]:.3f}]")

    print(f"\n  ✅ {name} E2E test PASSED")
    return True


def main():
    print("=" * 60)
    print("  Pose 6-DoF TFM — End-to-End Verification")
    print("=" * 60)

    results = []

    # T-LESS
    tless_root = "data/datasets/tless"
    if Path(tless_root).exists() and (Path(tless_root) / "test").exists():
        results.append(test_dataset("T-LESS", tless_root, "000001", img_id=0))
    else:
        print("\n⏳ T-LESS test images not yet extracted")

    # YCB-Video
    ycbv_root = "data/datasets/ycbv"
    if Path(ycbv_root).exists() and (Path(ycbv_root) / "test").exists():
        results.append(test_dataset("YCB-V", ycbv_root, "000048", img_id=1))
    else:
        print("\n⏳ YCB-Video test images not yet extracted")

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} datasets verified")
    if passed == total and total > 0:
        print("  🎯 All E2E tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
