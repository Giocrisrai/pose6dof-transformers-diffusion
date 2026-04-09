"""
Experiment 3: Ablation Study — Rotation Representations.

Compares quaternion, 6D continuous, axis-angle, and Euler
representations in terms of:
    1. Reconstruction error (roundtrip precision)
    2. Interpolation smoothness
    3. Gradient stability (PyTorch backprop)
    4. Singularity behavior (Gimbal lock for Euler)

Generates tables and figures for TFM Chapter 6.

Usage:
    cd repo_tfm
    python experiments/exp3_rotation_ablation.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.lie_groups import so3_exp, so3_log, geodesic_distance_SO3
from src.utils.rotations import (
    matrix_to_quat, quat_to_matrix,
    matrix_to_6d, sixd_to_matrix,
    matrix_to_axisangle, axisangle_to_matrix,
    euler_to_matrix, matrix_to_euler,
    sixd_to_matrix_torch, matrix_to_6d_torch,
)
from src.utils.dataset_loader import BOPDataset

OUTPUT_DIR = Path("experiments/results/exp3_rotation_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


def experiment_roundtrip_precision(rotations: list) -> dict:
    """Test roundtrip error for each representation."""
    print("\n[Exp 3.1] Roundtrip Precision (N={})".format(len(rotations)))

    results = {
        "quaternion": [],
        "6D_continuous": [],
        "axis_angle": [],
        "euler_ZYX": [],
    }

    for R in rotations:
        # Quaternion
        q = matrix_to_quat(R)
        R_rec = quat_to_matrix(q)
        results["quaternion"].append(np.linalg.norm(R - R_rec))

        # 6D continuous
        rep6 = matrix_to_6d(R)
        R_rec = sixd_to_matrix(rep6)
        results["6D_continuous"].append(np.linalg.norm(R - R_rec))

        # Axis-angle
        axis, angle = matrix_to_axisangle(R)
        R_rec = axisangle_to_matrix(axis, angle)
        results["axis_angle"].append(np.linalg.norm(R - R_rec))

        # Euler (ZYX)
        roll, pitch, yaw = matrix_to_euler(R)
        R_rec = euler_to_matrix(roll, pitch, yaw)
        results["euler_ZYX"].append(np.linalg.norm(R - R_rec))

    table = {}
    for name, errs in results.items():
        table[name] = {
            "mean": float(np.mean(errs)),
            "max": float(np.max(errs)),
            "std": float(np.std(errs)),
        }
        print(f"  {name:<16} mean={np.mean(errs):.2e}  "
              f"max={np.max(errs):.2e}  std={np.std(errs):.2e}")

    return table


def experiment_interpolation_smoothness(R1, R2, n_steps=100) -> dict:
    """Compare interpolation paths between representations."""
    print("\n[Exp 3.2] Interpolation Smoothness")

    # Ground truth: geodesic interpolation via exp/log maps
    omega = so3_log(R1.T @ R2)
    gt_path = [so3_exp(omega * t / n_steps) @ R1 for t in range(n_steps + 1)]
    gt_angles = [geodesic_distance_SO3(np.eye(3), R1.T @ R)
                 for R in gt_path]

    # Quaternion SLERP
    q1 = matrix_to_quat(R1)
    q2 = matrix_to_quat(R2)
    # Ensure shortest path
    if np.dot(q1, q2) < 0:
        q2 = -q2

    quat_path = []
    for i in range(n_steps + 1):
        t = i / n_steps
        dot = np.dot(q1, q2)
        dot = np.clip(dot, -1, 1)
        theta = np.arccos(dot)
        if theta < 1e-8:
            q_interp = q1
        else:
            q_interp = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
        quat_path.append(geodesic_distance_SO3(np.eye(3), R1.T @ quat_to_matrix(q_interp)))

    # Linear interpolation in 6D space (naive)
    r1_6d = matrix_to_6d(R1)
    r2_6d = matrix_to_6d(R2)
    sixd_path = []
    for i in range(n_steps + 1):
        t = i / n_steps
        r_interp = (1 - t) * r1_6d + t * r2_6d
        R_interp = sixd_to_matrix(r_interp)
        sixd_path.append(geodesic_distance_SO3(np.eye(3), R1.T @ R_interp))

    # Euler linear interpolation
    e1 = matrix_to_euler(R1)
    e2 = matrix_to_euler(R2)
    euler_path = []
    for i in range(n_steps + 1):
        t = i / n_steps
        e_interp = [(1 - t) * a + t * b for a, b in zip(e1, e2)]
        R_interp = euler_to_matrix(*e_interp)
        euler_path.append(geodesic_distance_SO3(np.eye(3), R1.T @ R_interp))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ts = np.linspace(0, 1, n_steps + 1)
    ax.plot(ts, np.degrees(gt_angles), "k-", linewidth=2.5, label="Geodésica (ground truth)")
    ax.plot(ts, np.degrees(quat_path), "--", linewidth=2, label="SLERP (quaternión)", color="#0098CD")
    ax.plot(ts, np.degrees(sixd_path), "-.", linewidth=2, label="Lineal en 6D", color="#006C8F")
    ax.plot(ts, np.degrees(euler_path), ":", linewidth=2, label="Lineal en Euler ZYX", color="#CC0000")

    ax.set_xlabel("Parámetro de interpolación t", fontsize=12)
    ax.set_ylabel("Ángulo desde R₁ (grados)", fontsize=12)
    ax.set_title("Suavidad de interpolación por representación", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "interpolation_smoothness.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/interpolation_smoothness.png")

    # Compute deviation from geodesic
    deviations = {
        "SLERP_quaternion": float(np.mean(np.abs(np.array(quat_path) - np.array(gt_angles)))),
        "linear_6D": float(np.mean(np.abs(np.array(sixd_path) - np.array(gt_angles)))),
        "linear_euler": float(np.mean(np.abs(np.array(euler_path) - np.array(gt_angles)))),
    }
    for name, dev in deviations.items():
        print(f"  {name:<20} mean deviation from geodesic: {np.degrees(dev):.4f}°")

    return deviations


def experiment_gradient_stability(n_samples=500) -> dict:
    """Compare gradient magnitudes through backprop for 6D vs quaternion."""
    print("\n[Exp 3.3] Gradient Stability (PyTorch)")

    grad_norms_6d = []
    grad_norms_quat = []

    for _ in range(n_samples):
        R_target = so3_exp(np.random.randn(3) * np.pi)

        # 6D: predict 6D → reconstruct R → compute loss
        rep6d = torch.randn(6, requires_grad=True)
        R_pred = sixd_to_matrix_torch(rep6d)
        R_tgt = torch.tensor(R_target, dtype=torch.float32)
        loss_6d = torch.nn.functional.mse_loss(R_pred, R_tgt)
        loss_6d.backward()
        grad_norms_6d.append(rep6d.grad.norm().item())

        # Quaternion: predict 4D → normalize → convert → loss
        q_raw = torch.randn(4, requires_grad=True)
        q_norm = q_raw / q_raw.norm()
        w, x, y, z = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
        R_q = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)]),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)]),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]),
        ])
        loss_q = torch.nn.functional.mse_loss(R_q, R_tgt)
        loss_q.backward()
        grad_norms_quat.append(q_raw.grad.norm().item())

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(grad_norms_6d, bins=30, alpha=0.7, color="#0098CD", label="6D continuo")
    axes[0].hist(grad_norms_quat, bins=30, alpha=0.7, color="#CC0000", label="Quaternión")
    axes[0].set_xlabel("Norma del gradiente")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de normas de gradiente")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    bp = axes[1].boxplot(
        [grad_norms_6d, grad_norms_quat],
        tick_labels=["6D continuo", "Quaternión"],
        patch_artist=True,
    )
    colors = ["#0098CD", "#CC0000"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Norma del gradiente")
    axes[1].set_title("Estabilidad del gradiente")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "gradient_stability.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/gradient_stability.png")

    result = {
        "6D_continuous": {
            "mean": float(np.mean(grad_norms_6d)),
            "std": float(np.std(grad_norms_6d)),
            "max": float(np.max(grad_norms_6d)),
        },
        "quaternion": {
            "mean": float(np.mean(grad_norms_quat)),
            "std": float(np.std(grad_norms_quat)),
            "max": float(np.max(grad_norms_quat)),
        },
    }
    for name, stats in result.items():
        print(f"  {name:<16} mean={stats['mean']:.4f} ± {stats['std']:.4f}  "
              f"max={stats['max']:.4f}")

    return result


def experiment_gimbal_lock() -> dict:
    """Demonstrate Gimbal lock in Euler angles near pitch=±π/2."""
    print("\n[Exp 3.4] Gimbal Lock Analysis")

    pitches = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 200)
    euler_errors = []
    quat_errors = []
    sixd_errors = []

    for pitch in pitches:
        R = euler_to_matrix(0.3, pitch, 0.7)

        # Euler roundtrip
        r, p, y = matrix_to_euler(R)
        R_rec = euler_to_matrix(r, p, y)
        euler_errors.append(np.linalg.norm(R - R_rec))

        # Quat roundtrip
        q = matrix_to_quat(R)
        R_rec = quat_to_matrix(q)
        quat_errors.append(np.linalg.norm(R - R_rec))

        # 6D roundtrip
        rep = matrix_to_6d(R)
        R_rec = sixd_to_matrix(rep)
        sixd_errors.append(np.linalg.norm(R - R_rec))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.degrees(pitches), euler_errors, "-", linewidth=2,
                label="Euler ZYX", color="#CC0000")
    ax.semilogy(np.degrees(pitches), quat_errors, "--", linewidth=2,
                label="Quaternión", color="#0098CD")
    ax.semilogy(np.degrees(pitches), sixd_errors, "-.", linewidth=2,
                label="6D continuo", color="#006C8F")

    ax.axvline(x=90, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=-90, color="gray", linestyle=":", alpha=0.5)
    ax.annotate("Gimbal lock\n(pitch = ±90°)", xy=(85, 1e-4),
                fontsize=10, color="gray")

    ax.set_xlabel("Pitch (grados)", fontsize=12)
    ax.set_ylabel("Error de reconstrucción (log)", fontsize=12)
    ax.set_title("Singularidad de Gimbal Lock — Error vs Pitch", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "gimbal_lock.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/gimbal_lock.png")

    return {
        "euler_max_error": float(np.max(euler_errors)),
        "quat_max_error": float(np.max(quat_errors)),
        "sixd_max_error": float(np.max(sixd_errors)),
    }


def experiment_real_poses() -> dict:
    """Evaluate representations on real T-LESS poses."""
    print("\n[Exp 3.5] Real Pose Analysis (T-LESS)")

    tless = BOPDataset("data/datasets/tless", split="test")
    all_results = {"quaternion": [], "6D_continuous": [], "axis_angle": [], "euler_ZYX": []}

    n_poses = 0
    for scene_id in tless.scenes[:5]:
        gt_data = tless.load_scene_gt(scene_id)
        for img_id_str in list(gt_data.keys())[:20]:
            for gt in gt_data[img_id_str][:1]:
                R = gt["cam_R_m2c"]

                q = matrix_to_quat(R)
                all_results["quaternion"].append(np.linalg.norm(R - quat_to_matrix(q)))

                r6 = matrix_to_6d(R)
                all_results["6D_continuous"].append(np.linalg.norm(R - sixd_to_matrix(r6)))

                ax, an = matrix_to_axisangle(R)
                all_results["axis_angle"].append(np.linalg.norm(R - axisangle_to_matrix(ax, an)))

                ro, pi, ya = matrix_to_euler(R)
                all_results["euler_ZYX"].append(np.linalg.norm(R - euler_to_matrix(ro, pi, ya)))

                n_poses += 1

    print(f"  Evaluated {n_poses} real poses")
    table = {}
    for name, errs in all_results.items():
        table[name] = {
            "mean": float(np.mean(errs)),
            "max": float(np.max(errs)),
            "n_failures": int(np.sum(np.array(errs) > 1e-6)),
        }
        print(f"  {name:<16} mean={np.mean(errs):.2e}  "
              f"max={np.max(errs):.2e}  failures(>1e-6)={table[name]['n_failures']}")

    return table


def main():
    print("=" * 60)
    print("  EXPERIMENT 3 — Rotation Representation Ablation")
    print("=" * 60)

    # Generate random rotations covering SO(3)
    rotations = [so3_exp(np.random.randn(3) * np.pi) for _ in range(1000)]

    results = {}

    # 3.1: Roundtrip precision
    results["roundtrip"] = experiment_roundtrip_precision(rotations)

    # 3.2: Interpolation smoothness
    R1 = so3_exp(np.array([0.1, 0.3, -0.2]))
    R2 = so3_exp(np.array([2.0, -1.0, 0.5]))
    results["interpolation"] = experiment_interpolation_smoothness(R1, R2)

    # 3.3: Gradient stability
    results["gradient"] = experiment_gradient_stability(n_samples=500)

    # 3.4: Gimbal lock
    results["gimbal_lock"] = experiment_gimbal_lock()

    # 3.5: Real poses
    results["real_poses"] = experiment_real_poses()

    # Save all results
    with open(OUTPUT_DIR / "exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  All results saved to {OUTPUT_DIR}/exp3_results.json")

    print(f"\n{'='*60}")
    print("  ✓ EXPERIMENT 3 COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
