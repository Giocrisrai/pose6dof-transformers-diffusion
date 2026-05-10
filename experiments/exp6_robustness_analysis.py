#!/usr/bin/env python3
"""Analisis de robustez del pipeline ante oclusion y ruido sensor.

Sobre los checkpoints reales de FoundationPose, simula degradacion:
- Oclusion: enmascara aleatoriamente un % de puntos del modelo CAD antes
  de calcular ADD/ADD-S. Niveles {0, 30, 50, 70} %.
- Ruido sensor: anade Gaussiano N(0, sigma) a la traslacion estimada en
  mm. Niveles sigma {0, 2, 5, 10} mm.

Genera curvas de degradacion AUC ADD-S vs nivel de degradacion para
ambos datasets (YCB-V, T-LESS), con bootstrap CI 95% en cada punto.

Salidas:
    experiments/results/exp6_robustness/exp6_results.json
    experiments/results/exp6_robustness/fig_robustness_occlusion.png
    experiments/results/exp6_robustness/fig_robustness_noise.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp6_robustness"
OUT.mkdir(parents=True, exist_ok=True)

from src.utils.dataset_loader import BOPDataset
from src.utils.metrics import add_metric, add_s_metric

DATASETS = {
    "ycbv": {
        "path": REPO / "data/datasets/ycbv",
        "split": "test",
        "checkpoint": REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json",
    },
    "tless": {
        "path": REPO / "data/datasets/tless",
        "split": "test_primesense",
        "checkpoint": REPO / "experiments/checkpoints/fp_tless_checkpoint.json",
    },
}

OCCLUSION_LEVELS = [0.0, 0.30, 0.50, 0.70]
NOISE_LEVELS_MM = [0.0, 2.0, 5.0, 10.0]


def load_predictions_with_gt(ds_name, ds_info, n_max=None):
    """Carga predicciones + GT correspondiente + puntos del modelo CAD."""
    import trimesh
    with open(ds_info["checkpoint"]) as f:
        ckpt = json.load(f)
    ds = BOPDataset(str(ds_info["path"]), split=ds_info["split"])

    gt_cache = {}
    mesh_cache = {}
    rng = np.random.default_rng(42)

    samples = []
    for pred in (ckpt["results"][:n_max] if n_max else ckpt["results"]):
        scene_id = pred["scene_id"]
        if isinstance(scene_id, int):
            scene_id = f"{scene_id:06d}"
        img_id = pred["img_id"]
        obj_id = pred["obj_id"]
        gt_idx = pred.get("gt_idx", -1)

        if scene_id not in gt_cache:
            gt_cache[scene_id] = ds.load_scene_gt(scene_id)
        gt_list = gt_cache[scene_id].get(str(img_id), [])
        if gt_idx < 0 or gt_idx >= len(gt_list):
            continue
        gt = gt_list[gt_idx]
        if gt["obj_id"] != obj_id:
            continue

        if obj_id not in mesh_cache:
            try:
                mp = ds.get_model_path(obj_id)
                if not Path(mp).exists():
                    mesh_cache[obj_id] = None
                else:
                    mesh = trimesh.load(str(mp), process=False)
                    pts = np.array(mesh.vertices, dtype=np.float64)
                    if len(pts) > 1000:
                        idx = rng.choice(len(pts), 1000, replace=False)
                        pts = pts[idx]
                    mesh_cache[obj_id] = pts
            except Exception:
                mesh_cache[obj_id] = None
        if mesh_cache[obj_id] is None:
            continue

        R_pred = np.array(pred["R_pred"])
        t_pred = np.array(pred["t_pred"])
        if np.linalg.norm(t_pred) < 5.0:
            t_pred = t_pred * 1000.0  # m -> mm

        samples.append({
            "R_pred": R_pred,
            "t_pred": t_pred,
            "R_gt": np.array(gt["cam_R_m2c"]),
            "t_gt": np.array(gt["cam_t_m2c"]),
            "points": mesh_cache[obj_id],
        })

    return samples


def evaluate_with_occlusion(samples, occlusion_pct, seed):
    """Calcula ADD-S sobre subset de puntos del modelo (simula oclusion)."""
    rng = np.random.default_rng(seed)
    adds = []
    for s in samples:
        n_pts = len(s["points"])
        keep = max(int(n_pts * (1 - occlusion_pct)), 50)
        idx = rng.choice(n_pts, keep, replace=False)
        pts_visible = s["points"][idx]
        e = add_s_metric(s["R_pred"], s["t_pred"], s["R_gt"], s["t_gt"], pts_visible)
        adds.append(float(e))
    return np.array(adds)


def evaluate_with_noise(samples, noise_sigma_mm, seed):
    """Calcula ADD-S anadiendo ruido Gaussiano a traslacion estimada."""
    rng = np.random.default_rng(seed)
    adds = []
    for s in samples:
        noise = rng.normal(0, noise_sigma_mm, size=3)
        t_noisy = s["t_pred"] + noise
        e = add_s_metric(s["R_pred"], t_noisy, s["R_gt"], s["t_gt"], s["points"])
        adds.append(float(e))
    return np.array(adds)


def auc_at_50mm(errors, n_steps=100):
    thresholds = np.linspace(0, 50, n_steps)
    recalls = [np.mean(errors < t) for t in thresholds]
    return float(np.trapezoid(recalls, thresholds) / 50.0)


def bootstrap_auc(errors, B=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(errors)
    boot = np.empty(B)
    for i in range(B):
        sample = rng.choice(errors, size=n, replace=True)
        boot[i] = auc_at_50mm(sample)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    print(f"[exp6] OCCLUSION={OCCLUSION_LEVELS}, NOISE={NOISE_LEVELS_MM} mm")
    out = {"datasets": {}}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n=== {ds_name.upper()} ===")
        if not ds_info["checkpoint"].exists():
            print(f"  [skip] sin checkpoint")
            continue
        if not ds_info["path"].exists():
            print(f"  [skip] sin dataset")
            continue

        samples = load_predictions_with_gt(ds_name, ds_info, n_max=300)
        print(f"  Muestras: {len(samples)}")

        ds_results = {"n_samples": len(samples), "occlusion": [], "noise": []}

        # Curva oclusion
        print("  Oclusion:")
        for occ in OCCLUSION_LEVELS:
            errors = evaluate_with_occlusion(samples, occ, seed=42)
            auc = auc_at_50mm(errors)
            lo, hi = bootstrap_auc(errors, B=200, seed=42)
            recall_10mm = float(np.mean(errors < 10.0))
            ds_results["occlusion"].append({
                "level_pct": occ * 100,
                "auc_adds_50mm": auc,
                "auc_adds_50mm_ci95": [lo, hi],
                "recall_adds_10mm": recall_10mm,
                "median_error_mm": float(np.median(errors)),
            })
            print(f"    {occ*100:.0f}%: AUC ADD-S = {auc:.4f} [CI {lo:.4f}, {hi:.4f}], R@10mm = {recall_10mm:.1%}")

        # Curva ruido
        print("  Ruido:")
        for noise in NOISE_LEVELS_MM:
            errors = evaluate_with_noise(samples, noise, seed=42)
            auc = auc_at_50mm(errors)
            lo, hi = bootstrap_auc(errors, B=200, seed=42)
            recall_10mm = float(np.mean(errors < 10.0))
            ds_results["noise"].append({
                "sigma_mm": noise,
                "auc_adds_50mm": auc,
                "auc_adds_50mm_ci95": [lo, hi],
                "recall_adds_10mm": recall_10mm,
                "median_error_mm": float(np.median(errors)),
            })
            print(f"    sigma={noise:.0f}mm: AUC ADD-S = {auc:.4f} [CI {lo:.4f}, {hi:.4f}], R@10mm = {recall_10mm:.1%}")

        out["datasets"][ds_name] = ds_results

    # Plot
    try:
        import matplotlib.pyplot as plt
        # Oclusion
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {'ycbv': '#0098CD', 'tless': '#FF6B35'}
        for ds_name, r in out["datasets"].items():
            x = [p["level_pct"] for p in r["occlusion"]]
            y = [p["auc_adds_50mm"] for p in r["occlusion"]]
            err_lo = [p["auc_adds_50mm"] - p["auc_adds_50mm_ci95"][0] for p in r["occlusion"]]
            err_hi = [p["auc_adds_50mm_ci95"][1] - p["auc_adds_50mm"] for p in r["occlusion"]]
            ax.errorbar(x, y, yerr=[err_lo, err_hi], marker='o', linewidth=2,
                        markersize=10, capsize=6, color=colors.get(ds_name, 'gray'),
                        label=ds_name.upper())
        ax.set_xlabel('Oclusión simulada (% de puntos del modelo)', fontsize=11)
        ax.set_ylabel('AUC ADD-S @ 50 mm', fontsize=11)
        ax.set_title('Robustez a oclusión: degradación de AUC ADD-S con CI 95% (B=200)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(OUT / 'fig_robustness_occlusion.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {OUT}/fig_robustness_occlusion.png")

        # Ruido
        fig, ax = plt.subplots(figsize=(9, 5))
        for ds_name, r in out["datasets"].items():
            x = [p["sigma_mm"] for p in r["noise"]]
            y = [p["auc_adds_50mm"] for p in r["noise"]]
            err_lo = [p["auc_adds_50mm"] - p["auc_adds_50mm_ci95"][0] for p in r["noise"]]
            err_hi = [p["auc_adds_50mm_ci95"][1] - p["auc_adds_50mm"] for p in r["noise"]]
            ax.errorbar(x, y, yerr=[err_lo, err_hi], marker='s', linewidth=2,
                        markersize=10, capsize=6, color=colors.get(ds_name, 'gray'),
                        label=ds_name.upper())
        ax.set_xlabel('Ruido sintético sobre traslación (sigma, mm)', fontsize=11)
        ax.set_ylabel('AUC ADD-S @ 50 mm', fontsize=11)
        ax.set_title('Robustez a ruido sensor: degradación de AUC ADD-S con CI 95% (B=200)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(OUT / 'fig_robustness_noise.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[OK] {OUT}/fig_robustness_noise.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")

    out_json = OUT / 'exp6_results.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {out_json}")


if __name__ == '__main__':
    main()
