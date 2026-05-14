#!/usr/bin/env python3
"""Experimento 15: comparativa licencia × performance.

Para cada alternativa open-license (FreeZeV2, MegaPose, Any6D, SamPose)
medimos:
- AUC ADD-S @50 mm con bootstrap CI 95 % (usando bop-bootstrap-ci)
- Recall@10 mm con bootstrap CI 95 %
- Degradacion vs baseline FoundationPose
- Status de comercializacion (licencia open)

Approach: usamos los checkpoints reales de FoundationPose sobre YCB-V y
T-LESS (1098 + 1012 instancias) y simulamos cada alternativa anadiendo
ruido R+t calibrado segun los numeros publicados de cada metodo.

Esto da una estimacion *cuantitativa* y *reproducible* de cuanta calidad
perderiamos al cambiar FP por una alternativa, sin necesidad de descargar
e integrar cada modelo (que tomaria semanas).

Salida:
    experiments/results/exp15_open_license/exp15_results.json
    experiments/results/exp15_open_license/fig_pareto.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bop_bootstrap_ci import bootstrap_auc_adds, bootstrap_recall
from src.perception.checkpoint_adapter import (
    NOISE_PROFILES,
    CheckpointPoseEstimator,
)
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

OUTPUT = REPO / "experiments/results/exp15_open_license"
OUTPUT.mkdir(parents=True, exist_ok=True)


def evaluate_method(method_name, dataset_name, dataset_path, split, checkpoint_path,
                     n_model_pts=500, seed=42, verbose=False):
    """Devuelve errores ADD y ADD-S por instancia para un metodo dado."""
    estimator = CheckpointPoseEstimator(checkpoint_path, method=method_name, seed=seed)
    ds = BOPDataset(str(dataset_path), split=split)
    rng = np.random.default_rng(seed)

    gt_cache = {}
    mesh_cache = {}
    add_errors, adds_errors = [], []
    skipped = 0

    # Iterar sobre el checkpoint para conocer scene/img/obj triples a evaluar
    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    for pred in ckpt["results"]:
        scene_id = pred["scene_id"]
        if isinstance(scene_id, int):
            scene_id = f"{scene_id:06d}"
        img_id = pred["img_id"]
        obj_id = pred["obj_id"]
        gt_idx = pred.get("gt_idx", -1)

        if scene_id not in gt_cache:
            gt_cache[scene_id] = ds.load_scene_gt(scene_id)
        gt_list = gt_cache[scene_id].get(str(img_id), [])
        if not gt_list or gt_idx < 0 or gt_idx >= len(gt_list):
            skipped += 1
            continue
        gt_match = gt_list[gt_idx]
        if gt_match["obj_id"] != obj_id:
            skipped += 1
            continue

        if obj_id not in mesh_cache:
            try:
                import trimesh
                model_path = ds.get_model_path(obj_id)
                if not Path(model_path).exists():
                    mesh_cache[obj_id] = None
                else:
                    mesh = trimesh.load(str(model_path), process=False)
                    pts = np.array(mesh.vertices, dtype=np.float64)
                    if len(pts) > n_model_pts:
                        idx = rng.choice(len(pts), n_model_pts, replace=False)
                        pts = pts[idx]
                    mesh_cache[obj_id] = pts
            except Exception:
                mesh_cache[obj_id] = None
        if mesh_cache[obj_id] is None:
            skipped += 1
            continue
        pts = mesh_cache[obj_id]

        try:
            est = estimator.predict_pose(scene_id=scene_id, img_id=img_id,
                                          obj_id=obj_id, gt_idx=gt_idx)
        except KeyError:
            skipped += 1
            continue

        R_gt = np.array(gt_match["cam_R_m2c"], dtype=np.float64)
        t_gt = np.array(gt_match["cam_t_m2c"], dtype=np.float64).flatten()

        try:
            a = add_metric(est.R, est.t, R_gt, t_gt, pts)
            adss = add_s_metric(est.R, est.t, R_gt, t_gt, pts)
            add_errors.append(float(a))
            adds_errors.append(float(adss))
        except Exception:
            skipped += 1

    return np.array(add_errors), np.array(adds_errors), skipped


def main():
    print("[exp15] Iniciando comparativa licencia x performance")
    methods = list(NOISE_PROFILES.keys())  # ['foundationpose','freezev2','megapose','any6d','sampose']

    results = {
        "description": "Comparativa de alternativas open-license para sustituir FoundationPose",
        "methodology": (
            "Para cada metodo se reproducen las predicciones reales del checkpoint "
            "FoundationPose y se anade ruido R+t calibrado segun los numeros publicados "
            "del metodo. Se computan AUC ADD-S @50mm y Recall@10mm con bootstrap CI 95 % "
            "(B=1000) usando bop-bootstrap-ci."
        ),
        "datasets": {},
    }

    for ds_name, info in DATASETS.items():
        print(f"\n=== {ds_name.upper()} ===")
        if not info["path"].exists():
            print(f"  [skip] dataset no disponible: {info['path']}")
            continue
        if not info["checkpoint"].exists():
            print(f"  [skip] checkpoint no disponible: {info['checkpoint']}")
            continue

        ds_results = {}
        for method in methods:
            profile = NOISE_PROFILES[method]
            print(f"\n  -> {method} (license: {profile['license']}, "
                  f"noise_t={profile['noise_t_mm_std']}mm, noise_R={profile['noise_R_rad_std']}rad)")
            t0 = time.time()
            add_e, adds_e, skipped = evaluate_method(
                method, ds_name, info["path"], info["split"], info["checkpoint"],
                seed=42 + hash(method) % 1000,
            )
            elapsed = time.time() - t0
            n = len(add_e)
            print(f"     n_evaluated={n}, skipped={skipped}, time={elapsed:.1f}s")

            if n == 0:
                continue

            # Bootstrap CI usando bop-bootstrap-ci (exploracion 1)
            auc_ci = bootstrap_auc_adds(adds_e, max_threshold_mm=50.0, B=1000, seed=42)
            rec10_ci = bootstrap_recall(adds_e, threshold=10.0, B=1000, seed=43)
            rec5_ci = bootstrap_recall(adds_e, threshold=5.0, B=1000, seed=44)

            ds_results[method] = {
                "license": profile["license"],
                "commercializable": profile["commercial"],
                "noise_t_mm_std": profile["noise_t_mm_std"],
                "noise_R_rad_std": profile["noise_R_rad_std"],
                "reference": profile["reference"],
                "n_evaluated": int(n),
                "add_mean_mm": float(np.mean(add_e)),
                "adds_median_mm": float(np.median(adds_e)),
                "auc_adds_50mm": auc_ci.as_dict(),
                "recall_adds_10mm": rec10_ci.as_dict(),
                "recall_adds_5mm": rec5_ci.as_dict(),
                "eval_time_s": elapsed,
            }
            print(f"     AUC ADD-S @50mm: {auc_ci.point:.4f} [{auc_ci.lo:.4f}, {auc_ci.hi:.4f}]")
            print(f"     Recall@10mm    : {rec10_ci.point:.1%} [{rec10_ci.lo:.1%}, {rec10_ci.hi:.1%}]")

        # Computar degradacion vs FP baseline
        if "foundationpose" in ds_results:
            fp_auc = ds_results["foundationpose"]["auc_adds_50mm"]["point"]
            fp_rec = ds_results["foundationpose"]["recall_adds_10mm"]["point"]
            for method, m_res in ds_results.items():
                m_res["degradation_auc_pp"] = round(
                    (fp_auc - m_res["auc_adds_50mm"]["point"]) * 100, 2)
                m_res["degradation_recall_pp"] = round(
                    (fp_rec - m_res["recall_adds_10mm"]["point"]) * 100, 2)

        results["datasets"][ds_name] = ds_results

    # Criterios del plan
    criteria = {
        "auc_adds_ycbv_min": 0.85,
        "auc_adds_tless_min": 0.90,
        "max_degradation_pct": 10.0,
    }

    # Identificar alternativas open-license que cumplen criterios
    candidates = []
    for ds_name, ds_results in results["datasets"].items():
        for method, m_res in ds_results.items():
            if not m_res["commercializable"]:
                continue  # Solo open-license
            auc = m_res["auc_adds_50mm"]["point"]
            min_auc = criteria[f"auc_adds_{ds_name}_min"]
            if auc >= min_auc:
                candidates.append({
                    "method": method,
                    "dataset": ds_name,
                    "auc": auc,
                    "license": m_res["license"],
                })

    results["criteria"] = criteria
    results["candidates_passing"] = candidates
    results["all_criteria_pass"] = len(candidates) > 0

    # Guardar
    out_json = OUTPUT / "exp15_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {out_json}")

    # Figura Pareto
    try:
        plot_pareto(results)
    except Exception as e:
        print(f"[warn] plot: {e}")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN — ALTERNATIVAS OPEN-LICENSE QUE PASAN CRITERIOS:")
    print("=" * 70)
    if not candidates:
        print("  Ninguna alternativa open-license alcanza los umbrales planteados.")
        print("  Ver detalles en exp15_results.json (degradaciones por metodo).")
    else:
        for c in candidates:
            print(f"  {c['method']:12s} en {c['dataset']:5s}: "
                  f"AUC={c['auc']:.4f}, license={c['license']}")


def plot_pareto(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ds_name, ds_results) in zip(axes, results["datasets"].items()):
        for method, m_res in ds_results.items():
            x = m_res["noise_t_mm_std"]  # proxy de "openness/risk"
            y = m_res["auc_adds_50mm"]["point"]
            ci_lo = m_res["auc_adds_50mm"]["lo"]
            ci_hi = m_res["auc_adds_50mm"]["hi"]
            color = "#35876B" if m_res["commercializable"] else "#FF6B35"
            marker = "o" if m_res["commercializable"] else "X"
            ax.errorbar(x, y, yerr=[[y-ci_lo], [ci_hi-y]],
                          fmt=marker, markersize=12, capsize=5,
                          color=color, label=f"{method} ({m_res['license']})")
            ax.annotate(method, (x, y), xytext=(8, 5), textcoords="offset points",
                            fontsize=9)
        ax.set_xlabel("Ruido translacion calibrado (mm)")
        ax.set_ylabel("AUC ADD-S @50 mm")
        ax.set_title(f"Pareto licencia × performance — {ds_name.upper()}")
        ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0.05, 0.86, "Umbral 0.85", color="red", fontsize=9, transform=ax.get_yaxis_transform())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    out_png = OUTPUT / "fig_pareto.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
