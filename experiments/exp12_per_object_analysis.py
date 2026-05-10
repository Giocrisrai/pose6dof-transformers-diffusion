#!/usr/bin/env python3
"""Analisis de error por categoria de objeto (per-object breakdown).

Para cada obj_id de YCB-V y T-LESS, calcula:
- AUC ADD-S por objeto
- Recall@10mm por objeto
- Numero de instancias evaluadas
- Identifica peores objetos (failure cases)

Salida:
    experiments/results/exp12_per_object/exp12_results.json
    experiments/results/exp12_per_object/fig_per_object_ycbv.png
    experiments/results/exp12_per_object/fig_per_object_tless.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp12_per_object"
OUT.mkdir(parents=True, exist_ok=True)

from src.utils.dataset_loader import BOPDataset
from src.utils.metrics import add_s_metric


def compute_per_object_errors(ds_name, ds_path, split, n_max=None):
    import trimesh
    with open(REPO / f"experiments/checkpoints/fp_{ds_name}_checkpoint.json") as f:
        ckpt = json.load(f)
    ds = BOPDataset(str(ds_path), split=split)

    gt_cache, mesh_cache = {}, {}
    rng = np.random.default_rng(42)
    errors_by_obj = defaultdict(list)

    preds = ckpt["results"][:n_max] if n_max else ckpt["results"]
    for pred in preds:
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
                mesh = trimesh.load(str(mp), process=False)
                pts = np.array(mesh.vertices, dtype=np.float64)
                if len(pts) > 1000:
                    idx = rng.choice(len(pts), 1000, replace=False)
                    pts = pts[idx]
                mesh_cache[obj_id] = pts
            except Exception:
                continue
        if obj_id not in mesh_cache:
            continue

        R_pred = np.array(pred["R_pred"])
        t_pred = np.array(pred["t_pred"])
        if np.linalg.norm(t_pred) < 5.0:
            t_pred = t_pred * 1000.0

        e = add_s_metric(R_pred, t_pred, np.array(gt["cam_R_m2c"]), np.array(gt["cam_t_m2c"]),
                         mesh_cache[obj_id])
        errors_by_obj[obj_id].append(float(e))

    return errors_by_obj


def auc_at_50mm(errors, n_steps=100):
    thresholds = np.linspace(0, 50, n_steps)
    recalls = [np.mean(errors < t) for t in thresholds]
    return float(np.trapezoid(recalls, thresholds) / 50.0)


def main():
    print("[exp12] Analisis de error por categoria de objeto")

    DATASETS = {
        "ycbv": {"path": REPO / "data/datasets/ycbv", "split": "test"},
        "tless": {"path": REPO / "data/datasets/tless", "split": "test_primesense"},
    }

    out = {"datasets": {}}

    for ds_name, info in DATASETS.items():
        if not info["path"].exists():
            continue
        print(f"\n=== {ds_name.upper()} ===")
        errors_by_obj = compute_per_object_errors(ds_name, info["path"], info["split"])
        print(f"  Objetos evaluados: {len(errors_by_obj)}")

        per_obj_stats = {}
        for obj_id, errs in sorted(errors_by_obj.items()):
            errs_arr = np.array(errs)
            per_obj_stats[obj_id] = {
                "n": len(errs),
                "auc_adds_50mm": auc_at_50mm(errs_arr),
                "recall_10mm": float(np.mean(errs_arr < 10.0)),
                "median_mm": float(np.median(errs_arr)),
                "p95_mm": float(np.percentile(errs_arr, 95)),
            }

        # Identificar peores y mejores
        sorted_by_auc = sorted(per_obj_stats.items(), key=lambda x: x[1]["auc_adds_50mm"])
        worst = sorted_by_auc[:3]
        best = sorted_by_auc[-3:]

        print(f"\n  Peores objetos (menor AUC):")
        for obj_id, s in worst:
            print(f"    obj_id={obj_id}: AUC={s['auc_adds_50mm']:.3f}, R@10mm={s['recall_10mm']:.1%}, n={s['n']}")
        print(f"\n  Mejores objetos:")
        for obj_id, s in best:
            print(f"    obj_id={obj_id}: AUC={s['auc_adds_50mm']:.3f}, R@10mm={s['recall_10mm']:.1%}, n={s['n']}")

        out["datasets"][ds_name] = {
            "n_objects": len(per_obj_stats),
            "per_object": {str(k): v for k, v in per_obj_stats.items()},
            "worst_3": [{"obj_id": o, **s} for o, s in worst],
            "best_3": [{"obj_id": o, **s} for o, s in best],
            "auc_adds_global_mean": float(np.mean([s["auc_adds_50mm"] for s in per_obj_stats.values()])),
            "auc_adds_std": float(np.std([s["auc_adds_50mm"] for s in per_obj_stats.values()])),
        }

    # Plot
    try:
        import matplotlib.pyplot as plt
        for ds_name, r in out["datasets"].items():
            objs = sorted(r["per_object"].keys(), key=lambda x: r["per_object"][x]["auc_adds_50mm"])
            aucs = [r["per_object"][o]["auc_adds_50mm"] for o in objs]
            recalls = [r["per_object"][o]["recall_10mm"] for o in objs]

            fig, ax = plt.subplots(figsize=(max(10, len(objs)*0.4), 5))
            x = np.arange(len(objs))
            colors = ['#FF6B35' if a < 0.85 else '#0098CD' if a < 0.95 else '#35876B' for a in aucs]
            ax.bar(x, aucs, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(objs, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('AUC ADD-S @ 50 mm')
            ax.set_xlabel('obj_id')
            ax.set_title(f'AUC ADD-S por objeto — {ds_name.upper()} ({r["n_objects"]} objetos)\n'
                         f'Media: {r["auc_adds_global_mean"]:.3f} ± {r["auc_adds_std"]:.3f}')
            ax.axhline(r["auc_adds_global_mean"], color='red', linestyle='--', linewidth=1, alpha=0.5,
                       label=f'Media: {r["auc_adds_global_mean"]:.3f}')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            plt.tight_layout()
            plt.savefig(OUT / f'fig_per_object_{ds_name}.png', dpi=180, bbox_inches='tight')
            plt.close()
            print(f"\n[OK] {OUT}/fig_per_object_{ds_name}.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")

    with open(OUT / 'exp12_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {OUT}/exp12_results.json")


if __name__ == '__main__':
    main()
