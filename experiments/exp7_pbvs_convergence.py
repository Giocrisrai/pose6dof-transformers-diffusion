#!/usr/bin/env python3
"""Validacion del controlador PBVS sobre poses reales del checkpoint FP.

Para cada instancia del checkpoint:
1. T_initial = pose estimada por FP (con perturbacion controlada)
2. T_target  = pose GT del dataset
3. Ejecutar bucle PBVS hasta convergencia
4. Reportar iteraciones, error final, tiempo de convergencia

Genera:
- Curva de convergencia tipica
- Histograma de iteraciones hasta convergencia
- Tabla de tasa de exito (% que converge en < N iteraciones)

Salidas:
    experiments/results/exp7_pbvs/exp7_results.json
    experiments/results/exp7_pbvs/fig_pbvs_convergence.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp7_pbvs"
OUT.mkdir(parents=True, exist_ok=True)

from src.control import PBVSController, simulate_pbvs_loop


def build_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t / 1000.0  # mm a m para PBVS
    return T


def main():
    print("[exp7] Validacion PBVS sobre poses reales FoundationPose")

    ckpt_path = REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json"
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    samples = ckpt["results"][:50]
    print(f"  Muestras: {len(samples)}")

    # Cargar GT
    sys.path.insert(0, str(REPO))
    from src.utils.dataset_loader import BOPDataset
    ds = BOPDataset(str(REPO / "data/datasets/ycbv"), split="test")
    gt_cache = {}

    controller = PBVSController(kp_lin=1.5, kp_ang=1.5,
                                eps_lin=0.002, eps_ang=0.01)
    print(f"  Controlador: kp_lin={controller.kp_lin}, kp_ang={controller.kp_ang}, "
          f"eps={controller.eps_lin*1000}mm/{np.degrees(controller.eps_ang):.1f}deg")

    results = []
    for pred in samples:
        scene_id = f"{int(pred['scene_id']):06d}" if isinstance(pred['scene_id'], int) else pred['scene_id']
        if scene_id not in gt_cache:
            gt_cache[scene_id] = ds.load_scene_gt(scene_id)
        gt_list = gt_cache[scene_id].get(str(pred["img_id"]), [])
        gt_idx = pred.get("gt_idx", -1)
        if gt_idx < 0 or gt_idx >= len(gt_list):
            continue
        gt = gt_list[gt_idx]

        R_pred = np.array(pred["R_pred"])
        t_pred = np.array(pred["t_pred"])
        if np.linalg.norm(t_pred) < 5.0:
            t_pred = t_pred * 1000.0  # m -> mm

        T_initial = build_T(R_pred, t_pred)
        T_target  = build_T(np.array(gt["cam_R_m2c"]), np.array(gt["cam_t_m2c"]))

        loop = simulate_pbvs_loop(T_initial, T_target, dt=0.05, max_iters=500,
                                   controller=controller)
        results.append({
            "obj_id": pred["obj_id"],
            "n_iters": loop["n_iters"],
            "converged": loop["converged"],
            "converged_at": loop["converged_at"],
            "final_error_lin_m": loop["errors_lin_m"][-1] if loop["errors_lin_m"] else None,
            "final_error_ang_rad": loop["errors_ang_rad"][-1] if loop["errors_ang_rad"] else None,
            "errors_lin_m": loop["errors_lin_m"],
            "errors_ang_rad": loop["errors_ang_rad"],
        })

    n_total = len(results)
    n_converged = sum(1 for r in results if r["converged"])
    iters_converged = [r["n_iters"] for r in results if r["converged"]]

    summary = {
        "n_samples": n_total,
        "n_converged": n_converged,
        "convergence_rate_pct": 100.0 * n_converged / max(n_total, 1),
        "median_iters": float(np.median(iters_converged)) if iters_converged else None,
        "p95_iters": float(np.percentile(iters_converged, 95)) if iters_converged else None,
        "mean_time_s_p95": float(np.percentile(iters_converged, 95) * 0.05) if iters_converged else None,
        "controller": {"kp_lin": controller.kp_lin, "kp_ang": controller.kp_ang,
                       "eps_lin_mm": controller.eps_lin * 1000,
                       "eps_ang_deg": float(np.degrees(controller.eps_ang))},
    }

    print(f"\n  Convergencia: {n_converged}/{n_total} ({summary['convergence_rate_pct']:.1f}%)")
    if iters_converged:
        print(f"  Iteraciones mediana: {summary['median_iters']:.0f}")
        print(f"  Iteraciones p95: {summary['p95_iters']:.0f}")
        print(f"  Tiempo p95 (dt=50ms): {summary['mean_time_s_p95']:.2f} s")

    # Plot convergencia
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Curvas tipicas (primeras 10 que convergieron)
        ax = axes[0]
        for r in [r for r in results if r["converged"]][:10]:
            iters = np.arange(len(r["errors_lin_m"])) * 0.05
            ax.plot(iters, np.array(r["errors_lin_m"]) * 1000, alpha=0.4, color='#0098CD', linewidth=1)
        # Promedio
        max_len = max((len(r["errors_lin_m"]) for r in results if r["converged"]), default=1)
        avg_curve = []
        for k in range(min(max_len, 200)):
            vals = [r["errors_lin_m"][k] for r in results if r["converged"] and k < len(r["errors_lin_m"])]
            if vals:
                avg_curve.append(np.median(vals) * 1000)
        t = np.arange(len(avg_curve)) * 0.05
        ax.plot(t, avg_curve, color='#FF6B35', linewidth=2.5, label='Mediana')
        ax.axhline(controller.eps_lin*1000, color='gray', linestyle='--', label=f'Tolerancia ({controller.eps_lin*1000:.0f} mm)')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Error de traslación (mm)')
        ax.set_title(f'Convergencia PBVS — Error lineal (n={len(results)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Histograma iters
        ax = axes[1]
        if iters_converged:
            ax.hist(np.array(iters_converged) * 0.05, bins=20, color='#35876B', edgecolor='black')
            ax.axvline(summary['median_iters'] * 0.05, color='#FF6B35', linewidth=2.5,
                       label=f"Mediana = {summary['median_iters']*0.05:.2f} s")
            ax.axvline(summary['p95_iters'] * 0.05, color='red', linestyle='--', linewidth=2,
                       label=f"p95 = {summary['p95_iters']*0.05:.2f} s")
        ax.set_xlabel('Tiempo de convergencia (s)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de tiempos PBVS — {n_converged}/{n_total} convergen')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT / 'fig_pbvs_convergence.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {OUT}/fig_pbvs_convergence.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")

    out_json = OUT / 'exp7_results.json'
    with open(out_json, 'w') as f:
        # No guardamos curvas completas (muy verbose), solo resumen
        compact = {
            "summary": summary,
            "samples": [{k: v for k, v in r.items() if k not in ("errors_lin_m", "errors_ang_rad")}
                        for r in results]
        }
        json.dump(compact, f, indent=2)
    print(f"[OK] {out_json}")


if __name__ == '__main__':
    main()
