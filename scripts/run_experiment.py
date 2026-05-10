#!/usr/bin/env python3
"""CLI unificada para ejecutar todos los experimentos del TFM.

Uso:
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py bootstrap         # exp1 (recompute + bootstrap)
    python scripts/run_experiment.py rotation          # exp3
    python scripts/run_experiment.py grasp             # exp4
    python scripts/run_experiment.py diffusion-steps   # exp5
    python scripts/run_experiment.py robustness        # exp6
    python scripts/run_experiment.py pbvs              # exp7
    python scripts/run_experiment.py diversity         # exp8
    python scripts/run_experiment.py viz3d             # exp9
    python scripts/run_experiment.py profiling         # exp10
    python scripts/run_experiment.py paired-stats      # exp11
    python scripts/run_experiment.py per-object        # exp12
    python scripts/run_experiment.py e2e-aggregate     # E2E offline
    python scripts/run_experiment.py e2e-live          # E2E con CoppeliaSim
    python scripts/run_experiment.py video             # grabar demo MP4
    python scripts/run_experiment.py all               # ejecutar TODOS los locales
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "bootstrap": {
        "script": "experiments/recompute_metrics_with_bootstrap.py",
        "desc": "Recomputar métricas FP con bootstrap CI 95% (B=1000)",
        "needs_gpu": False,
        "needs_coppelia": False,
        "duration_s": 60,
    },
    "rotation": {
        "script": "experiments/exp3_rotation_ablation.py",
        "desc": "Ablation representaciones rotación (quat/6D/euler/axis-angle)",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 30,
    },
    "grasp": {
        "script": "experiments/exp4_grasp_comparison.py",
        "desc": "Heurístico vs Diffusion grasp planners",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 60,
    },
    "diffusion-steps": {
        "script": "experiments/exp5_diffusion_steps_ablation.py",
        "desc": "Ablation n_diffusion_steps {25, 50, 100}",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 300,
    },
    "robustness": {
        "script": "experiments/exp6_robustness_analysis.py",
        "desc": "Robustez ante oclusión + ruido sensor",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 120,
    },
    "pbvs": {
        "script": "experiments/exp7_pbvs_convergence.py",
        "desc": "Convergencia PBVS sobre 50 poses reales",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 60,
    },
    "diversity": {
        "script": "experiments/exp8_diffusion_diversity.py",
        "desc": "Diversidad multimodal Diffusion (K-means + silhouette)",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 180,
    },
    "viz3d": {
        "script": "experiments/exp9_3d_trajectories_viz.py",
        "desc": "Visualización 3D de trayectorias multimodales",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 60,
    },
    "profiling": {
        "script": "experiments/exp10_profiling.py",
        "desc": "Profiling pipeline (cuello de botella)",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 60,
    },
    "paired-stats": {
        "script": "experiments/exp11_paired_stats.py",
        "desc": "Tests pareados Wilcoxon + Cohen's d",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 60,
    },
    "per-object": {
        "script": "experiments/exp12_per_object_analysis.py",
        "desc": "Análisis de error por categoría de objeto",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 90,
    },
    "e2e-aggregate": {
        "script": "experiments/aggregate_e2e_timings.py",
        "desc": "E2E offline (agregación de timings reales)",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 5,
    },
    "e2e-live": {
        "script": "experiments/run_e2e_live.py --n-instances 30 --ddim-steps 25 --sim-steps 50",
        "desc": "E2E live con CoppeliaSim corriendo",
        "needs_gpu": False, "needs_coppelia": True, "duration_s": 360,
    },
    "video": {
        "script": "experiments/record_e2e_video_v2.py --n-cycles 4 --fps 24",
        "desc": "Grabar video demo cinematográfico",
        "needs_gpu": False, "needs_coppelia": True, "duration_s": 300,
    },
    "consolidate": {
        "script": "experiments/run_chapter6_consolidation.py",
        "desc": "Consolidar resultados Cap 6 (figuras + tablas)",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 30,
    },
    "diagram": {
        "script": "experiments/generate_pipeline_diagram.py",
        "desc": "Generar diagrama de arquitectura del pipeline",
        "needs_gpu": False, "needs_coppelia": False, "duration_s": 5,
    },
}


def list_experiments():
    print("\nExperimentos disponibles:")
    print(f"{'Nombre':20} {'Tiempo':>8}  {'GPU':>4}  {'Coppelia':>9}  Descripción")
    print("─" * 100)
    for name, info in EXPERIMENTS.items():
        gpu = '✓' if info["needs_gpu"] else '·'
        coppelia = '✓' if info["needs_coppelia"] else '·'
        print(f"{name:20} {info['duration_s']:6}s  {gpu:>4}  {coppelia:>9}  {info['desc']}")
    print()


def run_experiment(name):
    if name not in EXPERIMENTS:
        print(f"❌ Experimento no reconocido: {name}")
        print(f"Disponibles: {', '.join(EXPERIMENTS.keys())}")
        return 1

    info = EXPERIMENTS[name]
    print(f"\n▶  Ejecutando: {name}")
    print(f"   {info['desc']}")
    print(f"   Tiempo estimado: {info['duration_s']}s")
    print()

    cmd = info["script"].split()
    cmd[0] = str(REPO / cmd[0])  # path absoluto
    full_cmd = [sys.executable] + cmd

    t0 = time.time()
    try:
        result = subprocess.run(full_cmd, cwd=REPO, check=False)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n✅  {name} completado en {elapsed:.1f}s")
            return 0
        else:
            print(f"\n❌  {name} falló con código {result.returncode}")
            return result.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️   {name} interrumpido por usuario")
        return 130


def run_all_local():
    """Ejecuta todos los experimentos que no requieren CoppeliaSim/GPU."""
    print("\n▶  Ejecutando todos los experimentos locales (sin CoppeliaSim, sin GPU)...\n")
    results = {}
    for name, info in EXPERIMENTS.items():
        if info["needs_coppelia"] or info["needs_gpu"]:
            print(f"⏭   Saltando {name} (requiere {'CoppeliaSim' if info['needs_coppelia'] else 'GPU'})")
            results[name] = "skipped"
            continue
        rc = run_experiment(name)
        results[name] = "ok" if rc == 0 else f"failed({rc})"

    print("\n" + "=" * 60)
    print("RESUMEN:")
    for name, status in results.items():
        marker = '✅' if status == 'ok' else ('⏭ ' if status == 'skipped' else '❌')
        print(f"  {marker} {name}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI unificada de experimentos TFM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("experiment", nargs='?', help="Nombre del experimento")
    parser.add_argument("--list", "-l", action="store_true", help="Listar experimentos disponibles")
    args = parser.parse_args()

    if args.list or not args.experiment:
        list_experiments()
        return 0

    if args.experiment == "all":
        return run_all_local()

    return run_experiment(args.experiment)


if __name__ == "__main__":
    sys.exit(main() or 0)
