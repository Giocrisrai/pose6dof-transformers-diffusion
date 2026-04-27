"""Aplicación de la planificación con Diffusion Policy a poses reales del run FoundationPose.

Carga `predictions_ycbv_*.json` y `predictions_tless_*.json` (poses reales
producidas por el run del 27-04-2026 en Colab T4) y para una muestra
representativa por dataset:

  1. Reconstruye SE(3) de la pose estimada (R_pred ∈ SO(3), t_pred ∈ R^3).
  2. Ejecuta `GraspSampler.sample()` (top-down) → candidatos de agarre.
  3. Genera trayectoria de aproximación con `generate_approach_trajectory()`.
  4. Genera trayectoria DDPM-style con `DiffusionGraspPlanner.plan_grasp_heuristic()`
     (la red neuronal de denoising no está entrenada en este TFM, así que
     usamos la trayectoria heurística que el wrapper expone para
     comparación cualitativa frente a la sampleada por la diffusion policy).
  5. Calcula métricas geométricas: longitud de trayectoria, suavidad
     (jerk discreto), profundidad de approach, score top-1 del sampler.
  6. Exporta JSON agregado + figura ilustrativa.

Salidas en `experiments/results/diffusion_real_poses/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.planning.diffusion_policy import DiffusionGraspPlanner  # noqa: E402
from src.planning.grasp_sampler import GraspCandidate, GraspSampler  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "experiments" / "results" / "foundationpose_eval"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "diffusion_real_poses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 30   # poses por dataset
N_GRASP_CANDIDATES = 64
HORIZON = 16       # waypoints DDPM
RNG_SEED = 42


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _trajectory_length(traj: np.ndarray) -> float:
    """Suma de distancias euclídeas entre waypoints consecutivos (m)."""
    if traj.ndim == 3:
        positions = traj[:, :3, 3]
    else:
        positions = traj[..., :3]
    diffs = np.diff(positions, axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())


def _trajectory_smoothness(traj: np.ndarray) -> float:
    """Norma media del jerk discreto (3ª derivada en posición). Más bajo = más suave."""
    if traj.ndim == 3:
        positions = traj[:, :3, 3]
    else:
        positions = traj[..., :3]
    if len(positions) < 4:
        return 0.0
    jerk = np.diff(positions, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def _process_one(pred: dict, sampler: GraspSampler, planner: DiffusionGraspPlanner) -> dict:
    R = np.array(pred["R_pred"], dtype=np.float64)
    t = np.array(pred["t_pred"], dtype=np.float64)
    T_obj = _se3(R, t)

    t0 = time.perf_counter()
    candidates = sampler.sample(
        object_pose=T_obj,
        n_candidates=N_GRASP_CANDIDATES,
        methods=["topdown", "side"],
    )
    sampling_ms = (time.perf_counter() - t0) * 1000.0

    if not candidates:
        return {
            "scene_id": pred["scene_id"],
            "img_id": pred["img_id"],
            "obj_id": pred["obj_id"],
            "ok": False,
            "reason": "no candidates after filtering",
        }

    best: GraspCandidate = candidates[0]

    t0 = time.perf_counter()
    approach_traj = sampler.generate_approach_trajectory(best, n_waypoints=HORIZON)
    approach_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    diffusion_traj = planner.plan_grasp_heuristic(T_obj)[0]  # (horizon, 7)
    diffusion_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "scene_id": pred["scene_id"],
        "img_id": pred["img_id"],
        "obj_id": pred["obj_id"],
        "ok": True,
        "n_candidates_after_filter": len(candidates),
        "best_grasp": {
            "score": float(best.score),
            "method": best.method,
            "width_m": float(best.width),
            "position_m": best.position().tolist(),
            "approach_vector": best.approach_vector().tolist(),
        },
        "approach_trajectory": {
            "n_waypoints": int(approach_traj.shape[0]),
            "length_m": _trajectory_length(approach_traj),
            "smoothness_jerk_mean": _trajectory_smoothness(approach_traj),
            "compute_ms": approach_ms,
        },
        "diffusion_trajectory": {
            "n_waypoints": int(diffusion_traj.shape[0]),
            "length_m": _trajectory_length(diffusion_traj),
            "smoothness_jerk_mean": _trajectory_smoothness(diffusion_traj),
            "gripper_open_at_start": bool(diffusion_traj[0, 6] > 0.5),
            "gripper_closed_at_end": bool(diffusion_traj[-1, 6] < 0.5),
            "compute_ms": diffusion_ms,
        },
        "sampling_ms": sampling_ms,
    }


def _aggregate(records: list[dict]) -> dict:
    ok = [r for r in records if r["ok"]]
    if not ok:
        return {"n_ok": 0, "n_total": len(records)}

    scores = np.array([r["best_grasp"]["score"] for r in ok])
    appr_len = np.array([r["approach_trajectory"]["length_m"] for r in ok])
    appr_jerk = np.array([r["approach_trajectory"]["smoothness_jerk_mean"] for r in ok])
    diff_len = np.array([r["diffusion_trajectory"]["length_m"] for r in ok])
    diff_jerk = np.array([r["diffusion_trajectory"]["smoothness_jerk_mean"] for r in ok])
    sampling_ms = np.array([r["sampling_ms"] for r in ok])

    gripper_open_start = sum(r["diffusion_trajectory"]["gripper_open_at_start"] for r in ok)
    gripper_closed_end = sum(r["diffusion_trajectory"]["gripper_closed_at_end"] for r in ok)

    return {
        "n_ok": len(ok),
        "n_total": len(records),
        "best_grasp_score": {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        },
        "approach_trajectory_length_m": {
            "mean": float(appr_len.mean()),
            "median": float(np.median(appr_len)),
        },
        "approach_jerk": {
            "mean": float(appr_jerk.mean()),
            "median": float(np.median(appr_jerk)),
        },
        "diffusion_trajectory_length_m": {
            "mean": float(diff_len.mean()),
            "median": float(np.median(diff_len)),
        },
        "diffusion_jerk": {
            "mean": float(diff_jerk.mean()),
            "median": float(np.median(diff_jerk)),
        },
        "sampling_ms": {
            "mean": float(sampling_ms.mean()),
            "median": float(np.median(sampling_ms)),
            "p95": float(np.percentile(sampling_ms, 95)),
        },
        "gripper_phase_consistency": {
            "open_at_start_pct": 100.0 * gripper_open_start / len(ok),
            "closed_at_end_pct": 100.0 * gripper_closed_end / len(ok),
        },
    }


def _select_sample(predictions: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    if len(predictions) <= n:
        return list(predictions)
    idx = rng.choice(len(predictions), size=n, replace=False)
    return [predictions[i] for i in sorted(idx.tolist())]


def _make_figure(per_dataset: dict, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Subplot 1: distribución de scores top-1 por dataset
    ax = axes[0]
    for name, color in [("ycbv", "tab:blue"), ("tless", "tab:orange")]:
        records = [r for r in per_dataset[name]["records"] if r["ok"]]
        scores = [r["best_grasp"]["score"] for r in records]
        ax.hist(scores, bins=20, alpha=0.55, label=f"{name.upper()} (n={len(scores)})", color=color)
    ax.set_xlabel("Score top-1 del grasp candidato (norm.)")
    ax.set_ylabel("# muestras")
    ax.set_title("Distribución de scores de agarre — poses reales FP")
    ax.grid(alpha=0.3)
    ax.legend()

    # Subplot 2: longitud de trayectoria (approach vs DDPM-heuristic)
    ax = axes[1]
    bar_data = []
    for name in ("ycbv", "tless"):
        agg = per_dataset[name]["aggregate"]
        bar_data.append([
            agg["approach_trajectory_length_m"]["median"],
            agg["diffusion_trajectory_length_m"]["median"],
        ])
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [b[0] for b in bar_data], w, label="GraspSampler approach", color="tab:green")
    ax.bar(x + w / 2, [b[1] for b in bar_data], w, label="DiffusionPlanner heur.", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(["YCB-V", "T-LESS"])
    ax.set_ylabel("Longitud mediana de trayectoria (m)")
    ax.set_title("Pipeline percepción → planificación con poses reales")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    fig.suptitle("Diffusion-style grasp planning sobre poses reales del run FP (2026-04-27)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    np.random.seed(RNG_SEED)

    sampler = GraspSampler(gripper_width=0.085, standoff_distance=0.10)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=HORIZON, device="cpu")

    per_dataset: dict[str, Any] = {}
    for ds in ("ycbv", "tless"):
        candidates = sorted(PREDICTIONS_DIR.glob(f"predictions_{ds}_*.json"))
        if not candidates:
            print(f"[WARN] sin predictions_{ds}_*.json en {PREDICTIONS_DIR}")
            continue
        latest = candidates[-1]
        with latest.open() as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        sample = _select_sample(preds, SAMPLE_SIZE, seed=RNG_SEED)
        print(f"[{ds.upper()}] cargado {latest.name} → {len(preds)} preds → "
              f"muestra de {len(sample)}")

        records = [_process_one(p, sampler, planner) for p in sample]
        agg = _aggregate(records)
        per_dataset[ds] = {
            "source_predictions": latest.name,
            "n_total_predictions": len(preds),
            "sample_size": len(sample),
            "records": records,
            "aggregate": agg,
        }

    out_json = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rng_seed": RNG_SEED,
            "horizon": HORIZON,
            "n_grasp_candidates": N_GRASP_CANDIDATES,
            "sampler": {
                "gripper_width_m": sampler.gripper_width,
                "standoff_distance_m": sampler.standoff_distance,
                "approach_min_angle_rad": sampler.approach_min_angle,
                "approach_max_angle_rad": sampler.approach_max_angle,
            },
            "planner": {
                "action_dim": planner.action_dim,
                "horizon": planner.horizon,
                "n_diffusion_steps": planner.scheduler.num_timesteps,
                "trained": planner._trained,
            },
            "source_run": "experiments/results/foundationpose_eval/comparison_20260427_084807.json",
        },
        "datasets": per_dataset,
    }

    out_path = OUTPUT_DIR / "trajectories_summary.json"
    with out_path.open("w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[OK] {out_path.relative_to(REPO_ROOT)}")

    fig_path = OUTPUT_DIR / "fig_diffusion_planning_real.png"
    _make_figure(per_dataset, fig_path)
    print(f"[OK] {fig_path.relative_to(REPO_ROOT)}")

    print("\n=== RESUMEN ===")
    for ds, blob in per_dataset.items():
        agg = blob["aggregate"]
        print(f"\n{ds.upper()}  ({agg['n_ok']}/{agg['n_total']} ok)")
        print(f"  best grasp score (median):     {agg['best_grasp_score']['median']:.3f}")
        print(f"  approach trajectory length:    {agg['approach_trajectory_length_m']['median']*100:.1f} cm")
        print(f"  diffusion trajectory length:   {agg['diffusion_trajectory_length_m']['median']*100:.1f} cm")
        print(f"  sampler latency (median):      {agg['sampling_ms']['median']:.1f} ms")
        print(f"  gripper open-at-start:         {agg['gripper_phase_consistency']['open_at_start_pct']:.0f} %")
        print(f"  gripper closed-at-end:         {agg['gripper_phase_consistency']['closed_at_end_pct']:.0f} %")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
