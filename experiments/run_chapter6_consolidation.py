"""Consolidador del cap. 6 — sustituye `notebooks/colab/03_results_analysis.ipynb`
para ejecución local (que requiere paths de Drive no presentes en repo).

Lee los tres JSON de evidencia experimental ya versionados:

  - experiments/results/foundationpose_eval/comparison_*.json
  - experiments/results/diffusion_real_poses/trajectories_summary.json
  - experiments/results/coppelia_smoke/smoke_test_result.json

Y genera:

  - experiments/results/chapter6_figures/fig_6_pipeline_dashboard.png
    (4 paneles: FP métricas + FP vs baselines + Diffusion + CoppeliaSim)
  - experiments/results/chapter6_figures/cap6_master_table.md
  - experiments/results/comparison_fp_vs_gdrnet.json
    (compatible con el formato que produciría el notebook 03)

Baselines oficiales (BOP Challenge 2022 + paper FoundationPose) tomados
literalmente de `docs/cap6_seccion_foundationpose.md` (Tabla 6.Y).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIG_DIR = RESULTS_DIR / "chapter6_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paleta UNIR (consistente con generate_chapter6_figures.py)
AZUL = "#0098CD"
NARANJA = "#FF6B35"
VERDE = "#2EA66B"
PURPURA = "#7E57C2"
GRIS = "#858591"

# Baselines oficiales — fuente única reconciliada (mismo set que notebook 02 /
# JSON producido en Colab el 2026-04-28). Ordenación BOP típica VSD < MSSD < MSPD.
# - GDR-Net++ : valores reportados en BOP Challenge 2022 leaderboard
#   (Sundermeyer et al., 2024) y replicados en notebook 02.
# - FoundationPose : valores reportados en Wen et al., CVPR 2024, Table 1,
#   columna correspondiente al subset BOP-19 evaluado.
GDR_BOP2022 = {
    "ycbv":  {"AR_VSD": 0.841, "AR_MSSD": 0.868, "AR_MSPD": 0.893, "Mean_AR": 0.867},
    "tless": {"AR_VSD": 0.712, "AR_MSSD": 0.764, "AR_MSPD": 0.825, "Mean_AR": 0.767},
}

FP_PAPER = {
    "ycbv":  {"AR_VSD": 0.872, "AR_MSSD": 0.898, "AR_MSPD": 0.921, "Mean_AR": 0.897},
    "tless": {"AR_VSD": 0.752, "AR_MSSD": 0.801, "AR_MSPD": 0.856, "Mean_AR": 0.803},
}


def _latest(glob_pattern: str, base: Path) -> Path | None:
    candidates = sorted(base.glob(glob_pattern))
    return candidates[-1] if candidates else None


def _load_fp_real() -> dict:
    """Aplana comparison_*.json al formato {ycbv: metrics, tless: metrics}."""
    path = _latest("comparison_*.json", RESULTS_DIR / "foundationpose_eval")
    if path is None:
        raise FileNotFoundError("comparison_*.json no encontrado")
    raw = json.loads(path.read_text())
    flat: dict = {"_meta": {"source": str(path.relative_to(REPO_ROOT))}}
    if "our_results" in raw:
        for ds in ("ycbv", "tless"):
            bucket = raw["our_results"].get(ds, {})
            metrics = bucket.get("metrics", {}) or {}
            if metrics:
                flat[ds] = metrics
    else:  # ya es plano
        for ds in ("ycbv", "tless"):
            if ds in raw:
                flat[ds] = raw[ds]
    return flat


def _load_diffusion() -> dict:
    path = RESULTS_DIR / "diffusion_real_poses" / "trajectories_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_coppelia() -> dict:
    path = RESULTS_DIR / "coppelia_smoke" / "smoke_test_result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ── Figura: dashboard de 4 paneles ───────────────────────────────────


def make_dashboard(fp_real: dict, diffusion: dict, coppelia: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: ADD/ADD-S recalls FP propio
    ax = axes[0, 0]
    thresholds = ["5mm", "10mm", "20mm"]
    x = np.arange(len(thresholds))
    w = 0.20
    for i, ds in enumerate(("ycbv", "tless")):
        m = fp_real.get(ds, {})
        add_vals = [m.get(f"recall_add_{t}", 0) * 100 for t in thresholds]
        adds_vals = [m.get(f"recall_adds_{t}", 0) * 100 for t in thresholds]
        offset = -1.5 + i * 2  # -1.5 y +0.5 → grupos por dataset
        ax.bar(x + (offset - 0.5) * w, add_vals, w, label=f"{ds.upper()} ADD",
               color=AZUL if ds == "ycbv" else NARANJA, alpha=0.6)
        ax.bar(x + (offset + 0.5) * w, adds_vals, w, label=f"{ds.upper()} ADD-S",
               color=AZUL if ds == "ycbv" else NARANJA, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Panel A — FoundationPose: recalls ADD / ADD-S (subset BOP-19)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Mean AR comparison (FP paper vs GDR BOP2022)
    ax = axes[0, 1]
    metrics = ["AR_VSD", "AR_MSSD", "AR_MSPD", "Mean_AR"]
    labels = ["VSD", "MSSD", "MSPD", "Mean"]
    x = np.arange(len(metrics))
    w = 0.18
    for i, ds in enumerate(("ycbv", "tless")):
        fp_vals = [FP_PAPER[ds][m] for m in metrics]
        gdr_vals = [GDR_BOP2022[ds][m] for m in metrics]
        offset_fp = -2 + i * 2
        offset_gdr = -1 + i * 2
        ax.bar(x + offset_fp * w, fp_vals, w,
               label=f"{ds.upper()} — FP paper", color=AZUL if ds == "ycbv" else NARANJA)
        ax.bar(x + offset_gdr * w, gdr_vals, w,
               label=f"{ds.upper()} — GDR-Net++ BOP22",
               color=AZUL if ds == "ycbv" else NARANJA, alpha=0.5, hatch="\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Recall")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Panel B — FoundationPose vs GDR-Net++ (baselines oficiales)")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Diffusion planning (con poses reales)
    ax = axes[1, 0]
    if diffusion:
        cats = ["Score top-1", "Trayec. approach (m)", "Latencia samp. (ms × 0.01)"]
        ycbv_agg = diffusion["datasets"]["ycbv"]["aggregate"]
        tless_agg = diffusion["datasets"]["tless"]["aggregate"]
        ycbv_vals = [
            ycbv_agg["best_grasp_score"]["median"],
            ycbv_agg["approach_trajectory_length_m"]["median"],
            ycbv_agg["sampling_ms"]["median"] / 100,
        ]
        tless_vals = [
            tless_agg["best_grasp_score"]["median"],
            tless_agg["approach_trajectory_length_m"]["median"],
            tless_agg["sampling_ms"]["median"] / 100,
        ]
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, ycbv_vals, w, label="YCB-V (n=30)", color=AZUL)
        ax.bar(x + w / 2, tless_vals, w, label="T-LESS (n=30)", color=NARANJA)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=10, fontsize=9)
        ax.set_title("Panel C — Diffusion + Grasp Sampler con poses reales FP")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "diffusion_real_poses no disponible",
                ha="center", va="center", transform=ax.transAxes)

    # Panel 4: CoppeliaSim infrastructure
    ax = axes[1, 1]
    if coppelia:
        labels = ["Conexión\nZMQ (ms)", "Latencia/step\n(ms)", "Sim time avanz.\n100 pasos (s ×10)"]
        vals = [
            coppelia.get("connect_ms", 0),
            coppelia.get("stepping", {}).get("step_ms_mean", 0),
            coppelia.get("stepping", {}).get("sim_time_advanced_s", 0) * 10,  # x10 para escala
        ]
        bars = ax.bar(labels, vals, color=[VERDE, PURPURA, GRIS])
        ax.set_title("Panel D — CoppeliaSim ZMQ Remote API (smoke test)")
        ax.set_ylabel("Magnitud")
        ax.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}", ha="center", fontsize=9)
        # Subtítulo con la versión del servidor
        v = coppelia.get("server_version", 0)
        ax.text(0.02, 0.95,
                f"Server v{v} — escena: {coppelia.get('scene_loaded', 'n/a')}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7))
    else:
        ax.text(0.5, 0.5, "coppelia_smoke no disponible",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(
        "TFM — Pipeline percepción → planificación → simulación "
        "(evidencia experimental real, run 2026-04-27)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / "fig_6_pipeline_dashboard.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Tabla maestra (Markdown) ─────────────────────────────────────────


def make_master_table(fp_real: dict, diffusion: dict, coppelia: dict) -> Path:
    lines: list[str] = []
    lines.append("# Tabla maestra del cap. 6 — TFM\n")
    lines.append(f"_Generada {datetime.now().strftime('%Y-%m-%d %H:%M')} por "
                 "`experiments/run_chapter6_consolidation.py`_\n")

    lines.append("\n## 1. FoundationPose — métricas reales (subset BOP-19)\n")
    lines.append("| Métrica | YCB-V | T-LESS |")
    lines.append("|---------|-------|--------|")

    def fmt(v: float, unit: str = "") -> str:
        return f"{v:.3f}{unit}" if isinstance(v, (int, float)) else "—"

    rows = [
        ("Objetos evaluados", "n_evaluated", "{:.0f}"),
        ("ADD mediana (mm)", "add_median_mm", "{:.2f}"),
        ("ADD-S mediana (mm)", "adds_median_mm", "{:.2f}"),
        ("AUC ADD", "auc_add", "{:.3f}"),
        ("AUC ADD-S", "auc_adds", "{:.3f}"),
        ("Recall@10mm ADD-S", "recall_adds_10mm", "{:.1%}"),
    ]
    for label, key, fmt_str in rows:
        ycbv_v = fp_real.get("ycbv", {}).get(key, "—")
        tless_v = fp_real.get("tless", {}).get(key, "—")
        ycbv_s = fmt_str.format(ycbv_v) if isinstance(ycbv_v, (int, float)) else "—"
        tless_s = fmt_str.format(tless_v) if isinstance(tless_v, (int, float)) else "—"
        lines.append(f"| {label} | {ycbv_s} | {tless_s} |")
    lines.append(f"\n_Fuente: `{fp_real.get('_meta', {}).get('source', 'n/a')}`_\n")

    lines.append("\n## 2. Comparación con baselines oficiales (Mean AR)\n")
    lines.append("| Dataset | FP propio (ADD med) | FP paper (Mean AR) | GDR-Net++ BOP22 (Mean AR) |")
    lines.append("|---------|---------------------|--------------------|---------------------------|")
    for ds in ("ycbv", "tless"):
        own = fp_real.get(ds, {}).get("add_median_mm", "—")
        own_s = f"{own:.2f} mm" if isinstance(own, (int, float)) else "—"
        lines.append(f"| {ds.upper()} | {own_s} | {FP_PAPER[ds]['Mean_AR']:.3f} | "
                     f"{GDR_BOP2022[ds]['Mean_AR']:.3f} |")
    lines.append("\n_Nota: ADD/ADD-S y Mean AR son métricas distintas; "
                 "comparación cualitativa documentada en cap. 6._\n")

    if diffusion:
        lines.append("\n## 3. Diffusion + Grasp Sampler con poses reales\n")
        lines.append("| Métrica | YCB-V | T-LESS |")
        lines.append("|---------|-------|--------|")
        ds_pairs = [
            ("Poses con éxito", lambda a: f"{a['n_ok']}/{a['n_total']}"),
            ("Score top-1 (mediana)", lambda a: f"{a['best_grasp_score']['median']:.3f}"),
            ("Approach length (mediana)", lambda a: f"{a['approach_trajectory_length_m']['median']*100:.1f} cm"),
            ("DDPM-style length (mediana)", lambda a: f"{a['diffusion_trajectory_length_m']['median']*100:.1f} cm"),
            ("Sampler latency p95", lambda a: f"{a['sampling_ms']['p95']:.1f} ms"),
            ("Pinza open-at-start / closed-at-end",
             lambda a: f"{a['gripper_phase_consistency']['open_at_start_pct']:.0f}% / "
                       f"{a['gripper_phase_consistency']['closed_at_end_pct']:.0f}%"),
        ]
        for label, fn in ds_pairs:
            ycbv_a = diffusion["datasets"]["ycbv"]["aggregate"]
            tless_a = diffusion["datasets"]["tless"]["aggregate"]
            lines.append(f"| {label} | {fn(ycbv_a)} | {fn(tless_a)} |")
        lines.append(f"\n_Fuente: `experiments/results/diffusion_real_poses/trajectories_summary.json`_\n")

    if coppelia:
        lines.append("\n## 4. Infraestructura CoppeliaSim (smoke test)\n")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Conexión ZMQ | {coppelia.get('connect_ms', '—')} ms |")
        lines.append(f"| Servidor sim | v{coppelia.get('server_version', '—')} |")
        lines.append(f"| Escena cargada | `{coppelia.get('scene_loaded', '—')}` |")
        step = coppelia.get("stepping", {})
        lines.append(f"| Latencia por step | {step.get('step_ms_mean', '—')} ms (mean) |")
        lines.append(f"| Sim time avanzado (100 pasos) | {step.get('sim_time_advanced_s', '—')} s |")
        vs = coppelia.get("vision_sensor", {})
        lines.append(f"| Render del sensor | {vs.get('resolution', ['—', '—'])[0]}×"
                     f"{vs.get('resolution', ['—', '—'])[1]}, "
                     f"intensidad media {vs.get('image_mean_intensity', '—'):.1f} |")
        lines.append("\n_Fuente: `experiments/results/coppelia_smoke/smoke_test_result.json`_\n")

    out = FIG_DIR / "cap6_master_table.md"
    out.write_text("\n".join(lines) + "\n")
    return out


# ── JSON consolidado (formato compatible con notebook 03) ────────────


def make_consolidated_json(fp_real: dict, diffusion: dict, coppelia: dict) -> Path:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "FoundationPose vs GDR-Net++ comparison + Diffusion planning + "
            "CoppeliaSim infrastructure for TFM Chapter 6"
        ),
        "foundationpose_own": {
            "ycbv": fp_real.get("ycbv", {}),
            "tless": fp_real.get("tless", {}),
            "_meta": fp_real.get("_meta", {}),
        },
        "foundationpose_paper": FP_PAPER,
        "gdrnet_leaderboard": GDR_BOP2022,
        "diffusion_planning": {
            "ycbv_aggregate": diffusion.get("datasets", {}).get("ycbv", {}).get("aggregate", {}),
            "tless_aggregate": diffusion.get("datasets", {}).get("tless", {}).get("aggregate", {}),
            "meta": diffusion.get("meta", {}),
        } if diffusion else None,
        "coppelia_smoke": coppelia or None,
    }
    out = RESULTS_DIR / "comparison_fp_vs_gdrnet.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def main() -> int:
    print("[INFO] consolidando evidencia experimental cap. 6 ...")

    fp_real = _load_fp_real()
    diffusion = _load_diffusion()
    coppelia = _load_coppelia()

    print(f"  FP real:       ycbv={'add_median_mm' in fp_real.get('ycbv', {})}, "
          f"tless={'add_median_mm' in fp_real.get('tless', {})}")
    print(f"  Diffusion:     {'OK' if diffusion else 'NO'}")
    print(f"  CoppeliaSim:   {'OK' if coppelia else 'NO'}")

    fig = make_dashboard(fp_real, diffusion, coppelia)
    print(f"[OK] {fig.relative_to(REPO_ROOT)}")

    table = make_master_table(fp_real, diffusion, coppelia)
    print(f"[OK] {table.relative_to(REPO_ROOT)}")

    js = make_consolidated_json(fp_real, diffusion, coppelia)
    print(f"[OK] {js.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
