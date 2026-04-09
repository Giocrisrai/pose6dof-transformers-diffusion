"""
Generate all figures and tables for TFM Chapter 6: Experiments and Results.

Consolidates outputs from all experiments into publication-ready figures.

Usage:
    cd repo_tfm
    python experiments/generate_chapter6_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("experiments/results")
OUTPUT_DIR = Path("experiments/results/chapter6_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UNIR brand colors
AZUL = "#0098CD"
AZUL_OSCURO = "#006C8F"
NEGRO = "#333333"
GRIS = "#858591"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig_noise_sensitivity():
    """Figure 6.1: Pose error vs noise level (from integration test)."""
    data = load_json(RESULTS_DIR / "integration_test" / "integration_results.json" if
                     (RESULTS_DIR / "integration_test").exists() else
                     Path("experiments/integration_test/integration_results.json"))

    noise_data = data.get("noise_analysis", {})
    if not noise_data:
        print("  [SKIP] No noise analysis data found")
        return

    noise_keys = sorted(noise_data.keys(), key=float)
    noise_levels = [float(k) for k in noise_keys]
    add_vals = [noise_data[k]["ADD"] for k in noise_keys]
    adds_vals = [noise_data[k]["ADD-S"] for k in noise_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, add_vals, "o-", color=AZUL, linewidth=2.5,
            markersize=8, label="ADD")
    ax.plot(noise_levels, adds_vals, "s-", color=AZUL_OSCURO, linewidth=2.5,
            markersize=8, label="ADD-S")

    ax.set_xlabel("Nivel de ruido (mm)", fontsize=12)
    ax.set_ylabel("Error de distancia (mm)", fontsize=12)
    ax.set_title("Fig. 6.1 — Sensibilidad al ruido en la estimación de pose", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "fig_6_1_noise_sensitivity.png"), dpi=200)
    plt.close(fig)
    print("  ✓ Fig 6.1: Noise sensitivity")


def fig_rotation_ablation():
    """Figures 6.2-6.4: Rotation representation ablation."""
    exp3_dir = RESULTS_DIR / "exp3_rotation_ablation"

    if not exp3_dir.exists():
        print("  [SKIP] Exp 3 results not found")
        return

    data = load_json(exp3_dir / "exp3_results.json")

    # Fig 6.2: Roundtrip precision table (as bar chart)
    roundtrip = data.get("roundtrip", {})
    if roundtrip:
        names = list(roundtrip.keys())
        means = [roundtrip[n]["mean"] for n in names]
        labels = ["Quaternión", "6D Continuo", "Axis-Angle", "Euler ZYX"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, means, color=[AZUL, AZUL_OSCURO, NEGRO, GRIS], alpha=0.8)
        ax.set_ylabel("Error de reconstrucción (Frobenius)", fontsize=12)
        ax.set_title("Fig. 6.2 — Precisión roundtrip por representación", fontsize=13)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                   f"{val:.1e}", ha="center", fontsize=9)

        plt.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "fig_6_2_roundtrip_precision.png"), dpi=200)
        plt.close(fig)
        print("  ✓ Fig 6.2: Roundtrip precision")

    # Copy interpolation and gimbal lock figures
    for src_name, dst_name, fig_num in [
        ("interpolation_smoothness.png", "fig_6_3_interpolation.png", "6.3"),
        ("gradient_stability.png", "fig_6_4_gradient_stability.png", "6.4"),
        ("gimbal_lock.png", "fig_6_5_gimbal_lock.png", "6.5"),
    ]:
        src = exp3_dir / src_name
        if src.exists():
            import shutil
            shutil.copy(str(src), str(OUTPUT_DIR / dst_name))
            print(f"  ✓ Fig {fig_num}: {src_name}")


def fig_grasp_comparison():
    """Figure 6.6: Grasp planning comparison."""
    exp4_dir = RESULTS_DIR / "exp4_grasp_comparison"
    if not exp4_dir.exists():
        print("  [SKIP] Exp 4 results not found")
        return

    # Copy the main comparison figure
    src = exp4_dir / "grasp_comparison.png"
    if src.exists():
        import shutil
        shutil.copy(str(src), str(OUTPUT_DIR / "fig_6_6_grasp_comparison.png"))
        print("  ✓ Fig 6.6: Grasp comparison")

    src2 = exp4_dir / "score_distribution.png"
    if src2.exists():
        import shutil
        shutil.copy(str(src2), str(OUTPUT_DIR / "fig_6_7_score_distribution.png"))
        print("  ✓ Fig 6.7: Score distribution")


def generate_results_table():
    """Generate LaTeX/Markdown table summarizing all experiments."""
    print("\n[Summary Table]")

    lines = []
    lines.append("| Experimento | Métrica | Resultado |")
    lines.append("|------------|---------|-----------|")

    # Integration test
    try:
        int_data = load_json(Path("experiments/integration_test/integration_results.json"))
        ms = int_data.get("multi_scene", {})
        lines.append(f"| Pipeline E2E (T-LESS) | ADD Recall@10mm | {ms.get('add_recall_at_10mm', 0)*100:.1f}% |")
        lines.append(f"| Pipeline E2E (T-LESS) | ADD-S Recall@10mm | {ms.get('adds_recall_at_10mm', 0)*100:.1f}% |")
        lines.append(f"| Pipeline E2E (T-LESS) | ADD AUC@50mm | {ms.get('add_auc_50mm', 0):.4f} |")
    except:
        pass

    # Exp 3
    try:
        exp3 = load_json(RESULTS_DIR / "exp3_rotation_ablation" / "exp3_results.json")
        interp = exp3.get("interpolation", {})
        lines.append(f"| Interpolación SLERP | Desviación geodésica | {interp.get('SLERP_quaternion', 0):.4f}° |")
        lines.append(f"| Interpolación 6D | Desviación geodésica | {interp.get('linear_6D', 0):.4f}° |")
        grad = exp3.get("gradient", {})
        if "6D_continuous" in grad:
            lines.append(f"| Gradiente 6D | Norma media | {grad['6D_continuous']['mean']:.4f} |")
            lines.append(f"| Gradiente Quat | Norma media | {grad['quaternion']['mean']:.4f} |")
    except:
        pass

    # Exp 4
    try:
        exp4 = load_json(RESULTS_DIR / "exp4_grasp_comparison" / "exp4_results.json")
        for method, key in [("Top-Down", "heuristic_topdown"),
                            ("Multi-Strategy", "heuristic_multi"),
                            ("Diffusion Policy", "diffusion_policy")]:
            r = exp4[key]
            lines.append(f"| Grasp {method} | Éxito | {r['success_rate']:.1f}% |")
            lines.append(f"| Grasp {method} | Tiempo | {r['avg_time_ms']:.1f}ms |")
    except:
        pass

    table = "\n".join(lines)
    print(table)

    with open(OUTPUT_DIR / "results_table.md", "w") as f:
        f.write("# Tabla de Resultados — Capítulo 6\n\n")
        f.write(table)
        f.write("\n")

    print(f"\n  Saved: {OUTPUT_DIR}/results_table.md")


def main():
    print("=" * 60)
    print("  Generating Chapter 6 Figures and Tables")
    print("=" * 60)

    fig_noise_sensitivity()
    fig_rotation_ablation()
    fig_grasp_comparison()
    generate_results_table()

    # List all generated files
    print(f"\n{'='*60}")
    print("  Generated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
