#!/usr/bin/env python3
"""Genera figuras 'hero' profesionales para usar en defensa, UIs y TFM.

Outputs:
    docs/figures_hero/01_pipeline_architecture.png    — diagrama arquitectonico
    docs/figures_hero/02_exploraciones_dashboard.png  — dashboard resumen
    docs/figures_hero/03_roadmap_timeline.png         — timeline post-TFM
    docs/figures_hero/04_modelos_comparativa.png      — comparativa modelos
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.plot_style import COLORS, apply_style

OUT = REPO / "docs/figures_hero"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FIGURA 1 — Diagrama de arquitectura del pipeline
# ============================================================================
def fig01_pipeline_architecture():
    apply_style("presentation")
    # Lienzo mas grande para que quepa todo con margen
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11)
    ax.axis("off")

    fig.suptitle(
        "Pipeline de Bin Picking 6-DoF — Arquitectura completa",
        fontsize=24,
        fontweight="bold",
        y=0.97,
        color=COLORS["ink"],
    )
    ax.text(
        11,
        10.0,
        "Camara RGB-D  →  Pose 6-DoF  →  Trayectoria multimodal  →  Control PBVS  →  Brazo robotico",
        ha="center",
        fontsize=15,
        style="italic",
        color=COLORS["muted"],
    )

    # Stages mas anchas (text-friendly): w=3.0 vs 2.4
    pipeline_y = 5.5
    stages = [
        (2.0, pipeline_y, 2.8, 2.4, "Camara\nRGB-D", COLORS["muted"], "Input"),
        (
            6.0,
            pipeline_y,
            3.2,
            2.4,
            "FoundationPose\n+ FreeZeV2",
            COLORS["primary"],
            "Percepcion 6-DoF",
        ),
        (10.5, pipeline_y, 3.2, 2.4, "Diffusion\nPolicy", COLORS["cat_2"], "Planificacion"),
        (14.5, pipeline_y, 2.8, 2.4, "PBVS\nSE(3)", COLORS["accent"], "Control fino"),
        (18.5, pipeline_y, 2.8, 2.4, "Brazo\nrobotico", COLORS["ink"], "Output"),
    ]
    for cx, cy, w, h, label, color, sublabel in stages:
        # Sombra
        shadow = FancyBboxPatch(
            (cx - w / 2 + 0.08, cy - h / 2 - 0.08),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="#00000015",
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(shadow)
        # Caja principal
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            cx,
            cy + 0.15,
            label,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white",
            zorder=3,
            linespacing=1.3,
        )
        # Etiqueta inferior (mas espacio)
        ax.text(
            cx,
            cy - h / 2 - 0.5,
            sublabel,
            ha="center",
            fontsize=12,
            color=COLORS["muted"],
            style="italic",
        )

    # Flechas entre etapas (mas largas porque hay mas espacio horizontal)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2] / 2
        x2 = stages[i + 1][0] - stages[i + 1][2] / 2
        ax.annotate(
            "",
            xy=(x2 - 0.05, pipeline_y),
            xytext=(x1 + 0.05, pipeline_y),
            arrowprops=dict(arrowstyle="->", lw=3, color=COLORS["ink_soft"], shrinkA=0, shrinkB=0),
        )

    # Outputs intermedios (debajo del pipeline, con mas espacio)
    outputs_y = 3.0
    outputs = [
        (6.0, "RGB + Depth\n640 x 480 px"),
        (10.5, "R en SO(3)\nt en mm"),
        (14.5, "Trayectoria\n16 pasos x 7"),
        (18.5, "Comando\nJoint pos"),
    ]
    for cx, label in outputs:
        ax.text(
            cx,
            outputs_y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["ink_soft"],
            linespacing=1.5,
            family="DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=COLORS["surface"],
                edgecolor=COLORS["border"],
                linewidth=1.5,
            ),
        )

    # Capa VLA-lite añadida (exploraciones)
    vla_box = FancyBboxPatch(
        (6.0, 1.0),
        9.0,
        1.1,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=COLORS["primary_light"],
        edgecolor=COLORS["primary"],
        linewidth=2.5,
        zorder=1,
    )
    ax.add_patch(vla_box)
    ax.text(
        10.5,
        1.55,
        "VLA-lite (exp 4-13)   'pick the leftmost red sphere'",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["primary_dark"],
    )

    # Metricas globales — arriba izquierda (mas ancho, mas legible)
    metrics_box = FancyBboxPatch(
        (0.3, 8.4),
        5.5,
        1.3,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=COLORS["success_light"],
        edgecolor=COLORS["success"],
        linewidth=2.5,
        zorder=1,
    )
    ax.add_patch(metrics_box)
    ax.text(
        3.05,
        9.40,
        "RESULTADOS VALIDADOS",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["success_dark"],
    )
    ax.text(
        3.05,
        9.00,
        "AUC ADD-S 0.91 YCB-V  /  0.96 T-LESS",
        ha="center",
        fontsize=12,
        color=COLORS["ink"],
    )
    ax.text(
        3.05,
        8.65,
        "Cycle p95 < 7 s   ·   255 tests passing",
        ha="center",
        fontsize=11,
        color=COLORS["ink_soft"],
    )

    # Hardware — arriba derecha
    hw_box = FancyBboxPatch(
        (16.2, 8.4),
        5.5,
        1.3,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=COLORS["accent_light"],
        edgecolor=COLORS["accent"],
        linewidth=2.5,
        zorder=1,
    )
    ax.add_patch(hw_box)
    ax.text(
        18.95,
        9.40,
        "HARDWARE",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent_dark"],
    )
    ax.text(18.95, 9.00, "Apple M1 Pro + MPS", ha="center", fontsize=12, color=COLORS["ink"])
    ax.text(
        18.95,
        8.65,
        "~2 000 USD  vs  150 000 USD industrial",
        ha="center",
        fontsize=11,
        color=COLORS["ink_soft"],
    )

    fig.savefig(
        OUT / "01_pipeline_architecture.png",
        dpi=140,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.4,
    )
    plt.close()
    print(f"  ✓ {OUT / '01_pipeline_architecture.png'}")


# ============================================================================
# FIGURA 2 — Dashboard resumen de las 13 exploraciones
# ============================================================================
def fig02_exploraciones_dashboard():
    apply_style("presentation")

    # Datos
    exploraciones = [
        ("1. Bootstrap-CI PyPI", "97 %", "cov"),
        ("2. Distillation 1-NFE", "x517", "speedup"),
        ("3. Pipeline open-license", "-3 pp", "AUC FreeZeV2"),
        ("4. VLA-lite color", "98.6 %", "selection"),
        ("5. Robustez linguistica", "100 %", "6 familias"),
        ("6. VLA color+forma", "99.9 %", "global"),
        ("7. Simulaciones 3D", "12/12", "escenas"),
        ("8. Multi-objeto N=2..5", "100 %", "todos N"),
        ("9. Atributo tamaño", "99.9 %", "8 templates"),
        ("10. Secuenciales multi-step", "8/8", "20 pasos"),
        ("11. CLIP-image visual", "100 %", "sin atributos"),
        ("12. Robustez DR", "12/12", "≥ 75 %"),
        ("13. Razonamiento espacial", "98.4 %", "13 templates"),
    ]

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Titulo
    fig.suptitle(
        "13 Exploraciones post-TFM — Resumen ejecutivo",
        fontsize=26,
        fontweight="bold",
        y=0.97,
        color=COLORS["ink"],
    )
    ax.text(
        10,
        14.5,
        "Todas mergeadas a main · 255 tests passing · paquete PyPI publicado oficialmente",
        ha="center",
        fontsize=15,
        style="italic",
        color=COLORS["muted"],
    )

    # Grid 4 columnas × 4 filas (cuarta fila solo 1 card — centrada)
    cols, rows = 4, 4
    box_w, box_h = 4.4, 2.4
    margin_x, margin_y = 0.45, 0.4
    start_x, start_y = 0.7, 13.0

    for i, (title, metric, sub) in enumerate(exploraciones):
        col = i % cols
        row = i // cols
        # Centrar la ultima fila si tiene menos de cols cards
        if row == rows - 1:
            cards_last_row = len(exploraciones) - row * cols
            row_x_offset = (cols - cards_last_row) * (box_w + margin_x) / 2
        else:
            row_x_offset = 0
        x = start_x + col * (box_w + margin_x) + row_x_offset
        y = start_y - row * (box_h + margin_y)
        # Color por categoria
        if i < 3:
            color = COLORS["primary"]
            dark = COLORS["primary_dark"]
        elif i < 8:
            color = COLORS["cat_2"]
            dark = "#1F5942"
        elif i < 11:
            color = COLORS["accent"]
            dark = COLORS["accent_dark"]
        else:
            color = COLORS["cat_4"]
            dark = "#6B21A8"

        # Sombra
        shadow = FancyBboxPatch(
            (x + 0.06, y - box_h - 0.06),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="#00000015",
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(shadow)
        # Card principal
        card = FancyBboxPatch(
            (x, y - box_h),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(card)
        # Banda superior coloreada (mas alta para que el texto respire)
        band_h = 0.55
        band = patches.Rectangle(
            (x + 0.08, y - band_h - 0.05),
            box_w - 0.16,
            band_h,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(band)
        ax.text(
            x + box_w / 2,
            y - band_h / 2 - 0.05,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        # Metrica grande (centrada en el resto de la card)
        ax.text(
            x + box_w / 2,
            y - band_h - (box_h - band_h) / 2 + 0.15,
            metric,
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=dark,
            zorder=4,
        )
        # Subtitulo (parte baja del card)
        ax.text(
            x + box_w / 2,
            y - box_h + 0.30,
            sub,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["muted"],
            style="italic",
            zorder=4,
        )

    # Footer resumen
    footer_box = FancyBboxPatch(
        (0.5, 0.4),
        19.0,
        1.5,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=COLORS["surface"],
        edgecolor=COLORS["border"],
        linewidth=1.5,
    )
    ax.add_patch(footer_box)
    ax.text(
        10,
        1.4,
        "Total acumulado: 13/13 exitos · 255 tests · 10 modelos Diffusion · 30+ renders 3D",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        10,
        0.75,
        "pip install bop-bootstrap-ci  ·  docs/exploraciones/ (13 cierres)  ·  listo para defensa",
        ha="center",
        fontsize=13,
        color=COLORS["muted"],
    )

    fig.savefig(
        OUT / "02_exploraciones_dashboard.png",
        dpi=140,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.4,
    )
    plt.close()
    print(f"  ✓ {OUT / '02_exploraciones_dashboard.png'}")


# ============================================================================
# FIGURA 3 — Timeline post-TFM (etapas hasta producto comercial)
# ============================================================================
def fig03_roadmap_timeline():
    apply_style("presentation")

    # Texto mas corto y compacto para que quepa
    stages = [
        ("Hoy", "TFM + 13 exploraciones", "Cerrado en simulacion", COLORS["success"]),
        ("3-6 m", "Etapa 1\nRobot fisico", "~10-20 k EUR", COLORS["primary"]),
        ("+2-3 m", "Etapa 2\nSim-to-real", "~2 k EUR (cloud)", COLORS["cat_2"]),
        ("+2-4 m", "Etapa 3\nDemos humanas", "~0 EUR (trabajo)", COLORS["cat_7"]),
        ("+3 m", "Etapa 4\nMVP producto", "~50 k EUR", COLORS["accent"]),
        ("+6 m", "Etapa 5\nPlanta piloto", "~50 k EUR", COLORS["warning"]),
        ("+6 m", "Etapa 6\nCertif. CE / ISO", "~30 k EUR", COLORS["cat_6"]),
        ("~2 anos", "Etapa 7\nSpin-off SaaS", "Total ~200 k EUR", COLORS["cat_4"]),
    ]

    # Layout vertical mas amplio para evitar solapamientos
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis("off")

    fig.suptitle(
        "Roadmap post-TFM — De simulacion a producto comercial",
        fontsize=24,
        fontweight="bold",
        y=0.97,
        color=COLORS["ink"],
    )
    ax.text(
        11,
        10.7,
        "Etapa 0 cerrada en simulacion. Etapas 1-7 requieren hardware, financiacion y certificacion.",
        ha="center",
        fontsize=14,
        style="italic",
        color=COLORS["muted"],
    )

    # Linea horizontal del timeline (mas baja para dejar espacio)
    line_y = 6.0
    ax.plot([1.5, 20.5], [line_y, line_y], color=COLORS["border"], linewidth=4, zorder=1)

    n = len(stages)
    x_positions = np.linspace(2.0, 20.0, n)
    card_w = 2.4  # mas ancho para que quepa el texto
    card_h = 2.3  # mas alto

    for i, ((time, title, cost, color), x) in enumerate(zip(stages, x_positions)):
        # Alternar arriba/abajo con MAS separacion vertical
        y_box = 8.0 if i % 2 == 0 else 4.0
        # Circulo en linea
        circle = patches.Circle(
            (x, line_y), 0.32, facecolor=color, edgecolor="white", linewidth=3, zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            x,
            line_y,
            str(i),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        # Linea de conexion al card
        if i % 2 == 0:
            ax.plot(
                [x, x],
                [line_y + 0.34, y_box - card_h / 2 - 0.05],
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.6,
            )
        else:
            ax.plot(
                [x, x],
                [line_y - 0.34, y_box + card_h / 2 + 0.05],
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.6,
            )
        # Card
        card = FancyBboxPatch(
            (x - card_w / 2, y_box - card_h / 2),
            card_w,
            card_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(card)
        # Tiempo (banda superior del card)
        ax.text(
            x,
            y_box + card_h / 2 - 0.35,
            time,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color,
        )
        # Titulo
        ax.text(
            x,
            y_box - 0.05,
            title,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["ink"],
            linespacing=1.3,
        )
        # Coste (banda inferior, con margen)
        ax.text(
            x,
            y_box - card_h / 2 + 0.32,
            cost,
            ha="center",
            va="center",
            fontsize=11,
            color=COLORS["muted"],
            style="italic",
        )

    # Flecha final mas pequeña, sin solapar
    ax.annotate(
        "",
        xy=(21.4, line_y),
        xytext=(20.5, line_y),
        arrowprops=dict(arrowstyle="->", lw=3, color=COLORS["ink_soft"]),
    )
    ax.text(
        21.6,
        line_y,
        "Producto\ncomercial",
        fontsize=12,
        color=COLORS["ink_soft"],
        fontweight="bold",
        va="center",
        ha="left",
    )

    # Leyenda inferior — texto mas legible
    legend_box = FancyBboxPatch(
        (0.5, 0.6),
        21.0,
        1.5,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=COLORS["surface"],
        edgecolor=COLORS["border"],
        linewidth=1.5,
    )
    ax.add_patch(legend_box)
    ax.text(
        11,
        1.5,
        "Detalle completo: docs/ROADMAP_POSTTFM.md",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        11,
        0.95,
        "Inversion inicial muy baja: ~10-20 k EUR para validar el sistema en robot fisico",
        ha="center",
        fontsize=12,
        color=COLORS["ink_soft"],
    )

    fig.savefig(
        OUT / "03_roadmap_timeline.png",
        dpi=140,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.4,
    )
    plt.close()
    print(f"  ✓ {OUT / '03_roadmap_timeline.png'}")


# ============================================================================
# FIGURA 4 — Comparativa de los 10 modelos Diffusion
# ============================================================================
def fig04_modelos_comparativa():
    apply_style("presentation")

    models = [
        ("Original", 0.020, "30 ep, 2K trajs", COLORS["muted_soft"]),
        ("Extended", 0.013, "50 ep, 5K trajs", COLORS["cat_5"]),
        ("Ultra", 0.0022, "100 ep, 10K trajs", COLORS["primary"]),
        ("Ultra-fast", 0.0124, "1 NFE distilled", COLORS["accent"]),
        ("CLIP color", None, "98.6 % VLA-lite", COLORS["cat_2"]),
        ("CLIP shapes", None, "99.9 % color+forma", COLORS["cat_7"]),
        ("CLIP multi", None, "100 % N=2..5", COLORS["cat_4"]),
        ("CLIP size", None, "99.9 % size attr", COLORS["cat_3"]),
        ("CLIP image", None, "100 % visual ground", COLORS["cat_6"]),
        ("CLIP spatial", None, "98.4 % spatial reas", COLORS["cat_8"]),
    ]

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # Panel izquierdo: MSE comparativa (4 estandar)
    ax1 = fig.add_subplot(gs[0])
    std_models = [(m[0], m[1], m[2], m[3]) for m in models if m[1] is not None]
    names = [m[0] for m in std_models]
    mses = [m[1] for m in std_models]
    colors = [m[3] for m in std_models]
    descs = [m[2] for m in std_models]

    bars = ax1.barh(names, mses, color=colors, edgecolor=COLORS["ink"], linewidth=1.5, height=0.7)
    for bar, mse, desc in zip(bars, mses, descs):
        ax1.text(
            mse + 0.0008,
            bar.get_y() + bar.get_height() / 2,
            f"  MSE={mse:.4f}\n  {desc}",
            va="center",
            fontsize=12,
            color=COLORS["ink_soft"],
        )
    ax1.set_xlabel("MSE de validacion (lower = mejor)", fontsize=14)
    ax1.set_title(
        "Modelos Diffusion estandar — Re-entrenamiento progresivo",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlim(0, max(mses) * 1.6)
    ax1.invert_yaxis()
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel derecho: capacidades de los 6 VLA-lite
    ax2 = fig.add_subplot(gs[1])
    vla_models = [(m[0], m[2], m[3]) for m in models if m[1] is None]
    names2 = [m[0] for m in vla_models]
    # Extraer accuracy del desc
    accs = []
    for m in vla_models:
        try:
            txt = m[1].split()[0]
            v = float(txt.replace("%", ""))
            accs.append(v if v <= 100 else 100)
        except (ValueError, IndexError):
            accs.append(100.0)
    colors2 = [m[2] for m in vla_models]
    descs2 = [m[1] for m in vla_models]

    bars2 = ax2.barh(
        names2, accs, color=colors2, edgecolor=COLORS["ink"], linewidth=1.5, height=0.7
    )
    for bar, a, desc in zip(bars2, accs, descs2):
        ax2.text(
            101,
            bar.get_y() + bar.get_height() / 2,
            f"  {desc}",
            va="center",
            fontsize=12,
            color=COLORS["ink_soft"],
        )
        ax2.text(
            a / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{a:.1f}%",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
    ax2.set_xlabel("Selection accuracy (%)", fontsize=14)
    ax2.set_title(
        "VLA-lite — 6 modelos con CLIP text/image", fontsize=17, fontweight="bold", pad=15
    )
    ax2.set_xlim(0, 160)
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Comparativa de los 10 modelos Diffusion entrenados",
        fontsize=22,
        fontweight="bold",
        y=0.985,
        color=COLORS["ink"],
    )

    fig.savefig(
        OUT / "04_modelos_comparativa.png",
        dpi=140,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.4,
    )
    plt.close()
    print(f"  ✓ {OUT / '04_modelos_comparativa.png'}")


def main():
    print("Generando 4 figuras hero profesionales...")
    fig01_pipeline_architecture()
    fig02_exploraciones_dashboard()
    fig03_roadmap_timeline()
    fig04_modelos_comparativa()
    print(f"\n[OK] Todas guardadas en {OUT}")


if __name__ == "__main__":
    main()
