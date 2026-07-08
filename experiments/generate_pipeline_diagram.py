#!/usr/bin/env python3
"""Genera el diagrama de arquitectura del pipeline TFM con Graphviz.

Reemplaza la version matplotlib previa (textos cortados, etiquetas
superpuestas en las flechas) por una version con layout automatico,
flechas tipadas y tipografia legible al proyectar.

Bloques: Imagen RGB-D + CAD -> FoundationPose (Transformer)
       -> SE(3) / SO(3) interfaz matematica
       -> Diffusion Policy (UNet1D + DDIM)
       -> Trayectoria 16 pasos
       -> CoppeliaSim + Ragnar (validacion E2E)

Salida: experiments/results/pipeline_e2e/fig_pipeline_arquitectura.png
"""
from __future__ import annotations

import os
from pathlib import Path

DOT_BIN = "/opt/homebrew/bin"
if DOT_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{DOT_BIN}:{os.environ.get('PATH', '')}"

import graphviz

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "experiments/results/pipeline_e2e"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = OUT_DIR / "fig_pipeline_arquitectura"  # graphviz anade .png

# Paleta consistente con scripts/make_diagrams.py y src/utils/plot_style.py
C = {
    "input":      "#1E4E8C",   # azul profundo para entrada
    "perception": "#0070A8",   # azul percepcion
    "math":       "#7E22CE",   # purpura interfaz matematica
    "planning":   "#0F766E",   # teal planificacion
    "execution":  "#A33D17",   # naranja ejecucion
    "output":     "#A16207",   # ambar validacion
    "ink":        "#0F172A",
    "ink_soft":   "#1E293B",
    "border":     "#CBD5E1",
    "surface":    "#F8FAFC",
}


def html_node(label: str, subtitle: str, ts: int = 16, ss: int = 12,
              color: str = "white") -> str:
    return (
        f"<<table border='0' cellpadding='6'>"
        f"<tr><td><font face='Helvetica-Bold' point-size='{ts}' color='{color}'>"
        f"<b>{label}</b></font></td></tr>"
        f"<tr><td><font face='Helvetica' point-size='{ss}' color='{color}'>"
        f"{subtitle}</font></td></tr>"
        f"</table>>"
    )


def build():
    g = graphviz.Digraph("pipeline_tfm", format="png")
    g.attr(rankdir="LR", bgcolor="white", pad="0.7",
           nodesep="0.7", ranksep="1.1", splines="spline", dpi="160")
    g.attr("node", shape="box", style="rounded,filled",
           fontname="Helvetica-Bold", fontsize="14",
           margin="0.28,0.20", penwidth="2.6")
    g.attr("edge", fontname="Helvetica-Bold", fontsize="12",
           color=C["ink"], fontcolor=C["ink"], penwidth="2.4", arrowsize="1.15")

    g.attr(label=(
        "<<font face='Helvetica-Bold' point-size='22' color='#0F172A'>"
        "<b>Pipeline TFM: FoundationPose + Diffusion Policy para Bin Picking 6-DoF</b>"
        "</font><br/>"
        "<font face='Helvetica-Oblique' point-size='13' color='#475569'>"
        "Marco matematico unificado SE(3) / SO(3) + SDEs</font>>"
    ), labelloc="t")

    # --- Entradas (cluster izquierdo) ---
    with g.subgraph(name="cluster_in") as c:
        c.attr(label="ENTRADAS", style="rounded,dashed",
               color=C["ink_soft"], fontcolor=C["ink_soft"],
               fontname="Helvetica-Bold", fontsize="13", labelloc="t")
        c.node("rgbd", label=html_node("Imagen RGB-D + CAD", "640 × 480 · .ply mesh"),
               fillcolor=C["input"], color=C["input"])
        c.node("bop", label=html_node("Datasets BOP", "T-LESS · YCB-V · subset BOP-19"),
               fillcolor=C["input"], color=C["input"])

    # --- Percepcion ---
    g.node("fp", label=html_node(
        "FoundationPose",
        "Wen et al. CVPR 2024<br/>Transformer cross-attn 2D-3D + ICP neural"),
        fillcolor=C["perception"], color=C["perception"])

    # --- Interfaz matematica ---
    g.node("se3", label=html_node(
        "SE(3) / SO(3)", "Grupo de Lie · T = (R, t)"),
        fillcolor=C["math"], color=C["math"])

    # --- Planificacion ---
    g.node("diff", label=html_node(
        "Diffusion Policy",
        "Chi et al. RSS 2023<br/>ConditionalUNet1D + DDIM-25 · SDE inversa"),
        fillcolor=C["planning"], color=C["planning"])

    # --- Salida de planificacion ---
    g.node("traj", label=html_node(
        "Trayectoria 7-DoF", "horizon = 16 · DDIM 25 pasos"),
        fillcolor=C["output"], color=C["output"], shape="folder")

    # --- Ejecucion ---
    g.node("sim", label=html_node(
        "CoppeliaSim Edu V4.10",
        "Robot Ragnar (delta) · simulacion fisica stepped"),
        fillcolor=C["execution"], color=C["execution"])

    # --- Validacion E2E ---
    g.node("val", label=html_node(
        "Validacion E2E",
        "H1 AUC ADD-S · H2 score multimodal<br/>H3 cycle p95 &lt; 10 s · Bootstrap CI 95 %"),
        fillcolor=C["output"], color=C["output"])

    # --- Aporte TFM (caja informativa) ---
    g.node("aporte", label=(
        "<<table border='0' cellpadding='4'>"
        "<tr><td align='center'><font face='Helvetica-Bold' point-size='14' color='#0F172A'>"
        "<b>Aporte TFM original</b></font></td></tr>"
        "<tr><td align='left'><font face='Helvetica' point-size='11' color='#1E293B'>"
        "• Integracion formal SE(3) + SDEs<br align='left'/>"
        "• Reproduccion SOTA sin GPU dedicada<br align='left'/>"
        "• Bootstrap CI 95 % (B = 1000)<br align='left'/>"
        "• Validacion visual reproducible</font></td></tr>"
        "</table>>"
    ), fillcolor=C["surface"], color=C["ink_soft"], shape="note", penwidth="2.0")

    # --- Conexiones del flujo principal ---
    g.edge("rgbd", "fp", label="RGB-D")
    g.edge("bop",  "fp", label="GT pose", style="dashed")
    g.edge("fp",   "se3", label="R, t")
    g.edge("se3",  "diff", label="cond")
    g.edge("diff", "traj", label="a0 ... aN")
    g.edge("traj", "sim",  label="trayectoria")
    g.edge("sim",  "val",  label="metricas")

    # Conexion informativa del aporte
    g.edge("aporte", "sim", style="dotted", arrowhead="none",
           color=C["ink_soft"], penwidth="1.5")

    # --- Leyenda como cluster inferior ---
    with g.subgraph(name="cluster_legend") as L:
        L.attr(label="Leyenda", style="rounded", color=C["border"],
               fontcolor=C["ink_soft"], fontname="Helvetica-Bold",
               fontsize="12", labelloc="t")
        items = [
            ("L1", "Entrada",                   C["input"]),
            ("L2", "Percepcion (Transformer)",  C["perception"]),
            ("L3", "Interfaz matematica",       C["math"]),
            ("L4", "Planificacion (Diffusion)", C["planning"]),
            ("L5", "Ejecucion (Simulacion)",    C["execution"]),
            ("L6", "Salida / Validacion",       C["output"]),
        ]
        for nid, txt, col in items:
            L.node(nid, label=(
                f"<<font face='Helvetica-Bold' point-size='12' color='white'>"
                f"<b>{txt}</b></font>>"
            ), fillcolor=col, color=col, shape="box",
                    style="rounded,filled", margin="0.18,0.10")
        # Invisibles para forzar orden horizontal
        for a, b in zip([i[0] for i in items], [i[0] for i in items[1:]]):
            L.edge(a, b, style="invis")

    g.render(str(OUT_STEM), format="png", cleanup=True)
    print(f"[OK] Diagrama generado: {OUT_STEM}.png")


if __name__ == "__main__":
    build()
