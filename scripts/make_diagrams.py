#!/usr/bin/env python3
"""Genera diagramas profesionales con Graphviz (via Python graphviz).

Salida en docs/figures_hero/ (siguen la misma carpeta que las figuras hero
de matplotlib, pero con flechas y layout automatico — calidad superior).

Diagramas:
- 10_pipeline_full.png        — pipeline completo con flechas direccionadas
- 11_vla_lite_flow.png         — flujo VLA-lite con CLIP text/image + gate
- 12_exploraciones_workflow.png — DAG de las 13 exploraciones (dependencias)
- 13_data_flow.png             — flujo de datos en runtime

Requiere:
- graphviz (instalado via brew o apt) → /opt/homebrew/bin/dot
- pip install graphviz diagrams
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Asegurar dot en PATH
DOT_BIN = "/opt/homebrew/bin"
if DOT_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{DOT_BIN}:{os.environ.get('PATH', '')}"

import graphviz

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/figures_hero"
OUT.mkdir(parents=True, exist_ok=True)

# === DESIGN TOKENS (replicando src/utils/plot_style.py) ===
COLORS = {
    "primary":  "#0098CD",
    "primary_dark": "#006B92",
    "accent":   "#FF6B35",
    "accent_dark": "#CC5429",
    "success":  "#16A34A",
    "success_dark": "#15803D",
    "warning":  "#F59E0B",
    "danger":   "#DC2626",
    "ink":      "#0F172A",
    "ink_soft": "#334155",
    "muted":    "#64748B",
    "border":   "#E2E8F0",
    "surface":  "#F8FAFC",
    "purple":   "#A855F7",
    "cyan":     "#06B6D4",
    "teal":     "#14B8A6",
    "amber":    "#EAB308",
}

COMMON_GRAPH_ATTR = {
    "rankdir": "LR",
    "bgcolor": "white",
    "pad": "0.5",
    "fontname": "Helvetica",
    "fontsize": "14",
    "splines": "spline",
    "concentrate": "false",
    "nodesep": "0.6",
    "ranksep": "0.8",
}

COMMON_NODE_ATTR = {
    "shape": "box",
    "style": "rounded,filled",
    "fontname": "Helvetica",
    "fontsize": "12",
    "margin": "0.2,0.15",
    "penwidth": "2",
}

COMMON_EDGE_ATTR = {
    "fontname": "Helvetica",
    "fontsize": "10",
    "color": COLORS["ink_soft"],
    "penwidth": "1.8",
    "arrowsize": "0.9",
}


def styled_node(g, name, label, color, text_color="white", subtitle=None):
    """Caja con titulo + subtitulo opcional en HTML-like."""
    if subtitle:
        html_label = (
            f"<<table border='0' cellpadding='6'>"
            f"<tr><td><font face='Helvetica' point-size='13' color='{text_color}'><b>{label}</b></font></td></tr>"
            f"<tr><td><font face='Helvetica' point-size='10' color='{text_color}'>{subtitle}</font></td></tr>"
            f"</table>>"
        )
        g.node(name, label=html_label, fillcolor=color, color=color)
    else:
        g.node(name, label=label, fillcolor=color, color=color, fontcolor=text_color)


# ============================================================================
# DIAGRAMA 10 — Pipeline completo (Camara → Brazo)
# ============================================================================
def diag10_pipeline_full():
    g = graphviz.Digraph("pipeline", format="png")
    g.attr(**COMMON_GRAPH_ATTR)
    g.attr("node", **COMMON_NODE_ATTR)
    g.attr("edge", **COMMON_EDGE_ATTR)

    # Titulo
    g.attr(label=("<<font point-size='20'><b>Pipeline de Bin Picking 6-DoF</b></font><br/>"
                    "<font point-size='12' color='#64748B'>"
                    "Camara → Pose 6-DoF → Trayectoria → Control → Brazo</font>>"),
              labelloc="t")

    # Pipeline principal
    styled_node(g, "camera",  "Camara RGB-D",      COLORS["muted"],   subtitle="Intel RealSense D435i")
    styled_node(g, "fp",       "FoundationPose",    COLORS["primary"], subtitle="Transformer cross-attn 2D-3D")
    styled_node(g, "freeze",  "FreeZeV2 (opcional)", COLORS["primary_dark"],
                  subtitle="Apache-2.0 · open-license")
    styled_node(g, "diff",     "Diffusion Policy",  COLORS["teal"],    subtitle="UNet1D + DDIM-25")
    styled_node(g, "fast",     "ultra_fast",         COLORS["accent"],  subtitle="1 NFE · ×517 speedup")
    styled_node(g, "pbvs",     "PBVS en SE(3)",     COLORS["accent_dark"], subtitle="Visual servoing")
    styled_node(g, "robot",   "Brazo robotico",    COLORS["ink"],     subtitle="UR3e / Kinova / xArm")

    # Flujo principal
    g.edge("camera", "fp", label="RGB + Depth")
    g.edge("fp", "diff", label="R, t (6-DoF)")
    g.edge("diff", "pbvs", label="Trayectoria\n16 pasos")
    g.edge("pbvs", "robot", label="Joint positions")

    # Alternativas con linea punteada
    g.edge("camera", "freeze", style="dashed", color=COLORS["muted"], label="alt (open)")
    g.edge("freeze", "diff", style="dashed", color=COLORS["muted"])
    g.edge("diff", "fast", style="dashed", color=COLORS["muted"], label="alt (rapido)")
    g.edge("fast", "pbvs", style="dashed", color=COLORS["muted"])

    # Capa VLA-lite arriba
    with g.subgraph(name="cluster_vla") as c:
        c.attr(label="VLA-lite (exp 4-13)",
                 style="rounded,filled", fillcolor=COLORS["surface"],
                 color=COLORS["primary"], penwidth="2",
                 fontname="Helvetica", fontsize="12",
                 fontcolor=COLORS["primary_dark"], labelloc="t")
        styled_node(c, "user",  "Usuario",          COLORS["muted"],   subtitle="texto en lenguaje natural")
        styled_node(c, "clip_t", "CLIP-text (frozen)", COLORS["purple"], subtitle="512-D embedding")
        styled_node(c, "clip_i", "CLIP-image (frozen)", COLORS["purple"], subtitle="768-D embedding")
        styled_node(c, "gate",   "TextGroundedGate / VisualGate", COLORS["cyan"],
                       subtitle="softmax sobre N objetos")

    g.edge("user", "clip_t", label="'pick the red sphere'")
    g.edge("camera", "clip_i", label="crops por objeto", style="dashed")
    g.edge("clip_t", "gate")
    g.edge("clip_i", "gate")
    g.edge("gate", "diff", label="objeto target", color=COLORS["primary"], penwidth="2.5")

    out = OUT / "10_pipeline_full"
    g.render(str(out), format="png", cleanup=True)
    print(f"  ✓ {out}.png")


# ============================================================================
# DIAGRAMA 11 — Flujo VLA-lite con CLIP
# ============================================================================
def diag11_vla_lite_flow():
    g = graphviz.Digraph("vla_lite", format="png")
    g.attr(**COMMON_GRAPH_ATTR)
    g.attr(rankdir="TB")  # vertical para este
    g.attr("node", **COMMON_NODE_ATTR)
    g.attr("edge", **COMMON_EDGE_ATTR)

    g.attr(label=("<<font point-size='20'><b>VLA-lite — Flujo end-to-end</b></font><br/>"
                    "<font point-size='12' color='#64748B'>"
                    "Lenguaje natural + apariencia visual → seleccion + trayectoria</font>>"),
              labelloc="t")

    # Inputs
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="ENTRADAS", style="dashed",
                 color=COLORS["muted"], fontcolor=COLORS["muted"], fontname="Helvetica")
        styled_node(c, "instr",   "Instruccion",       COLORS["amber"],
                       subtitle="'pick the red sphere'")
        styled_node(c, "scene",   "Escena (RGB)",       COLORS["muted"],
                       subtitle="cam cenital + segmentacion")

    # CLIP encoders (ambos frozen)
    with g.subgraph(name="cluster_clip") as c:
        c.attr(label="CLIP (frozen, 150 M params)", style="rounded,filled",
                 fillcolor="#F3E8FF", color=COLORS["purple"], penwidth="2",
                 fontname="Helvetica", fontsize="11",
                 fontcolor="#6B21A8", labelloc="t")
        styled_node(c, "tok",    "CLIPTokenizer",      COLORS["purple"])
        styled_node(c, "text_enc", "CLIPTextModel",     COLORS["purple"],
                       subtitle="output 512-D")
        styled_node(c, "vis_enc",  "CLIPVisionModel",   COLORS["purple"],
                       subtitle="output 768-D")

    # Gate entrenable
    with g.subgraph(name="cluster_gate") as c:
        c.attr(label="MODULOS ENTRENABLES (~150 K params)", style="rounded,filled",
                 fillcolor="#E0F2F1", color=COLORS["teal"], penwidth="2",
                 fontname="Helvetica", fontsize="11",
                 fontcolor="#0F766E", labelloc="t")
        styled_node(c, "proj_t", "text_proj\n(512 → 64)",  COLORS["teal"])
        styled_node(c, "proj_v", "vis_proj\n(768 → 64)",   COLORS["teal"])
        styled_node(c, "score",  "Scorer MLP\n+ softmax",  COLORS["cyan"])

    # Resultado
    styled_node(g, "gates",      "Gates por objeto",    COLORS["success"],
                  subtitle="probabilidades 0..1")
    styled_node(g, "sel_pos",    "Selected position",   COLORS["accent"],
                  subtitle="weighted average")
    styled_node(g, "diff",        "Diffusion Policy",    COLORS["primary"],
                  subtitle="DDIM-25 condicionado")
    styled_node(g, "traj_out",   "Trayectoria final",   COLORS["ink"],
                  subtitle="16 pasos × 7 DoF")

    # Conexiones
    g.edge("instr", "tok"); g.edge("tok", "text_enc")
    g.edge("scene", "vis_enc")
    g.edge("text_enc", "proj_t")
    g.edge("vis_enc", "proj_v")
    g.edge("proj_t", "score", label="64-D")
    g.edge("proj_v", "score", label="N × 64-D")
    g.edge("score", "gates")
    g.edge("gates", "sel_pos", label="agregar")
    g.edge("sel_pos", "diff", label="condicionamiento")
    g.edge("diff", "traj_out")

    out = OUT / "11_vla_lite_flow"
    g.render(str(out), format="png", cleanup=True)
    print(f"  ✓ {out}.png")


# ============================================================================
# DIAGRAMA 12 — DAG de las 13 exploraciones
# ============================================================================
def diag12_exploraciones_dag():
    g = graphviz.Digraph("exp_dag", format="png")
    g.attr(rankdir="LR", bgcolor="white", pad="0.5",
              nodesep="0.4", ranksep="1.0", splines="ortho",
              fontname="Helvetica")
    g.attr("node", **COMMON_NODE_ATTR)
    g.attr("edge", **COMMON_EDGE_ATTR)

    g.attr(label=("<<font point-size='20'><b>13 Exploraciones post-TFM — Dependencias</b></font><br/>"
                    "<font point-size='12' color='#64748B'>"
                    "Cada flecha indica que el resultado se reutiliza en la siguiente</font>>"),
              labelloc="t")

    # Cada exploracion como nodo
    exp_data = [
        ("exp1",  "1. bop-bootstrap-ci",          COLORS["primary"],     "publicado PyPI"),
        ("exp2",  "2. Distillation 1-NFE",        COLORS["primary"],     "×517 speedup"),
        ("exp3",  "3. Pipeline open-license",     COLORS["primary"],     "FreeZeV2 −3 pp"),
        ("exp4",  "4. VLA-lite color",            COLORS["teal"],        "98.6 %"),
        ("exp5",  "5. Robustez linguistica",      COLORS["teal"],        "100 % 6 familias"),
        ("exp6",  "6. VLA color + forma",         COLORS["teal"],        "99.9 % global"),
        ("exp7",  "7. Simulaciones 3D",           COLORS["teal"],        "12/12 escenas"),
        ("exp8",  "8. Multi-objeto N=2..5",       COLORS["accent"],      "100 % todos N"),
        ("exp9",  "9. Atributo tamano",           COLORS["accent"],      "99.9 %"),
        ("exp10", "10. Secuenciales multi-step",  COLORS["accent"],      "8/8 secuencias"),
        ("exp11", "11. CLIP-image visual",        COLORS["accent_dark"], "100 % sin atribs"),
        ("exp12", "12. Robustez DR",              COLORS["accent_dark"], "12/12 ≥ 75 %"),
        ("exp13", "13. Razonamiento espacial",    COLORS["purple"],      "98.4 %"),
    ]
    for (eid, label, color, sub) in exp_data:
        styled_node(g, eid, label, color, subtitle=sub)

    # Dependencias logicas
    edges = [
        ("exp1", "exp3"),    # bootstrap-ci se usa en exp3 comparativa licencias
        ("exp1", "exp12"),   # bootstrap-ci se usa en exp12 robustez
        ("exp4", "exp5"),    # exp5 evalua robustez linguistica de exp4
        ("exp4", "exp6"),    # exp6 extiende color a color+forma
        ("exp6", "exp7"),    # exp7 visualiza decisiones de exp6
        ("exp6", "exp8"),    # exp8 generaliza a N>2
        ("exp8", "exp9"),    # exp9 anade atributo tamano sobre multi-objeto
        ("exp8", "exp10"),   # exp10 ejecuta secuencias sobre multi-objeto
        ("exp8", "exp13"),   # exp13 anade dim espacial sobre multi-objeto
        ("exp11", "exp12"),  # exp12 evalua robustez del visual gate
    ]
    for src, dst in edges:
        g.edge(src, dst)

    out = OUT / "12_exploraciones_workflow"
    g.render(str(out), format="png", cleanup=True)
    print(f"  ✓ {out}.png")


# ============================================================================
# DIAGRAMA 13 — Flujo de datos en runtime
# ============================================================================
def diag13_data_flow():
    g = graphviz.Digraph("dataflow", format="png")
    g.attr(rankdir="TB", bgcolor="white", pad="0.5",
              nodesep="0.4", ranksep="0.7", splines="spline")
    g.attr("node", **COMMON_NODE_ATTR)
    g.attr("edge", **COMMON_EDGE_ATTR)

    g.attr(label=("<<font point-size='20'><b>Flujo de datos en runtime — Tipos y dimensiones</b></font><br/>"
                    "<font point-size='12' color='#64748B'>"
                    "Latencia total p95 &lt; 7 s en M1 Pro</font>>"),
              labelloc="t")

    # Inputs (oval = datos)
    g.node("rgb",    "RGB image\n(H × W × 3, uint8)",       shape="oval", style="filled",
              fillcolor="#FEF3C7", color=COLORS["warning"], fontsize="11")
    g.node("depth",  "Depth image\n(H × W, float32 mm)",    shape="oval", style="filled",
              fillcolor="#FEF3C7", color=COLORS["warning"], fontsize="11")
    g.node("text",   "Instruccion\nstring",                  shape="oval", style="filled",
              fillcolor="#FEF3C7", color=COLORS["warning"], fontsize="11")
    g.node("cad",    "CAD del objeto\n(.ply mesh)",         shape="oval", style="filled",
              fillcolor="#FEF3C7", color=COLORS["warning"], fontsize="11")

    # Procesos (rectangulos con color)
    styled_node(g, "fp_proc",   "FoundationPose\n~ 4 000 ms",   COLORS["primary"])
    styled_node(g, "clip_proc", "CLIP encode\n~ 50 ms",          COLORS["purple"])
    styled_node(g, "gate_proc", "VisualGate\n~ 1 ms",            COLORS["cyan"])
    styled_node(g, "diff_proc", "Diffusion DDIM-25\n~ 90 ms",    COLORS["teal"])
    styled_node(g, "pbvs_proc", "PBVS SE(3)\n~ 1 800 ms",        COLORS["accent_dark"])

    # Outputs
    g.node("pose_out",  "Pose 6-DoF\nR en SO(3), t en mm",  shape="oval", style="filled",
              fillcolor="#DCFCE7", color=COLORS["success"], fontsize="11")
    g.node("emb_out",   "CLIP embedding\n512-D / 768-D",      shape="oval", style="filled",
              fillcolor="#DCFCE7", color=COLORS["success"], fontsize="11")
    g.node("gates_out", "Gates softmax\nN × float",            shape="oval", style="filled",
              fillcolor="#DCFCE7", color=COLORS["success"], fontsize="11")
    g.node("traj_out",  "Trayectoria\n16 × 7 floats",          shape="oval", style="filled",
              fillcolor="#DCFCE7", color=COLORS["success"], fontsize="11")
    g.node("ctrl_out",  "Comando articular\n6-DoF joint pos",  shape="oval", style="filled",
              fillcolor="#DCFCE7", color=COLORS["success"], fontsize="11")

    # Flujo
    g.edge("rgb", "fp_proc")
    g.edge("depth", "fp_proc")
    g.edge("cad", "fp_proc")
    g.edge("fp_proc", "pose_out")

    g.edge("text", "clip_proc")
    g.edge("rgb", "clip_proc", style="dashed", label="crops")
    g.edge("clip_proc", "emb_out")

    g.edge("emb_out", "gate_proc")
    g.edge("gate_proc", "gates_out")

    g.edge("pose_out", "diff_proc")
    g.edge("gates_out", "diff_proc")
    g.edge("diff_proc", "traj_out")

    g.edge("traj_out", "pbvs_proc")
    g.edge("pose_out", "pbvs_proc", style="dashed", label="feedback")
    g.edge("pbvs_proc", "ctrl_out")

    out = OUT / "13_data_flow"
    g.render(str(out), format="png", cleanup=True)
    print(f"  ✓ {out}.png")


def main():
    print("Generando 4 diagramas con Graphviz...")
    diag10_pipeline_full()
    diag11_vla_lite_flow()
    diag12_exploraciones_dag()
    diag13_data_flow()
    print(f"\n[OK] Todas guardadas en {OUT}")


if __name__ == "__main__":
    main()
