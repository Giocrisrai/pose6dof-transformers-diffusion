#!/usr/bin/env python3
"""Genera el diagrama de arquitectura del pipeline TFM.

Bloques: Imagen RGB-D -> FoundationPose (Transformer) -> Pose 6-DoF SE(3)
                     -> Diffusion Policy -> Trayectoria multimodal
                     -> CoppeliaSim + Ragnar -> Pick&Place
Marca SE(3)/SO(3) como interfaz matematica entre componentes.

Salida: experiments/results/pipeline_e2e/fig_pipeline_arquitectura.png
"""
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/results/pipeline_e2e/fig_pipeline_arquitectura.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# Colores
COLOR_INPUT = '#E8F4F8'
COLOR_PERCEPTION = '#0098CD'
COLOR_PLANNING = '#35876B'
COLOR_EXECUTION = '#E66B00'
COLOR_OUTPUT = '#FFE4B5'
COLOR_MATH = '#FFE4E1'
COLOR_DARK = '#2C3E50'

def block(x, y, w, h, text, fc, fs=11, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=fc, edgecolor=COLOR_DARK, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, weight=weight, wrap=True)

def arrow(x1, y1, x2, y2, label='', label_y_off=0.3, color=COLOR_DARK):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->,head_width=0.4,head_length=0.6',
                          color=color, linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr)
    if label:
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2 + label_y_off
        ax.text(midx, midy, label, ha='center', va='bottom',
                fontsize=9, style='italic', color=color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.85))

# --- Titulo ---
ax.text(8, 8.5, 'Pipeline TFM: FoundationPose + Diffusion Policy para Bin Picking 6-DoF',
        ha='center', va='center', fontsize=15, weight='bold', color=COLOR_DARK)
ax.text(8, 8.05, 'Marco matemático unificado SE(3) / SO(3) + SDEs',
        ha='center', va='center', fontsize=11, style='italic', color=COLOR_DARK)

# --- Fila 1: ENTRADA ---
block(0.3, 6.0, 2.4, 1.2, 'Imagen RGB-D\n(640×480)\n+ Modelo CAD', COLOR_INPUT, fs=10)
block(0.3, 4.4, 2.4, 1.2, 'Datasets BOP\nT-LESS / YCB-V\n(subset BOP-19)', COLOR_INPUT, fs=10)

# --- PERCEPCION (azul) ---
block(3.4, 4.8, 3.5, 2.4,
      'FoundationPose\n(Wen et al., CVPR 2024)\n\nTransformer\ncross-attention 2D-3D\n+ ICP neural', COLOR_PERCEPTION, fs=10, bold=True)
arrow(2.7, 6.6, 3.4, 6.0, 'RGB-D')
arrow(2.7, 5.0, 3.4, 5.5, 'GT pose')

# --- INTERFAZ MATEMATICA SE(3) ---
block(7.2, 5.5, 1.5, 1.0, 'SE(3) / SO(3)\nGrupo de Lie',
      COLOR_MATH, fs=10, bold=True)
arrow(6.9, 6.0, 7.2, 6.0, 'R, t')

# --- PLANIFICACION (verde) ---
block(9.0, 4.8, 3.5, 2.4,
      'Diffusion Policy\n(Chi et al., RSS 2023)\n\nDDPM scheduler\n+ ConditionalUNet1D\nSDE inversa', COLOR_PLANNING, fs=10, bold=True)
arrow(8.7, 6.0, 9.0, 6.0, 'cond')

# --- TRAYECTORIA ---
block(13.0, 5.5, 2.7, 1.0, 'Trayectoria 7-DoF\n(horizon=16, DDIM 25)',
      COLOR_OUTPUT, fs=10)
arrow(12.5, 6.0, 13.0, 6.0, 'a₀…a₁₅')

# --- Fila 2: EJECUCION (naranja) ---
block(5.5, 1.8, 5.0, 2.0,
      'CoppeliaSim Edu V4.10\nEscena pickAndPlaceDemo\n\nRobot Ragnar (delta)\nSimulación física stepped', COLOR_EXECUTION, fs=10, bold=True)
arrow(14.4, 5.4, 10.5, 3.7, 'trayectoria', color=COLOR_DARK)

# --- VALIDACION ---
block(11.5, 1.8, 4.0, 2.0,
      'Validación E2E\n\nH1: AUC ADD-S\nH2: score multimodal\nH3: cycle p95 < 10s\n[Bootstrap CI 95%]', COLOR_OUTPUT, fs=10)
arrow(10.5, 2.8, 11.5, 2.8, 'metricas')

# --- TFM aporte ---
block(0.3, 1.8, 5.0, 2.0,
      'Aporte TFM original\n\n• Integración formal SE(3)+SDEs\n• Reproducción SOTA sin GPU dedicada\n• Bootstrap CI 95% B=1000\n• Validación visual reproducible',
      '#F5F5DC', fs=9, bold=False)
arrow(2.8, 3.8, 5.5, 3.8, '', color=COLOR_DARK)

# --- Math under FoundationPose ---
ax.text(5.15, 4.4, r'output: $T \in \mathrm{SE}(3) = \mathrm{SO}(3) \ltimes \mathbb{R}^3$',
        ha='center', va='center', fontsize=9, style='italic')

# --- Math under Diffusion ---
ax.text(10.75, 4.4, r'$dx = f(x,t)dt + g(t)\,d\bar{w}$ (SDE inversa)',
        ha='center', va='center', fontsize=9, style='italic')

# --- Leyenda colores ---
legend_elements = [
    mpatches.Patch(facecolor=COLOR_INPUT, edgecolor=COLOR_DARK, label='Entrada'),
    mpatches.Patch(facecolor=COLOR_PERCEPTION, edgecolor=COLOR_DARK, label='Percepción (Transformer)'),
    mpatches.Patch(facecolor=COLOR_MATH, edgecolor=COLOR_DARK, label='Interfaz matemática'),
    mpatches.Patch(facecolor=COLOR_PLANNING, edgecolor=COLOR_DARK, label='Planificación (Diffusion)'),
    mpatches.Patch(facecolor=COLOR_EXECUTION, edgecolor=COLOR_DARK, label='Ejecución (Simulación)'),
    mpatches.Patch(facecolor=COLOR_OUTPUT, edgecolor=COLOR_DARK, label='Salida / Validación'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
          bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Diagrama generado: {OUT}")
print(f"     Tamaño: {os.path.getsize(OUT)/1024:.0f} KB")
