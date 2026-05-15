"""Paleta de colores y estilo matplotlib unificado para todo el proyecto.

Inspirado en design tokens profesionales (Tailwind / Material / IBM Carbon).
Cohesion visual entre los renders de los experimentos, Gradio, Streamlit y
las figuras del TFM.

Uso:
    from src.utils.plot_style import apply_style, COLORS
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(..., color=COLORS["primary"])
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams


# ============================================================================
# PALETA — design tokens cohesionados
# ============================================================================
COLORS = {
    # Estados semanticos
    "primary": "#0098CD",       # Azul corporativo (UNIR / tech)
    "primary_dark": "#006B92",
    "primary_light": "#E6F4FA",

    "accent": "#FF6B35",        # Naranja energetico (trayectorias)
    "accent_dark": "#CC5429",
    "accent_light": "#FFF1EB",

    "success": "#16A34A",       # Verde (aciertos, target)
    "success_dark": "#15803D",
    "success_light": "#DCFCE7",

    "warning": "#F59E0B",       # Ambar
    "warning_light": "#FEF3C7",

    "danger": "#DC2626",        # Rojo (errores)
    "danger_light": "#FEE2E2",

    # Neutrales (slate / gris)
    "ink": "#0F172A",           # Texto principal
    "ink_soft": "#334155",
    "muted": "#64748B",
    "muted_soft": "#94A3B8",
    "border": "#E2E8F0",
    "surface": "#F8FAFC",
    "card": "#FFFFFF",

    # Categoricos (para mas de 5 series)
    "cat_1": "#0098CD",  # azul
    "cat_2": "#35876B",  # turquesa
    "cat_3": "#FF6B35",  # naranja
    "cat_4": "#A855F7",  # purpura
    "cat_5": "#EAB308",  # amarillo
    "cat_6": "#EF4444",  # rojo
    "cat_7": "#06B6D4",  # cyan
    "cat_8": "#84CC16",  # lime
}


# Paleta categorica para `cycle('color')` automatico
CATEGORICAL = [
    COLORS["cat_1"], COLORS["cat_2"], COLORS["cat_3"], COLORS["cat_4"],
    COLORS["cat_5"], COLORS["cat_6"], COLORS["cat_7"], COLORS["cat_8"],
]


def apply_style(scale: str = "presentation"):
    """Aplica estilo matplotlib unificado.

    scale:
        - 'paper':        densidad alta (figura academica)
        - 'presentation': fuentes grandes (defensa/slides) [default]
        - 'web':          equilibrado para UI Gradio/Streamlit
    """
    if scale == "paper":
        font_size = 11
        title = 12
        label = 11
        tick = 9
    elif scale == "web":
        font_size = 12
        title = 13
        label = 11
        tick = 10
    else:  # presentation
        font_size = 14
        title = 17
        label = 13
        tick = 12

    rcParams.update({
        # Tipografia (system fonts disponibles en Mac/Linux/Windows)
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": font_size,
        "font.weight": "normal",

        "axes.titlesize": title,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": label,
        "axes.labelweight": "normal",
        "axes.labelpad": 8,
        "axes.edgecolor": COLORS["border"],
        "axes.linewidth": 1.0,
        "axes.facecolor": "#FFFFFF",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler("color", CATEGORICAL),

        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "xtick.color": COLORS["ink_soft"],
        "ytick.color": COLORS["ink_soft"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,

        "grid.color": COLORS["border"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,

        "legend.fontsize": tick + 1,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": COLORS["border"],
        "legend.facecolor": "#FFFFFF",
        "legend.borderpad": 0.6,

        "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF",
        "figure.dpi": 110,
        "figure.titlesize": title + 4,
        "figure.titleweight": "bold",

        "savefig.dpi": 130,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def add_value_labels(ax, fmt: str = "{:.1%}", **kwargs):
    """Anota los valores encima de cada barra de un bar chart."""
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold", fontsize=kwargs.get("fontsize", 13),
            color=kwargs.get("color", COLORS["ink"]),
        )


def style_axes(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """Aplica estilo cohesionado a un axes individual."""
    if title:
        ax.set_title(title, color=COLORS["ink"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS["ink_soft"])
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["ink_soft"])
    ax.tick_params(colors=COLORS["ink_soft"])
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(COLORS["border"])
