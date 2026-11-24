"""Vector de condición de Diffusion Policy: una función por convención de entrenamiento.

Cada checkpoint se entrenó con una disposición concreta del vector de 64
dimensiones. Si en inferencia se arma de otra forma, el modelo recibe una entrada
que nunca vio y sus salidas dejan de corresponder a la pose pedida (medido en
`experiments/exp28_multimodality_per_pose.py`). Este módulo es el único sitio
donde se define cada disposición.

| Disposición | Modelos                         | Contenido                                   |
|-------------|---------------------------------|---------------------------------------------|
| `pose12`    | `grasp` (notebook 06)           | filas de [R | t] en cond[:12], t en metros  |
| `pos3`      | `extended`, `ultra`, `ultra_fast` | posición en cond[:3] (m), cond[3:6] libre |
"""
from __future__ import annotations

import numpy as np

COND_DIM = 64

# Disposición con la que se entrenó cada checkpoint servido por la API.
LAYOUT_BY_MODEL = {
    "original": "pose12",
    "grasp": "pose12",
    "extended": "pos3",
    "ultra": "pos3",
    "ultra_fast": "pos3",
}


def cond_pose12(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Filas de [R | t] en cond[:12]; equivale a `pose[:3, :].flatten()`.

    Args:
        R: rotación (3, 3).
        t: traslación (3,) en metros.
    """
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[:12] = np.hstack([np.asarray(R, dtype=np.float64).reshape(3, 3),
                           np.asarray(t, dtype=np.float64).reshape(3, 1)]).flatten()
    return cond


def cond_pos3(t: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Posición del objeto en cond[:3] (metros) y un vector opcional en cond[3:6].

    En el entrenamiento cond[3:6] fue ruido gaussiano pequeño (σ = 0.1) sin relación
    con la trayectoria, así que el modelo es insensible a él; cero es un valor válido.
    """
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[:3] = np.asarray(t, dtype=np.float64).reshape(3)
    if extra is not None:
        cond[3:6] = np.asarray(extra, dtype=np.float64).reshape(3)
    return cond


def condition_for_model(model_name: str, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Condición de una pose (R, t en metros) con la disposición de `model_name`."""
    layout = LAYOUT_BY_MODEL.get(model_name)
    if layout == "pose12":
        return cond_pose12(R, t)
    if layout == "pos3":
        return cond_pos3(t)
    raise ValueError(f"No se conoce la disposición de condición del modelo '{model_name}'")
