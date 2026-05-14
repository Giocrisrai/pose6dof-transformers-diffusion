"""Interfaz abstracta para estimadores de pose 6-DoF.

Define el contrato que cualquier estimador debe cumplir para integrarse
en el pipeline TFM. Esto permite intercambiar FoundationPose (licencia NC)
por alternativas open-license (FreeZeV2 Apache 2, MegaPose AGPL, Any6D MIT)
sin tocar el resto del pipeline.

Cualquier estimador concreto solo debe implementar `predict_pose`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class PoseEstimate:
    """Resultado de un estimador de pose 6-DoF."""
    R: np.ndarray              # (3, 3) rotacion
    t: np.ndarray              # (3,) translacion en mm (BOP convention)
    confidence: float = 1.0    # [0, 1]
    inference_time_s: float = 0.0
    method: str = "unknown"


@runtime_checkable
class PoseEstimator(Protocol):
    """Protocolo que define el contrato de cualquier estimador 6-DoF.

    Cualquier clase que implemente `predict_pose` puede sustituirse en el
    pipeline sin cambios. Esto permite probar alternativas open-license
    (FreeZeV2, MegaPose, Any6D, SamPose) sin re-escribir el resto del codigo.
    """
    name: str
    license: str  # "MIT" | "Apache-2.0" | "BSD" | "AGPL-3.0" | "NC" | ...

    def predict_pose(
        self,
        rgb: np.ndarray,           # (H, W, 3) uint8 — opcional para metodos model-free
        depth: Optional[np.ndarray],  # (H, W) float32 mm — opcional
        cam_K: np.ndarray,          # (3, 3) intrinsica
        obj_id: int,                # ID de objeto BOP
        scene_id: Optional[str] = None,
        img_id: Optional[int] = None,
    ) -> PoseEstimate:
        ...

    def is_commercializable(self) -> bool:
        """True si la licencia permite uso comercial directo."""
        ...
