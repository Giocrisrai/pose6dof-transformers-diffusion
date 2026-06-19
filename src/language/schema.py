"""Modelo de datos del subsistema de lenguaje natural.

Dataclasses puras, sin dependencias pesadas (numpy/torch/CLIP). Son el
contrato entre el parser (texto -> Instruction) y el grounder
(Instruction + objetos -> target).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TargetSpec:
    """Atributos que describen el objeto buscado."""
    color: Optional[Literal["red", "blue", "green", "yellow"]] = None   # normalizado: "red", "blue", "green", ...
    shape: Optional[Literal["cube", "sphere", "cylinder", "box"]] = None  # "cube", "sphere", "cylinder", "box"
    size: Optional[Literal["small", "large"]] = None                     # "small", "large" (atributo continuo, exp22)
    raw_noun: Optional[str] = None                                       # sustantivo crudo: "pieza", "objeto"

    def is_empty(self) -> bool:
        """True si no hay ningún atributo discriminativo."""
        return not (self.color or self.shape or self.size)


@dataclass
class SpatialRelation:
    """Relación espacial respecto a un ancla o a la escena (exp26)."""
    relation: Literal["left_of", "right_of", "nearest", "farthest", "on_top"]  # left_of/right_of/nearest/farthest/on_top
    anchor: Optional[TargetSpec] = None                                         # objeto de referencia (si aplica)


@dataclass
class Instruction:
    """Instrucción de lenguaje natural ya estructurada."""
    raw_text: str
    target: TargetSpec
    intent: Literal["pick", "pick_then_place", "sequence"] = "pick"   # pick | pick_then_place | sequence
    spatial: Optional[SpatialRelation] = None
    steps: list["Instruction"] = field(default_factory=list)  # exp23
    confidence: float = 1.0
    backend: str = "unknown"               # qué parser lo produjo


@dataclass
class ObjectView:
    """Vista ligera de un objeto detectado, consumible por el Grounder.

    Desacopla el grounder de PoseResult: el pipeline construye ObjectViews
    a partir de sus PoseResult (+ atributos de CLIP o de metadatos de escena).
    """
    obj_id: int
    centroid: tuple[float, float, float]   # posición 3D en metros (cámara/mundo)
    attributes: dict[str, str] = field(default_factory=dict)  # {"color","shape","size"}
    bbox: Optional[tuple[int, int, int, int]] = None           # (x1,y1,x2,y2) en imagen, para crops CLIP


@dataclass
class GroundingResult:
    """Resultado de asociar una Instruction a los objetos de la escena."""
    target_obj_id: Optional[int]
    scores: dict[int, float]               # {obj_id: score}
    method: str                            # "attribute" | "clip_image" | "spatial"
    rejected: list[int] = field(default_factory=list)
    ambiguous: bool = False
