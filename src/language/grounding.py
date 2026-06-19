"""Grounder: asocia una Instruction a los objetos detectados de la escena.

Método "attribute" (default, determinista): puntúa por coincidencia de
atributos (color/forma/tamaño) y aplica la relación espacial sobre los
centroides. Método "clip_image" (opcional, lazy): rellena atributos con
CLIP cuando no vienen dados. La lógica de selección es compartida.
"""
from __future__ import annotations

from src.language.schema import GroundingResult, Instruction, ObjectView, TargetSpec

# umbral para considerar dos scores "empatados" (ambigüedad)
_TIE_EPS = 1e-6


def _attr_score(spec: TargetSpec, attrs: dict) -> float:
    """Fracción de atributos especificados que coinciden (0..1)."""
    checks = []
    for key in ("color", "shape", "size"):
        want = getattr(spec, key)
        if want is not None:
            checks.append(1.0 if attrs.get(key) == want else 0.0)
    if not checks:
        return 0.0
    return sum(checks) / len(checks)


def _apply_spatial(relation: str, candidates: list[ObjectView]) -> ObjectView | None:
    """Elige un candidato según la relación espacial (centroide = (x,y,z))."""
    if not candidates:
        return None
    if relation == "left_of":
        return min(candidates, key=lambda o: o.centroid[0])
    if relation == "right_of":
        return max(candidates, key=lambda o: o.centroid[0])
    if relation == "nearest":
        return min(candidates, key=lambda o: o.centroid[2])
    if relation == "farthest":
        return max(candidates, key=lambda o: o.centroid[2])
    if relation == "on_top":
        return max(candidates, key=lambda o: o.centroid[1])
    return None


class Grounder:
    """Selecciona el objeto target descrito por una Instruction."""

    def __init__(self, method: str = "attribute", clip_model: str = "openai/clip-vit-base-patch32"):
        self.method = method
        self.clip_model = clip_model
        self._clip = None  # lazy

    def _ensure_attributes(self, objects: list[ObjectView], rgb=None) -> list[ObjectView]:
        """Si method=clip_image y faltan atributos, los rellena con CLIP.

        En method=attribute se asume que los ObjectView ya traen atributos
        (de metadatos de simulación o de una pasada previa).

        Nota: en modo clip_image este método MUTA cada ObjectView.attributes
        en el lugar (in-place), guardando en caché el resultado de CLIP para
        no volver a inferirlo en llamadas sucesivas al mismo objeto.
        """
        if self.method != "clip_image" or rgb is None:
            return objects
        from src.language._clip import clip_attributes  # lazy, opcional
        for o in objects:
            if not o.attributes and o.bbox is not None:
                o.attributes = clip_attributes(rgb, o.bbox, self.clip_model)
        return objects

    def ground(self, instruction: Instruction, objects: list[ObjectView], rgb=None, K=None) -> GroundingResult:
        """Asocia la instrucción a un objeto de la escena.

        Parámetros
        ----------
        instruction : Instruction
            Instrucción estructurada producida por el parser.
        objects : list[ObjectView]
            Objetos detectados en la escena con sus atributos.
        rgb : array-like, opcional
            Imagen RGB para modo clip_image.
        K : array-like, opcional
            Matriz intrínseca de cámara (reservado para uso futuro).

        Retorna
        -------
        GroundingResult
            Resultado con el obj_id seleccionado (o None si no hay match).
        """
        objects = self._ensure_attributes(objects, rgb)
        spec = instruction.target

        scores = {o.obj_id: _attr_score(spec, o.attributes) for o in objects}
        max_score = max(scores.values(), default=0.0)

        if max_score <= 0.0:
            return GroundingResult(
                target_obj_id=None, scores=scores, method=self.method,
                rejected=[o.obj_id for o in objects], ambiguous=False,
            )

        winners = [o for o in objects if abs(scores[o.obj_id] - max_score) <= _TIE_EPS]
        rejected = [o.obj_id for o in objects if abs(scores[o.obj_id] - max_score) > _TIE_EPS]

        if len(winners) == 1:
            return GroundingResult(
                target_obj_id=winners[0].obj_id, scores=scores,
                method=self.method, rejected=rejected, ambiguous=False,
            )

        # desempate por relación espacial
        if instruction.spatial is not None:
            chosen = _apply_spatial(instruction.spatial.relation, winners)
            if chosen is not None:
                return GroundingResult(
                    target_obj_id=chosen.obj_id, scores=scores,
                    method="spatial",
                    rejected=[o.obj_id for o in objects if o.obj_id != chosen.obj_id],
                    ambiguous=False,
                )

        # empate irresoluble -> ambiguo (se devuelve el primero como sugerencia)
        return GroundingResult(
            target_obj_id=winners[0].obj_id, scores=scores, method=self.method,
            rejected=rejected, ambiguous=True,
        )
