"""Pick guiado por lenguaje natural en CoppeliaSim.

Núcleo PURO (sin simulador): planifica la escena (specs de objetos),
mapea colores RGB→nombre, convierte specs a ObjectView, groundea la
instrucción al objeto y evalúa la selección. El cascarón SIM
(apply_scene/run_language_pick) se añade en una tarea posterior, con
imports del bridge perezosos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.language import make_parser
from src.language.grounding import Grounder
from src.language.schema import GroundingResult, Instruction, ObjectView

_RGB_TO_NAME = {
    (0.85, 0.15, 0.15): "red",
    (0.15, 0.30, 0.85): "blue",
    (0.20, 0.75, 0.20): "green",
}
_NAME_TO_RGB = {v: k for k, v in _RGB_TO_NAME.items()}
_DISTRACTOR_COLORS = ["blue", "green"]
_SHAPE_POOL = ["cube", "sphere", "cylinder"]

_BIN_X = (0.38, 0.55)
_BIN_Y = (-0.17, -0.02)
_Z = 0.033
_MIN_DIST = 0.06


@dataclass
class SimObjectSpec:
    """Especificación de un objeto de escena (sin handle de sim todavía)."""
    obj_id: int
    position: tuple[float, float, float]
    color: str
    shape: str
    size: str = "large"
    handle: Optional[int] = None
    alias: Optional[str] = None


def color_name_from_rgb(rgb, tol: float = 0.12) -> Optional[str]:
    """Mapea un RGB a su nombre canónico más cercano (o None).

    Parámetros
    ----------
    rgb : tuple[float, float, float]
        Color RGB a mapear.
    tol : float
        Tolerancia máxima (diferencia máxima en cualquier canal).

    Retorna
    -------
    str | None
        Nombre del color canónico o None si ninguno está dentro de tol.
    """
    best, best_d = None, tol
    for ref, name in _RGB_TO_NAME.items():
        d = max(abs(rgb[i] - ref[i]) for i in range(3))
        if d <= best_d:
            best, best_d = name, d
    return best


def _sample_positions(n: int, rng: np.random.Generator, max_retries: int = 100) -> list[tuple]:
    """Muestrea n posiciones dentro del bin con separación mínima entre ellas."""
    out: list[tuple] = []
    for _ in range(n):
        for _ in range(max_retries):
            x = float(rng.uniform(*_BIN_X))
            y = float(rng.uniform(*_BIN_Y))
            if all((x - qx) ** 2 + (y - qy) ** 2 >= _MIN_DIST ** 2 for qx, qy, _ in out):
                out.append((x, y, _Z))
                break
        else:
            raise RuntimeError(f"no se pudo muestrear {n} posiciones")
    return out


def plan_language_scene(rng: np.random.Generator, n_objects: int = 3,
                        with_shapes: bool = False,
                        target_color: str = "red",
                        target_shape: str = "cube") -> list[SimObjectSpec]:
    """Planifica una escena: obj 0 = target, resto distractores. Determinista dado rng.

    Parámetros
    ----------
    rng : np.random.Generator
        Generador de números aleatorios (determinista si se fija la semilla).
    n_objects : int
        Número total de objetos (target + distractores).
    with_shapes : bool
        Si True, los distractores pueden tener formas distintas al target.
    target_color : str
        Color del objeto target (obj_id=0).
    target_shape : str
        Forma del objeto target (obj_id=0).

    Retorna
    -------
    list[SimObjectSpec]
        Lista de specs con el target en posición 0.
    """
    positions = _sample_positions(n_objects, rng)
    specs = [SimObjectSpec(0, positions[0], target_color, target_shape, "large")]
    for i in range(1, n_objects):
        color = _DISTRACTOR_COLORS[int(rng.integers(0, len(_DISTRACTOR_COLORS)))]
        shape = (_SHAPE_POOL[int(rng.integers(0, len(_SHAPE_POOL)))]
                 if with_shapes else "cube")
        specs.append(SimObjectSpec(i, positions[i], color, shape, "large"))
    # garantizar al menos 2 formas distintas cuando with_shapes=True
    if with_shapes and n_objects >= 2 and {s.shape for s in specs} == {"cube"}:
        specs[-1].shape = "sphere"
    return specs


def sim_objects_to_views(specs: list[SimObjectSpec]) -> list[ObjectView]:
    """Convierte specs a ObjectView para el Grounder.

    Parámetros
    ----------
    specs : list[SimObjectSpec]
        Especificaciones de objetos de la escena.

    Retorna
    -------
    list[ObjectView]
        Vistas ligeras consumibles por el Grounder.
    """
    return [ObjectView(obj_id=s.obj_id, centroid=tuple(s.position),
                       attributes={"color": s.color, "shape": s.shape, "size": s.size})
            for s in specs]


def select_sim_target(instruction: str, specs: list[SimObjectSpec],
                      parser_backend: str = "deterministic",
                      grounder_method: str = "attribute"
                      ) -> tuple[Optional[SimObjectSpec], GroundingResult, Instruction]:
    """Groundea la instrucción contra los specs. Devuelve (spec|None, grounding, instr).

    Parámetros
    ----------
    instruction : str
        Instrucción en lenguaje natural (p.ej. "pick the red cube").
    specs : list[SimObjectSpec]
        Objetos presentes en la escena.
    parser_backend : str
        Backend del parser ("deterministic" | "llm_local" | "llm_api").
    grounder_method : str
        Método del grounder ("attribute" | "clip_image").

    Retorna
    -------
    tuple[SimObjectSpec | None, GroundingResult, Instruction]
        El spec seleccionado (o None si no hay match), el resultado del
        grounding y la instrucción estructurada.
    """
    parser = make_parser(parser_backend)
    grounder = Grounder(method=grounder_method)
    instr = parser.parse(instruction)
    res = grounder.ground(instr, sim_objects_to_views(specs))
    by_id = {s.obj_id: s for s in specs}
    chosen = by_id.get(res.target_obj_id) if res.target_obj_id is not None else None
    return chosen, res, instr


def evaluate_selection(specs: list[SimObjectSpec], instruction: str,
                       expected_id: int, parser_backend: str = "deterministic") -> dict:
    """Evalúa si el grounding seleccionó el objeto esperado (métrica pura, sin sim).

    Parámetros
    ----------
    specs : list[SimObjectSpec]
        Objetos presentes en la escena.
    instruction : str
        Instrucción en lenguaje natural.
    expected_id : int
        obj_id del objeto que debería haberse seleccionado.
    parser_backend : str
        Backend del parser a usar.

    Retorna
    -------
    dict
        Diccionario con keys: instruction, expected_id, selected_id,
        correct (bool), ambiguous (bool).
    """
    chosen, res, instr = select_sim_target(instruction, specs, parser_backend)
    sel = chosen.obj_id if chosen is not None else None
    return {"instruction": instruction, "expected_id": expected_id,
            "selected_id": sel, "correct": sel == expected_id,
            "ambiguous": res.ambiguous}
