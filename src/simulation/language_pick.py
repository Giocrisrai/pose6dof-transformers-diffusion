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
# Inverso nombre→RGB: lo usa el cascarón sim (apply_scene) para pintar primitivas.
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
        Número total de objetos (target + distractores). Rango soportado: 1..6.
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
    if not (1 <= n_objects <= 6):
        raise ValueError(f"n_objects fuera de rango soportado (1..6): {n_objects}")
    positions = _sample_positions(n_objects, rng)
    specs = [SimObjectSpec(0, positions[0], target_color, target_shape, "large")]
    for i in range(1, n_objects):
        color = _DISTRACTOR_COLORS[int(rng.integers(0, len(_DISTRACTOR_COLORS)))]
        shape = (_SHAPE_POOL[int(rng.integers(0, len(_SHAPE_POOL)))]
                 if with_shapes else "cube")
        specs.append(SimObjectSpec(i, positions[i], color, shape, "large"))
    # Garantiza variedad de formas: si todas coincidieron, cambia el último distractor.
    if with_shapes and n_objects >= 2 and len({s.shape for s in specs}) < 2:
        specs[-1].shape = "sphere" if specs[-1].shape != "sphere" else "cylinder"
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


# ── Cascarón SIM ────────────────────────────────────────────────────────────
# Imports del bridge y pick son PEREZOSOS (dentro de las funciones) para que
# el módulo sea importable sin CoppeliaSim instalado.

# Tamaño de primitiva por 'size' (lado en metros)
_SIZE_M = {"small": 0.04, "large": 0.06}
_SHAPE_PARAM = {"cube": "primitiveshape_cuboid",
                "sphere": "primitiveshape_spheroid",
                "cylinder": "primitiveshape_cylinder"}


def apply_scene(bridge, specs: list[SimObjectSpec]) -> list[SimObjectSpec]:
    """Crea/pinta primitivas en CoppeliaSim según specs. Rellena handle/alias.

    Patrón coherente con collect_diffusion_dataset_v9_clutter.py. Aparca los
    /object_* preexistentes para que no interfieran con el pick.

    Parámetros
    ----------
    bridge : CoppeliaSimBridge
        Conexión activa al simulador.
    specs : list[SimObjectSpec]
        Especificaciones de objetos a crear en la escena.

    Retorna
    -------
    list[SimObjectSpec]
        Los mismos specs con handle y alias rellenos tras la creación.
    """
    sim = bridge.sim
    try:
        from src.simulation.multi_object_scene import _list_existing_cubes, PARK_POSITION
        for h in _list_existing_cubes(sim):
            sim.setObjectPosition(h, -1, list(PARK_POSITION))
    except Exception:
        pass
    for s in specs:
        side = _SIZE_M.get(s.size, 0.06)
        ptype = getattr(sim, _SHAPE_PARAM[s.shape])
        h = sim.createPrimitiveShape(ptype, [side, side, side], 0)
        alias = f"object_lang_{s.obj_id}"
        sim.setObjectAlias(h, alias)
        sim.setObjectPosition(h, -1, [float(s.position[0]), float(s.position[1]),
                                      float(s.position[2])])
        rgb = _NAME_TO_RGB.get(s.color, (0.5, 0.5, 0.5))
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(rgb))
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        s.handle = h
        s.alias = f"/{alias}"
    return specs


def run_language_pick(instruction: str, scene: str = "multi",
                      parser_backend: str = "deterministic", render: bool = False,
                      n_objects: int = 3, with_shapes: Optional[bool] = None,
                      seed: int = 42) -> dict:
    """Orquesta el pick guiado por lenguaje en CoppeliaSim (ruta integration).

    Requiere CoppeliaSim en :23000. Construye la escena, groundea la
    instrucción, ejecuta el pick sobre el objeto elegido y devuelve un payload
    con parsing, grounding, escena, selection_correct y métricas del pick.

    Parámetros
    ----------
    instruction : str
        Instrucción en lenguaje natural (p.ej. "dame el cubo rojo").
    scene : str
        Tipo de escena ("multi" | "clutter"). Si "clutter", with_shapes=True.
    parser_backend : str
        Backend del parser ("deterministic" | "llm_local" | "llm_api").
    render : bool
        Si True, compila frames en MP4 tras el pick.
    n_objects : int
        Número total de objetos (target + distractores). Rango 1..6.
    with_shapes : bool | None
        Si None, se infiere: True cuando scene=="clutter", False en otro caso.
    seed : int
        Semilla para el generador aleatorio (determinismo de la escena).

    Retorna
    -------
    dict
        Payload con claves: instruction, parsed, grounding, scene,
        selection_correct, pick (o None), mp4_path.

    Raises
    ------
    ConnectionError
        Si CoppeliaSim no está accesible en localhost:23000.
    """
    from pathlib import Path as _Path
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
    from src.simulation.pick_sequence import run_pick_sequence, compile_mp4

    REPO = _Path(__file__).resolve().parents[2]
    scenes_dir = REPO / "data" / "scenes"
    out_dir = REPO / "experiments" / "results" / "language_pick"
    frames_dir = out_dir / "frames"

    if with_shapes is None:
        with_shapes = (scene == "clutter")
    rng = np.random.default_rng(seed)
    specs = plan_language_scene(rng, n_objects, with_shapes)

    with CoppeliaSimBridge() as bridge:
        bridge.set_stepping(True)
        bridge.load_scene(scenes_dir / "bin_base.ttt")
        apply_scene(bridge, specs)
        chosen, grounding, instr = select_sim_target(instruction, specs, parser_backend)
        payload = {
            "instruction": instruction,
            "parsed": {
                "color": instr.target.color,
                "shape": instr.target.shape,
                "size": instr.target.size,
                "spatial": instr.spatial.relation if instr.spatial else None,
                "backend": instr.backend,
            },
            "grounding": {
                "target_obj_id": grounding.target_obj_id,
                "method": grounding.method,
                "ambiguous": grounding.ambiguous,
                "scores": grounding.scores,
            },
            "scene": [
                {"obj_id": s.obj_id, "color": s.color, "shape": s.shape,
                 "pos": list(s.position)}
                for s in specs
            ],
        }
        if chosen is None:
            payload["selection_correct"] = False
            payload["pick"] = None
            return payload
        result = run_pick_sequence(
            bridge, frames_dir,
            target_object=chosen.alias,
            pose_override_xyz=list(chosen.position),
        )

    mp4 = (compile_mp4(frames_dir, out_dir / "language_pick.mp4", fps=25)
           if render else None)
    payload["selection_correct"] = (grounding.target_obj_id == 0)
    payload["pick"] = {
        "tip_grasp_proximity_m": round(result.tip_grasp_proximity_m, 3),
        "object_moved_m": round(result.obj_moved_m, 3),
        "grasp_plausible": result.grasp_plausible,
        "ik_converged": result.ik_converged,
    }
    payload["mp4_path"] = str(mp4.relative_to(REPO)) if mp4 else None
    return payload


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
