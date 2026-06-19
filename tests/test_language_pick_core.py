"""Tests del núcleo puro de language_pick (sin CoppeliaSim)."""
import numpy as np
from src.simulation.language_pick import (
    SimObjectSpec, color_name_from_rgb, plan_language_scene,
    sim_objects_to_views, select_sim_target, evaluate_selection,
)


def test_color_name_from_rgb():
    assert color_name_from_rgb((0.85, 0.15, 0.15)) == "red"
    assert color_name_from_rgb((0.15, 0.30, 0.85)) == "blue"
    assert color_name_from_rgb((0.20, 0.75, 0.20)) == "green"


def test_plan_scene_target_es_primero_y_rojo():
    rng = np.random.default_rng(0)
    specs = plan_language_scene(rng, n_objects=3, with_shapes=False)
    assert len(specs) == 3
    assert specs[0].obj_id == 0
    assert specs[0].color == "red"
    assert all(s.shape == "cube" for s in specs)
    xs = [s.position[0] for s in specs]
    assert len(set(round(x, 4) for x in xs)) == 3


def test_plan_scene_con_formas_incluye_no_cubos():
    rng = np.random.default_rng(1)
    specs = plan_language_scene(rng, n_objects=4, with_shapes=True)
    shapes = {s.shape for s in specs}
    assert shapes <= {"cube", "sphere", "cylinder"}
    assert len(shapes) >= 2


def test_sim_objects_to_views_mapea_atributos():
    specs = [SimObjectSpec(0, (-0.2, 0.0, 0.5), "red", "cube", "large"),
             SimObjectSpec(1, (0.2, 0.0, 0.5), "blue", "cube", "small")]
    views = sim_objects_to_views(specs)
    assert views[0].obj_id == 0
    assert views[0].centroid == (-0.2, 0.0, 0.5)
    assert views[0].attributes == {"color": "red", "shape": "cube", "size": "large"}


def test_select_sim_target_por_color():
    specs = [SimObjectSpec(0, (-0.2, 0.0, 0.5), "red", "cube", "large"),
             SimObjectSpec(1, (0.2, 0.0, 0.5), "blue", "cube", "small")]
    chosen, grounding, instr = select_sim_target("pick the red cube", specs)
    assert chosen.obj_id == 0
    assert grounding.target_obj_id == 0
    assert instr.target.color == "red"


def test_select_sim_target_sin_match():
    specs = [SimObjectSpec(0, (-0.2, 0.0, 0.5), "blue", "cube", "large")]
    chosen, grounding, instr = select_sim_target("pick the red sphere", specs)
    assert chosen is None
    assert grounding.target_obj_id is None


def test_evaluate_selection_correcto_e_incorrecto():
    specs = [SimObjectSpec(0, (-0.2, 0.0, 0.5), "red", "cube", "large"),
             SimObjectSpec(1, (0.2, 0.0, 0.5), "blue", "cube", "small")]
    ok = evaluate_selection(specs, "pick the red cube", expected_id=0)
    assert ok["correct"] is True and ok["selected_id"] == 0
    bad = evaluate_selection(specs, "pick the blue cube", expected_id=0)
    assert bad["correct"] is False and bad["selected_id"] == 1
