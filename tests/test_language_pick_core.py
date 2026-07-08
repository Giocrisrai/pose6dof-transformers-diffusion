"""Tests del núcleo puro de language_pick (sin CoppeliaSim)."""
import numpy as np

from src.simulation.language_pick import (
    SimObjectSpec,
    color_name_from_rgb,
    evaluate_selection,
    plan_language_scene,
    select_sim_target,
    sim_objects_to_views,
)


def test_color_name_from_rgb():
    assert color_name_from_rgb((0.85, 0.15, 0.15)) == "red"
    assert color_name_from_rgb((0.15, 0.30, 0.85)) == "blue"
    assert color_name_from_rgb((0.20, 0.75, 0.20)) == "green"


def test_color_name_from_rgb_miss():
    from src.simulation.language_pick import color_name_from_rgb
    assert color_name_from_rgb((0.5, 0.5, 0.5)) is None     # gris, fuera de paleta
    assert color_name_from_rgb((0.84, 0.16, 0.15)) == "red" # dentro de tolerancia


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


def test_plan_scene_target_shape_no_cubo_mantiene_variedad():
    rng = np.random.default_rng(7)
    specs = plan_language_scene(rng, n_objects=3, with_shapes=True,
                                target_color="red", target_shape="sphere")
    assert specs[0].shape == "sphere"
    assert len({s.shape for s in specs}) >= 2


def test_plan_scene_rechaza_n_objects_fuera_de_rango():
    import pytest
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        plan_language_scene(rng, n_objects=9)


import pytest


@pytest.mark.integration
def test_run_language_pick_smoke():
    # Requiere CoppeliaSim en :23000. Valida el contrato del payload.
    from src.simulation.language_pick import run_language_pick
    out = run_language_pick("dame el cubo rojo", n_objects=3, seed=0, render=False)
    assert "grounding" in out and "selection_correct" in out


def test_spec_matches_instruction_honesto():
    from src.language import make_parser
    from src.simulation.language_pick import spec_matches_instruction
    p = make_parser("deterministic")
    red_cube = SimObjectSpec(0, (0, 0, 0.5), "red", "cube", "large")
    assert spec_matches_instruction(red_cube, p.parse("pick the red cube")) is True
    assert spec_matches_instruction(red_cube, p.parse("pick the blue cube")) is False
    assert spec_matches_instruction(red_cube, p.parse("the red sphere")) is False
    assert spec_matches_instruction(None, p.parse("pick the red cube")) is False
    # instrucción sin atributos -> no es match significativo
    assert spec_matches_instruction(red_cube, p.parse("")) is False
