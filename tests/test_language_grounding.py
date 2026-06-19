"""Tests del Grounder (selección de target). Base 100% determinista."""
from src.language import make_parser
from src.language.grounding import Grounder
from src.language.schema import ObjectView


def _escena():
    # 3 objetos con atributos conocidos (como vendrían de metadatos de sim o CLIP)
    return [
        ObjectView(0, centroid=(-0.20, 0.0, 0.5), attributes={"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, centroid=( 0.00, 0.0, 0.5), attributes={"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, centroid=( 0.20, 0.0, 0.5), attributes={"color": "red", "shape": "sphere", "size": "small"}),
    ]


def test_selecciona_por_color_y_forma():
    g = Grounder(method="attribute")
    instr = make_parser().parse("pick the red cube")
    res = g.ground(instr, _escena())
    assert res.target_obj_id == 0
    assert not res.ambiguous


def test_relacion_espacial_desempata():
    # dos rojos; "el rojo de la izquierda" -> obj_id 0 (x más negativo)
    g = Grounder(method="attribute")
    instr = make_parser().parse("dame el rojo de la izquierda")
    res = g.ground(instr, _escena())
    assert res.target_obj_id == 0


def test_nearest_por_profundidad():
    g = Grounder(method="attribute")
    objs = [
        ObjectView(0, centroid=(0.0, 0.0, 0.8), attributes={"color": "red"}),
        ObjectView(1, centroid=(0.1, 0.0, 0.4), attributes={"color": "red"}),
    ]
    instr = make_parser().parse("the nearest red object")
    res = g.ground(instr, objs)
    assert res.target_obj_id == 1   # menor z = más cercano


def test_ambiguedad_detectada():
    g = Grounder(method="attribute")
    objs = [
        ObjectView(0, centroid=(-0.1, 0, 0.5), attributes={"color": "red", "shape": "cube"}),
        ObjectView(1, centroid=( 0.1, 0, 0.5), attributes={"color": "red", "shape": "cube"}),
    ]
    instr = make_parser().parse("pick the red cube")
    res = g.ground(instr, objs)
    assert res.ambiguous          # dos candidatos empatados, sin relación espacial


def test_sin_match_devuelve_none():
    g = Grounder(method="attribute")
    instr = make_parser().parse("pick the green cylinder")
    res = g.ground(instr, _escena())
    assert res.target_obj_id is None
