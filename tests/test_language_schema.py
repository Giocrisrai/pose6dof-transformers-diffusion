"""Tests para src/language/schema.py — dataclasses puras."""
from src.language.schema import (
    GroundingResult,
    Instruction,
    ObjectView,
    TargetSpec,
)


def test_targetspec_defaults_son_none():
    t = TargetSpec()
    assert t.color is None and t.shape is None and t.size is None
    assert t.raw_noun is None
    assert t.is_empty()


def test_targetspec_no_vacio_con_atributo():
    assert not TargetSpec(color="red").is_empty()


def test_instruction_minima():
    instr = Instruction(raw_text="pick the red cube", target=TargetSpec(color="red", shape="cube"))
    assert instr.intent == "pick"          # default
    assert instr.steps == []               # default
    assert instr.spatial is None
    assert instr.confidence == 1.0
    assert instr.backend == "unknown"


def test_objectview_y_grounding_result():
    ov = ObjectView(obj_id=2, centroid=(0.1, 0.2, 0.3), attributes={"color": "red"})
    assert ov.obj_id == 2
    g = GroundingResult(target_obj_id=2, scores={2: 0.9, 1: 0.1},
                        method="attribute", rejected=[1], ambiguous=False)
    assert g.target_obj_id == 2 and not g.ambiguous
