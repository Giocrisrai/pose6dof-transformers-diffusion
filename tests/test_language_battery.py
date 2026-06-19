"""Tests de la agregación de selection-accuracy (núcleo puro)."""
import numpy as np
from experiments.run_language_battery import build_cases, aggregate


def test_build_cases_genera_instruccion_y_expected():
    rng = np.random.default_rng(0)
    cases = build_cases(rng, n_scenes=5, n_objects=3, with_shapes=False)
    assert len(cases) == 5
    for c in cases:
        assert "instruction" in c and "expected_id" in c and "specs" in c
        assert c["expected_id"] == 0


def test_aggregate_calcula_accuracy():
    rows = [{"correct": True}, {"correct": True}, {"correct": False}]
    agg = aggregate(rows)
    assert agg["n"] == 3 and agg["n_correct"] == 2
    assert abs(agg["selection_accuracy"] - 2/3) < 1e-9


def test_build_cases_incluye_dificultades():
    rng = np.random.default_rng(3)
    cases = build_cases(rng, n_scenes=6, n_objects=3, with_shapes=False)
    diffs = {c["difficulty"] for c in cases}
    assert diffs == {"color", "shape", "spatial"}


def test_shape_case_requiere_forma_para_desambiguar():
    # En un caso 'shape', hay un distractor del mismo color que el target.
    rng = np.random.default_rng(4)
    cases = build_cases(rng, n_scenes=3, n_objects=3, with_shapes=False)
    shape_case = next(c for c in cases if c["difficulty"] == "shape")
    specs = shape_case["specs"]
    same_color = [s for s in specs if s.color == specs[0].color]
    assert len(same_color) >= 2                  # color no basta
    assert specs[0].shape not in [s.shape for s in same_color[1:]]  # forma sí distingue


def test_spatial_case_target_es_leftmost():
    rng = np.random.default_rng(5)
    cases = build_cases(rng, n_scenes=3, n_objects=3, with_shapes=False)
    sp = next(c for c in cases if c["difficulty"] == "spatial")
    specs = sp["specs"]
    target = next(s for s in specs if s.obj_id == sp["expected_id"])
    assert target.position[0] == min(s.position[0] for s in specs)
