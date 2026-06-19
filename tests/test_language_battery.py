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
