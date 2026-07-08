import numpy as np

from src.simulation.multi_object_scene import (
    BIN_X_RANGE,
    BIN_Y_RANGE,
    MIN_DIST_M,
    sample_non_overlapping_positions,
)


def test_sample_positions_count():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=5, rng=rng)
    assert positions.shape == (5, 3)


def test_sample_positions_min_dist():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=5, rng=rng)
    for i in range(5):
        for j in range(i + 1, 5):
            d = np.linalg.norm(positions[i, :2] - positions[j, :2])
            assert d >= MIN_DIST_M


def test_sample_positions_within_bin():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=8, rng=rng)
    for p in positions:
        assert BIN_X_RANGE[0] <= p[0] <= BIN_X_RANGE[1]
        assert BIN_Y_RANGE[0] <= p[1] <= BIN_Y_RANGE[1]


def test_sample_positions_8_extreme():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=8, rng=rng, max_retries=200)
    assert positions.shape == (8, 3)
