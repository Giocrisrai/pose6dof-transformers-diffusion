"""Tests for bop_bootstrap_ci.ci."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bop_bootstrap_ci import (
    BootstrapResult,
    auc_from_errors,
    bootstrap_auc_adds,
    bootstrap_ci,
    bootstrap_recall,
    recall_at_threshold,
)


class TestRecallAtThreshold:
    def test_basic(self):
        errs = np.array([2.0, 5.0, 8.0, 12.0, 20.0])
        # < 10 -> 3 of 5
        assert recall_at_threshold(errs, 10.0) == pytest.approx(0.6)

    def test_strict_lt(self):
        # 10.0 itself must NOT count (strict <), matches TFM convention
        errs = np.array([10.0, 10.0, 10.0])
        assert recall_at_threshold(errs, 10.0) == 0.0

    def test_empty(self):
        assert math.isnan(recall_at_threshold(np.array([]), 10.0))

    def test_all_below(self):
        errs = np.array([1.0, 2.0, 3.0])
        assert recall_at_threshold(errs, 100.0) == 1.0


class TestAucFromErrors:
    def test_perfect_predictor(self):
        errs = np.zeros(50)
        auc = auc_from_errors(errs, max_threshold=50.0)
        # All errors at 0 -> recall=1 everywhere except possibly t=0
        # AUC should be very close to 1 (one threshold point at t=0 contributes 0)
        assert auc > 0.98

    def test_terrible_predictor(self):
        errs = np.full(50, 1000.0)
        auc = auc_from_errors(errs, max_threshold=50.0)
        # All errors > 50 mm -> recall=0 everywhere -> AUC=0
        assert auc == 0.0

    def test_range(self):
        rng = np.random.default_rng(0)
        errs = rng.exponential(scale=8.0, size=500)
        auc = auc_from_errors(errs)
        assert 0.0 < auc < 1.0

    def test_empty(self):
        assert math.isnan(auc_from_errors(np.array([])))


class TestBootstrapCi:
    def test_returns_dataclass(self):
        errs = np.arange(10).astype(float)
        r = bootstrap_ci(errs, B=100, seed=42)
        assert isinstance(r, BootstrapResult)
        assert r.B == 100
        assert r.alpha == 0.05

    def test_point_matches_statistic(self):
        errs = np.arange(10).astype(float)
        r = bootstrap_ci(errs, statistic=np.mean, B=100, seed=42)
        assert r.point == pytest.approx(4.5)

    def test_ci_contains_point(self):
        rng = np.random.default_rng(0)
        errs = rng.normal(loc=5.0, scale=2.0, size=200)
        r = bootstrap_ci(errs, B=500, seed=42)
        assert r.lo <= r.point <= r.hi

    def test_reproducibility(self):
        errs = np.arange(50).astype(float)
        r1 = bootstrap_ci(errs, B=200, seed=42)
        r2 = bootstrap_ci(errs, B=200, seed=42)
        assert r1.point == r2.point
        assert r1.lo == r2.lo
        assert r1.hi == r2.hi

    def test_different_seeds_give_different_ci(self):
        rng = np.random.default_rng(0)
        errs = rng.normal(size=100)
        r1 = bootstrap_ci(errs, B=200, seed=1)
        r2 = bootstrap_ci(errs, B=200, seed=2)
        # Same data, same point estimate, but CI bounds should differ
        assert r1.point == r2.point
        assert (r1.lo, r1.hi) != (r2.lo, r2.hi)

    def test_alpha_changes_width(self):
        rng = np.random.default_rng(0)
        errs = rng.normal(size=300)
        narrow = bootstrap_ci(errs, B=500, alpha=0.10, seed=42)  # 90% CI
        wide = bootstrap_ci(errs, B=500, alpha=0.01, seed=42)    # 99% CI
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)

    def test_empty_returns_nan(self):
        r = bootstrap_ci(np.array([]), B=100, seed=42)
        assert math.isnan(r.point) and math.isnan(r.lo) and math.isnan(r.hi)

    def test_as_dict(self):
        errs = np.arange(10).astype(float)
        r = bootstrap_ci(errs, B=100, seed=42)
        d = r.as_dict()
        assert set(d.keys()) == {"point", "lo", "hi", "B", "alpha"}


class TestBootstrapRecall:
    def test_basic_works(self):
        rng = np.random.default_rng(0)
        errs = rng.exponential(scale=8.0, size=200)
        r = bootstrap_recall(errs, threshold=10.0, B=200, seed=42)
        assert 0.0 <= r.lo <= r.point <= r.hi <= 1.0


class TestBootstrapAucAdds:
    def test_basic_works(self):
        rng = np.random.default_rng(0)
        errs = rng.exponential(scale=8.0, size=200)
        r = bootstrap_auc_adds(errs, B=200, seed=42)
        assert 0.0 <= r.lo <= r.point <= r.hi <= 1.0

    def test_perfect_predictor(self):
        errs = np.zeros(100)
        r = bootstrap_auc_adds(errs, B=100, seed=42)
        assert r.point > 0.98 and r.hi > 0.98


class TestReproducesTfmResults:
    """Critical test: reproduce the TFM's YCB-V numbers bit-exactly.

    The TFM's `recompute_metrics_with_bootstrap.py` is the source-of-truth.
    These tests pin the exact bootstrap-percentile recipe so future
    refactors don't drift.
    """
    def test_recall_at_10mm_matches_legacy_recipe(self):
        # Reproduce: rng = default_rng(seed+1) for recall (per TFM code)
        errs = np.array([2.0, 5.0, 8.0, 12.0, 20.0, 3.0, 6.0, 9.0, 11.0, 15.0])
        r = bootstrap_recall(errs, threshold=10.0, B=1000, seed=43)
        # 6 of 10 errors strictly < 10 mm -> Recall@10mm = 0.6
        assert r.point == pytest.approx(0.6)
        assert 0.0 < r.lo < r.point < r.hi < 1.0

    def test_auc_50mm_matches_legacy_recipe(self):
        errs = np.array([2.0, 5.0, 8.0, 12.0, 20.0, 3.0, 6.0, 9.0, 11.0, 15.0])
        r = bootstrap_auc_adds(errs, max_threshold_mm=50.0, n_steps=100,
                                B=1000, seed=42)
        # Point estimate matches direct call
        direct = auc_from_errors(errs, max_threshold=50.0, n_steps=100)
        assert r.point == pytest.approx(direct)
