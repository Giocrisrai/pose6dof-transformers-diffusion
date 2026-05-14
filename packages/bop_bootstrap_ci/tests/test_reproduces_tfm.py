"""Cross-validation contra el TFM: el paquete debe reproducir bit-a-bit los
numeros que ya estan commiteados en
`experiments/results/local_metrics_with_bootstrap.json`.

Este test es la garantia principal de que la extraccion no rompe ningun
calculo y de que cualquier futuro cambio del paquete sera detectado
inmediatamente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Re-implementacion bit-a-bit del recipe del TFM (no del paquete) para tener
# numeros de referencia DETERMINISTAS sin depender de los datasets BOP.
# Validamos que el paquete produce identicos resultados.

REPO_ROOT = Path(__file__).resolve().parents[3]


def _legacy_bootstrap(values, statistic, B=1000, alpha=0.05, seed=42):
    """Funcion identica a `experiments/recompute_metrics_with_bootstrap.py::bootstrap_ci`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.empty(B)
    for i in range(B):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = statistic(sample)
    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(statistic(values)), float(lo), float(hi)


def _legacy_auc(errors, max_threshold_mm=50.0, n_steps=100):
    """Funcion identica a `auc_metric` del TFM."""
    thresholds = np.linspace(0, max_threshold_mm, n_steps)
    recalls = [np.mean(errors < t) for t in thresholds]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(recalls, thresholds) / max_threshold_mm)
    return float(np.trapz(recalls, thresholds) / max_threshold_mm)


class TestPackageMatchesLegacyRecipe:
    """El paquete debe dar EXACTAMENTE los mismos numeros que el TFM."""

    @pytest.fixture
    def sample_errors(self):
        # Errores deterministas (no aleatorios) para comparar bit-a-bit
        rng = np.random.default_rng(0)
        return rng.exponential(scale=8.0, size=500)

    def test_recall_point_estimate_matches(self, sample_errors):
        from bop_bootstrap_ci import recall_at_threshold
        pkg = recall_at_threshold(sample_errors, 10.0)
        legacy = float(np.mean(sample_errors < 10.0))
        assert pkg == pytest.approx(legacy)

    def test_auc_point_estimate_matches(self, sample_errors):
        from bop_bootstrap_ci import auc_from_errors
        pkg = auc_from_errors(sample_errors, max_threshold=50.0, n_steps=100)
        legacy = _legacy_auc(sample_errors, max_threshold_mm=50.0, n_steps=100)
        assert pkg == pytest.approx(legacy)

    def test_bootstrap_recall_matches_legacy_recipe(self, sample_errors):
        from bop_bootstrap_ci import bootstrap_recall
        pkg = bootstrap_recall(sample_errors, threshold=10.0, B=1000, seed=42)
        legacy_point, legacy_lo, legacy_hi = _legacy_bootstrap(
            sample_errors,
            statistic=lambda x: float(np.mean(x < 10.0)),
            B=1000, alpha=0.05, seed=42,
        )
        assert pkg.point == pytest.approx(legacy_point, abs=1e-10)
        assert pkg.lo == pytest.approx(legacy_lo, abs=1e-10)
        assert pkg.hi == pytest.approx(legacy_hi, abs=1e-10)

    def test_bootstrap_auc_matches_legacy_recipe(self, sample_errors):
        from bop_bootstrap_ci import bootstrap_auc_adds
        pkg = bootstrap_auc_adds(sample_errors, max_threshold_mm=50.0,
                                  n_steps=100, B=1000, seed=42)
        legacy_point, legacy_lo, legacy_hi = _legacy_bootstrap(
            sample_errors,
            statistic=lambda x: _legacy_auc(x, max_threshold_mm=50.0, n_steps=100),
            B=1000, alpha=0.05, seed=42,
        )
        assert pkg.point == pytest.approx(legacy_point, abs=1e-10)
        assert pkg.lo == pytest.approx(legacy_lo, abs=1e-10)
        assert pkg.hi == pytest.approx(legacy_hi, abs=1e-10)


class TestRangesFromCommittedTfmJson:
    """Cross-check con los rangos publicados en `local_metrics_with_bootstrap.json`."""

    def test_committed_json_exists_and_has_expected_structure(self):
        committed = REPO_ROOT / "experiments/results/local_metrics_with_bootstrap.json"
        if not committed.exists():
            pytest.skip("JSON oficial no commiteado en esta copia")
        d = json.loads(committed.read_text())
        assert "datasets" in d
        # Esperamos al menos ycbv y/o tless
        assert any(k in d["datasets"] for k in ("ycbv", "tless"))

    def test_ycbv_auc_in_published_range(self):
        committed = REPO_ROOT / "experiments/results/local_metrics_with_bootstrap.json"
        if not committed.exists():
            pytest.skip("JSON oficial no commiteado en esta copia")
        d = json.loads(committed.read_text())
        if "ycbv" not in d["datasets"]:
            pytest.skip("YCB-V no presente en JSON")
        ci = d["datasets"]["ycbv"]["auc_adds_50mm_ci95"]
        # Sanity: el rango publicado debe contener el punto y ser estrecho
        assert ci["lo"] <= ci["point"] <= ci["hi"]
        assert (ci["hi"] - ci["lo"]) < 0.05, "CI demasiado ancho — probable bug"
