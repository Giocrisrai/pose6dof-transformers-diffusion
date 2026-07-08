"""Bootstrap percentile confidence intervals for BOP-style metrics.

Public API
----------
- BootstrapResult: dataclass with .point, .lo, .hi, .B, .alpha
- bootstrap_ci: generic percentile bootstrap for any statistic over a 1-D array.
- recall_at_threshold: Recall@N mm from a vector of per-instance errors.
- auc_from_errors: Area Under the Recall Curve in [0, max_threshold].
- bootstrap_recall: convenience CI for Recall@N mm.
- bootstrap_auc_adds: convenience CI for AUC ADD-S (or any AUC of recall vs threshold).

All functions are deterministic given a seed and reproduce the TFM's
`recompute_metrics_with_bootstrap.py` output bit-exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap percentile CI.

    Attributes
    ----------
    point : float
        Point estimate of the statistic computed on the original sample.
    lo : float
        Lower bound of the (1 - alpha) percentile CI.
    hi : float
        Upper bound of the (1 - alpha) percentile CI.
    B : int
        Number of bootstrap resamples used.
    alpha : float
        Significance level (e.g. 0.05 for a 95 % CI).
    """
    point: float
    lo: float
    hi: float
    B: int
    alpha: float

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "B": self.B,
            "alpha": self.alpha,
        }


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Generic percentile bootstrap CI for a 1-D array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of per-instance values (errors, recalls, etc.).
    statistic : callable
        Function that maps an array to a scalar. Default: mean.
    B : int
        Number of bootstrap resamples. Default 1000.
    alpha : float
        Significance level. 0.05 yields a 95 % CI. Default 0.05.
    seed : int
        Seed for `np.random.default_rng`. Default 42 (reproducibility).

    Returns
    -------
    BootstrapResult

    Notes
    -----
    Uses the percentile method (not BCa) — same convention as the
    accompanying TFM and as is commonly reported in BOP-style papers.
    """
    values = np.asarray(values)
    n = values.size
    if n == 0:
        nan = float("nan")
        return BootstrapResult(point=nan, lo=nan, hi=nan, B=B, alpha=alpha)

    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=np.float64)
    for i in range(B):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = float(statistic(sample))

    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    point = float(statistic(values))
    return BootstrapResult(point=point, lo=lo, hi=hi, B=B, alpha=alpha)


def recall_at_threshold(errors: np.ndarray, threshold: float) -> float:
    """Fraction of errors strictly below `threshold`.

    Convention matches the TFM (strict `<`, not `<=`), and matches
    `recompute_metrics_with_bootstrap.py` exactly.
    """
    errors = np.asarray(errors)
    if errors.size == 0:
        return float("nan")
    return float(np.mean(errors < threshold))


def auc_from_errors(
    errors: np.ndarray,
    max_threshold: float = 50.0,
    n_steps: int = 100,
) -> float:
    """Area Under the Recall Curve, normalised to [0, 1].

    Computes Recall(t) at n_steps thresholds in [0, max_threshold] and
    integrates with the trapezoidal rule, dividing by max_threshold to
    yield a unitless score in [0, 1].

    Equivalent to the AUC ADD-S @50 mm convention from BOP and the
    TFM's `auc_metric` helper.
    """
    errors = np.asarray(errors)
    if errors.size == 0:
        return float("nan")
    thresholds = np.linspace(0.0, max_threshold, n_steps)
    recalls = np.array([np.mean(errors < t) for t in thresholds])
    # numpy >= 2.0 renamed trapz -> trapezoid; support both transparently
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(recalls, thresholds)
    else:
        integral = np.trapz(recalls, thresholds)  # type: ignore[attr-defined]  # numpy < 2.0
    return float(integral / max_threshold)


def bootstrap_recall(
    errors: np.ndarray,
    threshold: float,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for Recall@threshold.

    Example
    -------
    >>> import numpy as np
    >>> errs = np.array([2.1, 5.3, 8.9, 1.4, 22.0, 7.5])
    >>> r = bootstrap_recall(errs, threshold=10.0, seed=42)
    >>> r.point  # doctest: +SKIP
    0.833...
    """
    return bootstrap_ci(
        errors,
        statistic=lambda x: recall_at_threshold(x, threshold),
        B=B, alpha=alpha, seed=seed,
    )


def bootstrap_auc_adds(
    errors: np.ndarray,
    max_threshold_mm: float = 50.0,
    n_steps: int = 100,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for AUC of the recall curve (typically ADD-S @50 mm).

    Pass `errors` as per-instance ADD-S values in millimetres for the
    standard BOP convention; the function is dimensionally agnostic.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> errs = rng.exponential(scale=8.0, size=200)
    >>> res = bootstrap_auc_adds(errs, seed=0)
    >>> 0.0 < res.point < 1.0
    True
    """
    def _stat(x: np.ndarray) -> float:
        return auc_from_errors(x, max_threshold=max_threshold_mm, n_steps=n_steps)
    return bootstrap_ci(errors, statistic=_stat, B=B, alpha=alpha, seed=seed)
