"""Bootstrap percentile confidence intervals for BOP 6-DoF pose metrics.

Standalone library extracted from the TFM "Estimacion de Pose 6-DoF mediante
Transformers y Modelos de Difusion para Bin Picking Robotico" (UNIR, 2026).

Why this exists
---------------
BOP Challenge submissions typically report point estimates of AUC ADD-S,
Recall@N mm and similar metrics, but rarely include confidence intervals.
This package provides reproducible, semi-standard bootstrap percentile CIs
(B=1000, 95 %) over any user-supplied set of per-instance pose errors,
turning a single number into a defensible range.

Quick start
-----------
>>> import numpy as np
>>> from bop_bootstrap_ci import bootstrap_auc_adds, bootstrap_recall
>>> add_s_errors_mm = np.array([2.1, 5.3, 8.9, 1.4, 22.0, 7.5])
>>> result = bootstrap_auc_adds(add_s_errors_mm, max_threshold_mm=50.0)
>>> print(f"AUC ADD-S: {result.point:.3f} [{result.lo:.3f}, {result.hi:.3f}]")
"""
from .ci import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_recall,
    bootstrap_auc_adds,
    auc_from_errors,
    recall_at_threshold,
)

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_recall",
    "bootstrap_auc_adds",
    "auc_from_errors",
    "recall_at_threshold",
]

__version__ = "0.1.0"
