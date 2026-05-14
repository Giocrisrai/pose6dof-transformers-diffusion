# bop-bootstrap-ci

> Bootstrap percentile confidence intervals (95 %, B = 1000) for **BOP Challenge**
> 6-DoF object pose estimation metrics: ADD, ADD-S, AUC, Recall@N mm.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![BOP Challenge](https://img.shields.io/badge/BOP-Challenge-orange.svg)](https://bop.felk.cvut.cz/challenges/)

## Why this exists

BOP submissions typically report **point estimates** of AUC ADD-S, Recall@10 mm
and similar metrics, but rarely include **confidence intervals**. This package
provides a reproducible, semi-standard bootstrap percentile CI (B = 1000, 95 %)
over any user-supplied set of per-instance pose errors — turning a single number
into a defensible range that supports statistical comparisons across methods.

It was extracted from a TFM at UNIR (2026) on bin-picking with FoundationPose +
Diffusion Policy, where bootstrap CIs were used to validate AUC ADD-S
0.908 [0.901, 0.916] on YCB-Video and 0.957 [0.954, 0.959] on T-LESS.

## Install

```bash
pip install bop-bootstrap-ci
```

## Quick start

```python
import numpy as np
from bop_bootstrap_ci import bootstrap_auc_adds, bootstrap_recall

# Your per-instance ADD-S errors in millimetres, from your BOP run
add_s_errors_mm = np.array([2.1, 5.3, 8.9, 1.4, 22.0, 7.5, 3.2, 14.8, ...])

auc = bootstrap_auc_adds(add_s_errors_mm, max_threshold_mm=50.0, B=1000, seed=42)
print(f"AUC ADD-S @50mm: {auc.point:.4f} [{auc.lo:.4f}, {auc.hi:.4f}]")

recall10 = bootstrap_recall(add_s_errors_mm, threshold=10.0, B=1000, seed=42)
print(f"Recall ADD-S @10mm: {recall10.point:.1%} "
      f"[{recall10.lo:.1%}, {recall10.hi:.1%}]")
```

## API

### `bootstrap_ci(values, statistic=np.mean, B=1000, alpha=0.05, seed=42)`

Generic percentile bootstrap CI for any 1-D array and statistic. Returns
a `BootstrapResult` with fields `.point`, `.lo`, `.hi`, `.B`, `.alpha`.

### `bootstrap_recall(errors, threshold, ...)`

Bootstrap CI for the fraction of `errors` strictly below `threshold`. Equivalent
to `Recall@N mm` as defined in BOP.

### `bootstrap_auc_adds(errors, max_threshold_mm=50.0, ...)`

Bootstrap CI for the Area Under the Recall Curve in `[0, max_threshold_mm]`,
normalised to `[0, 1]`. The standard BOP AUC ADD-S @50 mm convention.

### `recall_at_threshold(errors, threshold)` and `auc_from_errors(errors, ...)`

Direct (non-bootstrapped) point estimates.

## Compatibility with `bop_toolkit`

This package consumes raw per-instance errors as `numpy.ndarray`. To use it
with the official BOP toolkit, run your evaluation as usual, then pass the
resulting per-instance ADD/ADD-S errors to the bootstrap functions:

```python
from bop_toolkit_lib import inout, pose_error
# ... compute add_s_errors_mm as you normally would ...

from bop_bootstrap_ci import bootstrap_auc_adds
ci = bootstrap_auc_adds(add_s_errors_mm).as_dict()
print(ci)  # {"point": 0.908, "lo": 0.901, "hi": 0.916, "B": 1000, "alpha": 0.05}
```

## Reproducibility

All public functions are deterministic given the `seed` argument and reproduce
the underlying TFM's results bit-exactly. Default seed = 42, default B = 1000,
default alpha = 0.05 (95 % CI), default method = percentile.

## Limitations

- **Percentile method only**, not BCa. BCa would correct for skew but is rarely
  reported in BOP-style papers and would break compatibility with our TFM.
- **i.i.d. assumption**: the bootstrap assumes per-instance errors are drawn
  i.i.d. from the underlying error distribution. If your evaluation has strong
  per-scene correlations, consider a block bootstrap instead.
- **No multiple-comparisons correction** built in. If comparing many methods,
  apply Bonferroni or Holm-Sidak externally.

## Citation

If you use this in academic work, please cite the underlying TFM:

```bibtex
@mastersthesis{godoy2026pose6dof,
  title  = {Estimaci\'on de Pose 6-DoF mediante Transformers y Modelos de
            Difusi\'on para Bin Picking Rob\'otico},
  author = {Godoy Bonillo, Giocrisrai and Carrasco, Jos\'e Miguel},
  school = {Universidad Internacional de La Rioja (UNIR)},
  year   = {2026},
  url    = {https://github.com/Giocrisrai/pose6dof-transformers-diffusion}
}
```

## License

MIT.
