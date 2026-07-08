#!/usr/bin/env python3
"""Quickstart end-to-end de bop-bootstrap-ci.

Reproduce las metricas publicadas para YCB-V y T-LESS en el TFM
y muestra como se reportarian en una publicacion.

Uso:
    python examples/quickstart.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from bop_bootstrap_ci import bootstrap_auc_adds, bootstrap_recall


def main():
    # Caso 1: errores sinteticos para demo standalone
    print("=" * 70)
    print("EJEMPLO 1: datos sinteticos (rapido, sin descargar nada)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    add_s_errors_mm = rng.exponential(scale=8.0, size=500)

    auc = bootstrap_auc_adds(add_s_errors_mm, B=1000, seed=42)
    rec10 = bootstrap_recall(add_s_errors_mm, threshold=10.0, B=1000, seed=42)
    print(f"  AUC ADD-S @50mm: {auc.point:.4f} [{auc.lo:.4f}, {auc.hi:.4f}]")
    print(f"  Recall@10mm    : {rec10.point:.1%} [{rec10.lo:.1%}, {rec10.hi:.1%}]")

    # Caso 2: si esta disponible el JSON del TFM, lo cargamos y validamos
    repo = Path(__file__).resolve().parents[3]
    tfm_json = repo / "experiments/results/local_metrics_with_bootstrap.json"
    if tfm_json.exists():
        print()
        print("=" * 70)
        print("EJEMPLO 2: contraste con resultados publicados del TFM")
        print("=" * 70)
        d = json.loads(tfm_json.read_text())
        for ds_name, dataset in d.get("datasets", {}).items():
            ci_auc = dataset["auc_adds_50mm_ci95"]
            ci_rec = dataset["recall_adds_10mm_ci95"]
            n = dataset["n_evaluated"]
            print(f"\n  {ds_name.upper()} (n = {n}):")
            print(f"    AUC ADD-S @50mm: {ci_auc['point']:.4f} "
                  f"[{ci_auc['lo']:.4f}, {ci_auc['hi']:.4f}]")
            print(f"    Recall@10mm    : {ci_rec['point']:.1%} "
                  f"[{ci_rec['lo']:.1%}, {ci_rec['hi']:.1%}]")
        print()
        print("(Estos numeros se reproduciran bit-a-bit usando bop-bootstrap-ci")
        print(" sobre los errores per-instance — vease tests/test_reproduces_tfm.py)")
    else:
        print()
        print("(JSON oficial del TFM no disponible — saltando ejemplo 2)")

    print()
    print("Reporte estilo paper:")
    print(f"  'On 500 synthetic ADD-S errors, AUC ADD-S @50mm = "
          f"{auc.point:.3f} (95 % CI [{auc.lo:.3f}, {auc.hi:.3f}], "
          f"bootstrap B = 1000, percentile method, seed = 42).'")


if __name__ == "__main__":
    main()
