"""Garantía de consistencia del TFM completo.

Cruza los números canónicos de las hipótesis (AUC ADD-S, cycle p95, recall,
Δ vs baseline, IC95) entre los reports JSON (fuente de verdad), la Entrega 3,
la Entrega 4 y el dashboard. Falla si hay cualquier contradicción.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "scripts/verify_tfm_consistency.py"


def test_tfm_consistency():
    r = subprocess.run(
        [sys.executable, str(VERIFIER)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert r.returncode == 0, (
        f"Consistencia del TFM FALLÓ (rc={r.returncode}):\n{r.stdout}\n{r.stderr}"
    )
    assert "comprobaciones OK" in r.stdout, r.stdout
