"""Garantía de consistencia de exp27 (text-to-CAD).

Ejecuta el verificador que cruza los números entre reports JSON, README,
Entrega 4 (docx) y dashboard, comprueba el determinismo del CAD y la
plausibilidad de la física registrada. La física EN VIVO (CoppeliaSim) es
opt-in vía EXP27_LIVE_PHYSICS=1 y se omite aquí.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "experiments/results/exp27_text_to_cad/verify_consistency.py"


def test_exp27_consistency():
    r = subprocess.run(
        [sys.executable, str(VERIFIER)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert r.returncode == 0, (
        f"Verificador de consistencia FALLÓ (rc={r.returncode}):\n"
        f"{r.stdout}\n{r.stderr}"
    )
    assert "comprobaciones OK" in r.stdout, r.stdout
