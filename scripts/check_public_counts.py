#!/usr/bin/env python3
"""Comprueba que las cifras de garantía publicadas coinciden con lo que mide el CI.

El README y los dos dashboards citan cuántos tests pasan, cuántas comprobaciones
de consistencia hay y cuántos archivos tipa mypy. Esas cifras se escriben a mano,
así que se quedan atrás cuando la suite crece. Este script las compara con las
salidas reales de la ejecución y falla si alguna no coincide.

Uso (lo llama .github/workflows/ci.yml):

    pytest tests/ --junitxml=reports/tests.xml
    pytest packages/bop_bootstrap_ci/tests/ --junitxml=reports/paquete.xml
    mypy ... | tee reports/mypy.txt
    python scripts/verify_tfm_consistency.py | tee reports/consistencia.txt
    python scripts/check_public_counts.py reports/
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def tests_que_pasan(junit: Path) -> int:
    """Tests que pasan en un informe JUnit: total - fallos - errores - omitidos."""
    total = 0
    for suite in ET.parse(junit).getroot().iter("testsuite"):
        a = suite.attrib
        total += int(a["tests"]) - int(a["failures"]) - int(a["errors"]) - int(a["skipped"])
    return total


def medido(reports: Path) -> dict[str, int]:
    consistencia = re.search(r"(\d+)/(\d+) comprobaciones OK", (reports / "consistencia.txt").read_text())
    mypy = re.search(r"no issues found in (\d+) source files", (reports / "mypy.txt").read_text())
    if not consistencia or not mypy:
        raise SystemExit("No se encontraron los totales en consistencia.txt o mypy.txt")
    return {
        "tests": tests_que_pasan(reports / "tests.xml") + tests_que_pasan(reports / "paquete.xml"),
        "checks": int(consistencia.group(2)),
        "mypy": int(mypy.group(1)),
    }


# (archivo, patrón con un grupo numérico, cifra medida con la que debe coincidir)
PUBLICADO = [
    ("README.md", r"badge/tests-(\d+)%20passing", "tests"),
    ("README.md", r"# (\d+) passing", "tests"),
    ("README.es.md", r"badge/tests-(\d+)%20passing", "tests"),
    ("README.es.md", r"# (\d+) passing", "tests"),
    ("docs/dashboard_executive.html", r'k-val">(\d+)<small> tests', "tests"),
    ("docs/dashboard_executive.html", r'mono">(\d+)</span> passed', "tests"),
    ("docs/dashboard_executive.html", r"(\d+) consistency checks", "checks"),
    ("docs/dashboard_executive.html", r'mono">\d+/(\d+)</span> checks', "checks"),
    ("docs/dashboard_executive.html", r"errors · (\d+) files", "mypy"),
    ("docs/dashboard_ejecutivo.html", r'k-val">(\d+)<small> tests', "tests"),
    ("docs/dashboard_ejecutivo.html", r'mono">(\d+)</span> passed', "tests"),
    ("docs/dashboard_ejecutivo.html", r"(\d+) checks de consistencia", "checks"),
    ("docs/dashboard_ejecutivo.html", r'mono">\d+/(\d+)</span> checks', "checks"),
    ("docs/dashboard_ejecutivo.html", r"errores · (\d+) files", "mypy"),
]


def main(reports: Path) -> int:
    real = medido(reports)
    print(f"Medido en esta ejecución: {real}")
    fallos = 0
    for archivo, patron, clave in PUBLICADO:
        encontrado = re.search(patron, (REPO / archivo).read_text())
        if not encontrado:
            print(f"  [FALLA] {archivo}: no aparece el patrón {patron!r}")
            fallos += 1
        elif int(encontrado.group(1)) != real[clave]:
            print(f"  [FALLA] {archivo}: publica {encontrado.group(1)} {clave}, se midieron {real[clave]}")
            fallos += 1
        else:
            print(f"  [PASS] {archivo}: {clave} = {real[clave]}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "reports"))
