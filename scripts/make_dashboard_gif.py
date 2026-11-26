#!/usr/bin/env python3
"""Regenera docs/media/dashboard.gif a partir del dashboard ejecutivo actual.

Toma una captura de página completa de docs/dashboard_executive.html con Chrome
sin ventana y la recorta en una secuencia de vistas (cabecera, H1, H2, H3,
robustez, extensión y garantías de CI). Así el GIF del README no queda con
cifras antiguas: basta volver a ejecutarlo cuando cambie el dashboard.

Uso:
    python scripts/make_dashboard_gif.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PAGINA = REPO / "docs/dashboard_executive.html"
SALIDA = REPO / "docs/media/dashboard.gif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ANCHO, ALTO_PAGINA = 1100, 5200
ALTO_VISTA = 566                      # misma proporción que el GIF (960 x 494)
TAMANO_GIF = (960, 494)
# desplazamiento vertical de cada vista, en píxeles de la captura
VISTAS = (0, 450, 1000, 1560, 1980, 2490, 4130)
MS_POR_VISTA = 1800


def capturar(destino: Path) -> None:
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=1", f"--window-size={ANCHO},{ALTO_PAGINA}",
         "--virtual-time-budget=4000", f"--screenshot={destino}", PAGINA.as_uri()],
        check=True, capture_output=True,
    )


def main() -> int:
    if not Path(CHROME).exists():
        print("No se encontró Google Chrome", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        captura = Path(tmp) / "pagina.png"
        capturar(captura)
        pagina = Image.open(captura).convert("RGB")
    cuadros = [
        pagina.crop((0, y, ANCHO, y + ALTO_VISTA)).resize(TAMANO_GIF, Image.Resampling.LANCZOS)
        for y in VISTAS
    ]
    paleta = [c.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
              for c in cuadros]
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    paleta[0].save(SALIDA, save_all=True, append_images=paleta[1:], duration=MS_POR_VISTA,
                   loop=0, optimize=True)
    print(f"{SALIDA.relative_to(REPO)}: {len(paleta)} vistas, {SALIDA.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
