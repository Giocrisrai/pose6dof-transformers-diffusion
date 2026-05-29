# Demo reel curado + limpieza del repo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generar un demo reel curado (clips anotados por etapa + reel resumen ~75–90 s) con overlays cv2 sobre los MP4 existentes, y dejar el repo limpio/ordenado de forma reversible.

**Architecture:** Helpers de overlay reutilizables en `src/simulation/reel_overlay.py` (cv2, sin `drawtext`); orquestador `experiments/build_demo_reel.py` con config inline que lee cada MP4 con `cv2.VideoCapture`, normaliza a 1280×720, dibuja barra de título + métricas + nota de honestidad, recompila con ffmpeg y concatena en un reel resumen con tarjetas. Limpieza vía manifiesto (`docs/CLEANUP_MANIFEST.md`) + movimientos reversibles, borrado solo con OK explícito.

**Tech Stack:** Python 3.12 (`.venv`), OpenCV (`cv2` 4.13), NumPy, ffmpeg (homebrew, SIN libfreetype → texto solo por cv2), pytest.

**Spec:** `docs/superpowers/specs/2026-05-28-demo-reel-y-limpieza-design.md`

**Decisiones cerradas:**
- Métrica DP en overlay = **~165 ms** (live, conservadora).
- MP4 generados del reel = **gitignored** (regenerables).
- Resolución de salida = **1280×720 @ 24 fps**.

---

## File Structure

**Create:**
- `src/simulation/reel_overlay.py` — helpers: `normalize_frame`, `truncate`, `draw_title_bar`, `draw_metrics`, `draw_honesty_tag`, `make_title_card`.
- `experiments/build_demo_reel.py` — config `CLIPS` + orquestación (annotate → compile → concat).
- `tests/test_reel_overlay.py` — unit tests de helpers + smoke build.
- `docs/CLEANUP_MANIFEST.md` — generado por script en Task 6 (read-only, sin borrar).

**Modify:**
- `.gitignore` — agregar `experiments/results/demo_reel/` y `experiments/results/**/frames/`.

**Persisted (gitignored):**
- `experiments/results/demo_reel/clips/01_percepcion.mp4 … 04_robustez.mp4`
- `experiments/results/demo_reel/reel_resumen.mp4`
- `experiments/results/demo_reel/README.md`

---

## Pre-flight checks

- [ ] cwd y herramientas:

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
which ffmpeg && .venv/bin/python -c "import cv2, numpy; print('cv2', cv2.__version__)"
ls experiments/results/pipeline_e2e/demo_v2.mp4 \
   experiments/results/pick_with_fp_pose/demo.mp4 \
   experiments/results/pick_with_diffusion/demo.mp4 \
   experiments/results/pick_battery/base/demo.mp4
```

Expected: ruta de ffmpeg, `cv2 4.13.x`, y los 4 MP4 fuente listados sin error.

- [ ] Branch (si `repo_tfm` es git repo):

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git rev-parse --is-inside-work-tree 2>/dev/null && git checkout -b feat/demo-reel-y-limpieza || echo "(no git — continuar sin branch)"
```

---

## Task 1: `reel_overlay.py` — helpers de overlay (TDD)

**Files:**
- Create: `src/simulation/reel_overlay.py`
- Test: `tests/test_reel_overlay.py`

### Step 1.1: Write the failing test

- [ ] Crear `tests/test_reel_overlay.py`:

```python
"""Tests para src/simulation/reel_overlay.py."""
import numpy as np

from src.simulation.reel_overlay import (
    normalize_frame,
    truncate,
    draw_title_bar,
    draw_metrics,
    draw_honesty_tag,
    make_title_card,
)


def test_normalize_frame_640x480_to_720p():
    src = np.full((480, 640, 3), 128, dtype=np.uint8)
    out = normalize_frame(src)
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    # Letterbox: filas superiores/inferiores deben quedar negras (pad)
    assert out[0, 0].sum() == 0


def test_normalize_frame_already_720p_unchanged_shape():
    src = np.full((720, 1280, 3), 200, dtype=np.uint8)
    out = normalize_frame(src)
    assert out.shape == (720, 1280, 3)


def test_truncate_adds_ellipsis():
    assert truncate("abcdefghij", 5) == "abcd…"
    assert truncate("abc", 5) == "abc"


def test_draw_title_bar_modifies_top_band_only():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_title_bar(frame, "①", "FoundationPose")
    # La banda superior (y<70) debe tener pixels no nulos
    assert out[20, 100].sum() > 0
    # El centro de la imagen no debe haberse tocado
    assert out[400, 640].sum() == 0


def test_draw_metrics_green_when_ok():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_metrics(frame, [("IK convergido ✓", True)])
    # Región lower-left debe tener pixels dibujados
    assert out[620:700, 16:600].sum() > 0


def test_draw_honesty_tag_modifies_bottom_right():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_honesty_tag(frame, "grasp por attach")
    assert out[690:715, 800:1280].sum() > 0


def test_make_title_card_shape_and_text():
    card = make_title_card([("Bin-picking 6-DoF", 1.2), ("subtitulo", 0.7)])
    assert card.shape == (720, 1280, 3)
    # Centro vertical debe tener texto (pixels no nulos)
    assert card[300:420, :].sum() > 0
```

### Step 1.2: Run test to verify it fails

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_reel_overlay.py -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'src.simulation.reel_overlay'`.

### Step 1.3: Write minimal implementation

- [ ] Crear `src/simulation/reel_overlay.py`:

```python
"""Overlays para el demo reel (cv2, sin ffmpeg drawtext).

El ffmpeg local no tiene libfreetype/drawtext, así que todo el texto se
renderiza con cv2.putText (fuentes Hershey) sobre los frames. Reutiliza el
estilo de experiments/record_e2e_video_v2.py.

Convención de color: BGR (como cv2).
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 1280, 720
_FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX
_FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_MUTED = (150, 150, 150)
_GREEN = (90, 220, 120)   # BGR
_ACCENT = (205, 152, 10)  # ámbar BGR


def normalize_frame(frame: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    """Escala manteniendo aspecto y pad (letterbox) a (h, w, 3)."""
    src_h, src_w = frame.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def truncate(text: str, max_chars: int) -> str:
    """Trunca con elipsis si excede max_chars."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _alpha_band(frame, x0, y0, x1, y1, color, alpha):
    """Blend de un rectángulo relleno semi-transparente, in-place."""
    sub = frame[y0:y1, x0:x1]
    overlay = np.full_like(sub, color, dtype=np.uint8)
    frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0)


def draw_title_bar(frame: np.ndarray, number: str, title: str,
                   accent=_ACCENT) -> np.ndarray:
    """Banda superior semi-transparente + franja de acento + título."""
    _alpha_band(frame, 0, 0, W, 70, (30, 30, 38), 0.55)
    cv2.rectangle(frame, (0, 0), (8, 70), accent, -1)
    label = f"{number} {truncate(title, 60)}".strip()
    cv2.putText(frame, label, (24, 46), _FONT_TITLE, 0.95, _WHITE, 2, cv2.LINE_AA)
    return frame


def draw_metrics(frame: np.ndarray, lines, accent=_ACCENT) -> np.ndarray:
    """Panel lower-left con métricas. lines = list[(texto, pasa_threshold)].

    Si pasa_threshold es True, el texto va en verde; si no, en blanco.
    """
    n = len(lines)
    panel_h = 18 + 30 * n
    y_top = H - 20 - panel_h
    _alpha_band(frame, 0, y_top, 620, H - 8, (30, 30, 38), 0.55)
    cv2.rectangle(frame, (0, y_top), (8, H - 8), accent, -1)
    y = y_top + 34
    for text, ok in lines:
        color = _GREEN if ok else _WHITE
        cv2.putText(frame, truncate(text, 52), (24, y), _FONT_BODY, 0.62,
                    color, 1, cv2.LINE_AA)
        y += 30
    return frame


def draw_honesty_tag(frame: np.ndarray, text: str) -> np.ndarray:
    """Texto chico y tenue abajo-derecha."""
    label = truncate(text, 60)
    (tw, _), _ = cv2.getTextSize(label, _FONT_BODY, 0.5, 1)
    x = W - tw - 20
    cv2.putText(frame, label, (x, H - 18), _FONT_BODY, 0.5, _MUTED, 1, cv2.LINE_AA)
    return frame


def make_title_card(lines, w: int = W, h: int = H, accent=_ACCENT) -> np.ndarray:
    """Frame oscuro con líneas centradas. lines = list[(texto, escala)]."""
    card = np.full((h, w, 3), (28, 28, 34), dtype=np.uint8)
    cv2.rectangle(card, (0, h // 2 - 90), (10, h // 2 + 90), accent, -1)
    total = sum(int(40 * s) + 18 for _, s in lines)
    y = h // 2 - total // 2 + 30
    for text, scale in lines:
        (tw, th), _ = cv2.getTextSize(text, _FONT_TITLE, scale, 2)
        x = (w - tw) // 2
        cv2.putText(card, text, (x, y), _FONT_TITLE, scale, _WHITE, 2, cv2.LINE_AA)
        y += int(40 * scale) + 18
    return card
```

### Step 1.4: Run test to verify it passes

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_reel_overlay.py -v
```

Expected: 7 passed.

### Step 1.5: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add src/simulation/reel_overlay.py tests/test_reel_overlay.py
git commit -m "feat(sim): reel_overlay — helpers cv2 para overlays del demo reel

normalize_frame (letterbox 1280x720), draw_title_bar, draw_metrics
(verde si pasa threshold), draw_honesty_tag, make_title_card. Sin
ffmpeg drawtext (libfreetype ausente). 7 tests.

Refs: spec demo-reel, plan Task 1." || echo "(no git)"
```

---

## Task 2: `build_demo_reel.py` — anotar clips individuales

**Files:**
- Create: `experiments/build_demo_reel.py`

### Step 2.1: Create the script with config + per-clip annotation

- [ ] Crear `experiments/build_demo_reel.py`:

```python
#!/usr/bin/env python3
"""Construye el demo reel curado a partir de los MP4 existentes.

- Anota cada clip fuente con título de etapa + métricas + nota de honestidad
  (overlays cv2, porque ffmpeg local no tiene drawtext).
- Genera tarjetas de intro/cierre.
- Concatena todo en reel_resumen.mp4.

Uso:
    .venv/bin/python experiments/build_demo_reel.py
    .venv/bin/python experiments/build_demo_reel.py --only-clips   # sin reel resumen
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.reel_overlay import (
    normalize_frame, draw_title_bar, draw_metrics, draw_honesty_tag,
    make_title_card,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_demo_reel")

OUT = REPO / "experiments" / "results" / "demo_reel"
CLIPS_OUT = OUT / "clips"
FPS = 24

# Config del reel. metrics = list[(texto, pasa_threshold)].
CLIPS = [
    {
        "key": "01_percepcion",
        "source": REPO / "experiments/results/pick_with_fp_pose/demo.mp4",
        "number": "①",
        "title": "FoundationPose -> pose 6-DoF",
        "metrics": [("1098 poses YCBV  -  ~4.2 s/pose", True)],
        "honesty": "pose estimada offline (Colab T4)",
    },
    {
        "key": "02_planificacion",
        "source": REPO / "experiments/results/pick_with_diffusion/demo.mp4",
        "number": "②",
        "title": "Diffusion Policy -> trayectoria (16 waypoints)",
        "metrics": [("inferencia ~165 ms  -  IK convergido ✓", True)],
        "honesty": "la DP imita la heuristica (Iter 1)",
    },
    {
        "key": "03_e2e",
        "source": REPO / "experiments/results/pipeline_e2e/demo_v2.mp4",
        "number": "③",
        "title": "Pipeline end-to-end",
        "metrics": [
            ("ciclo p95 5.2 s (FP 4.2 / DP 0.2 / sim 1.0)", True),
            ("aceptacion <10 s ✓", True),
        ],
        "honesty": "grasp por attach (estandar en sims comerciales)",
    },
    {
        "key": "04_robustez",
        "source": REPO / "experiments/results/pick_battery/base/demo.mp4",
        "number": "④",
        "title": "Robustez -> 3 escenarios",
        "metrics": [("grasp_proximity 0.8 cm  -  IK ✓ en los 3", True)],
        "honesty": "grasp por attach (estandar en sims comerciales)",
    },
]

INTRO = [("Bin-picking 6-DoF", 1.3),
         ("Percepcion  ->  Planificacion  ->  Ejecucion", 0.7)]
OUTRO = [("Validado end-to-end  -  ciclo <10 s", 1.0),
         ("honestidad declarada", 0.7),
         ("potencial: grasp fisico / FoundationPose en vivo", 0.55)]


def _compile(frames_dir: Path, mp4_path: Path) -> Path | None:
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado — skip MP4")
        return None
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", str(frames_dir / "%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           str(mp4_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"ffmpeg falló: {r.stderr[-500:]}")
        return None
    return mp4_path


def annotate_clip(clip: dict) -> Path | None:
    """Lee el MP4 fuente, aplica overlays y compila el clip anotado."""
    src = clip["source"]
    if not src.exists():
        logger.warning(f"[{clip['key']}] fuente faltante: {src} — skip")
        return None
    cap = cv2.VideoCapture(str(src))
    out_mp4 = CLIPS_OUT / f"{clip['key']}.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = normalize_frame(frame)
            draw_title_bar(frame, clip["number"], clip["title"])
            draw_metrics(frame, clip["metrics"])
            draw_honesty_tag(frame, clip["honesty"])
            cv2.imwrite(str(td / f"{idx:06d}.png"), frame)
            idx += 1
        cap.release()
        if idx == 0:
            logger.warning(f"[{clip['key']}] 0 frames leídos — skip")
            return None
        CLIPS_OUT.mkdir(parents=True, exist_ok=True)
        result = _compile(td, out_mp4)
    if result:
        logger.info(f"[{clip['key']}] clip anotado: {out_mp4.name} ({idx} frames)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-clips", action="store_true",
                        help="Solo anota clips; no genera el reel resumen.")
    args = parser.parse_args()

    CLIPS_OUT.mkdir(parents=True, exist_ok=True)
    annotated = []
    for clip in CLIPS:
        p = annotate_clip(clip)
        if p:
            annotated.append(p)

    if not annotated:
        logger.error("0 clips anotados — abortando")
        return 1
    logger.info(f"{len(annotated)}/{len(CLIPS)} clips anotados en {CLIPS_OUT}")

    if args.only_clips:
        return 0
    # El reel resumen se arma en Task 3 (build_reel).
    from experiments.build_demo_reel import build_reel  # noqa: import tardío
    build_reel(annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Nota:** `build_reel` se define en Task 3; este `main` lo importa. Hasta
entonces, correr con `--only-clips`.

### Step 2.2: Run the per-clip annotation

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/build_demo_reel.py --only-clips 2>&1 | tail -10
```

Expected: log `4/4 clips anotados` y archivos en `experiments/results/demo_reel/clips/`.

### Step 2.3: Verify dimensions

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
for f in experiments/results/demo_reel/clips/*.mp4; do
  echo "$f"; ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$f"; done
```

Expected: cada clip `1280,720`.

### Step 2.4: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/build_demo_reel.py
git commit -m "feat(reel): build_demo_reel — anota clips por etapa con overlays

Config CLIPS inline (4 etapas). Lee cada MP4 fuente con cv2, normaliza
a 1280x720, aplica titulo + metricas + nota honestidad, recompila.
Fuente faltante -> skip con warning.

Refs: spec demo-reel, plan Task 2." || echo "(no git)"
```

---

## Task 3: Tarjetas intro/cierre + concatenar reel resumen

**Files:**
- Modify: `experiments/build_demo_reel.py` (agregar `build_reel` + `_make_card_clip`)

### Step 3.1: Add card + concat functions

- [ ] En `experiments/build_demo_reel.py`, ANTES de `def main`, agregar:

```python
def _make_card_clip(lines, key: str, seconds: float = 3.0) -> Path | None:
    """Genera un MP4 corto (card estática) de `seconds` a FPS."""
    card = make_title_card(lines)
    n = int(round(seconds * FPS))
    out_mp4 = CLIPS_OUT / f"{key}.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n):
            cv2.imwrite(str(td / f"{i:06d}.png"), card)
        return _compile(td, out_mp4)


def build_reel(annotated: list[Path]) -> Path | None:
    """Concatena intro + clips anotados + outro en reel_resumen.mp4."""
    intro = _make_card_clip(INTRO, "00_intro")
    outro = _make_card_clip(OUTRO, "99_outro")
    sequence = [p for p in [intro, *annotated, outro] if p is not None]
    if not sequence:
        logger.error("sin clips para concatenar")
        return None
    # ffmpeg concat demuxer (todos mismo codec/res/fps -> -c copy)
    list_file = OUT / "_concat_list.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in sequence))
    reel = OUT / "reel_resumen.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(reel)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if r.returncode != 0:
        logger.error(f"concat falló: {r.stderr[-500:]}")
        return None
    logger.info(f"reel resumen: {reel}")
    return reel
```

### Step 3.2: Run the full build

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/build_demo_reel.py 2>&1 | tail -12
```

Expected: log `reel resumen: .../reel_resumen.mp4`.

### Step 3.3: Verify the reel

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -show_entries format=duration \
  -of default=noprint_wrappers=1 experiments/results/demo_reel/reel_resumen.mp4
```

Expected: `width=1280`, `height=720`, `duration` entre ~60 y ~120 s.

### Step 3.4: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/build_demo_reel.py
git commit -m "feat(reel): tarjetas intro/cierre + concat del reel resumen

_make_card_clip (card estatica N frames) + build_reel (concat demuxer
de ffmpeg, -c copy porque todo es 1280x720 @24fps). reel_resumen.mp4.

Refs: spec demo-reel, plan Task 3." || echo "(no git)"
```

---

## Task 4: README del reel (mapa clip→slide + guión hablado)

**Files:**
- Create: `experiments/results/demo_reel/README.md`

### Step 4.1: Write the README

- [ ] Crear `experiments/results/demo_reel/README.md`:

```markdown
# Demo reel — mapa de uso

Generado por `experiments/build_demo_reel.py` (regenerable). Overlays cv2.

## Archivos

- `reel_resumen.mp4` — reel continuo (~75–90 s) para abrir/cerrar la charla.
- `clips/01_percepcion.mp4` — FoundationPose → pose 6-DoF.
- `clips/02_planificacion.mp4` — Diffusion Policy → trayectoria.
- `clips/03_e2e.mp4` — pipeline end-to-end.
- `clips/04_robustez.mp4` — robustez en 3 escenarios.

## Mapa clip → slide + guión hablado

| Slide | Clip | Qué decir (con honestidad) |
|---|---|---|
| Percepción | `clips/01_percepcion.mp4` | "FoundationPose estima la pose 6-DoF del objeto. Validado sobre 1098 instancias de YCBV, ~4.2 s por pose. Las poses se calcularon offline en Colab." |
| Planificación | `clips/02_planificacion.mp4` | "La Diffusion Policy genera la trayectoria de 16 waypoints en ~165 ms. En esta iteración la política imita la heurística; el aporte es cerrar el lazo percepción→planificación." |
| Ejecución | `clips/03_e2e.mp4` | "El pipeline completo corre con ciclo p95 de 5.2 s, bien bajo el umbral de 10 s. El grasp usa la técnica de attach, estándar en simuladores comerciales — lo declaramos explícitamente." |
| Robustez | `clips/04_robustez.mp4` | "Mismo pipeline en 3 escenarios (iluminación/colores): grasp_proximity 0.8 cm y IK convergido en los tres." |

## Plan B (modo en vivo)

Si la corrida en vivo en CoppeliaSim falla, proyectar `reel_resumen.mp4`.

## Regenerar

```bash
.venv/bin/python experiments/build_demo_reel.py
```
```

### Step 4.2: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add -f experiments/results/demo_reel/README.md
git commit -m "docs(reel): README con mapa clip->slide y guion hablado honesto

Refs: spec demo-reel, plan Task 4." || echo "(no git)"
```

**Nota:** `-f` porque la carpeta `demo_reel/` está gitignored (Task 5); el README
sí lo queremos versionar.

---

## Task 5: `.gitignore` del reel + frames

**Files:**
- Modify: `.gitignore`

### Step 5.1: Append ignore rules

- [ ] Agregar al final de `.gitignore`:

```
# Demo reel: MP4 generados (regenerables con build_demo_reel.py)
experiments/results/demo_reel/
!experiments/results/demo_reel/README.md
# Frames intermedios de los demos (regenerables re-corriendo el sim)
experiments/results/**/frames/
```

### Step 5.2: Verify

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git check-ignore experiments/results/demo_reel/reel_resumen.mp4 \
  experiments/results/pick_battery/base/frames 2>/dev/null
git status --short | grep -E "demo_reel|frames" || echo "(nada nuevo trackeado: OK)"
```

Expected: las dos rutas listadas por `check-ignore`; `README.md` NO ignorado.

### Step 5.3: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add .gitignore
git commit -m "chore: gitignore demo_reel/ (salvo README) y **/frames/

Refs: spec demo-reel, plan Task 5." || echo "(no git)"
```

---

## Task 6: Generar `CLEANUP_MANIFEST.md` (read-only, sin borrar)

**Files:**
- Create: `docs/CLEANUP_MANIFEST.md` (generado)

### Step 6.1: Generate the manifest

- [ ] Correr este comando (clasifica cada carpeta; NO mueve ni borra nada):

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python - <<'PY'
import subprocess, pathlib
REPO = pathlib.Path(".").resolve()
RESULTS = REPO / "experiments" / "results"
KEEP_DEMO = {"pipeline_e2e", "pick_with_fp_pose", "pick_with_diffusion",
             "pick_battery", "scenario_battery", "foundationpose_eval", "demo_reel"}
rows = []
for d in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    name = d.name
    size = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True).stdout.split("\t")[0]
    refs = subprocess.run(["grep", "-rl", name, "docs"], capture_output=True, text=True).stdout.strip()
    nrefs = len([x for x in refs.splitlines() if x])
    if name in KEEP_DEMO or nrefs > 0:
        klass, action = "KEEP", "no tocar (solo mover sus frames/ -> _frames_archive)"
    else:
        klass, action = "HUERFANO", "REVISAR uno por uno (candidato a archivar)"
    rows.append((name, size, nrefs, klass, action))
lines = ["# CLEANUP_MANIFEST — revisión antes de mover/borrar",
         "",
         "> Nada se mueve ni borra hasta que apruebes este manifiesto.",
         "> Movimientos reversibles (a _frames_archive / _orphans_review).",
         "> Borrado definitivo = paso final separado, con OK explícito.",
         "",
         "| Carpeta | Tamaño | #refs docs | Clasificación | Acción propuesta |",
         "|---|---|---:|---|---|"]
for name, size, nrefs, klass, action in rows:
    lines.append(f"| {name} | {size} | {nrefs} | {klass} | {action} |")
lines += ["",
          "## Grupo INTERMEDIO (frames regenerables, ~1.4 GB)",
          "Mover `experiments/results/{pick_battery,pick_with_fp_pose,pick_demo,pick_with_diffusion}/**/frames/`",
          "a `experiments/results/_frames_archive/` (reversible). El reel no los necesita."]
pathlib.Path("docs/CLEANUP_MANIFEST.md").write_text("\n".join(lines) + "\n")
print("escrito: docs/CLEANUP_MANIFEST.md")
print("\n".join(lines))
PY
```

Expected: imprime la tabla y escribe `docs/CLEANUP_MANIFEST.md`. NO mueve nada.

### Step 6.2: Commit the manifest

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add docs/CLEANUP_MANIFEST.md
git commit -m "docs: CLEANUP_MANIFEST — inventario clasificado de results/

Read-only. KEEP (demos + referenciadas en docs) vs HUERFANO (0 refs).
Nada movido/borrado. Acciones propuestas pendientes de aprobacion.

Refs: spec demo-reel, plan Task 6." || echo "(no git)"
```

### Step 6.3: STOP — gate de aprobación humana

- [ ] **Presentar `docs/CLEANUP_MANIFEST.md` al usuario y ESPERAR su OK
  explícito.** No ejecutar Task 7 hasta entonces.

---

## Task 7: Limpieza reversible (SOLO tras aprobar el manifiesto)

**Files:** ninguno de código — operaciones de filesystem reversibles.

> **GATE:** ejecutar únicamente si el usuario aprobó `CLEANUP_MANIFEST.md`.

### Step 7.1: Mover frames intermedios (reversible)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
ARCHIVE=experiments/results/_frames_archive
mkdir -p "$ARCHIVE"
for d in pick_battery pick_with_fp_pose pick_demo pick_with_diffusion; do
  find "experiments/results/$d" -type d -name frames -print0 2>/dev/null | \
  while IFS= read -r -d '' fr; do
    dest="$ARCHIVE/$(echo "$fr" | sed 's#experiments/results/##; s#/#__#g')"
    mkdir -p "$(dirname "$dest")"
    mv "$fr" "$dest"
    echo "movido: $fr -> $dest"
  done
done
du -sh experiments/results experiments/results/_frames_archive
```

Expected: frames movidos; el tamaño activo de `results` baja ~1.4 GB. Reversible
(`mv` de vuelta). Los MP4 del reel siguen intactos.

### Step 7.2: Verify reel still builds from MP4 (no dependía de frames)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/build_demo_reel.py --only-clips 2>&1 | tail -5
```

Expected: `4/4 clips anotados` (confirma que mover frames no rompió el reel).

### Step 7.3 (OPCIONAL, GATE FINAL): borrar `_frames_archive`

- [ ] **Solo con OK explícito adicional del usuario.** Hasta entonces, dejar
  `_frames_archive/` en disco (reversible).

```bash
# Ejecutar SOLO si el usuario confirma el borrado definitivo:
# rm -rf experiments/results/_frames_archive
echo "Borrado definitivo pendiente de OK explícito del usuario."
```

---

## Verificación final

### Step F.1: Tests

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_reel_overlay.py -v
```

Expected: 7 passed.

### Step F.2: Artefactos del reel presentes

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
ls -lh experiments/results/demo_reel/reel_resumen.mp4 \
       experiments/results/demo_reel/clips/*.mp4 \
       experiments/results/demo_reel/README.md
```

Expected: reel resumen + 4 clips + README.

### Step F.3: Push del branch (si git)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git rev-parse --is-inside-work-tree 2>/dev/null && {
  git log --oneline main..HEAD
  git push -u origin feat/demo-reel-y-limpieza
} || echo "(no git — nada que pushear)"
```

Expected: ~6-7 commits.

---

## Checklist final del plan

- [ ] **Task 1:** `reel_overlay.py` + 7 tests
- [ ] **Task 2:** `build_demo_reel.py` — anota 4 clips
- [ ] **Task 3:** tarjetas + concat reel resumen
- [ ] **Task 4:** README del reel (mapa + guión)
- [ ] **Task 5:** `.gitignore` demo_reel + frames
- [ ] **Task 6:** `CLEANUP_MANIFEST.md` (read-only) + **gate de aprobación**
- [ ] **Task 7:** limpieza reversible (solo tras OK) + borrado final (segundo OK)
- [ ] **F.1–F.3:** tests + artefactos + push
