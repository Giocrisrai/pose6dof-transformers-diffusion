# Demo reel curado + limpieza del repo — Design

**Fecha:** 2026-05-28
**Estado:** diseño aprobado, pendiente plan de implementación
**Autor:** ggodoy@mtechsol.com
**Branch sugerido:** `feat/demo-reel-y-limpieza`

## Contexto y propósito

Charla en <1 mes con **doble propósito**: (1) preparar la **defensa del TFM** y
(2) sentar base para convertir el desarrollo en un **producto reutilizable**.
Audiencia mixta (académica, empresa, técnica). Criterio de defensa acordado:
**pipeline end-to-end funcional** (percepción FoundationPose → planificación
Diffusion Policy → IK del UR5 → ejecución en CoppeliaSim), NO validez física
del grasp ni superioridad de la DP sobre la heurística (ver
`docs/PICK_LIMITATIONS.md`).

Entregable priorizado: **demo reel curado** — clips de simulación anotados con
título de etapa + métricas clave + nota de honestidad, en formato **híbrido**
(MP4 pre-grabado como base segura + corrida en vivo opcional con el video de
respaldo). Adicionalmente, **limpieza y orden** del repo (1.5 GB en
`experiments/results`) para dejarlo presentable y liviano.

## Objetivos

- Producir **clips anotados independientes** (uno por etapa) embebibles en
  slides y usables como respaldo en vivo.
- Producir un **reel resumen** continuo (~75–90 s) que une todas las etapas con
  tarjetas de intro y cierre.
- Overlays con **título de etapa + métricas reales + nota de honestidad**,
  renderizados con cv2/PIL (el ffmpeg local NO tiene `drawtext`/libfreetype).
- Todo **regenerable por script** (config-driven), sin edición manual de video.
- **Limpiar y ordenar** `experiments/results` de forma **segura y reversible**:
  manifiesto primero, movimientos reversibles, borrado solo con OK explícito.

## No-objetivos

- Editar slides/deck (entregable posterior).
- Re-entrenar modelos o re-correr experimentos (usamos los MP4 existentes).
- Branding/logo en los overlays (descartado por ahora).
- Live coding del setup en vivo (solo se documenta el plan B con el reel).
- Borrar evidencia del TFM sin confirmación humana explícita.

## Restricción técnica clave

`ffmpeg` (homebrew, `/opt/homebrew/bin/ffmpeg`) está compilado **sin
libfreetype** → **no hay filtro `drawtext`**. Sí tiene `drawbox`, `scale`,
`concat`. Por lo tanto **todo el texto se renderiza con cv2** (`cv2.putText`,
fuentes Hershey) o PIL sobre los frames, reutilizando el patrón ya probado en
`experiments/record_e2e_video_v2.py` (panel con título, subtítulos de etapa,
métricas key/value, barra de progreso). PIL 12.x y cv2 4.13 disponibles en
`.venv`.

---

## Sección 1 — Guión del reel y mapa de clips

| # | Etapa | Clip fuente | Título overlay | Métrica overlay (real) | Nota honestidad |
|---|---|---|---|---|---|
| 0 | Tarjeta intro | — | "Bin-picking 6-DoF — Percepción → Planificación → Ejecución" | TFM · mtechsol | — |
| 1 | Percepción | `pick_with_fp_pose/demo.mp4` | "① FoundationPose → pose 6-DoF" | "1098 poses YCBV · ~4.2 s/pose" | "pose estimada offline (Colab T4)" |
| 2 | Planificación | `pick_with_diffusion/demo.mp4` | "② Diffusion Policy → trayectoria (16 waypoints)" | "inferencia ~165 ms · IK ✓" | "la DP imita la heurística (Iter 1)" |
| 3 | Ejecución E2E | `pipeline_e2e/demo_v2.mp4` | "③ Pipeline end-to-end" | "ciclo p95 5.2 s (FP 4.2 / DP 0.2 / sim 1.0) · <10 s ✓" | "grasp por attach (estándar en sims comerciales)" |
| 4 | Robustez | `pick_battery/{base,easy,hard}/demo.mp4` | "④ Robustez — 3 escenarios" | "grasp_proximity 0.8 cm · IK ✓ en los 3" | misma nota attach |
| 5 | Tarjeta cierre | — | "Validado end-to-end · ciclo <10 s · honestidad declarada" | + "potencial: grasp físico / FP en vivo" | — |

**Fuentes de las métricas (reales, ya medidas):**
- Ciclo E2E p95 = 5.18 s (YCBV), FP ~4.15 s, DP ~165 ms (live) / ~1.9 ms
  (aggregated), sim ~0.9–1.3 s; H3 acceptance `p95<10s` PASSED.
  Fuente: `experiments/results/pipeline_e2e/e2e_aggregated_metrics.json` y
  `e2e_live_metrics.json`.
- FoundationPose: n=1098 (YCBV), n=1012 (TLESS).
- pick_battery: grasp_proximity 0.8 cm, deposit 4–6 cm, ik_converged ✓ en
  base/easy/hard. Fuente: `experiments/results/pick_battery/report.json`.

Las filas 1–4 salen también como **clips anotados independientes**.
El **reel resumen** une 0→1→2→3→4→5 con cortes secos.

---

## Sección 2 — Diseño visual, empaque y herramienta

### Overlay (cv2, sin drawtext)

- Todos los clips normalizados a **1280×720**. `pipeline_e2e/demo_v2.mp4` ya
  está en 1280×720; `pick_battery/*` y los demás en 640×480 → escalar con
  **letterbox** (scale manteniendo aspecto + pad a 1280×720).
- **Barra superior:** banda oscura semi-transparente (alpha ~0.55) + franja de
  acento, con número + título de etapa.
- **Lower-third (métricas):** panel inferior-izquierdo con métricas clave; ✓ en
  verde para lo que pasa threshold (ej. `<10 s ✓`, `IK ✓`).
- **Tag de honestidad:** texto chico, color tenue, abajo-derecha.
- **Tarjetas 0 y 5:** frame 1280×720 oscuro, texto centrado, ~3 s (= fps × 3
  frames repetidos).
- **Transiciones:** cortes secos (sin fades).
- **Fuentes:** Hershey de cv2 (`FONT_HERSHEY_DUPLEX`/`SIMPLEX`), consistente con
  `record_e2e_video_v2.py`.

### Estructura de archivos

**Crear:**
- `src/simulation/reel_overlay.py` — helpers reutilizables y testeables:
  - `normalize_frame(frame, w=1280, h=720) -> np.ndarray` (scale + letterbox)
  - `draw_title_bar(frame, number, title) -> frame`
  - `draw_metrics(frame, lines: list[tuple[str, bool]]) -> frame` (bool = pasa
    threshold → verde)
  - `draw_honesty_tag(frame, text) -> frame`
  - `make_title_card(lines, w=1280, h=720) -> np.ndarray`
- `experiments/build_demo_reel.py` — config inline (lista `CLIPS` con
  source/título/métricas/nota) + orquestación: por cada clip, lee MP4 con
  `cv2.VideoCapture`, aplica overlays frame a frame, escribe PNGs temporales,
  compila el clip anotado con ffmpeg; luego genera tarjetas y concatena todo en
  `reel_resumen.mp4` vía `ffmpeg concat`.
- `tests/test_reel_overlay.py` — unit tests de los helpers sobre frame sintético
  (shapes correctos, regiones no vacías tras dibujar) + smoke del build con 1
  clip corto (output existe, ffprobe dims = 1280×720).

**Reutiliza:**
- Patrón de overlay de `experiments/record_e2e_video_v2.py`.
- `compile_mp4` / convención ffmpeg de `src/simulation/pick_sequence.py`.

### Salida en `experiments/results/demo_reel/`

- `clips/01_percepcion.mp4 … 04_robustez.mp4` (anotados, independientes)
- `reel_resumen.mp4` (~75–90 s, con tarjetas 0 y 5)
- `README.md` — mapa clip→slide + guión hablado con la línea de honestidad de
  cada etapa

### Config-driven / regenerable

La lista `CLIPS` vive inline en `build_demo_reel.py` (Python list de dicts; sin
formato externo — YAGNI). Cambiar una métrica o título = editar la config y
re-correr. Sin edición manual de video.

---

## Sección 3 — Limpieza y orden (segura, con confirmación)

**Principio rector: nada se borra sin que el usuario lo apruebe viendo un
manifiesto. Los movimientos son reversibles. El borrado definitivo es el último
paso, con OK explícito.**

`experiments/results` pesa **1.5 GB**. Clasificación en 3 grupos:

1. **KEEP** — usadas por el reel + referenciadas en docs/specs: `pipeline_e2e`,
   `pick_with_fp_pose`, `pick_with_diffusion`, `pick_battery`,
   `scenario_battery`, `foundationpose_eval`, y toda carpeta `exp*` que aparezca
   en `docs/`. No se tocan (salvo sus `frames/`, ver grupo 2).
2. **INTERMEDIO regenerable** — `frames/*.png` dentro de las 4 carpetas de demos
   (~1.4 GB del total). El reel (Approach A) trabaja desde los MP4, así que los
   frames no son necesarios. Acción: mover a `experiments/results/_frames_archive/`
   (reversible) y/o agregar `**/frames/` a `.gitignore`. Recupera ~1.4 GB.
3. **HUÉRFANO (0 refs en docs)** — candidatos a revisar (ej. `local_notebooks`,
   `exp9_3d_viz`, `exp8_diversity`, `exp11/12/13`, `e2e_verification`,
   `drive_chapter6_figs`). NO se borran; se listan en el manifiesto para decisión
   uno por uno.

**Entregable de limpieza:** `docs/CLEANUP_MANIFEST.md` con tabla completa
(carpeta · tamaño · #refs en docs · clasificación · acción propuesta). El
usuario aprueba el manifiesto y recién ahí se ejecutan los movimientos. Borrado
definitivo = paso final separado, con OK explícito.

**Generación del manifiesto:** script o comando que recorre
`experiments/results/*`, calcula tamaño (`du`), cuenta referencias
(`grep -rl <dir> docs/`), y emite la tabla clasificada.

---

## Manejo de errores

| Punto | Falla | Comportamiento |
|---|---|---|
| Build reel | MP4 fuente faltante | Skip ese clip con warning; el reel se arma con los disponibles. |
| Build reel | ffmpeg ausente | Warning y abort del compile (patrón existente en `compile_mp4`). |
| Overlay | Resolución distinta de la esperada | `normalize_frame` escala + letterbox a 1280×720. |
| Overlay | Texto más largo que el panel | Truncar con elipsis (helper); no romper layout. |
| Limpieza | Carpeta candidata resulta estar referenciada | Reclasificar a KEEP automáticamente (el grep manda sobre la heurística). |
| Limpieza | Usuario no aprobó manifiesto | NO ejecutar ningún movimiento ni borrado. |

## Testing

- `tests/test_reel_overlay.py`:
  - `normalize_frame` devuelve (720, 1280, 3) para entradas 640×480 y 1280×720.
  - `draw_title_bar`/`draw_metrics`/`draw_honesty_tag` modifican la región
    esperada (pixels no vacíos en la banda; resto intacto).
  - `make_title_card` devuelve frame del tamaño correcto.
  - Smoke: `build_demo_reel` sobre 1 clip corto sintético → MP4 existe, ffprobe
    width=1280 height=720.
- Tests existentes deben seguir pasando.
- Validación manual: abrir `reel_resumen.mp4` y verificar legibilidad de
  overlays y orden de etapas.

## Métricas de éxito

| Métrica | Threshold |
|---|---|
| Clips anotados generados | 4 (percepción, planificación, E2E, robustez) |
| Reel resumen | 1 MP4, 1280×720, ~75–90 s, con tarjetas intro/cierre |
| Overlays legibles | Título + métrica + nota de honestidad visibles en cada clip |
| Regenerable | `build_demo_reel.py` re-corre sin edición manual |
| Espacio recuperado (limpieza) | ~1.4 GB tras aprobar manifiesto + mover frames |
| Seguridad limpieza | 0 borrados sin OK explícito; movimientos reversibles |
| Tests | `test_reel_overlay.py` pasa + suite existente verde |

## Estimación de superficie

- ~250–300 líneas nuevas (`reel_overlay.py` ~120, `build_demo_reel.py` ~150,
  tests ~60).
- 1 doc nuevo (`CLEANUP_MANIFEST.md`) + 1 README del reel.
- ~5–8 MB de salida nueva (clips + reel) ; ~1.4 GB recuperados tras limpieza.
- `.gitignore` actualizado (`**/frames/`, `experiments/results/demo_reel/` si
  se decide no versionar los MP4 generados).
