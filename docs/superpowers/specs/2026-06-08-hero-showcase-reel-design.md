# Hero Showcase Reel (Iter 7c) — Diseño

**Fecha:** 2026-06-08
**Estado:** aprobado (diseño), pendiente de plan de implementación

## Objetivo

Un video nuevo y separado (`reel_showcase.mp4`, ~90-120 s) de alto impacto que combina (A) una toma cinematográfica pulida del mejor pick-and-place de Iter 7c con (B) un segmento narrativo de valor y aplicaciones. NO modifica el reel técnico actual (`reel_resumen.mp4`), que se mantiene para la defensa formal.

Resuelve la brecha actual: el reel técnico usa una cámara fija de baja resolución (la de percepción), tomas cortas (~15 s) y no comunica la potencialidad/valor del sistema.

**Entregado (2026-06-08):** `reel_showcase.mp4` de **~90 s, 1280×720**, en tres segmentos: (A) hero pick cinematográfico (~24,5 s), (B) replay en cámara lenta del close-up del grasp (~25,8 s), (C) 5 tarjetas de valor/aplicaciones (8 s c/u). Cámara dedicada con eje óptico +Z y coreografía órbita→seguimiento→retroceso vía matriz look-at.

## Decisiones de diseño (confirmadas con el usuario)

1. **Objetivo:** balanceado — cinematografía mejorada + narrativa de valor.
2. **Cámara:** combinación cinematográfica (establecimiento → seguimiento → retroceso).
3. **Alcance:** hero pick cinematográfico + segmento de valor, enfocado. Video separado.

## Arquitectura

### Cámara cinematográfica dedicada

- Se usa un vision sensor **dedicado** `/cine_camera`, **separado de `/rgb_camera`** (percepción). La cámara de percepción queda fija para que la Diffusion Policy reciba el RGB-D correcto — no se corrompe la entrada del modelo.
- Resolución cinematográfica alta (objetivo 1280×720).
- **Método de creación (riesgo a validar primero):** crear el vision sensor vía API ZMQ en runtime (`sim.createVisionSensor` o equivalente en la versión instalada). **Fallback:** si la creación en runtime es inestable, agregar `/cine_camera` a la escena `bin_base.ttt` una vez (manual, documentado). El primer paso de la implementación valida cuál de los dos camina.

### Coreografía de cámara (sincronizada a las fases del pick)

La acción del pick son 16 waypoints; las fases (de `reward_fn`/pick): approach+grasp (k=0..5), lift (k=6..8), deposit (k=9..15). La cámara se mueve por keyframes interpolados anclados a esos índices:

- **Establecimiento** (approach, k≈0..5): órbita amplia y lenta alrededor del workspace. Da contexto 3D.
- **Seguimiento** (descend+grasp, k≈5..8): la cámara se acerca y apunta al TCP/gripper. Detalle del agarre.
- **Retroceso** (lift+deposit, k≈9..15): pull-back que revela el cubo cayendo en el bin.

Implementado como trayectorias de posición+orientación (look-at al TCP o al workspace) interpoladas por substep, en `src/simulation/cine_camera.py`.

### Más tiempo de grabación / pacing

- Más `steps_per_substep` y más frames de settle → movimiento fluido.
- Playback a 30 fps.
- Toma hero sola ~50-70 s (vs ~15 s actuales).

### Segmento de valor (parte B)

Tarjetas + overlays (reusa `src/simulation/reel_overlay.py`, cv2), ancladas en datos reales y honestas:

- **Qué se logró:** FoundationPose + Diffusion Policy integrados; pick-and-place E2E 84 %, IK 100 %, ciclo p95 < 10 s.
- **Por qué es mejor:** hardware accesible (~USD 1.920: Apple M1 Pro + Colab T4) vs setups industriales (USD 15k-150k) — 1-2 órdenes de magnitud más barato.
- **Aplicaciones:** bin-picking en logística (order-picking e-commerce), alimentación de líneas de manufactura, clasificación.
- **Honestidad:** grasp por snap+attach (estándar en sims comerciales); falta validar grasp físico + robot real.

### Honestidad (se mantiene)

Tag declarado en el reel: el grasp usa snap+attach; valida la cadena percepción→planificación→IK→ejecución, no la mecánica física del agarre por fricción.

## Componentes / archivos

| Archivo | Propósito | Depende de |
|---------|-----------|------------|
| `src/simulation/cine_camera.py` | Crea/posiciona la cámara dedicada; helpers de coreografía (orbit, track, pull-back, look-at) | bridge ZMQ, sim API |
| `experiments/make_showcase_reel.py` | Corre el hero pick con captura cine (best-of-8 + fix IK, pose i=49 seed 3) + ensambla parte A + parte B | cine_camera, pick_with_dp, reel_overlay, compile_mp4 |
| `src/simulation/reel_overlay.py` | (reuso) overlays cv2 para tarjetas/métricas del segmento de valor | — |

**Salida:** `experiments/results/demo_reel/reel_showcase.mp4` (regenerable, gitignored como `reel_resumen.mp4`).

## Flujo de datos

1. `make_showcase_reel.py` carga policy v7a_phase2 + encoder_iter5 (igual que el eval 7c).
2. Crea `/cine_camera` (o usa la de la escena).
3. Corre el pick (pose i=49, torch-seed 3) con `pick_with_dp(best_of_n=8)`; por cada substep, además de ejecutar la trayectoria, posiciona la cine_camera según la coreografía y captura un frame de alta resolución.
4. Compila la parte A (hero pick) a MP4.
5. Genera la parte B (tarjetas de valor) con reel_overlay.
6. Concatena A + B → `reel_showcase.mp4`.

## Manejo de errores

- Si la creación de `/cine_camera` en runtime falla → fallback a cámara de escena; si tampoco existe → error claro pidiendo agregarla.
- Si el pick elegido (i=49/seed 3) no sale limpio en esta corrida (estocasticidad) → reintentar o reusar el barrido; el script reporta métricas y aborta si grasp/deposit no son plausibles, para no publicar un clip fallido.
- ffmpeg ausente → error claro (ya manejado en `compile_mp4`).

## Testing

- La generación de video requiere CoppeliaSim (no CI-friendly): la verificación es manual (revisar `reel_showcase.mp4`) + asserts de métricas del pick (grasp/deposit/IK plausibles) en el script.
- Helpers puros de coreografía de cámara (interpolación de posición/orientación, look-at) en `cine_camera.py` → tests unitarios sin sim (geometría), estilo `test_reel_overlay.py`.

## Fuera de alcance (YAGNI)

- Re-filmar percepción/robustez cinematográficamente (percepción es offline; alcance acotado a la toma hero).
- Voz en off / audio.
- Edición no lineal / múltiples versiones de duración.
