# Videos para proyectar — defensa y charla divulgativa

Copias sueltas de los videos de simulación, listas para **proyectar** desde
cualquier reproductor (VLC, QuickTime) si la reproducción embebida en el
`.pptx` falla en la máquina del proyector. Todos están además **embebidos**
en las presentaciones; esta carpeta es el plan B.

| # | Archivo | Qué muestra | Dur. | Defensa | Divulgativa |
|---|---------|-------------|------|---------|-------------|
| 01 | `01_pipeline_en_accion.mp4` | Reel del pipeline E2E (percepción → planificación → ejecución) | 1:30 | slide 12 «Demo: el pipeline en acción» | slide 2 «50 segundos…» |
| 02 | `02_lenguaje_natural.mp4` | **Lenguaje natural**, reel completo a velocidad real (5 instrucciones) | 2:54 | (versión larga) | (versión larga) |
| 02b | `02b_lenguaje_natural_corto.mp4` | **Lenguaje natural**, corte a 4.5x — las 5 instrucciones en 39s. **Es el embebido** en el slide 13 de ambas | 0:39 | **slide 13** (embebido) | **slide 13** (embebido) |
| 03 | `03_e2e_con_telemetria.mp4` | Ciclo E2E grabado con panel de telemetría real (latencias, fase, H3) | 0:30 | (apoyo H3) | slide 14 «El mismo sistema, grabado» |
| 04 | `04_agarre_diffusion.mp4` | Agarre generado por Diffusion Policy (Iter 7c) | 0:29 | slide 11 «Progresión Iter 5 → 7c» (apoyo) | slide 12 «DEMO 2» (respaldo si la demo en vivo falla) |
| 05 | `05_resumen_cierre.mp4` | Reel de resumen para el cierre | 1:30 | (cierre) | slide 22 «¿Preguntas?» |

## Guion de proyección (defensa, ~20 min)

- **Slide 12** — lanza `01` mientras explicas el ciclo completo. Déjalo en bucle.
- **Slide 13** — lanza `02`: es la contribución más nueva. Di la frase en voz
  alta («dame el cubo rojo de la izquierda») justo cuando el overlay la muestra.
- **Si te preguntan por el agarre real (H2)** — ten `04` a mano para mostrar la
  trayectoria de Diffusion Policy.
- **Cierre** — `05` en bucle de fondo durante el turno de preguntas.

## Notas técnicas

- Formato: MP4 (H.264 / AAC), 1280×720 los reels, listos para 16:9.
- Si proyectas desde VLC: `Reproducción → Repetir` para dejarlos en bucle.
- Origen reproducible: `experiments/results/{demo_reel,language_reel,pipeline_e2e,pick_with_diffusion}/`.
- Ya embebidos en las presentaciones finales (`Presentacion_Defensa_TFM.pptx` y `Presentacion_Robotica_IA.pptx`).
