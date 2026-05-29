# Demo reel — mapa de uso

Generado por `experiments/build_demo_reel.py` (regenerable). Overlays cv2.

## Archivos

- `reel_resumen.mp4` — reel continuo (~2.4 min: 4 clips completos + tarjetas) para abrir/cerrar la charla.
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
