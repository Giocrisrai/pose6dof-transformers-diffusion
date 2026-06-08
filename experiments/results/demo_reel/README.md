# Demo reel — mapa de uso

Generado por `experiments/build_demo_reel.py` (regenerable). Overlays cv2.

## Archivos

- `reel_resumen.mp4` — reel continuo (~2.4 min: 4 clips completos + tarjetas) para abrir/cerrar la charla.
- `clips/01_percepcion.mp4` — FoundationPose → pose 6-DoF.
- `clips/02_planificacion.mp4` — Diffusion Policy v7 (Iter 7c: curriculum RL + best-of-N + fix IK) → trayectoria.
- `clips/03_e2e.mp4` — pipeline end-to-end.
- `clips/04_robustez.mp4` — robustez en 3 escenarios.

## Mapa clip → slide + guión hablado

| Slide | Clip | Qué decir (con honestidad) |
|---|---|---|
| Percepción | `clips/01_percepcion.mp4` | "FoundationPose estima la pose 6-DoF del objeto. Validado sobre 1098 instancias de YCBV, ~4.2 s por pose. Las poses se calcularon offline en Colab." |
| Planificación | `clips/02_planificacion.mp4` | "La Diffusion Policy genera 16 waypoints. Partiendo del conditioning visual ResNet-18 RGB-D, dos refinamientos cierran el problema: un currículo de aprendizaje por refuerzo que maximiza el depósito, y una selección best-of-N en inferencia que recupera la precisión del agarre usando la pose de FoundationPose. En este pick, proximity al cubo de 3.9 cm y depósito a 4.9 cm — ambos plausibles. Sobre 50 picks con seed 2026: pick-and-place end-to-end del 84 %, superando al baseline supervisado del 60 %, con IK convergido al 100 % tras corregir la semilla del solver." |
| Ejecución | `clips/03_e2e.mp4` | "El pipeline completo corre con ciclo p95 de 5.2 s, bien bajo el umbral de 10 s. El grasp usa la técnica de attach, estándar en simuladores comerciales — lo declaramos explícitamente." |
| Robustez | `clips/04_robustez.mp4` | "Mismo pipeline en 3 escenarios (iluminación/colores): grasp_proximity 0.8 cm y IK convergido en los tres." |

## Plan B (modo en vivo)

Si la corrida en vivo en CoppeliaSim falla, proyectar `reel_resumen.mp4`.

## Regenerar

```bash
.venv/bin/python experiments/build_demo_reel.py
```

## Reel showcase (alto impacto)

`reel_showcase.mp4` — hero pick cinematográfico de Iter 7c (cámara dedicada
1280×720, coreografía órbita→seguimiento→retroceso, eje óptico +Z) + segmento
de valor y aplicaciones (logros, costo accesible, casos de uso, honestidad).
Separado del `reel_resumen.mp4` técnico (que queda para la defensa formal); este
es para producto/pitch. Regenerar:

    .venv/bin/python experiments/make_showcase_reel.py
