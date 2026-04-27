# Planificación con poses reales — evidencia experimental Cap. 5

Salidas del script `experiments/run_diffusion_planning_real.py` ejecutado
sobre las poses reales producidas por el run FoundationPose del 2026-04-27.

## Origen de las poses

Los waypoints aquí publicados consumen las predicciones serializadas en
`experiments/results/foundationpose_eval/predictions_{ycbv,tless}_20260427_084807.json`,
es decir poses 6-DoF estimadas por FoundationPose sobre el subset BOP-19
(YCB-V test, T-LESS test_primesense).

## Configuración

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `SAMPLE_SIZE` | 30 poses por dataset | Submuestreo aleatorio reproducible (`RNG_SEED=42`) sobre 1098 / 1012 poses |
| `N_GRASP_CANDIDATES` | 64 | Tras filtros de ángulo de approach, top-1 = mejor scored |
| `HORIZON` | 16 waypoints | Coincide con `DiffusionGraspPlanner.horizon` por defecto |
| Métodos del sampler | `topdown`, `side` | El modo `antipodal/surface` requiere nube de puntos por instancia, no disponible aquí |
| `gripper_width` | 0.085 m | Apertura típica de pinza paralela industrial (Robotiq 2F-85) |
| `standoff_distance` | 0.10 m | Aproximación 10 cm sobre el objeto |

## Métricas reportadas

- **Best grasp score (top-1)**: producido por `_score_candidates()` del
  `GraspSampler`, considera score base por método + bonus por estabilidad.
- **Approach trajectory length**: longitud euclídea acumulada de la
  trayectoria de aproximación del sampler (5 waypoints de standoff a contacto).
- **Diffusion trajectory length**: longitud de la trayectoria
  DDPM-style (`plan_grasp_heuristic`): approach → grasp → close → lift,
  16 waypoints, 7-DoF (xyz + axis-angle + gripper).
- **Smoothness (jerk)**: norma media del jerk discreto (3ª derivada de
  posición); más bajo = trayectoria más suave.
- **Sampling latency**: tiempo de pared del muestreo + filtrado + scoring,
  no incluye la planificación posterior.
- **Gripper phase consistency**: ¿la pinza está abierta al inicio y
  cerrada al final?

## Resultados (resumen del JSON)

Sobre 30 poses por dataset, todas procesadas con éxito (60/60):

| Métrica | YCB-V | T-LESS |
|---------|-------|--------|
| Best grasp score (mediana) | 0.964 | 0.964 |
| Std del score top-1 | 0.005 | 0.004 |
| Approach trajectory length (mediana) | 10.0 cm | 10.0 cm |
| Diffusion trajectory length (mediana) | 25.0 cm | 25.0 cm |
| Sampler latency p95 | 2.0 ms | 2.0 ms |
| Gripper open-at-start | 100 % | 100 % |
| Gripper closed-at-end | 100 % | 100 % |

Los valores idénticos entre datasets son esperables: ambos consumen el
mismo planner heurístico (`plan_grasp_heuristic`) — la única diferencia
con poses reales aparecerá cuando la red `ConditionalUNet1D` esté
entrenada y la inferencia sea estocástica multimodal (línea futura).

## Ficheros

- `trajectories_summary.json` — métricas agregadas + 60 records con
  scores, longitudes, latencias, posiciones de top-1 grasp.
- `fig_diffusion_planning_real.png` — distribución de scores y barras de
  longitud comparativas (PNG 160 dpi).
- `README.md` — este documento.

## Reproducibilidad

```bash
python experiments/run_diffusion_planning_real.py
```

Con seed fija (42) y predictions checkeados en git, los resultados son
determinísticos para `topdown` con jitter gaussiano (que también está
seedeado vía `np.random.seed(RNG_SEED)`).

## Limitaciones reconocidas

1. **Red de difusión no entrenada**: `DiffusionGraspPlanner` lleva una
   `ConditionalUNet1D` random-init en este TFM. Para evaluación
   cualitativa usamos `plan_grasp_heuristic` que el wrapper expone como
   baseline determinista del mismo formato. Entrenar la red requeriría
   un dataset de demostraciones que está fuera del alcance del TFM.
2. **Sin nube de puntos por instancia**: el sampler cae en `topdown` /
   `side`. Para `antipodal` real se necesitaría reproyectar el depth +
   máscara por `gt_idx` — dejado como trabajo futuro.
3. **Sin chequeo cinemático del UR5/Panda**: los waypoints son
   geométricamente plausibles en el frame de la cámara, pero no se
   verifica reachability del brazo. Esa validación corresponde a Fase 2
   (CoppeliaSim).
