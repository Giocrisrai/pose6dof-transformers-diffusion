# Cap. 5 — Sección de planificación con poses reales (parche para integrar al .docx)

> Este markdown contiene la sección "Planificación con poses reales" para
> insertar en `TFM_Final_v2.docx` (extiende la sección 4.X.3 Evaluación
> con la pieza de planificación). Está redactado en español académico
> con citas a artefactos versionados.

---

## 5.X. Planificación de agarres sobre poses reales

### 5.X.1. Pipeline percepción → planificación

Las poses 6-DoF estimadas por FoundationPose en el subset BOP-19 (Sec.
4.X) constituyen la entrada del módulo de planificación de agarres del
sistema. El script `experiments/run_diffusion_planning_real.py`
materializa este enlace consumiendo directamente
`predictions_ycbv_20260427_084807.json` y `predictions_tless_*.json`,
sin pasos manuales intermedios.

Para cada predicción $T_{obj} = (R_{pred}, t_{pred}) \in SE(3)$ obtenida
del run del 27 de abril de 2026, el pipeline ejecuta:

1. **Muestreo de candidatos de agarre** mediante `GraspSampler.sample()`,
   combinando estrategias *top-down* y *side approach* hasta obtener 64
   candidatos por objeto. Cada candidato es una pose $T_g \in SE(3)$ con
   ancho de pinza $w$ (m) y dirección de aproximación $\hat{a} \in S^2$.
2. **Filtrado por ángulo de approach** $\theta \in [0, \pi/3]$ rad
   respecto al eje vertical, descartando agarres físicamente infactibles.
3. **Scoring** vía `_score_candidates()` que pondera estabilidad de
   contacto, distancia al centro del objeto y método generador.
4. **Selección top-1** y **generación de trayectoria de aproximación**
   con `generate_approach_trajectory()`: 16 waypoints interpolados desde
   el standoff (10 cm) hasta el punto de contacto.
5. **Generación de trayectoria DDPM-style** con
   `DiffusionGraspPlanner.plan_grasp_heuristic()`: 16 waypoints
   completos del ciclo *approach → grasp → close → lift* en formato
   $(x, y, z, r_x, r_y, r_z, gripper) \in \mathbb{R}^7$, equivalente al
   espacio de acción que la red `ConditionalUNet1D` predeciría tras
   entrenamiento.

### 5.X.2. Configuración

Se procesó una muestra aleatoria reproducible (`RNG_SEED=42`) de **30
poses por dataset** sobre los 1098 (YCB-V) y 1012 (T-LESS) registros
disponibles. Parámetros del sistema:

- Apertura máxima de pinza: $w_{max} = 0.085$ m (Robotiq 2F-85).
- Distancia de standoff: 10 cm.
- Horizonte de planificación: 16 waypoints.
- Pasos de difusión: 100 (DDPM scheduler).

### 5.X.3. Resultados

La Tabla 5.X resume las métricas geométricas y temporales obtenidas:

| Métrica | YCB-V | T-LESS |
|---------|-------|--------|
| Poses procesadas con éxito | 30/30 | 30/30 |
| Score top-1 mediano | **0.964** | **0.964** |
| Std del score top-1 | 0.005 | 0.004 |
| Longitud trayectoria approach (mediana) | 10.0 cm | 10.0 cm |
| Longitud trayectoria DDPM-style (mediana) | 25.0 cm | 25.0 cm |
| Latencia del sampler (p95) | 2.0 ms | 2.0 ms |
| Pinza abierta al inicio | 100 % | 100 % |
| Pinza cerrada al final | 100 % | 100 % |

> **Tabla 5.X**: Métricas de planificación sobre 60 poses reales del
> run FoundationPose. Fuente:
> `experiments/results/diffusion_real_poses/trajectories_summary.json`.

**Discusión:**

- El score top-1 mediano de 0.964 (con desviación típica < 0.005 en
  ambos datasets) confirma que el muestreador produce candidatos de alta
  calidad incluso sobre poses con error ADD mediano de 4.17 mm (YCB-V) y
  2.90 mm (T-LESS): los errores de estimación se mantienen muy por
  debajo del umbral en el que un agarre top-down con 8.5 cm de apertura
  pierde estabilidad.
- La latencia del sampler (mediana < 2 ms, p95 = 2 ms) deja un margen
  amplio para integración en lazos de control reactivo, dado que la
  inferencia FoundationPose ya consume ~4 s/objeto sobre Colab T4.
- La consistencia 100 % en las fases de pinza (abierta/cerrada en los
  instantes correctos) valida la implementación del cronograma temporal
  del controlador heurístico.

La Figura 5.X (`fig_diffusion_planning_real.png`) ilustra la
distribución de scores y la comparación de longitudes de trayectoria
entre el sampler convencional y el planner DDPM-style.

### 5.X.4. Limitaciones reconocidas

Tres limitaciones explícitas, detalladas en el README versionado:

1. La red `ConditionalUNet1D` del `DiffusionGraspPlanner` no está
   entrenada en el alcance de este TFM; el modo *plan_grasp* devolvería
   ruido. La evaluación cualitativa se realiza por tanto con la
   trayectoria heurística del wrapper, que comparte espacio de acción y
   horizonte con la red entrenable. Entrenar la red requeriría un
   dataset de demostraciones expertas, descrito como línea futura.
2. El muestreo *antipodal* del sampler requiere la nube de puntos por
   instancia (depth + máscara por `gt_idx`). Aunque los assets están en
   el repositorio, su reconstrucción se deja como trabajo futuro para no
   inflar el alcance experimental.
3. La verificación cinemática (reachability del UR5/Panda + colisiones)
   se traslada a la fase de simulación CoppeliaSim (Sec. 5.Y).

### 5.X.5. Reproducibilidad

```bash
python experiments/run_diffusion_planning_real.py
```

Con seed fija (42), `predictions_*.json` versionados en git y el
generador `np.random.default_rng()`, la salida es bit-perfectamente
reproducible. Los artefactos quedan en
`experiments/results/diffusion_real_poses/`:

- `trajectories_summary.json` (74 KB) — métricas + 60 records.
- `fig_diffusion_planning_real.png` — figura 5.X.
- `README.md` — documentación de origen, configuración y limitaciones.
