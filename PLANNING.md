# Planificación TFM — Pose 6-DoF con Transformers y Difusión

> **Plazo:** Abril 2026 → Julio 2026 (12 semanas)
> **Equipo:** Giocrisrai Godoy · José Miguel Carrasco
> **Directora:** Profesora Benítez
> **Hardware local:** MacBook Pro M1 Pro + Docker

---

## Fase 0 · Entorno de Trabajo (Semana 1: 4–10 abril)

### Setup Local
- [ ] Crear entorno virtual Python 3.10+ con dependencias del TFM
- [ ] Verificar PyTorch con soporte MPS (GPU Apple Silicon)
- [ ] Instalar Open3D, OpenCV, trimesh para manipulación 3D
- [ ] Instalar Docker Desktop y verificar que funciona en ARM64
- [ ] Clonar repositorio en máquina de José Miguel

### Docker para ROS 2
- [ ] Crear `Dockerfile` con ROS 2 Humble (base `ros:humble-desktop`)
- [ ] Añadir MoveIt 2, CoppeliaSim EDU dentro del contenedor
- [ ] Crear `docker-compose.yml` con volúmenes para `src/` y `data/`
- [ ] Verificar que CoppeliaSim abre GUI (XQuartz/X11 forwarding)
- [ ] Ejecutar demo MoveIt 2 `move_group` con robot genérico

### Datasets
- [ ] Descargar T-LESS (subset de test, ~2 GB) desde BOP Challenge
- [ ] Descargar YCB-Video (subset de test, ~3 GB) desde BOP Challenge
- [ ] Organizar en `data/datasets/{tless,ycb_video}/`
- [ ] Verificar carga de imágenes RGB-D y anotaciones con script de prueba
- [ ] (Opcional) Descargar modelos CAD de objetos T-LESS para visualización

---

## Fase 1 · Baselines y Reproducción (Semanas 2–4: 11 abril – 1 mayo)

### 1A — GDR-Net (Baseline Comparativo)
- [ ] Clonar repo oficial GDR-Net: `github.com/shanice-l/gdrnpp_bop2022`
- [ ] Configurar entorno con dependencias (puede requerir Docker x86 con Rosetta)
- [ ] Descargar pesos pre-entrenados para T-LESS y YCB-Video
- [ ] Ejecutar inferencia en imágenes de test de T-LESS
- [ ] Ejecutar inferencia en imágenes de test de YCB-Video
- [ ] Registrar métricas BOP: VSD, MSSD, MSPD (usar `bop_toolkit`)
- [ ] Guardar resultados en `experiments/gdrnet/results_tless.json`
- [ ] Guardar resultados en `experiments/gdrnet/results_ycb.json`
- [ ] Documentar en notebook: `notebooks/01_gdrnet_baseline.ipynb`

### 1B — FoundationPose (Método Principal)
- [ ] Clonar repo oficial: `github.com/NVlabs/FoundationPose`
- [ ] Configurar con Docker (requiere GPU NVIDIA → usar Colab para training)
- [ ] Descargar pesos pre-entrenados del model zoo
- [ ] Ejecutar inferencia en T-LESS (en Colab si no hay GPU local)
- [ ] Ejecutar inferencia en YCB-Video
- [ ] Registrar métricas BOP: VSD, MSSD, MSPD
- [ ] Guardar resultados en `experiments/foundationpose/results_tless.json`
- [ ] Guardar resultados en `experiments/foundationpose/results_ycb.json`
- [ ] Documentar en notebook: `notebooks/02_foundationpose_eval.ipynb`
- [ ] **Comparar vs GDR-Net** — tabla comparativa de métricas

### 1C — Diffusion Policy (Planificación)
- [ ] Clonar repo oficial: `github.com/real-stanford/diffusion_policy`
- [ ] Ejecutar demo Push-T en local (no requiere GPU pesada)
- [ ] Entender API: observación → acción con diffusion sampling
- [ ] Documentar en notebook: `notebooks/03_diffusion_policy_demo.ipynb`

---

## Fase 2 · Fundamentos Matemáticos + Capítulo 4 (Semanas 3–5: 18 abril – 15 mayo)

### Implementaciones Matemáticas (src/utils/)
- [ ] `src/utils/lie_groups.py` — Funciones SE(3), SO(3): exp, log, adjoint
- [ ] `src/utils/rotations.py` — Conversiones: quaternion ↔ matrix ↔ 6D ↔ axis-angle
- [ ] `src/utils/metrics.py` — Implementar VSD, MSSD, MSPD (wrapper de bop_toolkit)
- [ ] `src/utils/visualization.py` — Visualizar poses sobre imagen RGB, nubes de puntos
- [ ] Tests unitarios para cada módulo (`pytest`)
- [ ] Notebook demostrativo: `notebooks/04_math_foundations.ipynb`
  - [ ] Visualizar rotaciones en SO(3) con Open3D
  - [ ] Demostrar exp/log maps
  - [ ] Score matching conceptual en 2D (toy example)

### Redacción Capítulo 4 — Marco Matemático
- [ ] 4.1 Grupos de Lie: SE(3) como grupo de isometrías rígidas
- [ ] 4.2 Representaciones de rotación y singularidades (Gimbal lock, continuidad 6D)
- [ ] 4.3 Mecanismo de atención multi-cabeza como operador en SE(3)
- [ ] 4.4 SDEs, score matching y dinámica de Langevin para difusión
- [ ] 4.5 Conexión matemática: FoundationPose (atención en SE(3)) + Diffusion Policy (SDE en espacio de acciones)
- [ ] Revisión cruzada del capítulo (José Miguel ↔ Giocrisrai)

---

## Fase 3 · Pipeline Integrado (Semanas 5–8: 2 mayo – 29 mayo)

### 3A — Percepción: FoundationPose Wrapper
- [ ] `src/perception/foundation_pose.py` — Clase wrapper con API unificada
  - [ ] `estimate_pose(rgb, depth, mask, cad_model) → SE(3)`
  - [ ] `track_pose(rgb_sequence, ...) → SE(3) trajectory`
- [ ] `src/perception/gdrnet.py` — Clase wrapper GDR-Net (mismo API)
- [ ] `src/perception/detector.py` — Detección 2D (CNOS/SAM para segmentación)
- [ ] Test de integración: imagen → detección → pose 6-DoF

### 3B — Planificación: Diffusion Policy para Bin Picking
- [ ] `src/planning/diffusion_grasp.py` — Adaptar Diffusion Policy para grasping
  - [ ] Input: pose 6-DoF del objeto + estado del gripper
  - [ ] Output: trayectoria de agarre (secuencia de waypoints SE(3))
- [ ] `src/planning/grasp_sampler.py` — Muestreo de agarres candidatos
- [ ] Entrenar/fine-tune en datos sintéticos (Colab GPU)
- [ ] Notebook: `notebooks/05_grasp_planning.ipynb`

### 3C — Integración End-to-End
- [ ] `src/pipeline.py` — Orquestador del pipeline completo:
  1. Captura RGB-D
  2. Detección + Segmentación
  3. Estimación de Pose (FoundationPose)
  4. Planificación de Agarre (Diffusion Policy)
  5. Ejecución (envío a simulador)
- [ ] Test end-to-end con imágenes estáticas de T-LESS
- [ ] Notebook: `notebooks/06_full_pipeline.ipynb`

---

## Fase 4 · Simulación CoppeliaSim + ROS 2 (Semanas 6–9: 9 mayo – 5 junio)

### Escena de Simulación
- [ ] Crear escena CoppeliaSim: mesa + bin + objetos T-LESS
- [ ] Configurar cámara RGB-D virtual (simular Intel RealSense)
- [ ] Añadir robot manipulador (UR5e o Franka Panda)
- [ ] Configurar gripper (paralelo o vacuum)
- [ ] Verificar comunicación CoppeliaSim ↔ ROS 2 via ZMQ Remote API

### Visual Servoing (IBVS/PBVS)
- [ ] `src/simulation/visual_servoing.py` — Implementar IBVS y PBVS
- [ ] Integrar con pose estimada por FoundationPose
- [ ] Test: servo hacia objeto con pose conocida (ground truth)
- [ ] Test: servo con pose estimada (pipeline real)

### MoveIt 2 + Ejecución
- [ ] Configurar MoveIt 2 para el robot elegido (URDF/SRDF)
- [ ] Planificación de trayectoria con MoveIt 2
- [ ] Ejecución de agarre completo en simulación
- [ ] Grabar videos de demos exitosas para la memoria

---

## Fase 5 · Experimentos y Resultados (Semanas 8–10: 23 mayo – 19 junio)

### Experimentos Cuantitativos
- [ ] **Exp 1:** FoundationPose vs GDR-Net en T-LESS (VSD, MSSD, MSPD)
- [ ] **Exp 2:** FoundationPose vs GDR-Net en YCB-Video
- [ ] **Exp 3:** Ablation — efecto de la representación de rotación (quat vs 6D)
- [ ] **Exp 4:** Diffusion Policy vs heuristic grasping (success rate en sim)
- [ ] **Exp 5:** Pipeline end-to-end — tasa de éxito de bin picking en CoppeliaSim
- [ ] Generar tablas de resultados (LaTeX/Markdown)
- [ ] Generar gráficos comparativos (matplotlib/plotly)
- [ ] Guardar todo en `experiments/results/`

### Experimentos Cualitativos
- [ ] Visualizaciones de poses estimadas sobre imágenes reales
- [ ] Visualización de trayectorias de agarre generadas por Diffusion Policy
- [ ] Videos de simulación CoppeliaSim (ciclos de bin picking)
- [ ] Análisis de casos de fallo (oclusión, objetos simétricos, reflectantes)

---

## Fase 6 · Redacción Final + Entrega (Semanas 10–12: 13 junio – 3 julio)

### Capítulos Pendientes
- [ ] Capítulo 5 — Diseño del Pipeline y Arquitectura
- [ ] Capítulo 6 — Experimentos y Resultados
- [ ] Capítulo 7 — Conclusiones y Trabajo Futuro
- [ ] Revisar y pulir Capítulos 1–4 (ya escritos)
- [ ] Añadir todas las figuras y tablas de resultados
- [ ] Bibliografía completa (30 papers + refs adicionales)
- [ ] Resumen / Abstract final

### Entregables Finales
- [ ] Memoria TFM completa en formato UNIR (.docx)
- [ ] Código fuente limpio en GitHub (hacer público si la licencia lo permite)
- [ ] Presentación de defensa (.pptx, ~15 slides)
- [ ] Video demo del pipeline funcionando
- [ ] README.md final con instrucciones de reproducción
- [ ] Revisar normativa UNIR de entrega y formato
- [ ] **ENTREGA** 🎯

---

## Resumen de Semanas

| Semana | Fechas | Foco Principal | Entregable |
|--------|--------|---------------|------------|
| 1 | 4–10 abr | Setup entorno + Docker + Datasets | Entorno funcional |
| 2 | 11–17 abr | GDR-Net baseline | Métricas baseline T-LESS/YCB |
| 3 | 18–24 abr | FoundationPose eval + Matemáticas | Comparativa FP vs GDR-Net |
| 4 | 25 abr–1 may | Diffusion Policy demo + Cap. 4 | Notebooks 01–04 |
| 5 | 2–8 may | Pipeline percepción integrado | src/perception/ completo |
| 6 | 9–15 may | CoppeliaSim escena + ROS 2 Docker | Simulación básica |
| 7 | 16–22 may | Diffusion grasp + MoveIt 2 | Agarre en simulación |
| 8 | 23–29 may | Integración E2E + Experimentos | Pipeline completo |
| 9 | 30 may–5 jun | Experimentos finales | Tablas y gráficos |
| 10 | 6–12 jun | Redacción Cap. 5–6 | Borrador completo |
| 11 | 13–19 jun | Redacción Cap. 7 + Revisión | Memoria casi final |
| 12 | 20 jun–3 jul | Pulido + Presentación + Entrega | **TFM ENTREGADO** |

---

## Notas Técnicas

### Hardware
- **Local (M1 Pro):** Simulación, inferencia ligera (MPS), desarrollo, redacción
- **Colab (T4 GPU):** Training/fine-tune FoundationPose, Diffusion Policy
- **Docker:** ROS 2 Humble + MoveIt 2 + CoppeliaSim (contenedor ARM64)

### Repartición Sugerida
- **Giocrisrai:** Percepción (FoundationPose, GDR-Net), Pipeline, Matemáticas
- **José Miguel:** Simulación (CoppeliaSim, ROS 2), Visual Servoing, MoveIt 2
- *(Ajustar según fortalezas de cada uno)*

### Repos Externos Clave
- FoundationPose: `https://github.com/NVlabs/FoundationPose`
- GDR-Net++: `https://github.com/shanice-l/gdrnpp_bop2022`
- Diffusion Policy: `https://github.com/real-stanford/diffusion_policy`
- BOP Toolkit: `https://github.com/thodan/bop_toolkit`
- CoppeliaSim: `https://www.coppeliarobotics.com/downloads`
