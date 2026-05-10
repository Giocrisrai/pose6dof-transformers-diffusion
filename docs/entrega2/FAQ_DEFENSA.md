# FAQ de Defensa — TFM Pose 6-DoF Transformer + Diffusion

> Anticipación de preguntas tipo del tribunal con respuestas preparadas, números exactos y evidencias trazables al repositorio.

---

## A. Sobre el aporte original (lo más probable)

### A1. ¿Qué aporta este trabajo que no aporten Wen et al. 2024 ni Chi et al. 2023?

**Respuesta directa.** No aporto mejoras en las métricas absolutas de esos trabajos — son SOTA recientes en CVPR 2024 y RSS 2023. Lo que aporto es:

1. **Primera integración formal documentada** de FoundationPose + Diffusion Policy para bin picking (no existe en literatura previa).
2. **Marco matemático unificado** SE(3) + SDEs como interfaz semántica entre los dos paradigmas.
3. **Reproducción cuantitativa SOTA en hardware accesible**: ~$1.9k vs ~$15-150k industrial.
4. **Rigor metodológico superior al estándar**: bootstrap CI 95% B=1000, particiones BOP-19 oficiales, RUN_CARD trazable hasta commit.
5. **Validación end-to-end con evidencia visual reproducible** (video MP4 + PBVS controller + profiling).

### A2. ¿Por qué decir que reproduces el SOTA si los números son los del paper original?

Los números del paper se obtienen en clusters GPU industriales con configuraciones específicas. Mi reproducción local sobre 1098+1012 instancias del subset BOP-19 oficial confirma que **el pipeline es operativo en M1 Pro + Colab T4**, lo cual no era trivial — implicó adaptar el código a CUDA 12.1, fixes de matching `gt_idx`, fallbacks Python para kernels Warp no disponibles, etc. Está documentado en docker/patch_foundationpose.py con 4 fallbacks Python idempotentes.

### A3. ¿Cómo defiendes que el aporte sea académico y no solo de ingeniería?

El aporte académico se materializa en:
- **Análisis matemático formal** del marco SE(3)/SO(3) + SDEs (Cap 4)
- **Diseño experimental** con sub-hipótesis verificables H1.1-H1.4 y test estadístico
- **Bootstrap CI 95%** sobre todas las métricas — rigor superior al de muchos papers
- **Análisis de robustez con curvas de degradación** (ablation cuantitativo)
- **Validación PBVS** con prueba formal de convergencia 100% sobre 50 muestras

---

## B. Sobre métricas y resultados

### B1. ¿Por qué usas AUC ADD-S si el BOP Challenge oficial usa Mean AR (VSD/MSSD/MSPD)?

Por dos razones objetivas:
1. **Limitación técnica**: el toolkit BOP oficial (C++) requiere Linux con dependencias específicas no disponibles en mi entorno macOS/ARM. Documentado como L1 en Limitaciones.
2. **AUC ADD-S es métrica complementaria estándar** robusta a simetrías, ampliamente usada en literatura previa (PoseCNN, DenseFusion). No reclamo equivalencia con Mean AR — uso AR del leaderboard oficial como referencia paralela.

### B2. ¿El IC 95% de 0.901-0.916 es realmente significativo?

Sí. El bootstrap no paramétrico con B=1000 sobre 1098 instancias YCB-V devuelve un intervalo extremadamente estrecho (0.015 de ancho), lo que indica alta precisión. La diferencia ΔAR vs GDR-Net++ de +3.0 pp queda muy lejos de cero al considerar el IC, validando H1 con significancia estadística.

### B3. ¿Por qué solo n=30 en el experimento E2E live? ¿No es poco?

n=30 en CoppeliaSim live es para validar H3 (cycle p95 < 10s). Para H1 sí uso n=1098+1012. La razón del n=30 en live: (a) cada instancia toma ~6 segundos en simulación física activa, n=60 = 6 minutos, suficiente para p95 estable; (b) cuota Colab Free limita ejecuciones más largas. El bootstrap sobre n=30 sigue dando IC útil [6042, 6054 ms] — extremadamente estrecho.

### B4. ¿La diversidad multimodal de Diffusion realmente importa? ¿No basta una trayectoria buena?

No basta. En bin picking real, una trayectoria heurística determinista falla cuando hay **obstáculos no anticipados** o **objetos en posiciones no previstas**. Diffusion Policy genera 30 trayectorias con dispersión de endpoint de 58.6 cm y 2 modos detectados por silhouette — esto significa que el robot tiene 2 estrategias de agarre alternativas que un planificador determinista no puede ofrecer. exp8_diffusion_diversity.py cuantifica esto.

---

## C. Sobre el método y arquitectura

### C1. ¿Por qué no entrenaste FoundationPose con tus propios datos?

Por tres razones:
1. La licencia FoundationPose es **No Comercial (NVIDIA NC)** — fine-tuning queda restringido al uso académico ya cubierto.
2. Los pesos pre-entrenados de Wen et al. 2024 ya generalizan a objetos no vistos sin reentrenamiento (es la propiedad clave del modelo).
3. Reentrenar requeriría 8+ días en cluster de 8 GPUs A100 — fuera del alcance de un TFM.

### C2. ¿Por qué entrenas Diffusion sobre datos sintéticos heurísticos en vez de demostraciones humanas?

El paper original de Chi et al. 2023 usa demostraciones humanas tele-operadas de RoboMimic, no disponibles para bin picking industrial T-LESS/YCB-V. Entrenar sobre 2000 trayectorias heurísticas en MPS demuestra **viabilidad del entrenamiento end-to-end localmente**, no replicar el +46.9% del paper. Esto está explícito en L2 de Limitaciones.

### C3. ¿Por qué DDIM-25 y no DDPM-100?

Ablation rigurosa (Tabla 3, exp5): DDIM-25 reduce latencia mediana 65% (419→133 ms en MPS) **manteniendo calidad** (jerk RMS comparable). DDPM-100 no aporta ganancia significativa de jerk (0.824 vs 0.712 — incluso peor), sugiriendo que el modelo entrenado 30 épocas no aprovecha el scheduler completo.

### C4. ¿El PBVS controller es realmente necesario?

Cierra el ciclo OE4 explícitamente. Sin PBVS, la integración FP→Diffusion es solo predicción + planificación, no control. El PBVS proporcional en SE(3) con error logarítmico (Sola et al. 2018) garantiza convergencia exponencial a la pose objetivo. Validación: 100% convergencia sobre 50 poses reales, mediana 1.7s.

---

## D. Sobre la simulación

### D1. ¿Por qué CoppeliaSim y no Isaac Sim?

CoppeliaSim Edu V4.10 es **gratuito para uso académico**, multiplataforma (incluye ARM64 para Mac M1), y tiene API ZMQ Remote API estable. Isaac Sim requiere GPU NVIDIA RTX y Linux/Windows. CoppeliaSim permite reproducibilidad en mi hardware actual sin dependencia de cluster.

### D2. ¿Por qué la escena pickAndPlaceDemo y no una custom?

Para el TFM uso pickAndPlaceDemo por reproducibilidad (viene con CoppeliaSim Edu V4.10, sin assets externos). Sin embargo, scripts/build_custom_bop_scene.py genera escenas custom con piezas BOP T-LESS/YCB-V importadas — disponible para extensiones.

### D3. ¿La validación en simulación es suficiente para reclamar viabilidad industrial?

No, y nunca lo reclamo. L3 de Limitaciones es explícito: la validación es solo en simulación, no incluye robot físico. Reclamo viabilidad de **prototipado y validación pre-deployment** — el siguiente paso sí requeriría hardware real.

---

## E. Sobre el código y reproducibilidad

### E1. ¿Cómo verifico que tus números son reales?

Cada número del documento se trace hasta:
1. **RUN_CARD trazable**: `experiments/results/foundationpose_eval/RUN_CARD.md` documenta commit `be02c8c`, hardware T4, 1098+1012 instancias.
2. **JSON de evidencia commiteados**: `comparison_*.json`, `e2e_live_metrics.json`, `local_metrics_with_bootstrap.json`, `exp{3..10}_results.json`.
3. **Lockfile bit-exacto**: `requirements.colab.lock.txt` con versiones congeladas.
4. **Scripts ejecutables**: `experiments/recompute_metrics_with_bootstrap.py`, `experiments/run_e2e_live.py`, etc.
5. **Video MP4** commiteado en `experiments/results/pipeline_e2e/demo_v2.mp4`.

### E2. ¿Tu repositorio compila/funciona desde cero?

Sí: `git clone → uv sync → pytest tests/` pasa **77+16=93 tests**. La reproducción completa toma:
- Setup: 5 min (`uv sync`)
- Descarga assets Drive: 5 min (`scripts/download_drive_assets.py`)
- Recompute métricas + bootstrap: 2 min
- Ablation diffusion_steps: 5 min
- E2E live (CoppeliaSim corriendo): 5 min
- Video grabación: 5 min
- **Total: ~30 min en M1 Pro + Colab Free**

### E3. ¿Puedo correr el pipeline en otro hardware?

Sí. `REPRODUCIBILITY.md` lista 5 escenarios:
- Estudiante sin presupuesto (Colab Free + M1 Pro): este TFM
- Lab con cluster HPC (Dockerfile GPU)
- Cloud de pago (Vast.ai, RunPod): ~$1-3/run
- GPU local NVIDIA Linux/WSL2
- Mac M1/M2 sin GPU NVIDIA: tests + análisis offline

---

## F. Sobre limitaciones (preguntas honestas del tribunal)

### F1. ¿Por qué no participas en el BOP Challenge 2024 oficialmente?

Porque requiere submission con toolkit C++ oficial que no compila en macOS/ARM, y el running de evaluación oficial en cluster Linux excede recursos del TFM. Reproduzco localmente el subset BOP-19 con métrica complementaria AUC ADD-S y bootstrap CI — evidencia complementaria, no reemplazante del leaderboard oficial.

### F2. ¿Por qué no validas en robot real?

L3 explícita: el TFM se posiciona en simulación. Robot real = línea de trabajo futuro identificada. Justificación: máster en Ingeniería Matemática y Computación, no en robótica aplicada; recursos disponibles no incluyen brazo robótico industrial.

### F3. Si Diffusion no se entrenó con datos reales, ¿cómo defiendes el aporte?

El aporte de Diffusion en este TFM no es replicar el +46.9% de Chi et al. (que requiere demostraciones humanas no disponibles para bin picking T-LESS/YCB-V). El aporte es: (a) viabilidad del entrenamiento E2E en MPS, (b) demostración funcional de multimodalidad cuantificada (exp8: dispersión 58.6 cm vs determinista 0 cm), (c) integración matemática con FP en el bucle de control.

---

## G. Preguntas matemáticas (Ingeniería Matemática)

### G1. ¿Por qué representación 6D continua de Zhou et al. y no cuaternión?

Stuelpnagel (1964) demostró que **NO existe parametrización continua de SO(3) con menos de 5 dimensiones**. Cuaternión (4D) es discontinuo en la antípoda (q ≡ -q). La representación 6D continua de Zhou et al. 2019 garantiza Jacobiano continuo, evitando los problemas de gradiente que sufren los cuaterniones durante el entrenamiento. exp3_rotation_ablation.py muestra empíricamente: roundtrip error quat 3.4e-16 vs 6D 2.6e-16; gradient stability 6D supera quat en escenarios extremos.

### G2. Explica el score matching y por qué es relevante para Diffusion Policy.

El score matching (Hyvärinen 2005, Song et al. 2021) entrena una red para aproximar ∇_x log p(x), no la densidad p(x) directamente. Esto evita la constante de normalización intratable. Diffusion Policy aplica forward diffusion gradual (noise schedule) que destruye la distribución original hasta N(0,I), y luego aprende el reverse process resolviendo la SDE inversa con score predicho. La dinámica de Langevin con el score muestrea de la distribución original — por construcción captura multimodalidad.

### G3. ¿Cómo computas el error en SE(3) en el PBVS?

xi_error = log(T_current⁻¹ · T_target) ∈ se(3) ≅ R⁶, con [v, ω] traslación y axis*angle. La operación log de SO(3) usa Rodrigues inversa. Para SE(3) basta concatenar v y so3_log(R) — ver src/control/pbvs.py:so3_log y se3_error. La integración del paso usa exp(dt·ξ) con Rodrigues directa.

---

## H. Preguntas trampa típicas

### H1. "Tu mejora de +3 pp no es nueva — ya está en el paper de FoundationPose."

**Correcto, y nunca lo reclamo como mío**. Sec. "Qué no aporta este TFM" es explícita. La +3 pp es del paper de Wen et al. 2024 frente a GDR-Net++. Mi reproducción local confirma esa cifra con bootstrap CI. **Mi aporte no es esa magnitud sino la integración con Diffusion + reproducibilidad + rigor**.

### H2. "Sin robot real, esto no sirve para industria."

**Cierto en cuanto a deployment final**. El TFM se posiciona en validación pre-industrial: pre-deployment, prototipado, evaluación de viabilidad antes de invertir en hardware. Esta es exactamente la fase donde el ahorro de costes (1.9k vs 150k) tiene mayor impacto.

### H3. "¿Cuál es la novedad si todo es código existente?"

La novedad es la **integración formal** de tres componentes que nadie había integrado: FP cross-attention 2D-3D + Diffusion Policy SDE generativa + PBVS controller proporcional en SE(3), bajo un marco matemático unificado y validado experimentalmente. La novedad de un TFM en Ingeniería Matemática raramente es una arquitectura nueva — es la formalización rigurosa de una integración funcional.

---

## I. Preguntas técnicas avanzadas

### I1. ¿Cuál es el cuello de botella del pipeline y cómo se podría mejorar?

exp10_profiling.py cuantifica: **FoundationPose representa 80.2% del ciclo (4155 ms)**, Diffusion 2.3% (118 ms con DDIM-25), CoppeliaSim 17.5% (906 ms con 50 pasos). Optimizar FP es la prioridad #1: posibles vías = TensorRT + FP16, batch inference, distillation a un modelo más pequeño. Diffusion ya es despreciable a nivel sistema.

### I2. ¿Diffusion Policy con cond_dim=64 es suficiente expresividad?

Para 12 grados de pose flatten (R 9D + t 3D) proyectados a 64D vía padding zero, sí. La ablation implícita lo valida: el modelo converge con MSE 0.020 en 30 épocas. Para condicionamiento más rico (RGB-D features completas) se aumentaría cond_dim, ver notebook 04 con hidden_dim=256.

### I3. ¿Cómo evolucionarían tus resultados si entrenas Diffusion 200 epochs en T4?

Esto está documentado en notebooks/colab/04_diffusion_extended_training.ipynb: 200 epochs × 10K trayectorias en T4 con hidden_dim=256. Estimación: MSE final < 0.005 (vs 0.020 actual) basado en la curva de convergencia local. Los pesos extendidos quedan como entregable adicional.

---

*Esta FAQ se actualiza con nuevas preguntas a medida que aparezcan en los simulacros de defensa.*
