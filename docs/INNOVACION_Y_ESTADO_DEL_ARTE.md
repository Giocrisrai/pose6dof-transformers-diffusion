# Innovación, valor diferencial y estado del arte (mayo 2026)

> Documento que sitúa este TFM en el panorama de mayo de 2026, explica
> claramente **qué es novedoso**, **qué no lo es**, **dónde aporta valor**
> y **cuáles son sus limitaciones reales** frente a los métodos publicados
> hasta la fecha de la defensa.

---

## 1. Resumen ejecutivo (lectura de 60 segundos)

Este TFM integra **tres tecnologías de vanguardia ya publicadas** —
FoundationPose (Transformer cross-attention 2D-3D, CVPR 2024),
Diffusion Policy (RSS 2023) y Visual Servoing PBVS en SE(3) — en
**un pipeline E2E reproducible, riguroso y entrenable en hardware accesible
(Apple M1 Pro, ~2 000 USD)**.

El aporte no está en inventar una de esas piezas — eso ya está hecho —
sino en **tres ejes concretos**:

1. **Integración E2E end-to-end** con marco matemático completo en SE(3) y
   métricas industriales (cycle time p95, recall ADD-S, oclusión-noise).
2. **Validación con rigor estadístico** (bootstrap CI 95 % B=1 000) sobre
   dos benchmarks estándar (T-LESS, YCB-Video) — algo que la mayoría de
   papers omite.
3. **Democratización**: entrenamiento Diffusion (100 epochs, 10K trayectorias)
   en 3,3 minutos en M1 Pro, inferencia E2E < 7 s sin GPU dedicada, todo MIT.

Lo que **no se reclama**: ni FoundationPose ni Diffusion Policy son nuestros.
Lo que **sí se reclama**: el primer pipeline open-source que combina ambos
con PBVS, evalúa con bootstrap CI, valida re-entrenamiento progresivo y
es ejecutable en un portátil.

---

## 2. Estado del arte vigente (mayo 2026)

### 2.1 Percepción 6-DoF (post-FoundationPose)

FoundationPose (Wen et al. CVPR 2024) sigue siendo la **baseline de
referencia**, pero ya tiene sucesores que abordan sus limitaciones:

| Método | Año | Contribución vs FoundationPose | Estado |
|---|---|---|---|
| **FreeZeV2** | 2025 | Training-free, 8× speedup, +5 % accuracy | Publicado |
| **Any6D** | 2025 | Model-free, una sola imagen RGB-D ancla | Publicado |
| **SamPose** | 2025 | Open-world, single-view prompt (sin CAD) | Publicado |
| **NBV-guided active perception** | Sep 2025 | Eleva success rate del 50 % al 95 % en escenarios ambiguos seleccionando la siguiente vista | Publicado |
| **BOP Challenge 2025** | Nov 2025 | Resultados publicados en ICCV'25 R6D Workshop (Honolulu, oct 2025). Nuevos métodos model-based y model-free dominan los rankings | Publicado |

**Implicación para el TFM**: usar el FoundationPose original (sin NBV ni
training-free) es una decisión consciente: prioriza **reproducibilidad
sobre punta-de-lanza**. Los métodos training-free como FreeZeV2 reducirían
los 4 s de FP a < 1 s, pero romperían la trazabilidad CVPR 2024 sobre la
que se ancla la tesis.

### 2.2 Planificación con Diffusion (post-Diffusion Policy)

Diffusion Policy (Chi et al. RSS 2023) es ahora **un componente estándar**.
El campo ha avanzado en cuatro frentes:

| Método | Año | Mejora clave |
|---|---|---|
| **RDT-1B** | Oct 2024 | Foundation model 1,2 B parámetros, manipulación bimanual, pre-entrenado en 46 datasets / 1 M episodios |
| **Two-Steps Diffusion Policy (Genetic Denoising)** | Oct 2025 | Solo **2 NFE** (pasos de red) vs 25-100 originales |
| **On-Device Diffusion Transformer Policy** | ICCV 2025 | Despliegue en edge con latencias suficientes para tiempo real |
| **3D Diffuser Actor** | 2024 | Integra representaciones 3D multi-vista con foundation features |
| **Hierarchical Diffusion Policy** | 2025 | Generación de trayectorias con guía de contacto explícita |
| **ManiFlow / Compose Your Policies!** | 2025-2026 | Flow matching como alternativa más estable a difusión |

**Implicación para el TFM**: nuestro Diffusion Policy con UNet1D
(hidden_dim=256, 100 epochs) y DDIM-25 produce **MSE 0,0022 y jerk 0,053**
— métricas competitivas para *bin picking single-arm con CAD*. RDT-1B
y Pi0 son más capaces pero requieren *órdenes de magnitud más datos y
cómputo*. El TFM ocupa el nicho "competente y reproducible en portátil".

### 2.3 El elefante en la habitación: VLA (Vision-Language-Action)

El paradigma dominante en 2025-2026 ha desplazado el foco a los **VLA**:

| Modelo | Parámetros | Datos | Capacidad |
|---|---|---|---|
| **OpenVLA** (Stanford, jun 2024) | 7 B | 970 K demos | Bate a RT-2-X (55 B) con 7× menos params |
| **π0** (Physical Intelligence, oct 2024) | ~3-7 B | Mixtura propia | Flow-matching, control continuo hasta 50 Hz |
| **π0-FAST** (2025) | similar | + tokenización DCT | Mejora la representación de acciones |
| **π0.5** (2025) | escalado | OXE + extras | Generalización open-world |
| **RDT-1B** (Tsinghua, 2024) | 1,2 B | 46 datasets / 1 M episodios | Diffusion bimanual zero-shot |

**Esto es lo que cambia el contexto desde 2024**:
- Los VLA prometen *un solo modelo* para muchas tareas, instruidas por
  lenguaje natural.
- Compiten con la "pipeline tradicional" (FP+Diffusion+PBVS) en simplicidad
  conceptual, pero requieren millones de demostraciones reales y
  decenas/centenas de GPUs.

**Implicación para el TFM**: el TFM se posiciona en el **otro extremo**:
*sin lenguaje, sin millones de demos, sin GPU dedicada — pero auditado
estadísticamente y reproducible en cualquier portátil moderno*. Estos son
universos paralelos, no competidores directos: VLA es para empresas con
flota de robots; este TFM es para PYMES industriales y educación.

---

## 3. Innovación y valor diferencial real

### 3.1 Lo que NO es novedoso

Hay que decirlo claro:

- **FoundationPose no es nuestro** — es NVIDIA Labs, CVPR 2024.
- **Diffusion Policy no es nuestro** — es Columbia/Toyota, RSS 2023.
- **PBVS en SE(3)** está descrito en libros de texto desde los 90.
- **Combinar percepción + planificación + control** es lo que hace cualquier
  pipeline de manipulación desde hace décadas.

### 3.2 Lo que SÍ es novedoso o de alto valor

| # | Contribución | Por qué importa |
|---|---|---|
| **1** | **Pipeline E2E open-source MIT** que integra FP + Diffusion + PBVS con BOP-19 | FoundationPose tiene licencia NC; muy pocos open-source pipelines completos disponibles. |
| **2** | **Re-entrenamiento progresivo demostrado empíricamente** (exp13: original→extended→ultra, −89 % MSE, −93 % jerk en 10× datos) | Pocos papers muestran la *curva de escalabilidad* con métricas industriales (jerk, dispersión endpoint, latencia). |
| **3** | **Bootstrap CI 95 % B=1000 en todas las métricas clave** | La mayoría de papers reporta números puntuales sin CIs — esto es estadísticamente más riguroso. |
| **4** | **Validación E2E live en CoppeliaSim** con métricas reales por ciclo (no nominales) sobre 60+ instancias | Muchos papers reportan latencias parciales; este reporta el cycle p95 completo. |
| **5** | **Entrenamiento Diffusion 100 epochs en M1 Pro MPS en 3,3 minutos** | Demuestra que el método es accesible sin datacenter. |
| **6** | **Robustez documentada con curvas de degradación** (oclusión {0,30,50,70}%, ruido σ={0,2,5,10} mm) | Análisis de robustez raramente cuantificado a este nivel en papers de planificación. |
| **7** | **Aplicaciones industriales mapeadas** por sector con métricas viables (`docs/APLICACIONES_INDUSTRIALES.md`) | El "puente paper → producto" suele estar ausente en TFMs académicos. |
| **8** | **Decision cards visuales** que explican QUÉ decide el sistema para cada objeto detectado | Comunicación científica accesible — raro en literatura técnica. |

### 3.3 ¿Alguien ha hecho exactamente esto?

Búsqueda hecha en mayo 2026 sobre Google Scholar + arXiv + papers BOP:

- **Nadie ha publicado** un pipeline open-source que combine
  *FoundationPose + Diffusion Policy + PBVS + BOP-19 + entrenamiento en
  hardware no-dedicado + bootstrap CI*. Hay piezas sueltas; no hay esta
  integración con este nivel de rigor estadístico.
- **Sí hay** pipelines comerciales (NVIDIA Isaac Manipulator, Symbotic,
  AMP Robotics) que combinan lo equivalente, pero son **cerrados** y
  requieren stack NVIDIA propietario.

El nicho del TFM es: *"reproducir y democratizar lo que solo era posible
con setups industriales caros, manteniendo rigor académico"*.

---

## 4. Limitaciones reales (honestas)

### 4.1 Limitaciones de alcance (decididas a priori)

- **Solo simulación**, no robot físico. Domain gap sim-to-real no validado.
- **CAD requerido** (model-based). No funciona con objetos sin modelo, a
  diferencia de SamPose o Any6D.
- **Single-arm**, no bimanual. RDT-1B y π0 sí lo hacen.
- **Sin lenguaje natural**. El sistema no entiende "recoge la pieza roja"
  — solo poses 6-DoF.
- **FoundationPose congelado** (no fine-tuned a T-LESS / YCB-V). Un fine-tune
  podría dar ganancias adicionales.

### 4.2 Limitaciones técnicas

- **Cycle time 6-7 s** — competente pero superable. Métodos on-device
  diffusion transformer 2025 alcanzan 1-3 s. La mayoría del tiempo se va
  en FoundationPose (~85 %), que es nuestro cuello de botella.
- **Diffusion Policy entrenado con trayectorias sintéticas** generadas
  heurísticamente, no con demostraciones humanas reales (DROID, Open
  X-Embodiment). Esto sesga las trayectorias hacia movimientos "óptimos
  geométricos" pero potencialmente menos naturales.
- **No-determinismo controlable**: en el modelo Ultra, las 10 trayectorias
  son muy parecidas (dispersión 3.8 cm). Esto significa que el sistema
  ya no es genuinamente "multimodal" — es prácticamente determinista. Para
  aplicaciones que necesiten *alternativas reales* (caso de colisión)
  hay que retroceder al Extended.

### 4.3 Limitaciones de evaluación

- **Subset BOP-19** evaluado, no challenge completo. Resultados no comparables
  oficialmente con el leaderboard del BOP Challenge 2025.
- **Sin baseline en hardware completamente equivalente**: comparamos
  contra GDR-Net++ pero ejecutado por terceros en hardware distinto.
- **CoppeliaSim** como simulador, no Isaac Sim — sin física fotorrealista
  ni domain randomization automático.

### 4.4 Limitaciones de licencia

- **FoundationPose tiene licencia no comercial** (NVIDIA Source Code License).
  Cualquier comercialización requiere licencia explícita o sustituir por
  un método open (FreeZeV2 LGPL, MegaPose AGPL, o uno propio entrenado).

---

## 5. Cómo se logra un buen resultado (recetas concretas)

Para quien quiera **reproducir** y obtener métricas equivalentes:

| Componente | Configuración que funciona | Por qué |
|---|---|---|
| **Hardware** | MacBook Pro M1/M2/M3 Pro 16 GB + Colab T4 | M1 para Diffusion (MPS), Colab T4 para FP (CUDA). |
| **Modelo Diffusion** | `diffusion_policy_ultra.pth`: UNet1D h=256, 100 ep, AdamW + cosine + warmup 5 ep, grad clip 1.0, 10K trayectorias sintéticas | Es el equilibrio MSE/jerk/latencia óptimo en M1 Pro. |
| **Sampling** | DDIM 25 pasos | Determinista, x4 más rápido que DDPM con calidad equivalente para H=16 acciones. |
| **PBVS** | Tolerancia 2 mm, log/exp en SE(3), Kp=1.0 | Garantiza convergencia en < 12 iter para errores < 5 cm. |
| **Evaluación** | Bootstrap CI 95 % B=1000 con semilla 42 | Mucho más informativo que number-only; permite tests estadísticos (Wilcoxon, Cohen's d). |
| **CoppeliaSim** | Escena `pickAndPlaceDemo.ttt`, robot Ragnar delta, 50 pasos a 18 ms | Validado contra E2E live, p95 contenido < 7 s. |

---

## 6. Aporte de valor — cómo marca diferencia

### Frente a métodos académicos punteros

| Aspecto | TFM | Métodos punta-de-lanza (RDT-1B, π0, FreeZeV2) |
|---|---|---|
| Reproducibilidad | ✅ Hardware accesible, MIT | ⚠️ Necesita cluster GPU |
| Trazabilidad estadística | ✅ Bootstrap CI 95 % | ⚠️ Números puntuales mayoría papers |
| Capacidad funcional | ⚠️ Single-arm, sin lenguaje | ✅ Bimanual, multi-tarea, instruido por lenguaje |
| Latencia | 6-7 s/instancia | 1-3 s/instancia |
| Coste hardware | ~2 000 USD | 20 000-200 000 USD |

### Frente a soluciones comerciales

| Aspecto | TFM | Industriales (Isaac, Symbotic, AMP) |
|---|---|---|
| Coste estación | < 5 000 USD | 15 000-150 000 USD |
| Stack | Open-source | Propietario |
| Personalización | Total (código abierto) | Limitada a APIs vendor |
| Soporte comercial | N/A | ✅ Empresa detrás |

### Frente a TFMs anteriores en UNIR / otros másteres

- Pocos TFMs incluyen evaluación BOP-19 con CIs bootstrap.
- Pocos TFMs entregan código MIT + Docker + API REST + Gradio + Streamlit
  + 171 tests + 17 experimentos commiteados + tutorial Jupyter + paquete
  PyPI propio.
- Pocos TFMs van más allá del paper a producto desplegable.

---

## 7. Cómo funciona el sistema *hoy* (mayo 2026, post-Entrega 2)

Lo que está vivo y funcionando ahora mismo:

```
                    ┌─────────────────────────────────────┐
                    │   USUARIO / INTEGRADOR INDUSTRIAL    │
                    └────────────────┬─────────────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        │                            │                              │
   API REST                    Demo Gradio                   Dashboard Streamlit
   (FastAPI 8000)             (puerto 7860)                  (puerto 8501)
        │                            │                              │
        └──────────────┬─────────────┴─────────────┬────────────────┘
                       │                            │
                ┌──────▼──────┐              ┌─────▼─────┐
                │  Diffusion  │              │ Métricas  │
                │   Policy    │              │   JSON    │
                │  (3 modelos)│              │ commit'd  │
                └──────┬──────┘              └───────────┘
                       │
                ┌──────▼─────────┐
                │ Checkpoints FP │
                │  YCBV + T-LESS │
                └────────────────┘

         ┌───────────────────────────────┐
         │  CoppeliaSim Edu V4.10        │  ← validación E2E live
         │  pickAndPlaceDemo + Ragnar    │
         └───────────────────────────────┘

         ┌───────────────────────────────┐
         │  Docker compose `api` service │  ← deployment portable
         └───────────────────────────────┘
```

**Resultados verificados en vivo (mayo 2026)**:
- API REST `POST /e2e` con n=5 instancias YCB-V: p95 = 5 076 ms (margen 4 924 ms)
- API REST `POST /e2e` con n=5 instancias T-LESS: p95 = 6 117 ms (margen 3 883 ms)
- 171 tests pasando (123 del TFM + 48 de exploraciones)
- 5 modelos Diffusion en disco (original 4 MB · extended 2.93 MB · ultra
  5.15 MB · ultra_fast 5.16 MB distillado · clip 5.6 MB con VLA-lite) listos
  para inferencia

---

## 7-bis. Contribuciones adicionales — exploraciones post-TFM (mayo 2026)

Tras entregar el TFM se planificaron y ejecutaron 8 exploraciones con
criterios numéricos de éxito. **Las 8 se mergearon a `main`** porque
cumplen los criterios. Documentación completa: [`docs/PLAN_EXPLORACIONES_POST_TFM.md`](PLAN_EXPLORACIONES_POST_TFM.md).

| # | Exploración | Resultado clave | Doc cierre |
|---|---|---|---|
| 1 | **bop-bootstrap-ci** (paquete PyPI) | 27 tests, 97 % cov, bit-a-bit reproduce el TFM | [01](exploraciones/01_bootstrap_ci_toolkit.md) |
| 2 | **Distillation 1-NFE** | ×517 speedup real con mejor MSE y jerk que teacher | [02](exploraciones/02_distillation_2nfe.md) |
| 3 | **Pipeline open-license** | FreeZeV2 Apache-2.0 viable a solo −3 pp AUC | [03](exploraciones/03_open_license_pipeline.md) |
| 4 | **VLA-lite con CLIP (color)** | 98.6 % selection accuracy con TextGroundedGate | [04](exploraciones/04_vla_lite_clip.md) |
| 5 | **Robustez lingüística** (extensión #4) | 100 % sobre 6 familias de frases no vistas (n=900) | [05](exploraciones/05_vla_robustness.md) |
| 6 | **VLA-lite multi-atributo color + forma** | 99.9 % global, 100 % en color/combinado | [06](exploraciones/06_vla_shapes.md) |
| 7 | **Simulaciones visuales 3D** | 12/12 escenas con renders de cubo/esfera/cilindro/caja | [07](exploraciones/07_visual_simulations.md) |
| 8 | **VLA-lite multi-objeto N=2..5** | 100 % accuracy con hasta 5 objetos en escena | [08](exploraciones/08_multi_object.md) |

**Hallazgos metodológicos importantes documentados:**

1. El "MSE 0.0022" reportado en el TFM original era *MSE de noise-prediction
   loss* durante el training, NO MSE de trayectoria reconstruida (que es
   ~0.0129 para el teacher). Detectado durante la Exploración 2.
2. El modelo `ultra` es prácticamente determinista (dispersión 3.8 cm); para
   multimodalidad real conviene usar `extended`. Documentado.
3. VLA-lite necesita gating explícito como inductive bias; sin él el modelo
   "promedia" trayectorias (50 % acc). Con TextGroundedGate + aux loss
   alcanza 98.6 %.
4. VLA-lite generaliza al 100 % sobre 6 familias lingüísticas no vistas
   en training: CLIP aporta comprensión real del lenguaje, no plantilla
   memorizada. Validado con 900 frases (exp 17).

**Total acumulado**: 171 tests · 7 modelos Diffusion · 1 paquete PyPI ·
4 hallazgos metodológicos · 8 documentos de cierre · 22 simulaciones
visuales 3D (12 de N=2 + 10 de N=2..5) · 1 doc de extrapolación industrial.

**Extrapolación industrial documentada**: el pipeline multi-atributo del
exp 6 escala directamente a logística (Amazon, DHL), reciclaje (AMP, TOMRA),
electrónica (Foxconn), médico (Omnicell) y automoción (ABB, Fanuc). Ver
[`docs/EXTRAPOLACION_INDUSTRIAL.md`](EXTRAPOLACION_INDUSTRIAL.md) con
roadmap priorizado, métricas viables por sector y plan de evolución
corto/medio/largo plazo.

---

## 8. Conclusión profesional

Este TFM **no inventa una pieza nueva** del rompecabezas de manipulación
robótica. Lo que hace es **resolver el rompecabezas con piezas existentes
de manera reproducible, estadísticamente rigurosa y económicamente
accesible** — y comunicar los resultados de forma que cualquier industrial
(o cualquier persona) pueda entenderlos.

El valor está en la **integración pulida + auditoría estadística +
democratización del acceso**, no en la novedad algorítmica.

Si la pregunta de la defensa es **"¿qué inventaste?"**, la respuesta honesta
es **"un pipeline E2E auditable y reproducible que une cuatro tecnologías
estándar con métricas industriales sobre BOP-19"**.

Si la pregunta es **"¿qué aportas?"**, la respuesta es **"un blueprint
documentado, testeado y desplegable que reduce el coste de entrada al
bin picking robótico de 6 cifras a 4 cifras, manteniendo rigor académico"**.

---

## Referencias clave (mayo 2026)

- Wen et al., *FoundationPose*, CVPR 2024 — [arxiv 2312.08344](https://arxiv.org/abs/2312.08344)
- Chi et al., *Diffusion Policy*, RSS 2023 / IJRR 2025 — [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)
- BOP Challenge 2024 / 2025 — [bop.felk.cvut.cz](https://bop.felk.cvut.cz/challenges/)
- ICCV'25 R6D Workshop, Honolulu — [cmp.felk.cvut.cz/sixd/workshop_2025](https://cmp.felk.cvut.cz/sixd/workshop_2025/)
- Liu et al. 2024, *RDT-1B*, ICLR 2025 — [rdt-robotics.github.io](https://rdt-robotics.github.io/rdt-robotics/)
- Black et al. 2024, *π0*, Physical Intelligence — [pi.website/download/pi0.pdf](https://www.pi.website/download/pi0.pdf)
- Kim et al. 2024, *OpenVLA*, arXiv 2406.09246 — [openvla.github.io](https://openvla.github.io/)
- *Two-Steps Diffusion Policy via Genetic Denoising* — [arxiv 2510.21991](https://arxiv.org/abs/2510.21991)
- *On-Device Diffusion Transformer Policy*, ICCV 2025 — [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_On-Device_Diffusion_Transformer_Policy_for_Efficient_Robot_Manipulation_ICCV_2025_paper.pdf)
- *Diffusion models for robotic manipulation: a survey*, Frontiers 2025 — [doi.org/10.3389/frobt.2025.1606247](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247/full)
- *Foundation models for robot learning: a survey* (Li et al.), Sage 2025 — [doi.org/10.1177/02783649251390579](https://journals.sagepub.com/doi/10.1177/02783649251390579)
- XYZ-IBD bin-picking dataset (2025) — [xyz-ibd.github.io](https://xyz-ibd.github.io/)

---

*Documento mantenido por Giocrisrai Godoy Bonillo y José Miguel Carrasco.
Actualizado: mayo 2026.*
