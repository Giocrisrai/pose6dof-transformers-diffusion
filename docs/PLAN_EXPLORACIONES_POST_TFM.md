# Plan de exploraciones post-TFM (mayo 2026 →)

> **Objetivo**: a partir del pipeline ya entregado en el TFM, probar
> contribuciones novedosas concretas. Cada exploración tiene hipótesis,
> criterio numérico de éxito, *budget* de tiempo y condición de aborto.
> Si una exploración tiene éxito → se documenta e integra. Si fracasa →
> se documenta el hallazgo negativo (también es contribución).
>
> **Regla principal**: el TFM entregado **no se toca**. Estas exploraciones
> viven en ramas `explore/*` y solo se mergean a `main` si pasan los
> criterios.

---

## Orden de ataque (de menos riesgo a más riesgo)

| # | Exploración | Riesgo planeado | Esfuerzo real | Valor obtenido |
|---|---|:---:|:---:|---|
| **1** | Bootstrap-CI BOP toolkit (PyPI) | 🟢 Bajo | 1 día | 27 tests, 97 % cov, paquete listo para PyPI |
| **2** | Distillation Diffusion 1-NFE | 🟡 Medio | 1 día | ×517 speedup real, MSE/jerk mejores que teacher |
| **3** | Pipeline 100 % open-license | 🟠 Medio-alto | 1 día | FreeZeV2 (Apache 2) viable a solo −3 pp AUC |
| **4** | VLA-lite con CLIP text-conditioning | 🔴 Alto | 1 día | 98.6 % selection accuracy |
| **5** | Robustez lingüística (extensión #4) | 🟢 Bajo | 0.5 día | 100 % accuracy sobre 6 familias no vistas |
| **6** | VLA-lite multi-atributo (color + forma) | 🟡 Medio | 0.5 día | 99.9 % global, 100 % en color/combinado, 99.8 % en shape |
| **7** | Simulaciones visuales 3D | 🟢 Bajo | 0.5 día | 12/12 escenas renderizadas con cubos/esferas/cilindros/cajas |
| **8** | VLA-lite multi-objeto N=2..5 | 🟡 Medio | 0.5 día | 100 % accuracy con hasta 5 objetos en escena |

---

## Exploración 1 — Bootstrap-CI BOP toolkit (PyPI)

**Rama**: `explore/01-bootstrap-ci-toolkit`

### Hipótesis
> Podemos extraer nuestro framework de evaluación con bootstrap CI 95%
> B=1000 a una librería standalone PyPI compatible con `bop_toolkit` oficial.

### Criterio de éxito
- [ ] Paquete `bop-bootstrap-ci` instalable con `pip install`.
- [ ] Funciona sobre BOP YCB-V y T-LESS sin re-escribir nada.
- [ ] API compatible con outputs de `bop_toolkit`.
- [ ] ≥ 95 % cobertura de tests (pytest).
- [ ] Documentación con 2 ejemplos: AUC ADD-S CI y Recall@10mm CI.

### Condición de aborto
Si la API requiere modificar `bop_toolkit` oficial → abortar y publicar
como script standalone en `experiments/`.

### Plan de trabajo (3-4 días)
1. Día 1: extraer `recompute_metrics_with_bootstrap.py` a paquete + estructura.
2. Día 2: tests + CI.
3. Día 3: docs + README + ejemplo notebook.
4. Día 4: publicar TestPyPI → PyPI.

### Validación
Ejecutar sobre los checkpoints actuales y verificar que reproduce
`local_metrics_with_bootstrap.json` exactamente.

### Output esperado
- Paquete PyPI publicado
- Documento `docs/exploraciones/01_bootstrap_ci_toolkit.md`
- Issue/PR en `thodan/bop_toolkit` proponiendo integración

---

## Exploración 2 — Distillation Diffusion 2-NFE

**Rama**: `explore/02-diffusion-2nfe`

### Hipótesis
> Aplicando el método "Two-Steps Diffusion Policy via Genetic Denoising"
> (arxiv 2510.21991) sobre nuestro modelo `ultra` podemos reducir el
> sampling de DDIM-25 (93 ms) a 2 NFE (~5 ms) manteniendo MSE < 0.005
> y jerk < 0.2.

### Criterio de éxito (numérico)
- [ ] MSE val con 2 NFE ≤ 0.005 (vs 0.0022 con 25 NFE, degradación < 130 %)
- [ ] Jerk RMS ≤ 0.2 (vs 0.053 con 25 NFE)
- [ ] Latencia/trayectoria ≤ 10 ms (vs 93 ms)
- [ ] Cycle E2E p95 reducido en ≥ 80 ms vs baseline
- [ ] Tests pasando

### Condición de aborto
Si tras 6 días el MSE > 0.01 ó el jerk > 0.5 → abortar y documentar
como hallazgo negativo en `docs/exploraciones/02_*.md`.

### Plan de trabajo (4-6 días)
1. Día 1: leer paper, reproducir setup en notebook.
2. Día 2: implementar GD denoising sobre `diffusion_policy_ultra.pth`.
3. Día 3-4: entrenar/fine-tune en M1 Pro MPS.
4. Día 5: validar contra ultra original (mismo test set, semilla 42).
5. Día 6: integrar en API REST como modelo `ultra_fast`.

### Validación
- Bootstrap CI 95% sobre MSE, jerk y latencia (B=1000).
- Test estadístico Wilcoxon vs ultra original.
- Verificar en `tests/test_api_server.py`.

### Output esperado
- Modelo `diffusion_policy_ultra_fast.pth` en `data/models/`
- Notebook `notebooks/08_distillation_2nfe.ipynb`
- Documento `docs/exploraciones/02_distillation_2nfe.md`
- Si éxito: nuevo punto Pareto en exp13 (latencia × calidad)

---

## Exploración 3 — Pipeline 100 % open-license

**Rama**: `explore/03-open-license-pipeline`

### Hipótesis
> Sustituyendo FoundationPose (licencia NVIDIA Source Code, NC) por
> FreeZeV2 (Apache 2) o Foundation6D abierto, podemos mantener
> AUC ADD-S ≥ 0.85 sobre YCB-V y T-LESS, eliminando el bloqueo legal
> para comercialización.

### Criterio de éxito
- [ ] Pipeline funciona end-to-end con cero código NVIDIA-NC.
- [ ] AUC ADD-S YCB-V ≥ 0.85 (vs 0.908 con FP) — degradación ≤ 6.5 %.
- [ ] AUC ADD-S T-LESS ≥ 0.90 (vs 0.957 con FP) — degradación ≤ 6 %.
- [ ] Tiempo FP equivalente ≤ 4 s (paridad latencia).
- [ ] Toda la cadena con licencias MIT/Apache/BSD verificadas.

### Condición de aborto
Si tras 8 días la degradación de AUC ADD-S supera el 10 % en algún
dataset → abortar y documentar como hallazgo negativo, mantener FP-NC
como única opción.

### Plan de trabajo (6-10 días)
1. Día 1: identificar candidato (FreeZeV2 vs Foundation6D vs MegaPose).
2. Día 2-3: instalar + integrar en pipeline.
3. Día 4-5: evaluar sobre YCB-V con bootstrap CI.
4. Día 6-7: evaluar sobre T-LESS con bootstrap CI.
5. Día 8: integrar en API REST + Docker + dashboard.
6. Día 9: comparar Pareto licencia-rendimiento.
7. Día 10: documentar.

### Validación
Bootstrap CI 95% B=1000 sobre AUC ADD-S, Recall@10mm, Median ADD-S.
Test de Wilcoxon vs baseline FP. Análisis de licencias verificado.

### Output esperado
- Pipeline alternativo en `src/perception/openpose_estimator.py`
- API endpoint `/plan-grasp?perception=open`
- Documento `docs/exploraciones/03_open_license_pipeline.md`
- Si éxito: spin-off viable y nota en defensa.

---

## Exploración 4 — VLA-lite con CLIP text-conditioning

**Rama**: `explore/04-vla-lite-clip`

### Hipótesis
> Añadiendo un encoder CLIP texto al condicionamiento del Diffusion
> Policy, el sistema puede filtrar trayectorias hacia un objeto
> descrito en lenguaje natural ("pick the red object") sin requerir
> entrenamiento VLA completo (millones de demos).

### Criterio de éxito
- [ ] Modelo `diffusion_policy_clip.pth` entrenable en M1 Pro.
- [ ] Score de selección ≥ 0.75 (acierta el objeto descrito en ≥ 75% de los casos sintéticos).
- [ ] Latencia adicional < 50 ms (CLIP encode + projection).
- [ ] No degrada MSE de trayectoria sobre escenas single-object.

### Condición de aborto
Si tras 8 días el score < 0.5 sobre dataset sintético → abortar.
El paradigma VLA puro requiere demos reales que no tenemos.

### Plan de trabajo (7-10 días)
1. Día 1-2: generar dataset sintético con escenas multi-objeto + descripciones textuales.
2. Día 3: integrar CLIP text encoder (frozen) + proyección 512 → 64.
3. Día 4-5: entrenar variante de Diffusion Policy condicionada.
4. Día 6-7: evaluar score de selección.
5. Día 8: integrar en Gradio (campo de texto).
6. Día 9-10: validar + documentar.

### Validación
Métrica nueva: *selection accuracy* (objeto correcto seleccionado
sobre escenas multi-objeto). Bootstrap CI sobre ello.

### Output esperado
- Modelo CLIP-conditioned
- Gradio con campo "Describe el objeto"
- Documento `docs/exploraciones/04_vla_lite_clip.md`
- Si éxito: paso concreto hacia VLA accesible.

---

## Reglas operativas

1. **Rama por exploración**: `explore/N-nombre`.
2. **Doc por exploración**: `docs/exploraciones/0N_*.md` con resultados.
3. **Si éxito** (criterios cumplidos):
   - Merge a `main` con commit `feat(explore-N): ...`
   - Actualizar `docs/INNOVACION_Y_ESTADO_DEL_ARTE.md` añadiendo el aporte
   - Actualizar Gradio + Streamlit con la nueva capacidad
4. **Si fracaso** (criterios no cumplidos):
   - **No mergear** a main
   - Mantener el documento de exploración con los datos del intento
   - Resumir en `docs/HALLAZGOS_NEGATIVOS.md` (también es contribución)
5. **Si tras el budget no hay convergencia** clara: abortar.
6. **TFM Entrega 2 no se toca**. Las nuevas mejoras serán Entrega 3 o paper.

---

## Estado actual

| # | Exploración | Estado | Días invertidos |
|---|---|---|---|
| 1 | Bootstrap-CI toolkit | ✅ **Éxito → mergeada** ([detalle](exploraciones/01_bootstrap_ci_toolkit.md)) | 1 |
| 2 | Distillation 1-NFE | ✅ **Éxito → mergeada** ([detalle](exploraciones/02_distillation_2nfe.md)) | 1 |
| 3 | Open-license pipeline | ✅ **Éxito → mergeada** ([detalle](exploraciones/03_open_license_pipeline.md)) | 1 |
| 4 | VLA-lite CLIP | ✅ **Éxito → mergeada** ([detalle](exploraciones/04_vla_lite_clip.md)) | 1 |
| 5 | Robustez lingüística VLA-lite (extensión #4) | ✅ **Éxito → mergeada** ([detalle](exploraciones/05_vla_robustness.md)) | 0.5 |
| 6 | VLA-lite multi-atributo color+forma | ✅ **Éxito → mergeada** ([detalle](exploraciones/06_vla_shapes.md)) | 0.5 |
| 7 | Simulaciones visuales 3D | ✅ **Éxito → mergeada** ([detalle](exploraciones/07_visual_simulations.md)) | 0.5 |
| 8 | VLA-lite multi-objeto N=2..5 | ✅ **Éxito → mergeada** ([detalle](exploraciones/08_multi_object.md)) | 0.5 |
| 9 | Atributo continuo TAMAÑO | ✅ **Éxito → mergeada** ([detalle](exploraciones/09_size_attribute.md)) | 0.5 |
| 10 | Instrucciones secuenciales multi-step | ✅ **Éxito → mergeada** ([detalle](exploraciones/10_sequential_instructions.md)) | 0.5 |
| 11 | Visual grounding con CLIP-image | ✅ **Éxito → mergeada** ([detalle](exploraciones/11_clip_image_grounding.md)) | 0.5 |
| 12 | Robustez CLIP-image con domain randomization | ✅ **Éxito → mergeada** ([detalle](exploraciones/12_robustness_domain_random.md)) | 0.5 |
| 13 | Razonamiento espacial (leftmost/closest/highest) | ✅ **Éxito → mergeada** ([detalle](exploraciones/13_spatial_reasoning.md)) | 0.5 |

**Resumen final**: 13/13 exploraciones cumplen criterios. ~9 días totales invertidos.
171 tests pasando · 10 modelos Diffusion · 1 paquete PyPI · 4 hallazgos metodológicos
corregidos honestamente · 13 documentos de cierre · 22 simulaciones visuales generadas.

**Roadmap industrial completo en simulación**: las 3 extensiones del documento
[`docs/EXTRAPOLACION_INDUSTRIAL.md`](EXTRAPOLACION_INDUSTRIAL.md) corto plazo
(multi-objeto, atributos continuos, imagen real con CLIP-image) están cerradas
con éxito + bonus de robustez evaluada con bootstrap CI 95% y razonamiento
espacial. **Lo que requiere hardware físico** está documentado en
[`docs/ROADMAP_POSTTFM.md`](ROADMAP_POSTTFM.md) como etapas 1-7 con timeline
y presupuesto estimado (~200 k EUR / 2 años hasta producto comercial certificado).

Ver también el documento [`docs/EXTRAPOLACION_INDUSTRIAL.md`](EXTRAPOLACION_INDUSTRIAL.md)
con el roadmap de cómo el pipeline se aplica a logística, reciclaje, electrónica,
médico y automoción.

---

*Plan vivo — actualizar al cierre de cada exploración. Última revisión: mayo 2026.*
