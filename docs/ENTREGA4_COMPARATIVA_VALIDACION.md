# Entrega 4 — Comparativa y validación de la contribución de lenguaje natural

> **Propósito**: documento de análisis que compara el estado de la **Entrega 3**
> con el de la **Entrega 4**, valida la contribución nueva (bin picking guiado por
> lenguaje natural + ejecución E2E en CoppeliaSim) con cifras trazables, y la
> posiciona frente al estado del arte. Es la base de cifras para el capítulo de
> tesis y para las presentaciones de la Entrega 4.
>
> **Regla de honestidad**: cada cifra del proyecto enlaza su fuente (fichero,
> JSON de resultados o ejecución en vivo). Las cifras de terceros se citan de su
> publicación o se marcan `n/d`. Nada inventado.

---

## 1. Resumen ejecutivo

La Entrega 3 entregó un pipeline de bin picking (pose 6-DoF + Diffusion Policy +
visual servoing) validado sobre YCB-V y T-LESS. La **Entrega 4 añade una capa de
lenguaje natural open-source**: el sistema acepta una instrucción ("dame el cubo
rojo de la izquierda") y selecciona y manipula el objeto descrito.

- **Comprensión + grounding**: parser determinista ES/EN + `Grounder` por atributos
  (color/forma/tamaño) con desempate espacial; backends LLM local (Ollama) y API
  enchufables con *fallback* determinista. Consolida 9 exploraciones (exp16–26).
- **Ejecución E2E real**: `run_language_pick` construye una escena multi-objeto con
  **variedad de formas reales** (cubo/esfera/cilindro), groundea la instrucción y
  ejecuta el pick-and-place real en CoppeliaSim, con métrica de selección honesta.
- **Validado en vivo** (19-jun-2026): pick E2E con grasp a 4 mm y IK convergente;
  batería de selección 100 % (pura n=90 y en simulador n=9); reel de demo de 174 s.

---

## 2. Entrega 3 vs Entrega 4

| Capacidad | Entrega 3 | Entrega 4 |
|---|---|---|
| Pose 6-DoF + grasp + servoing | ✅ (H1/H2/H3 validadas) | ✅ (sin cambios; se preserva) |
| Selección por **lenguaje natural** | ❌ | ✅ `pipeline.run(instruction=…)` |
| Parser de instrucciones | ❌ | ✅ determinista ES/EN + LLM local/API (fallback) |
| Grounding atributo + **espacial** | ❌ | ✅ color/forma/tamaño + left/right/nearest/… |
| Variedad de formas en simulador | cubos | ✅ cubo + esfera + cilindro (`createPrimitiveShape`) |
| **E2E por instrucción** en CoppeliaSim | ❌ | ✅ `run_language_pick` (pick real del objeto descrito) |
| Métrica de selección | — | ✅ honesta (coincide con todos los atributos pedidos) |
| Demo de lenguaje | ❌ | ✅ CLI `--dry-run`/E2E, tab de dashboard, reel |
| Superficie de docs | pipeline | + `LENGUAJE_NATURAL.md`, comparativa SOTA |

**Lo que NO cambia**: el pipeline base de la Entrega 3 se mantiene intacto; la capa
de lenguaje es *opt-in* (`PipelineConfig.language_enabled`).

### Baseline Entrega 3 (referencia, sin cambios)

| Hipótesis | Resultado | Fuente |
|---|---|---|
| H1 — Precisión pose (AUC ADD-S) | 0.908 [0.901, 0.916] YCB-V / 0.957 [0.954, 0.959] T-LESS | README.md §hipótesis |
| H2 — Multimodalidad | score 0.96, MSE 0.020 | README.md §hipótesis |
| H3 — Cycle E2E p95 | 6.12 s YCB-V / 6.86 s T-LESS (< 10 s) | README.md §hipótesis |
| Robustez | T-LESS 70 % oclusión → −1 pp AUC; PBVS 100 %/50 | README.md |

---

## 3. Validación de la contribución (Entrega 4)

### 3.1 Reproducible sin simulador (núcleo + batería pura)

| Prueba | Resultado | Fuente |
|---|---|---|
| Suite base de tests | 278 passed, 5 deselected | `pytest -m "not slow and not integration"` |
| Test CLIP (zero-shot atributos) | 1 passed | `pytest -m slow` |
| Batería **selection-accuracy pura** n=90 | **100 %** global; color 30/30, forma 30/30, espacial 30/30 | `experiments/results/language_battery/report_pure_n90.json` |

> La batería pura valida el *plumbing* end-to-end (parse → ground → selección) y el
> **desempate espacial** sobre un *benchmark controlado* donde el objeto descrito es
> unívoco; por construcción el determinista acierta el 100 %. La robustez frente a
> lenguaje **ambiguo/no visto** está evidenciada por exp16–26 sobre el modelo VLA
> aprendido (sección 4), no por esta batería.

### 3.2 Requiere CoppeliaSim (`:23000`) — ejecución E2E real

| Prueba | Resultado | Fuente |
|---|---|---|
| Test de integración `run_language_pick` | passed (98 s) | `tests/test_language_pick_core.py -m integration` |
| CLI E2E "dame el cubo rojo de la izquierda" (clutter) | escena cubo-rojo / esfera-verde / cilindro-azul; `selection_correct=true`; **grasp 4 mm**; IK convergió; objeto movido 0.895 m; MP4 generado | `experiments/run_pick_language.py` (run en vivo 19-jun) |
| Batería **selection-accuracy en simulador** n=9 | **100 %** (`sim_selection_accuracy=1.0`); color 3/3, forma 3/3, espacial 3/3 | `experiments/results/language_battery/report_sim_n9.json` |
| Reel de demo (crescendo de 5 instrucciones) | 174 s, 4350 frames, H.264 1280×720 | `experiments/results/language_reel/language_reel.mp4` |

> El grasp es cinemático (técnica snap+attach, ver `PICK_LIMITATIONS.md`): la métrica
> honesta de calidad es la **proximidad tip↔objeto** (4 mm < umbral 5 cm = plausible)
> y la **convergencia de IK**, no el desplazamiento del objeto.

---

## 4. Resultados consolidados de lenguaje (exp16–26)

Evidencia de robustez de selección del modelo VLA-lite sobre datos sintéticos/
controlados (cada cifra trazada en `docs/LENGUAJE_NATURAL.md` §6):

| Exp | Capacidad | Selection accuracy | Fuente |
|---|---|---|---|
| exp16 | CLIP text-conditioning (color) | 98.6 % | exploraciones/04 |
| exp17 | Robustez lingüística (6 familias no vistas) | 100 % (900/900) | exploraciones/05 |
| exp18 | Multi-atributo color+forma | 99.9 % | exploraciones/06 |
| exp20 | Multi-objeto N=2..5 | 100 % | exploraciones/08 |
| exp22 | Atributo continuo tamaño | 99.9 % | exploraciones/09 |
| exp23 | Instrucciones secuenciales | 8/8 secuencias, 20/20 pasos | exploraciones/10 |
| exp24 | CLIP-image grounding | 100 % | exploraciones/11 |
| exp25 | Robustez CLIP-image (domain rand.) | media 95.3 % | exploraciones/12 |
| exp26 | Razonamiento espacial | 98.4 % | exploraciones/13 |

---

## 5. Posicionamiento frente al SOTA

Resumen de `docs/COMPARATIVA_SOTA_LENGUAJE.md` (detalle y referencias allí):

| Eje | Esta contribución | CLIPort / VoxPoser / SayCan / OWL-ViT |
|---|---|---|
| Licencia | Open-license (Apache/MIT en deps) | variable |
| Hardware | Corre en portátil M1 (16 GB), sin GPU dedicada | clúster / GPU de gama alta (típico) |
| Re-entrenamiento por vocabulario | No (determinista + LLM enchufable) | a menudo sí |
| Interpretabilidad | Caja blanca (parse + grounding inspeccionables) | caja negra (típico) |
| Accuracy de selección | exp16 98.6 %, exp24/20 100 %, exp26 98.4 % (sintético) | `n/d` cifra única comparable (suites distintas) |

> Las celdas `n/d` reflejan que esos trabajos reportan métricas no homologables
> (task success en RAVENS/RLBench, mAP open-vocab, éxito de plan en cocina móvil);
> no se inventa una cifra comparable.

**Argumento de posicionamiento**: una solución de bin picking guiado por lenguaje
**100 % open-license que corre en un portátil**, con grounding interpretable a
98–100 % en los atributos evaluados y comprensión LLM enchufable, es competitiva en
los ejes que importan para reproducibilidad y despliegue de bajo coste.

---

## 6. Recomendación de incorporación a la Entrega 4

**Entra en la tesis (SP-2 — capítulo nuevo)**
- Motivación + arquitectura Parser→Grounder→Pipeline (con diagrama).
- Tabla E3-vs-E4 (sección 2) y validación E2E (sección 3.2).
- Consolidación exp16–26 (sección 4) y posicionamiento SOTA (sección 5).
- Limitaciones honestas (grasp cinemático; benchmark de selección controlado;
  `llm_api` es stub; tamaño por CLIP es el atributo menos fiable).

**Entra en las presentaciones (SP-3)**
- Defensa: 1–2 slides de la demo de lenguaje + frame del reel + tabla E3-vs-E4.
- Divulgativa: la narrativa "háblale al robot" con el reel embebido.
- Actualizar guion y FAQ (pregunta esperable: "¿el grasp es real?" → snap+attach,
  4 mm de proximidad, IK convergente).

**Trabajo futuro (no en Entrega 4)**
- Benchmark de selección con casos genuinamente ambiguos (mismo color+forma) para
  que la métrica en sim discrimine, no solo valide *plumbing*.
- Grounding por CLIP-image sobre la cámara real del sim (hoy usa metadatos de escena).
- Backend `llm_api` con un proveedor concreto.

---

*Documento backbone de la Entrega 4. Fuentes de cifras del proyecto: README.md,
docs/LENGUAJE_NATURAL.md, docs/COMPARATIVA_SOTA_LENGUAJE.md, docs/exploraciones/,
experiments/results/language_battery/*.json, y ejecuciones en vivo del 19-jun-2026.*
