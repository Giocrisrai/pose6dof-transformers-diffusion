# Comparativa con el estado del arte — bin picking guiado por lenguaje

Esta página posiciona la *feature* de lenguaje natural de este proyecto (ver
[`LENGUAJE_NATURAL.md`](LENGUAJE_NATURAL.md)) frente a sistemas de referencia que
combinan visión, lenguaje y acción para manipulación robótica.

> **Regla de honestidad**: las cifras de terceros se citan de sus publicaciones
> (sección [Referencias](#referencias)). Donde no se dispone de una cifra
> directamente comparable o no se puede afirmar con certeza, se marca **n/d** (no
> disponible). Las métricas de selección de estos sistemas se miden en *task
> suites* (CLIPort/RAVENS, RLBench, etc.) **distintas** de nuestras escenas
> sintéticas, por lo que las comparaciones de *accuracy* no son homologables y se
> aportan solo como orden de magnitud.

---

## Sistemas comparados

- **Este proyecto** — *grounding* determinista + CLIP, comprensión LLM
  enchufable, ejecutable en portátil M1 Pro. Código MIT.
- **CLIPort** (Shridhar et al., CoRL 2021) — política de manipulación que combina
  la semántica de CLIP con la precisión espacial de Transporter Networks.
- **VoxPoser** (Huang et al., CoRL 2023) — usa un LLM grande para componer mapas
  de valor 3D que guían un planificador de movimiento (zero-shot, sin entrenar la
  política).
- **SayCan** (Ahn et al., CoRL 2022) — un LLM propone *skills* de alto nivel y un
  modelo de *affordance* (entrenado con RL) decide cuáles son factibles.
- **OWL-ViT / CLIP-Fields** — detección/grounding open-vocabulary: OWL-ViT
  (Minderer et al., ECCV 2022) para detección guiada por texto; CLIP-Fields
  (Shafiullah et al., RSS 2023) como memoria espacial-semántica destilada de
  modelos de visión-lenguaje. Aquí se usan como representantes de la línea de
  *open-vocabulary grounding* (percepción, no política completa).

---

## Tabla comparativa

| Eje | **Este proyecto** | CLIPort | VoxPoser | SayCan | OWL-ViT / CLIP-Fields |
|---|---|---|---|---|---|
| **Licencia / open-source** | Código **MIT** | **Apache-2.0** (repo cliport) | Código abierto (**MIT**) | Sin pesos abiertos oficiales; reimplementaciones de comunidad | OWL-ViT **Apache-2.0** (HF/Scenic); CLIP-Fields código abierto (**MIT**) |
| **Hardware** | **Portátil M1 Pro** (CPU/MPS), sin GPU dedicada | GPU para entrenar/inferir la política | LLM grande (API o GPU) + percepción; no diseñado para portátil | LLM grande + política RL; **clúster** para entrenar | GPU para los encoders de visión-lenguaje (inferencia) |
| **Re-entrenamiento por vocabulario** | **No** en el determinista (basta ampliar léxico) y el LLM es zero-shot; el *gate* CLIP de exp16–26 sí se entrena por conjunto de atributos | Política entrenada por dominio (RAVENS/tareas) | **No** (zero-shot vía LLM + CLIP/OWL) | *Skills* + *affordances* entrenadas; el lenguaje es zero-shot | **No** (open-vocabulary por diseño) |
| **Interpretabilidad** | **Caja blanca**: parser → `Instruction` legible + tabla de *scores* por objeto | Parcial: mapas de *pick/place* visualizables, política aprendida opaca | Media: los mapas de valor 3D son inspeccionables; el LLM que los genera, no | Media: el LLM expone su *chain* de skills; affordance opaca | Alta a nivel de detección (cajas/heatmaps por texto) |
| **Accuracy de selección reportada** | exp16 **98.6 %**; exp18 **99.9 %**; exp20/exp24 **100 %**; exp26 **98.4 %** (sintético, ver [LENGUAJE_NATURAL.md](LENGUAJE_NATURAL.md)) | Tasas de éxito por tarea en RAVENS; **n/d** una cifra única comparable | Tasas de éxito por tarea (real+sim); **n/d** una cifra única comparable | Éxito de plan en cocina móvil; **n/d** una cifra única comparable a "selección" | mAP de detección open-vocab; **n/d** como "selection accuracy" de manipulación |

Notas:

- Las celdas **n/d** se deben a que esos trabajos no reportan una "selection
  accuracy" homologable a la nuestra (usan *task success rate* sobre *suites*
  distintas) o a que no se dispone de una cifra única indiscutible. No se ha
  inventado ningún número.
- Las afirmaciones de licencia se refieren al **código** publicado por los
  autores; los modelos de lenguaje grandes que algunos usan (p. ej. en SayCan o
  VoxPoser) tienen sus propias licencias/condiciones de API.

---

## Argumento de posicionamiento

El valor diferencial de esta feature no es superar a los grandes VLA/LLM en
cobertura lingüística, sino ofrecer un **camino accesible y auditable**:

1. **Open-license ejecutable en portátil.** El núcleo (parser determinista +
   grounder por atributos/espacial) corre en un M1 Pro sin GPU ni servicios
   externos, con código MIT. Los sistemas basados en LLM grandes (VoxPoser,
   SayCan) asumen acceso a un LLM potente (API o clúster).
2. **Grounding determinista + CLIP.** El método por atributos es reproducible y
   sin red; cuando se necesita inferir atributos de la imagen, se activa CLIP
   (`clip_image`) como en exp24. Esto da una ruta gradual de "sin pesos" a
   "visión real".
3. **Comprensión LLM enchufable.** El backend `llm_local` (Ollama, zero-shot) se
   activa para frases fuera del léxico controlado, con **fallback** automático al
   determinista; `llm_api` queda como punto de extensión. El resto del pipeline
   es agnóstico al parser.
4. **Interpretabilidad de extremo a extremo.** La `Instruction` parseada y la
   tabla de *scores* del `GroundingResult` hacen explícito *por qué* se eligió un
   objeto — frente a la opacidad de una política VLA aprendida.

En síntesis: una alternativa **caja blanca, open-license y de bajo coste de
cómputo**, que cubre el caso de bin picking guiado por instrucciones sin requerir
el presupuesto de datos/cómputo de los VLA grandes, a costa de un dominio más
restringido (vocabulario y escenas controladas).

---

## Referencias

- **Shridhar, M., Manuelli, L., Fox, D. (2021).** *CLIPort: What and Where
  Pathways for Robotic Manipulation.* Conference on Robot Learning (CoRL).
- **Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., Fei-Fei, L. (2023).**
  *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language
  Models.* Conference on Robot Learning (CoRL).
- **Ahn, M. et al. (2022).** *Do As I Can, Not As I Say: Grounding Language in
  Robotic Affordances (SayCan).* Conference on Robot Learning (CoRL).
- **Minderer, M. et al. (2022).** *Simple Open-Vocabulary Object Detection with
  Vision Transformers (OWL-ViT).* European Conference on Computer Vision (ECCV).
- **Shafiullah, N. M. M., Paxton, C., Pinto, L., Chintala, S., Szlam, A.
  (2023).** *CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory.*
  Robotics: Science and Systems (RSS).
- **Radford, A. et al. (2021).** *Learning Transferable Visual Models From
  Natural Language Supervision (CLIP).* International Conference on Machine
  Learning (ICML).

> Los años indicados corresponden a la *venue* de publicación principal de cada
> trabajo. Si alguna referencia se citara en el TFM con un año de *preprint*
> (arXiv) distinto, debe armonizarse con la bibliografía del documento.

---

*Comparativa de posicionamiento. Detalle de la feature en
[`LENGUAJE_NATURAL.md`](LENGUAJE_NATURAL.md).*
