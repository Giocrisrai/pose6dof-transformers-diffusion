# Percepción intercambiable: FoundationPose vs. alternativas comercializables

> **TL;DR.** El bloque de percepción (estimación de pose 6-DoF) del pipeline es
> **intercambiable por diseño**. FoundationPose (licencia *no comercial* de
> NVIDIA) puede sustituirse por un estimador de **licencia abierta y uso
> comercial** sin tocar el resto del pipeline. Y no es solo posible: hoy el
> **ganador del BOP Challenge 2024 es FreeZeV2.1, con licencia Apache-2.0** — es
> decir, **se pueden igualar e incluso superar los resultados de FoundationPose
> con un modelo comercializable**. Lo que aporta este TFM (la integración
> matemática SE(3) + difusión + control + reproducibilidad) **es independiente
> del estimador concreto**.

---

## 1. El problema (y por qué importa para comercializar)

FoundationPose (Wen et al., CVPR 2024) es estado del arte y **generaliza a
objetos nuevos** (zero-shot con CAD), pero tiene **licencia NVIDIA *no
comercial***. Sirve para investigar y reproducir; **no** para vender un producto
industrial directamente. Para cualquiera que quiera *usar* esto en producción,
ese es el bloqueo a resolver.

## 2. El pipeline ya está diseñado para el cambio

No es una promesa: el contrato está en el código.

`src/perception/pose_estimator.py` define un **protocolo** `PoseEstimator` que
cualquier estimador debe cumplir:

```python
@runtime_checkable
class PoseEstimator(Protocol):
    name: str
    license: str  # "MIT" | "Apache-2.0" | "AGPL-3.0" | "NC" | ...

    def predict_pose(self, rgb, depth, cam_K, obj_id, ...) -> PoseEstimate: ...
    def is_commercializable(self) -> bool: ...
```

- **Entrada/salida fijas:** recibe RGB(-D) + intrínseca + `obj_id`; devuelve
  `PoseEstimate(R, t, confidence, ...)` — una pose en SE(3). Todo lo que viene
  *después* (planificación por difusión, IK, PBVS, depósito) **solo depende de
  esa pose**, no del método que la produjo.
- **La comercialización es un atributo de primera clase:** el protocolo expone
  `is_commercializable()` y un campo `license`. El pipeline puede *avisar* si la
  configuración elegida es apta para uso comercial.
- **Selección por configuración:** `PipelineConfig.pose_method` ya conmuta entre
  `"foundationpose"` y `"gdrnet"` en `src/pipeline.py`; añadir un tercero es
  registrar una clase nueva, no reescribir el pipeline.

> Cambiar de estimador = implementar `predict_pose` en una clase nueva y poner
> `pose_method="<nuevo>"`. El resto del pipeline no se entera.

## 3. Alternativas open-license / comercializables

| Estimador | Licencia | ¿Uso comercial? | Generaliza a objetos nuevos | Rendimiento vs FoundationPose | Esfuerzo de integración |
|---|---|---|---|---|---|
| **FoundationPose** (NVIDIA) | **NC** (no comercial) | ❌ | ✅ zero-shot (CAD) | referencia (SOTA) | ya integrado |
| **FreeZeV2.1** (Caraffa et al. 2025) | **Apache-2.0** | ✅ | ✅ sin entrenamiento | **≥ FoundationPose** (ganó BOP 2024) | medio |
| **GDR-Net++ / GDRNPP** (Liu et al. 2025c) | **Apache-2.0** | ✅ | ⚠️ model-based (entrena por catálogo) | −3.0/−3.6 pp Mean AR *en este TFM*; **cerrable con fine-tuning** | ya scaffolded (`gdrnet.py`) |
| **MegaPose** (Labbé et al. 2022) | **AGPL-3.0** (copyleft) | ⚠️ con obligaciones AGPL | ✅ render-and-compare | comparable | medio |
| **Any6D** | **MIT** | ✅ | ✅ | comparable (verificar) | medio |
| **Entrenar uno propio** (BlenderProc + datasets abiertos) | tuya | ✅ | según diseño | según esfuerzo/datos | alto |

Datos de entrenamiento abiertos disponibles para la última fila: T-LESS
(CC BY 4.0), YCB-Video, BOP `train_pbr` sintético. Las piezas industriales sin
textura (el caso difícil) son justo donde un modelo *fine-tuneado al catálogo*
rinde muy bien.

> **Nota honesta:** las licencias y métricas exactas deben **verificarse en el
> momento de integrar** (versión del repo, del checkpoint y de los datos). La
> tabla refleja lo declarado por cada proyecto y por la documentación de este
> TFM; AGPL permite uso comercial pero impone *copyleft de red* (publicar
> cambios), por eso va marcado con ⚠️.

## 4. ¿Se pueden igualar o superar los resultados? (análisis honesto)

Sí, y por dos caminos complementarios:

1. **Reemplazo directo por SOTA abierto.** FreeZeV2.1 **ganó el BOP Challenge
   2024** y es **Apache-2.0**. Sustituir FoundationPose por FreeZeV2.1 mantiene —
   o mejora— la precisión **y** habilita uso comercial. Es el camino más directo
   a "igual o mejor".
2. **Especializar un model-based open-license.** FoundationPose gana por ser
   *generalista* (cualquier objeto). Pero en una planta real el catálogo de
   piezas es **acotado y conocido**. Un model-based Apache-2.0 como GDR-Net++,
   **fine-tuneado a ese catálogo**, suele **igualar o superar** a un generalista
   en su dominio, cerrando los −3 pp medidos en este TFM. Menos generalidad, más
   precisión donde importa.

En ambos casos, **el aporte del TFM se conserva intacto**: el puente matemático
percepción→acción sobre SE(3), la planificación por difusión, el control PBVS y
la reproducibilidad no dependen del estimador.

## 5. Cómo se hace el cambio (pasos concretos)

1. **Implementar la clase** `NuevoEstimator` en `src/perception/` que cumpla el
   protocolo: `predict_pose(...) -> PoseEstimate` + `license` +
   `is_commercializable()`.
2. **Registrarla** en `BinPickingPipeline.initialize()` (`src/pipeline.py`),
   junto a las ramas `foundationpose` / `gdrnet`.
3. **Conmutar** con `PipelineConfig(pose_method="nuevo", ...)`.
4. **Re-validar** con el mismo arnés del TFM (sin cambiar nada más).

## 6. Plan de validación A/B (reusa el rigor del TFM)

- Mismos datasets (YCB-Video, T-LESS) y **métricas oficiales BOP** (Mean AR; VSD,
  MSSD, MSPD), con los splits oficiales.
- **Intervalos de confianza 95 % por bootstrap** con el paquete propio
  `bop-bootstrap-ci` (ya publicado en PyPI) — comparación estadística honesta,
  no números sueltos.
- **Criterio de aceptación del swap:** `Mean AR(nuevo) ≥ Mean AR(FoundationPose) − δ`
  (δ pequeño y declarado), o bien superarlo; y `is_commercializable() == True`.
- Reportar también latencia p95 del ciclo E2E (objetivo < 10 s, ya cumplido) para
  confirmar que el cambio no rompe H3.

## 7. Recomendación

- **Para un producto comercial:** swap a **FreeZeV2.1 (Apache-2.0)** como primera
  opción (SOTA + comercial), o **GDR-Net++ fine-tuneado** si el catálogo de
  piezas es cerrado y se busca máxima precisión en dominio.
- **Para seguir investigando/divulgando:** mantener FoundationPose (es SOTA y
  zero-shot), dejando claro el matiz de licencia y la ruta de swap.

> En una frase para defensa/charla: *"FoundationPose es no comercial, pero mi
> aporte no es el estimador: es el puente matemático y el pipeline reproducible.
> El estimador es un bloque que se cambia por uno open-license —incluso por el
> ganador del BOP 2024, que es Apache-2.0— igualando o mejorando los
> resultados."*

---

*Referencias del contrato en código:* `src/perception/pose_estimator.py`
(protocolo), `src/perception/gdrnet.py` (alternativa Apache-2.0 scaffolded),
`src/pipeline.py` (`PipelineConfig.pose_method`). Estado del arte y licencias:
ver la memoria del TFM (`docs/entrega3/`) y `docs/COMPARATIVA_SOTA_LENGUAJE.md`.
