# OE3: Estrategia de Evaluacion FoundationPose + Diffusion Policy

**Fecha**: 2026-04-11
**Objetivo**: Definir la estrategia incremental para cumplir OE3 del TFM
**Alcance**: Evaluacion de percepcion (FoundationPose vs GDR-Net) + planificacion (Diffusion Policy)

---

## Contexto

OE3 del TFM dice textualmente:
> "Configurar FoundationPose con evaluacion en BOP datasets (T-LESS, YCB-Video).
> Configurar Diffusion Policy para generacion multimodal de trayectorias de agarre.
> Comparativa cuantitativa con baseline GDR-Net.
> Metricas BOP: VSD, MSSD, MSPD, ADD, ADD-S."

### Estado actual
- Infraestructura de codigo completa (pipeline.py, perception/, planning/, utils/)
- Modulos matematicos testeados (lie_groups, rotations, metrics — 64/64 tests CI)
- Datasets BOP cargados en Colab (T-LESS 20 escenas, YCB-V 12 escenas)
- GPU T4 verificada en Colab (41K samples/s throughput)
- Pesos de FoundationPose y GDR-Net: NO descargados aun

### Hardware
- **Colab T4 (15GB VRAM)**: inferencia FoundationPose, training Diffusion Policy
- **Local M1 Pro**: desarrollo, visualizacion, documentacion

---

## Estrategia: Enfoque B con escalado a A

### Fase 1 — Subconjunto representativo (Semanas 2-4)

**Objetivo**: Resultados propios de FoundationPose en subconjunto + baseline leaderboard.

**1.1 FoundationPose en Colab**
- Clonar repo oficial NVlabs/FoundationPose
- Instalar dependencias CUDA (nvdiffrast, mycuda extensions)
- Descargar pesos pre-entrenados (score predictor + pose refiner)
- Ejecutar inferencia en:
  - T-LESS: 5 escenas (escenas 1-5), ~30 imagenes por escena
  - YCB-V: 5 escenas (escenas 48-52), ~50 imagenes por escena
- Metricas propias con nuestro `src/utils/metrics.py`:
  - ADD (Average Distance of model points)
  - ADD-S (Symmetric variant)
  - AUC (Area Under Curve) a distintos umbrales
- Sistema de checkpoints (guardar en Drive cada 2 escenas)

**1.2 GDR-Net baseline**
- Usar resultados oficiales BOP Challenge 2022 leaderboard:
  - YCB-V: AR_VSD=0.841, AR_MSSD=0.868, AR_MSPD=0.893
  - T-LESS: AR_VSD=0.712, AR_MSSD=0.764, AR_MSPD=0.825
- Justificacion: GDR-Net es el baseline comparativo, no el metodo propuesto. Usar resultados oficiales es practica estandar.

**1.3 Tabla comparativa**
- Generar tabla: FoundationPose (propio) vs GDR-Net (leaderboard) vs FoundationPose (paper)
- Grafico de barras por metrica y dataset
- Guardar en `experiments/results/` y Drive

**Entregable Fase 1**: Tabla comparativa con resultados propios de FP, figuras para Cap. 6.

### Fase 2 — Escalar evaluacion (Semanas 5-7)

**Gate**: Solo si Fase 1 produce resultados sin errores criticos.

**2.1 FoundationPose completo**
- Ampliar a TODAS las escenas de test:
  - T-LESS: 20 escenas completas
  - YCB-V: 12 escenas completas
- Checkpoints permiten retomar entre sesiones de Colab

**2.2 GDR-Net en Colab (intento)**
- Instalar mmcv + mmdetection via openmim
- Descargar pesos del model zoo
- Si funciona: resultados propios para comparativa directa
- Si falla (mmcv incompatible): mantener leaderboard, documentar intento

**2.3 Metricas BOP completas**
- Calcular VSD, MSSD, MSPD con nuestro evaluador
- Comparativa directa FP propio vs GDR-Net propio/leaderboard

**Entregable Fase 2**: Evaluacion completa en ambos datasets, comparativa exhaustiva.

### Fase 3 — Reproduccion completa (Semanas 7-8, opcional)

**Gate**: Solo si Fase 2 completa y hay margen de tiempo.

- Evaluacion en todas las escenas con todos los objetos
- Submission al BOP evaluation server (credibilidad extra)
- Resultados publicables

### Fase Paralela — Diffusion Policy (Semanas 4-6)

**Independiente de las fases de percepcion.**

**DP.1 Training en datos sinteticos**
- Generar datos de entrenamiento: poses estimadas -> trayectorias de agarre
- Entrenar ConditionalUNet1D en Colab GPU
- 100 epochs, batch_size=64, horizon=16, action_dim=7

**DP.2 Evaluacion comparativa**
- Diffusion Policy vs heuristic grasping (ya implementado en grasp_sampler.py)
- Metricas: success rate, diversidad de trayectorias, smoothness
- Generar figuras comparativas

**DP.3 Integracion con poses reales**
- Conectar poses de FoundationPose -> Diffusion Policy
- Pipeline completo: imagen -> pose -> trayectoria de agarre

---

## Arquitectura de evaluacion

```
Colab GPU (T4)
  |
  +-- 01_foundationpose_eval.ipynb
  |     |-- Clona NVlabs/FoundationPose
  |     |-- Descarga pesos (Drive cache)
  |     |-- Inferencia con checkpoints
  |     |-- Calcula ADD, ADD-S, AUC
  |     |-- Guarda resultados en Drive
  |
  +-- 02_gdrnet_eval.ipynb
  |     |-- Intenta instalar mmcv+mmdet
  |     |-- Si OK: inferencia propia
  |     |-- Si falla: usa leaderboard
  |     |-- Comparativa final
  |
  +-- 03_diffusion_training.ipynb (nuevo)
        |-- Genera datos sinteticos de grasp
        |-- Entrena ConditionalUNet1D
        |-- Evalua vs heuristic baseline
        |-- Guarda modelo en Drive

Local (M1 Pro)
  |
  +-- Analisis de resultados (descargados de Drive)
  +-- Generacion de figuras para Cap. 6
  +-- Integracion con CoppeliaSim (OE4, Jose Miguel)
```

---

## Metricas y criterios de exito

| Metrica | Fuente | Umbral minimo |
|---------|--------|---------------|
| ADD Recall@10mm | Propia (FP) | >50% en T-LESS |
| ADD-S Recall@10mm | Propia (FP) | >70% en T-LESS |
| AUC ADD@50mm | Propia (FP) | >0.5 |
| FP vs GDR-Net delta | Comparativa | FP >= GDR-Net en Mean AR |
| Diffusion vs Heuristic | Propia | Diffusion >= Heuristic en success rate |
| Pipeline E2E | Propia | Imagen -> pose -> trayectoria sin errores |

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|--------|-------------|------------|
| FoundationPose no compila en Colab | Media | Docker alternativo, o usar demo simplificada del repo |
| Pesos FP no disponibles publicamente | Baja | Contactar autores, usar version model-free |
| GDR-Net mmcv incompatible | Alta | Mantener leaderboard como baseline (defendible) |
| Colab timeout durante inferencia | Media | Checkpoints cada 2 escenas en Drive |
| Diffusion Policy no converge | Baja | Heuristic baseline como fallback (ya funciona) |

---

## Dependencias entre objetivos

```
OE2 (matematicas) -----> independiente, ya avanzado
OE3 (pipeline) --------> Fase 1 -> Fase 2 -> Fase 3
OE4 (simulacion) ------> depende de OE3 para poses reales
                          Jose Miguel lidera, Giocrisrai integra
```

---

## Timeline integrado con hitos UNIR

| Semana | Fecha | Foco | Entregable |
|--------|-------|------|------------|
| 2 | 11-17 abr | Fase 1: FP setup + primeras inferencias | FP corriendo en Colab |
| 3 | 18-24 abr | Fase 1: FP 5 escenas T-LESS + YCB-V | Primeros resultados propios |
| 4 | 25 abr-1 may | Fase 1: comparativa + DP training | Tabla comparativa v1 |
| **5** | **2-8 may** | **Borrador inicial (Cap 2-3)** | **Entrega UNIR** |
| 6 | 9-15 may | Fase 2: escalar FP + intentar GDR-Net | Evaluacion ampliada |
| 7 | 16-22 may | Fase 2: metricas BOP + DP integracion | Comparativa completa |
| 8 | 23-29 may | Fase 3 (si aplica) + integracion OE4 | Pipeline E2E |
| 9 | 30 may-5 jun | Experimentos finales | Todas las figuras Cap 6 |
| **10** | **6-12 jun** | **Borrador intermedio (Cap 4 resultados)** | **Entrega UNIR** |
| 11 | 13-19 jun | Revision + conclusiones | Cap 7 finalizado |
| 12 | 20-26 jun | Pulido final | Memoria casi lista |
| 13 | 27 jun-3 jul | Presentacion defensa | Slides + video demo |
| **14** | **4-10 jul** | **Entrega final** | **TFM ENTREGADO** |
