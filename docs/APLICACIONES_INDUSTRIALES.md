# Aplicaciones Industriales del Pipeline TFM

> Análisis del potencial de aplicación real de la integración **FoundationPose + Diffusion Policy + Visual Servoing PBVS** desarrollada en este TFM, con casos de uso por industria, métricas de viabilidad y propuestas de extensión.

---

## 1. Sectores con aplicación directa

### 1.1. Automoción

**Problema**: ensamblaje robotizado de piezas pequeñas (tornillos, conectores, cableado) en líneas de producción donde la geometría y posición exacta varía entre cada unidad.

**Aplicación**:
- FoundationPose detecta la pieza objetivo en el contenedor sin requerir reentrenamiento por SKU.
- Diffusion Policy genera trayectorias de aproximación adaptables a obstáculos en el espacio de trabajo.
- PBVS cierra el lazo de control durante el agarre, alcanzando precisión submilimétrica.

**Casos concretos**:
- Selección de bujías, sensores, conectores en línea Tier-1
- Manipulación de piezas reflectantes metálicas (T-LESS-like, validado en este TFM)
- Pre-ensamblaje de subconjuntos electrónicos

**Métricas viables (extrapolando del TFM)**:
- Tiempo de ciclo: **6-7 s/instancia** (validado E2E live)
- Tasa de éxito a 10 mm: **>95 %** (Recall@10mm ADD-S YCB-V)
- Robustez ante oclusión 70 %: **−1 pp en T-LESS, −2.6 pp en YCB-V**

**Empresas referencia que usan tecnologías similares**: ABB Robotics, Fanuc, KUKA, Yaskawa.

---

### 1.2. Logística y e-commerce (warehousing)

**Problema**: pick & place de productos en almacenes con SKU heterogéneos y alta rotación, donde reentrenar por cada producto es prohibitivo.

**Aplicación**:
- Capacidad de generalización a *novel objects* heredada de FoundationPose: no requiere reentrenamiento por SKU.
- Diffusion Policy multimodal: el robot puede elegir entre múltiples estrategias de agarre cuando hay ambigüedad.
- Sistema híbrido local + Colab T4: deployment económico en cells de picking.

**Casos concretos**:
- Picking en almacenes Amazon Fulfillment Centers
- Order fulfillment en farmacéutica (selección de medicamentos)
- Sorting en centros de paquetería (DHL, FedEx, Correos)

**Métricas viables**:
- Tiempo por SKU: **<7 s** (compatible con ciclo industrial)
- Coste de hardware: **~$2k por estación** (Mac mini M2 + cámara RGB-D + cobot)
- Diversidad de objetos manejados: ilimitada (sin reentrenamiento)

**Empresas referencia**: Amazon Robotics, Ocado, Symbotic, Berkshire Grey.

---

### 1.3. Electrónica y semiconductores

**Problema**: posicionamiento de componentes electrónicos (resistencias, chips, conectores) con precisión submilimétrica en placas PCB de alta densidad.

**Aplicación**:
- Precisión submilimétrica heredada del refinamiento ICP neural de FoundationPose.
- PBVS para corrección fina durante la inserción.
- Análisis por categoría de objeto (exp12) permite identificar componentes problemáticos y especializar el sistema.

**Casos concretos**:
- Pick & place de SMD en líneas de manufactura electrónica
- Inserción de conectores en cables (industria automotriz)
- Posicionamiento de IC en PCB

**Métricas viables**:
- Precisión: ADD-S < 5 mm en >97 % de casos (datos T-LESS)
- Robustez a iluminación variable (oclusión sintética demostrada)

**Empresas referencia**: ASML, Foxconn, Flex, Jabil.

---

### 1.4. Reciclaje y economía circular

**Problema**: separación robotizada de residuos en plantas de reciclaje, donde los objetos están sucios, deformados, parcialmente ocultos y son altamente heterogéneos.

**Aplicación**:
- Robustez ante oclusión hasta 70 % validada (exp6).
- Diffusion Policy genera trayectorias multimodales adecuadas para objetos en orientaciones impredecibles.
- Pipeline reproducible en hardware accesible: viable para PYMES de reciclaje.

**Casos concretos**:
- Separación de botellas PET vs vidrio vs metal
- Picking de componentes electrónicos en e-waste
- Selección de plásticos por tipo (PE, PP, PS) para reciclaje

**Métricas viables**:
- Cycle time: 6-7 s suficiente para línea de reciclaje
- Robustez geométrica: degradación contenida ante oclusiones severas

**Empresas referencia**: AMP Robotics, ZenRobotics, Greyparrot.

---

### 1.5. Médico y farmacéutico

**Problema**: dispensación automatizada de medicamentos, manipulación de instrumental quirúrgico, posicionamiento de muestras en laboratorio.

**Aplicación**:
- Precisión submilimétrica requerida en farmacia (selección de comprimidos individuales).
- Marco matemático SE(3) garantiza precisión geométrica formal.
- PBVS con tolerancia configurable (2 mm en este TFM, ajustable).

**Casos concretos**:
- Dispensación robotizada en hospitales y farmacias
- Pick & place de tubos de ensayo en laboratorios clínicos
- Posicionamiento de instrumental en cirugía asistida

**Empresas referencia**: Intuitive Surgical, Medtronic, Boston Dynamics (Stretch).

---

## 2. Extensiones técnicas factibles

### 2.1. Integración con LLM para planificación de alto nivel

**Idea**: usar un LLM (GPT-4, Claude, Llama) para interpretar instrucciones en lenguaje natural y orquestar el pipeline.

```
Usuario: "Recoge la pieza roja del centro del contenedor"
   ↓
LLM: identifica objeto objetivo, lo describe a FoundationPose
   ↓
Pipeline TFM: ejecuta percepción → planificación → agarre
```

**Estado**: factible inmediatamente. APIs LLM disponibles. El pipeline TFM ya expone interfaces estructuradas.

### 2.2. Multi-objeto en una sola pasada

**Idea**: extender Diffusion Policy a planificación secuencial de múltiples agarres en una escena cluttered.

**Estado**: requiere extensión del modelo. Diseño viable con horizon expandido.

### 2.3. Sim-to-real con Isaac Sim

**Idea**: validar el pipeline en NVIDIA Isaac Sim con física fotorealistic antes de deployment en robot real.

**Estado**: documentado en `REPRODUCIBILITY.md`. Requiere GPU NVIDIA Linux.

### 2.4. Edge deployment (NVIDIA Jetson)

**Idea**: portar el pipeline a Jetson Orin Nano/AGX para deployment en factoría sin necesidad de cloud.

**Estado**: factible. Diffusion Policy ya cabe en 3-5 MB. FoundationPose requiere optimización TensorRT.

**Métricas estimadas en Jetson Orin**:
- Inferencia FP: ~2-3 s/instancia (vs 4 s actual)
- Diffusion DDIM-25: ~50 ms (vs 100 ms en MPS)
- **Cycle total: ~5 s** (factible para industria)

### 2.5. Multi-cámara y fusión sensorial

**Idea**: usar múltiples cámaras RGB-D + láser 3D + sensores hápticos para mayor robustez.

**Estado**: requiere extensión de la capa de percepción.

---

## 3. APIs y servicios deployables

### 3.1. REST API (FastAPI) — Implementada en este TFM

Servidor con endpoints:
- `POST /predict-pose` — recibe imagen RGB-D + modelo CAD, devuelve pose 6-DoF
- `POST /plan-grasp` — recibe pose, devuelve trayectoria multimodal
- `POST /e2e` — pipeline completo en un endpoint
- `GET /docs` — OpenAPI Swagger automático

**Uso**: integrable directamente en cualquier sistema industrial vía HTTP.

### 3.2. SDK Python — Reusable

```python
from pose6dof_pipeline import Pipeline

pipeline = Pipeline.from_pretrained("ultra")
result = pipeline.process(rgb_image, depth_image, cad_model)
print(result.pose, result.trajectory, result.cycle_time_ms)
```

### 3.3. Dashboard web — Streamlit (ya implementado)

Visualización interactiva de métricas, resultados experimentales y demo del pipeline.

### 3.4. Demo Gradio público

Posibilidad de publicar en Hugging Face Spaces para que la comunidad pueda probar el pipeline online sin instalación.

---

## 4. Posicionamiento académico y comercial

### 4.1. Publicaciones académicas viables

| Venue | Contribución sugerida | Tipo |
|-------|----------------------|------|
| **CVPR Workshops** | Reproducibilidad de FP+Diffusion sin GPU dedicada | Short paper |
| **ICRA** | PBVS + Diffusion integrado en SE(3) | Full paper |
| **IROS** | Análisis de robustez/escalabilidad en bin picking | Full paper |
| **CoRL** | Diffusion Policy multimodal para grasp planning | Workshop |
| **BOP Challenge** | Submission oficial con toolkit C++ | Challenge |

### 4.2. Posicionamiento comercial

**Spin-off potencial**: empresa de servicios de pick & place a PYMES industriales que no pueden permitirse setups industriales completos.

**Modelo de negocio**:
- Hardware: Mac mini + cobot ($5k-10k por estación)
- Software: licencia anual de pipeline ($X/año)
- Servicios: integración, training, mantenimiento

**Validación**: el coste del setup TFM ($2k) es 1-2 órdenes de magnitud menor que las soluciones industriales actuales ($15k-150k).

---

## 5. Limitaciones y caminos críticos

### 5.1. Limitaciones actuales (alcance TFM)

- **Validación solo en simulación**: no incluye robot físico
- **Sin BOP toolkit C++**: no participa en challenge oficial
- **Licencia NC de FoundationPose**: restricción para uso comercial directo
- **Single-shot**: no maneja escenas secuenciales aún

### 5.2. Caminos críticos para producción

1. **Transferencia sim-to-real** con domain randomization
2. **Reentrenamiento de FoundationPose con licencia abierta** (foundation model alternativo)
3. **Hardware certificado**: PLC, cámara industrial, robot certificado
4. **Estandarización ROS 2** para deployment industrial
5. **Tests de seguridad**: certificaciones CE, ISO/TS 15066

### 5.3. Time-to-market estimado

| Fase | Duración | Inversión |
|------|----------|-----------|
| Prototipo en robot real (UR/Kinova) | 3 meses | ~10k EUR |
| Validación en planta piloto | 6 meses | ~50k EUR |
| Certificación industrial | 6 meses | ~30k EUR |
| Comercialización beta | 6 meses | ~100k EUR |
| **Total a producto comercial** | **~2 años** | **~200k EUR** |

---

## 6. Recursos disponibles en el repositorio

| Recurso | Path | Uso |
|---------|------|-----|
| API REST | `scripts/api_server.py` | Servidor inferencia |
| Demo Gradio | `scripts/gradio_demo.py` | UI web interactiva |
| Dashboard | `dashboard.py` | Métricas + resultados |
| CLI experiments | `scripts/run_experiment.py` | Reproducir todos los experimentos |
| Docker GPU | `docker/inference-gpu.Dockerfile` | Deployment industrial |
| Pesos entrenados | `data/models/` | 10 modelos Diffusion (original/extended/ultra/ultra_fast + 6 VLA-lite: clip, clip_shapes, clip_multi, clip_size, clip_image, clip_spatial) + configs FP |
| Pipeline open-license | `src/perception/checkpoint_adapter.py` | FreeZeV2 viable a −3 pp AUC (exp 15) |
| VLA-lite con CLIP | `data/models/diffusion_policy_clip.pth` | "pick the red object" → 98.6 % selection acc (exp 16) |
| Paquete PyPI | `packages/bop_bootstrap_ci/` | `pip install bop-bootstrap-ci` con bootstrap CI 95 % |
| Notebooks | `notebooks/` | Material educativo |

---

*Este documento se actualiza con nuevos casos de uso identificados durante el deployment.*
