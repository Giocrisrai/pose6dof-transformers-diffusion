# Roadmap post-TFM — etapas físicas y comerciales

> Documento que delimita claramente lo que **está cerrado en simulación**
> (lo que el TFM y sus 13 exploraciones validan cuantitativamente) y lo
> que **requiere hardware físico, certificación o financiación** para ir
> a producción real. Las etapas se ordenan por dependencias y se acompañan
> de timeline + presupuesto estimado.
>
> *Toda la base teórica, algorítmica y arquitectónica está validada
> en simulación. Las etapas siguientes son ingeniería de despliegue,
> no investigación.*

---

## Cerrado en simulación — disponible hoy en `main`

### Pipeline core (TFM, junio 2026)

✅ FoundationPose 6-DoF · AUC ADD-S 0.908 YCB-V / 0.957 T-LESS
✅ Diffusion Policy multimodal · MSE 0.0022 noise pred
✅ PBVS en SE(3) · convergencia 100 % en 50 muestras
✅ Cycle E2E p95 < 7 s en CoppeliaSim con robot Ragnar
✅ Bootstrap CI 95 % B=1000 en todas las métricas clave
✅ 123 tests TFM pasando

### Exploraciones post-TFM (mayo 2026, validadas en sim)

| # | Hito | Resultado |
|---|---|---|
| 1 | `bop-bootstrap-ci` paquete PyPI | 27 tests, 97 % cov |
| 2 | Distillation Diffusion 1-NFE | ×517 speedup real |
| 3 | Pipeline open-license | FreeZeV2 Apache-2.0 −3 pp AUC |
| 4-7 | VLA-lite color + forma + multi-objeto | 100 % global |
| 8 | Multi-objeto N=2..5 | 100 % accuracy |
| 9 | Atributo continuo TAMAÑO | 99.9 % |
| 10 | Instrucciones secuenciales | 8/8 secuencias completas |
| 11 | CLIP-image visual grounding | 100 % sin atributos declarados |
| 12 | Robustez domain randomization | 12/12 condiciones ≥ 75 % |
| 13 | Razonamiento espacial | 98.4 % sobre 13 templates |

**Capacidades del sistema actual (sólo simulación, todo open-source MIT):**

- Toma una pose 6-DoF y produce trayectoria de agarre multimodal en < 100 ms
- Entiende instrucciones en lenguaje natural sobre color, forma, tamaño,
  cantidad, orden secuencial, posición espacial relativa
- Procesa imágenes visuales reales (CLIP-image) — no requiere atributos
  declarados a mano
- Sobrevive a oclusión 60 %, ruido sensor σ=50, iluminación 0.5x-2.0x
- Funciona con hasta 5 objetos simultáneos en la escena
- Cycle E2E budget compatible con ciclos industriales (< 10 s)

---

## Etapa 1 — Validación en robot físico (3-6 meses, ~10 000 EUR)

**Objetivo**: cerrar el gap sim-to-real.

### Hardware necesario
| Componente | Coste estimado | Justificación |
|---|---|---|
| Cobot 6-DoF (UR3e, xArm 7, Kinova Gen3) | 8 000-15 000 EUR | Brazo seguro para uso académico/PYME |
| Cámara RGB-D (Intel RealSense D435i o ZED 2) | 400-1 200 EUR | Replicable, calibrada |
| PC host (Mac mini M2 o NUC) | 800-1 500 EUR | Inference local |
| Pinza paralela o ventosa | 200-1 500 EUR | Configurable por tarea |
| Mesa de trabajo + bins + objetos test | 500 EUR | Replicabilidad |
| **Total hardware** | **~10 000-20 000 EUR** | |

### Hitos
1. **Mes 1-2**: integración driver del brazo (ROS 2) + calibración cámara.
2. **Mes 2-3**: validar pose estimation real vs sintética. Domain randomization
   en sim para reducir gap.
3. **Mes 3-4**: ejecutar Diffusion + PBVS en brazo real con FP-replay de poses
   precomputadas (validar control sin perception loop).
4. **Mes 4-5**: pipeline completo: cámara → FP → Diffusion → PBVS → ejecución real.
5. **Mes 5-6**: tasa de éxito sobre 100 picks reales con bootstrap CI.

### Criterios de éxito
- Tasa de éxito ≥ 85 % sobre 100 picks de objetos T-LESS / YCB-V
- Cycle p95 real ≤ 12 s (margen sobre H3 de 10 s en sim)
- Sin colisiones documentadas durante la fase de test

### Riesgos
- Domain gap mayor del esperado (sim→real degradación > 10 pp)
  → mitigación: aumentar domain randomization en exp 25, fine-tune con demos
- Latencia GPU vs MPS si se usa hardware sin Apple Silicon
  → mitigación: distillation a 1-NFE ya validada (exp 2)

---

## Etapa 2 — Domain randomization sim-to-real (2-3 meses, paralelizable con Etapa 1)

**Objetivo**: minimizar el delta entre simulación y robot real.

### Trabajo
1. **Photo-realistic rendering en Isaac Sim** (NVIDIA): regenerar dataset
   con luz/textura/cámara variando aleatoriamente.
2. **Augmentations agresivos** sobre crops CLIP-image:
   - Motion blur, lens distortion
   - Reflexiones / specular highlights (para objetos metálicos T-LESS)
   - Oclusiones por gripper (parte del propio robot)
3. **Fine-tune con pares (sim_pose, real_pose)** una vez con datos del paso 1.

### Hardware adicional
- GPU NVIDIA (mínimo RTX 4080 o uso de Vast.ai / Lambda Cloud): 100-500 EUR/mes

### Criterios de éxito
- Reducción del domain gap medido en Etapa 1 en ≥ 50 %.

---

## Etapa 3 — Aprendizaje a partir de demos humanas (LfD) (2-4 meses)

**Objetivo**: sustituir trayectorias sintéticas por demostraciones humanas reales.

### Trabajo
1. **Captura por teleoperación**: VR controller o gamepad para guiar el brazo.
   Recoger ~500-1 000 demos diversas.
2. **Captura kinestésica**: mover el brazo a mano (modo gravity compensation).
3. **Re-train Diffusion Policy** sobre las trayectorias reales. El gate
   (TextGroundedGate/MultiAttributeGate/SpatialGate) no necesita cambios.
4. **Comparación A/B**: trayectorias sintéticas vs demos humanas. Métricas:
   tasa de éxito, jerk, naturalidad subjetiva.

### Criterios de éxito
- Tasa de éxito ≥ 90 % (mejora sobre Etapa 1).
- Jerk medio reducido vs sintético (movimientos más naturales).

### Resultado esperado
Modelo que **se mueve como un humano lo haría**, no como un optimizador
puramente geométrico. Crítico para aceptación operario.

---

## Etapa 4 — Producto MVP (3 meses, ~50 000 EUR)

**Objetivo**: convertir el sistema en algo demostrable a clientes potenciales.

### Componentes
1. **UI operario**: tablet/web con campo de texto + visualización en vivo del pipeline.
2. **API documentada** + SDK Python para integradores.
3. **Docker images** firmadas con todas las dependencias (CUDA / CPU / MPS).
4. **Monitoring + telemetría** (Prometheus + Grafana) sobre cycle p95, tasa de éxito, gripper failures.
5. **Documentación operacional**: instalación, calibración, primer pick en < 1 día.

### Personal
- 1 ingeniero ML/robótica fulltime · ~40 000 EUR/3 meses
- 0.5 ingeniero frontend · ~10 000 EUR/3 meses

### Criterios de éxito
- Demo del MVP convence a ≥ 3 PYMES industriales a hacer piloto pagado.

---

## Etapa 5 — Validación en planta piloto (6 meses, ~50 000 EUR)

**Objetivo**: instalar el sistema en una planta real con un cliente.

### Trabajo
1. **Selección del partner**: PYME de logística/reciclaje/manufactura con ≥ 1 cinta donde aplicar.
2. **Integración**: PLC, sistema de visión existente, interfaz operario.
3. **Fine-tune in-situ** con objetos reales del cliente.
4. **Operación supervisada** durante 3 meses con métricas comparables a baseline humano.

### Métricas piloto
- Tasa de éxito real ≥ 90 %.
- Throughput ≥ 80 % del operario humano.
- Tiempo medio entre fallos > 8 h.
- ROI demostrado al cliente (ahorro €/año).

---

## Etapa 6 — Certificación industrial (6 meses, ~30 000 EUR)

**Objetivo**: cumplir normativa para venta comercial.

### Normas aplicables
- **CE marking** (Directiva Máquinas 2006/42/CE)
- **ISO/TS 15066** — Robots colaborativos
- **ISO 10218** — Safety requirements for industrial robots
- **IEC 61508** (si software safety-critical)

### Hitos
1. Análisis de riesgos completo.
2. Pruebas en laboratorio acreditado (TÜV / SGS).
3. Documentación técnica + manual de usuario.
4. Marcado CE.

### Coste estimado
- Auditorías + tests: ~20 000 EUR
- Documentación legal: ~10 000 EUR

---

## Etapa 7 — Spin-off / SaaS (continuo, post-certificación)

### Modelo de negocio
- **Hardware**: ~5 000 EUR/estación (Mac mini + cobot + cámara + pinza)
- **Software**: licencia anual SaaS — propuesta inicial 5 000 EUR/año/estación
- **Servicios**: integración 8-15 k EUR + mantenimiento 2 k EUR/año

### Mercado objetivo
- PYMES industriales europeas sin presupuesto para soluciones de 150 000 EUR
- Educación / universidades (research kit)
- Reciclaje (alto volumen, márgenes ajustados — sweet spot para nuestro coste)

---

## Resumen ejecutivo timeline + presupuesto

| Etapa | Duración | Hardware/Inversión | Acumulado |
|---|---|---|---|
| 1. Robot físico | 3-6 meses | ~10-20 k EUR | 20 k |
| 2. Sim-to-real | 2-3 meses (paralelo) | ~2 k EUR (GPU cloud) | 22 k |
| 3. Demos humanas | 2-4 meses | ~0 EUR (sólo trabajo) | 22 k |
| 4. MVP producto | 3 meses | ~50 k EUR (personal) | 72 k |
| 5. Planta piloto | 6 meses | ~50 k EUR | 122 k |
| 6. Certificación | 6 meses | ~30 k EUR | 152 k |
| 7. Spin-off | continuo | inversión seed | 200 k |
| **Total a producto comercial** | **~2 años** | **~200 k EUR** | |

### Comparativa
- Inversión típica para llegar a producto industrial: 500 k - 2 M EUR
- Nuestra estimación: **200 k EUR**, en parte porque la I+D
  algorítmica ya está hecha (TFM + 13 exploraciones validadas en sim).

---

## Lo que NO requiere etapa posterior — disponible HOY en `main`

- Reproducir todos los resultados del TFM
- Probar Diffusion Policy distillada a 1-NFE (×517 speedup)
- Sustituir FP por FreeZeV2 documentado (open-license)
- Dar instrucciones en lenguaje natural: color, forma, tamaño, espacial, secuencial
- Robustez evaluada con bootstrap CI 95 % sobre 12 condiciones
- API REST + Gradio + Streamlit + Docker funcionando
- Paquete PyPI `bop-bootstrap-ci` listo para publicar

---

## Riesgos honestos

1. **Domain gap sim-to-real**: nunca cero. Mitigación: domain randomization + LfD.
2. **Coste real del cobot**: si el cliente quiere certificación industrial,
   sube a 25-40 k EUR. Mitigación: empezar por aplicaciones menos críticas.
3. **Competencia VLA fundación model**: RDT-1B / π0 / OpenVLA podrían
   absorber este nicho con sus propios proveedores. Mitigación: el TFM se
   diferencia por **coste de hardware** y **rigor estadístico**, no por
   capacidad pura.
4. **Aceptación operario**: los operarios pueden rechazar instrucciones en
   lenguaje natural. Mitigación: tablet con preset común + lenguaje libre opcional.

---

## Métricas de éxito por etapa

```
Etapa 1 (robot real):     85 % éxito sobre 100 picks
Etapa 2 (sim-to-real):    delta < 5 pp vs sim
Etapa 3 (LfD):            90 % éxito + jerk natural
Etapa 4 (MVP):            3 PYMES interesadas
Etapa 5 (piloto):         ROI positivo demostrado
Etapa 6 (certif):         CE marking obtenido
Etapa 7 (spin-off):       primera venta comercial
```

---

*Documento de roadmap post-TFM mantenido por Giocrisrai Godoy y José Miguel Carrasco.
Actualizado: mayo 2026.*
