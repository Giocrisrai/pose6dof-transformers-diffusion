# exp27 · text-to-CAD → simulación → pose 6-DoF

Mini-experimento que integra generación **text-to-CAD** con el pipeline del TFM.
Traduce una descripción en lenguaje a una pieza CAD paramétrica con geometría de
*ground-truth* exacta y la usa como (1) objeto de bin picking en CoppeliaSim y
(2) modelo 3D para estimación de pose 6-DoF.

Inspirado en el repositorio [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad)
(colección de *agent skills* de CAD/robótica). Usamos su motor subyacente,
**build123d** (sobre OpenCascade), no el plugin completo.

## Motivación

FoundationPose es *model-based*: necesita el modelo 3D del objeto. Generar piezas
desde texto permite **objetos de test con geometría exacta conocida**, ideales para
medir el error de pose de forma controlada y para enriquecer las escenas más allá
de YCB-Video / T-LESS.

## Pieza generada

> *"escuadra en L de 60×40 mm, espesor 5 mm, ala vertical de 45 mm, dos agujeros
> de 6 mm en la base y uno en el ala"*

La asimetría (forma en L + agujeros) hace la **pose 6-DoF inequívoca**.

| Propiedad | Valor |
|-----------|-------|
| Bounding box | 60 × 40 × 45 mm (= especificación) |
| Sólidos / caras | 1 / 11 |
| Volumen CAD | 19 575.9 mm³ |
| Malla STL | **watertight**, 1 cuerpo, 1544 triángulos |
| Volumen malla vs CAD | 19 576.1 vs 19 575.9 mm³ → error < 0.01 % |
| Formatos | STEP · STL · GLB · OBJ (`assets/`) |

![preview](figs/preview_bracket.png)

## Paso 1 — Física en CoppeliaSim ✅

Import del OBJ (escala mm→m) en una escena con bin; se suelta desde 18 cm y se
registra `z(t)`. La pieza **cae, impacta a ~0.2 s y se asienta estable** sobre la
base del bin (7.0 mm), sin atravesar el suelo y dentro del contenedor.

| z inicial | z mínimo | z final | estable | dentro del bin |
|-----------|----------|---------|---------|----------------|
| 167 mm | 6.8 mm | 7.0 mm | sí (σ < 1 µm) | sí |

![trayectoria](figs/sim_z_trajectory.png) ![captura](figs/sim_drop_final.png)

## Paso 2 — Recuperación de pose 6-DoF (proxy model-based) ✅

> **Nota de honestidad.** FoundationPose (red neuronal) requiere CUDA y se ejecuta
> en Google Colab; en el M1 Pro no corre localmente (ver
> `src/perception/foundation_pose.py`, líneas 15-17 y 74-78). Este experimento
> **no ejecuta la red**: es un **proxy local** del mismo principio *model-based*
> (modelo CAD → hipótesis global FPFH+RANSAC → refinamiento ICP point-to-plane →
> selección por *fitness*, el rol del *scorer*), sobre vistas parciales sintéticas
> con ruido de profundidad (~1 mm). Sí confirma que el mesh carga por el wrapper
> real del repo (`FoundationPoseEstimator.load_cad_model`).

Evaluación sobre **N = 12** poses ground-truth exactas:

| Métrica | Resultado |
|---------|-----------|
| Error de traslación | **2.6 ± 1.0 mm** (mediana 2.9) |
| Error de rotación (mediana) | **6.7°** |
| Rotaciones correctas (< 8°) | 8–10 / 12 (según semilla RANSAC) |
| *Flips* de 180° | 2–3 / 12 |

![eval](figs/pose_recovery_eval.png)

**Lectura.** La traslación se recupera con precisión de milímetros en todos los
casos. La mayoría de rotaciones también, pero **2-3 casos caen en un *flip* de
180°**: son vistas donde el ala vertical queda auto-ocluida y solo se observa la
placa base (casi un rectángulo), es decir la **ambigüedad de simetría bajo
observación parcial**. Es exactamente el fallo que motiva el *scorer* aprendido de
FoundationPose —y que el registro geométrico clásico no resuelve—, en línea con el
análisis de oclusión/simetría del TFM.

## Reproducción

Entorno: `.venv` del repo (uv) + CoppeliaSim Edu V4.10 en `localhost:23000`.
Dependencias extra: `build123d` (`uv pip install build123d --python .venv/bin/python`).

```bash
python gen_part.py            # genera assets/test_bracket.{step,stl,glb,obj}
# (abrir CoppeliaSim_Edu antes del siguiente)
python sim_drop_test.py       # física en el bin -> figs/sim_z_traj.npy + captura
python pose_recovery_proxy.py # pose 6-DoF vs ground-truth -> figs/pose_recovery_results.npy
python make_figures.py        # regenera las figuras desde los .npy
```

## Archivos

- `gen_part.py` — generador text-to-CAD (build123d)
- `sim_drop_test.py` — import + física en CoppeliaSim
- `pose_recovery_proxy.py` — recuperación de pose model-based (Open3D)
- `make_figures.py` — figuras a partir de los `.npy`
- `assets/` — CAD exportado (STEP/STL/GLB/OBJ)
- `figs/` — figuras y datos crudos (`.npy`)
