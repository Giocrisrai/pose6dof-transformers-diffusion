# Cap. 6 — Sección de evaluación FoundationPose (parche para integrar al .docx)

> Este markdown contiene la sección "Resultados FoundationPose en BOP" para
> insertar en `TFM_Capitulo_6_Experimentos.docx`. Está redactado en español
> académico con citas y referencias a las figuras y tablas del repo.

---

## 6.X. Evaluación de FoundationPose en datasets BOP

### 6.X.1. Configuración experimental

Se evaluó el modelo FoundationPose (Wen et al., CVPR 2024) sobre los
datasets BOP `ycbv` (YCB-Video) y `tless` (T-LESS), siguiendo el
protocolo del BOP Challenge 2022. La inferencia se ejecutó en Google
Colab con GPU NVIDIA Tesla T4 (16 GB), CUDA 12.1 y torch 2.1.2+cu121
(versiones congeladas en `requirements.colab.lock.txt`). El notebook de
referencia es `notebooks/colab/01_foundationpose_eval.ipynb` y la
tarjeta de identidad del run se documenta en
`experiments/results/foundationpose_eval/RUN_CARD.md`.

Los frames evaluados corresponden al subconjunto oficial
`test_targets_bop19.json` de cada dataset, que define las imágenes y
objetos sobre los que se reportan los resultados del leaderboard BOP.
Sin este filtrado, las métricas no serían comparables al estado del
arte. Se evaluaron 5 escenas por dataset (modo `dev`) con 50 imágenes
por escena, configuración `MAX_SCENES=5`, `MAX_IMAGES_PER_SCENE=50`,
`REGISTER_ITERATIONS=5`.

Cabe destacar que la configuración del pipeline requirió la corrección
de tres errores no triviales durante el desarrollo (todos versionados
en commits anteriores a la ejecución reportada):

1. **Cargado de máscaras por instancia (`gt_idx`)**: el loader devolvía
   sistemáticamente la máscara del primer ground-truth de cada imagen.
   En YCB-V (multiobjeto) esto provocaba que FoundationPose registrara
   los objetos en localizaciones incorrectas, inflando el ADD a valores
   ~130 mm.
2. **Filtrado por subset BOP-19**: la versión inicial iteraba sobre
   todos los frames RGB en lugar de los listados en
   `test_targets_bop19.json`, produciendo métricas no comparables.
3. **Emparejamiento GT por instancia múltiple**: en T-LESS, donde las
   escenas contienen varias instancias del mismo `obj_id`, la métrica
   emparejaba la N-ésima predicción siempre con la primera instancia
   GT. Se introdujo un contador de instancias por `obj_id`.

Estos errores se documentan en los commits `ea8fc2e`, `280c9f3` y
`7b4c7c6` para auditoría.

### 6.X.2. Resultados YCB-Video

La Tabla 6.X resume las métricas obtenidas sobre las 5 escenas de test
de YCB-V (250 imágenes, 1098 objetos evaluados):

| Métrica | Valor |
|---------|-------|
| ADD mediano | **3.5 mm** |
| Tiempo por objeto | 4125 ms |
| Throughput | 0.2 objetos/s |
| Tiempo total | 1 h 18 min |

> **Tabla 6.X**: Métricas de FoundationPose en YCB-V (subset BOP-19,
> 5 escenas × 50 imágenes). El valor de ADD mediano se sitúa por debajo
> del rango reportado en el paper original (10-20 mm), lo que sugiere
> que la combinación específica de escenas evaluadas (relativamente
> sencillas en oclusión) puede sesgar el resultado a la baja respecto
> al cómputo sobre las 12 escenas completas.

Los valores agregados de AUC ADD, AUC ADD-S y recalls a umbrales 5/10/20
mm se reportan en el archivo `comparison_<timestamp>.json` y se ilustran
en la Figura 6.X (`fig_6_X_fp_real_add_metrics.png`), generada
automáticamente por el script `experiments/generate_chapter6_figures.py`.

### 6.X.3. Resultados T-LESS

> *Sección pendiente de completar tras la ejecución del bloque T-LESS
> del notebook. Estructura simétrica a 6.X.2.*

### 6.X.4. Comparación con baselines

Como referencia cualitativa al estado del arte, la Tabla 6.Y reproduce
los valores oficiales de:

- **GDR-Net++** (Liu et al., 2022) en el BOP Challenge 2022 Leaderboard.
- **FoundationPose** (Wen et al., 2024) en el paper original.

| Dataset | Método | AR_VSD | AR_MSSD | AR_MSPD | Mean AR |
|---------|--------|--------|---------|---------|---------|
| YCB-V   | GDR-Net++ (BOP 2022)     | 0.842 | 0.819 | 0.874 | 0.845 |
| YCB-V   | FoundationPose (CVPR 2024) | 0.882 | 0.862 | 0.907 | 0.884 |
| T-LESS  | GDR-Net++ (BOP 2022)     | 0.736 | 0.685 | 0.773 | 0.731 |
| T-LESS  | FoundationPose (CVPR 2024) | 0.774 | 0.725 | 0.832 | 0.777 |

> **Tabla 6.Y**: Métricas BOP oficiales (Average Recall sobre VSD/MSSD/MSPD).

La comparación directa con nuestra ejecución no es estricta: nosotros
reportamos ADD/ADD-S porque el toolkit oficial BOP que calcula VSD
requiere dependencias nativas no disponibles en el entorno Colab. La
relación entre métricas (ADD/ADD-S vs AR) se discute en el Apéndice C.

### 6.X.5. Reproducibilidad

Todos los artefactos del experimento están en el repositorio del TFM:

- **Código del pipeline**: `notebooks/colab/01_foundationpose_eval.ipynb`,
  `src/utils/dataset_loader.py`, `src/perception/evaluator.py`.
- **Tarjeta del run**: `experiments/results/foundationpose_eval/RUN_CARD.md`.
- **JSON de resultados**: `experiments/results/foundationpose_eval/comparison_<timestamp>.json`.
- **Predicciones por frame**: `experiments/results/foundationpose_eval/predictions_{ycbv,tless}_<timestamp>.json`.
- **Figuras y tabla LaTeX generadas**: `experiments/results/chapter6_figures/`.
- **Lockfile de versiones Colab**: `requirements.colab.lock.txt`.
- **Contenedor GPU equivalente**: `docker/inference-gpu.Dockerfile` con
  versiones idénticas pinneadas (ver `docker/README-GPU.md`).

Para replicar el experimento basta con ejecutar el notebook en Colab
(o el contenedor en una máquina con GPU NVIDIA), bajar los JSON
generados al repositorio y ejecutar
`python experiments/generate_chapter6_figures.py` para regenerar
figuras y tabla.
