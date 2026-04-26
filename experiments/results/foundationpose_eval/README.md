# Resultados FoundationPose — Evaluación BOP (YCB-V / T-LESS)

Esta carpeta contiene los **artefactos de salida** del notebook
`notebooks/colab/01_foundationpose_eval.ipynb` ejecutado en Google Colab
sobre los datasets BOP `ycbv` y `tless`. Es el respaldo material de los
resultados reportados en el Capítulo 6 del TFM.

## Cómo reproducir

1. Abrir el notebook en Colab y ejecutar todas las celdas.
2. Los JSON quedan automáticamente en
   `/content/drive/MyDrive/TFM/experiments/foundationpose_eval/`.
3. Descargar manualmente esos JSON aquí (Drive → repo).
4. Ejecutar:

   ```bash
   python experiments/generate_chapter6_figures.py
   ```

   Genera figuras y tabla LaTeX en `experiments/results/chapter6_figures/`.

## Archivos esperados

| Archivo | Origen | Contenido |
|---------|--------|-----------|
| `comparison_<timestamp>.json` | celda 24 del notebook | Métricas agregadas (ADD/ADD-S mean+median+AUC, recalls@5/10/20mm), config del run, baselines BOP/paper para contraste. **Es el archivo principal**. |
| `predictions_ycbv_<timestamp>.json` | celda 24 | Predicciones por frame YCB-V (R, t, score) + métricas. Útil para auditoría y re-cálculo. |
| `predictions_tless_<timestamp>.json` | celda 24 | Idem para T-LESS. |
| `fp_ycbv_checkpoint.json` | celda 18 (streaming) | Checkpoint reanudable, schema `v2_bop_targets_mask_per_gt_idx`. Si el run se interrumpe, retoma desde aquí. |
| `fp_tless_checkpoint.json` | celda 22 (streaming) | Idem T-LESS. |
| `RUN_CARD.md` | manual | Metadata del run (commit, fecha, GPU, MODE) — **versionado en git**. |

## Schema del comparison JSON

```json
{
  "timestamp": "YYYYMMDD_HHMMSS",
  "gpu": "Tesla T4",
  "config": {
    "max_scenes": 5,
    "max_images_per_scene": 50,
    "register_iterations": 5
  },
  "baselines": {
    "gdrnet": {...},
    "foundationpose_paper": {...}
  },
  "our_results": {
    "ycbv": {
      "metrics": {
        "add_mean_mm":   ..., "add_median_mm":   ...,
        "adds_mean_mm":  ..., "adds_median_mm":  ...,
        "auc_add":       ..., "auc_adds":        ...,
        "n_evaluated":   ...,
        "recall_add_5mm":  ..., "recall_add_10mm":  ..., "recall_add_20mm":  ...,
        "recall_adds_5mm": ..., "recall_adds_10mm": ..., "recall_adds_20mm": ...
      },
      "n_predictions":   ...,
      "timing_total_s":  ...
    },
    "tless": { ... idem ... }
  }
}
```

## Política de versionado

- **`comparison_*.json` y `RUN_CARD.md`**: versionar en git (son la
  evidencia del TFM, ~kB).
- **`predictions_*.json`**: pueden ser grandes (varios MB a cientos por
  miles de predicciones); versionar solo el más reciente referenciado
  por el TFM. Los anteriores van a `.gitignore` o se dejan en Drive.
- **`*_checkpoint.json`**: artefactos de ejecución, **no** versionar
  (`.gitignore` los excluye).

## Trazabilidad

El notebook guarda el campo `comparison.config.commit_hash` (si está
disponible) y `gpu`. Cualquier resultado citado en el TFM debe
referenciar el `comparison_*.json` exacto.
