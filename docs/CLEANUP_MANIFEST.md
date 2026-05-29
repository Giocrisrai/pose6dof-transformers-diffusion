# CLEANUP_MANIFEST — revisión antes de mover/borrar

> Nada se mueve ni borra hasta que apruebes este manifiesto.
> Movimientos reversibles (a _frames_archive / _orphans_review).
> Borrado definitivo = paso final separado, con OK explícito.

| Carpeta | Tamaño | #refs docs | Clasificación | Acción propuesta |
|---|---|---:|---|---|
| chapter6_figures | 1.1M | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| coppelia_smoke | 140K | 4 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| demo_reel |  17M | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| diffusion_real_poses | 160K | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| diffusion_training | 440K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| drive_chapter6_figs | 160K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| e2e_verification | 1.1M | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp10_profiling | 104K | 3 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp11_paired_stats |  84K | 0 | HUERFANO | REVISAR uno por uno (candidato a archivar) |
| exp12_per_object | 124K | 0 | HUERFANO | REVISAR uno por uno (candidato a archivar) |
| exp13_model_comparison | 144K | 0 | HUERFANO | REVISAR uno por uno (candidato a archivar) |
| exp14_distillation |  36K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp15_open_license | 168K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp16_vla_lite | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp17_vla_robustness | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp18_vla_shapes | 4.0K | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp19_visual_sims | 3.3M | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp20_vla_multi_object | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp21_visual_multi | 3.0M | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp22_vla_size | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp23_sequential | 2.3M | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp24_clip_image | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp25_robustness |  72K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp26_spatial | 4.0K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp3_rotation_ablation | 420K | 3 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp4_grasp_comparison | 108K | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp5_diffusion_steps | 108K | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp6_robustness | 136K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp7_pbvs | 160K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp8_diversity | 188K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| exp9_3d_viz | 744K | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| foundationpose_eval | 1.3M | 8 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| local_notebooks | 8.4M | 1 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| pick_battery | 726M | 4 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| pick_demo | 239M | 2 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| pick_with_diffusion | 204M | 7 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| pick_with_fp_pose | 239M | 4 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| pipeline_e2e |  69M | 10 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |
| scenario_battery | 688K | 4 | KEEP | no tocar (solo mover sus frames/ -> _frames_archive) |

## Grupo INTERMEDIO (frames regenerables, ~1.4 GB)
Mover `experiments/results/{pick_battery,pick_with_fp_pose,pick_demo,pick_with_diffusion}/**/frames/`
a `experiments/results/_frames_archive/` (reversible). El reel no los necesita.

---

## Verificación de "HUÉRFANO" (2026-05-28)

El heurístico "0 refs en docs" marcó 3 carpetas como huérfanas, pero la
verificación a nivel de TODO el repo (`.py`/`.ipynb`/código) mostró que **NO
lo son** — son evidencia viva consumida por código. **Reclasificadas a KEEP, no
se tocan:**

| Carpeta | Referenciada por |
|---|---|
| exp11_paired_stats | `experiments/exp11_paired_stats.py`, `scripts/run_experiment.py` |
| exp12_per_object | `dashboard.py`, `experiments/exp12_per_object_analysis.py`, `scripts/run_experiment.py` |
| exp13_model_comparison | `scripts/api_server.py`, `experiments/exp13_diffusion_models_comparison.py` |

Lección: el conteo de refs solo sobre `docs/` es insuficiente; verificar contra
el código antes de archivar.

## Ejecutado (2026-05-28)

- **Movidos** 6 dirs `frames/` (de `pick_battery/{base,easy,hard}`,
  `pick_with_fp_pose`, `pick_demo`, `pick_with_diffusion`) a
  `experiments/results/_frames_archive/` (~1.4 GB **staged**, reversible).
- Verificado: `build_demo_reel.py --only-clips` sigue generando 4/4 clips sin
  los frames (el reel trabaja desde los MP4).
- Demos activos: de cientos de MB → pocos MB cada uno.

### Cómo revertir (si hiciera falta)

```bash
cd experiments/results/_frames_archive
for f in *; do
  dest="../$(echo "$f" | sed 's#__#/#g')"
  mkdir -p "$(dirname "$dest")"; mv "$f" "$dest"
done
```

### Borrado definitivo (PENDIENTE — requiere 2º OK explícito)

```bash
# Solo con confirmación adicional del usuario:
# rm -rf experiments/results/_frames_archive   # libera ~1.4 GB
```
