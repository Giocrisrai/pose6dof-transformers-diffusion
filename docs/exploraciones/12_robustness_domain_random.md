# Exploración 12 — Robustez CLIP-image con domain randomization

**Estado**: ✅ **ÉXITO** — **12/12 condiciones perturbativas** mantienen accuracy ≥ 75 %. Validado con **bootstrap CI 95 %** (`bop-bootstrap-ci`). Mergeado a `main`.

**Fecha**: mayo 2026

---

## Hipótesis

> El modelo CLIP-image del exp 24 (entrenado con crops sintéticos limpios)
> sobrevive a las condiciones realistas de oclusión, ruido de sensor e
> iluminación variable que tendría una cámara industrial.

## Setup

300 escenas evaluadas en cada una de 12 condiciones:

- **Oclusión**: 0 %, 20 %, 40 %, 60 % del crop tapado por un cuadrado gris.
- **Ruido gaussiano** sobre intensidad: σ ∈ {0, 10, 25, 50}.
- **Iluminación** (gain multiplicativo): {0.5×, 0.75×, 1.0×, 1.5×, 2.0×}.
- **Combinación realista**: occ=20 % + σ=15 + gain=0.8 (típico industrial).

Bootstrap CI 95 % B=1000 sobre la media de aciertos, usando el paquete
`bop-bootstrap-ci` de la Exploración 1 (**cierra el ciclo**).

## Resultados (12 condiciones)

| Condición | Accuracy | CI 95% | Confianza media |
|---|:---:|:---:|:---:|
| `occ_0` (baseline) | 100.0 % | [1.000, 1.000] | 100.00 % |
| `occ_20` | 98.0 % | [0.960, 0.993] | 98.20 % |
| `occ_40` | 91.0 % | [0.877, 0.940] | 96.48 % |
| `occ_60` | **77.7 %** | [0.730, 0.820] | 90.27 % |
| `noise_10` | 100.0 % | [1.000, 1.000] | 99.94 % |
| `noise_25` | 100.0 % | [1.000, 1.000] | 99.88 % |
| `noise_50` | 99.3 % | [0.983, 1.000] | 99.32 % |
| `illum_0.5` | 100.0 % | [1.000, 1.000] | 99.96 % |
| `illum_0.75` | 100.0 % | [1.000, 1.000] | 99.96 % |
| `illum_1.5` | 93.0 % | [0.900, 0.957] | 96.11 % |
| `illum_2.0` | 91.0 % | [0.877, 0.940] | 94.91 % |
| `combined_realistic` | 93.7 % | [0.907, 0.963] | 97.85 % |

**Resumen**: media **95.3 %** | mín 77.7 % | máx 100.0 % | **12/12 ≥ 75 %**.

## Lecciones

1. **Oclusión es el modo de degradación principal**: 60 % de oclusión baja
   al 77.7 % accuracy. Razonable — un objeto ocluido 60 % es difícil incluso
   para humanos.
2. **Ruido sensor es prácticamente inocuo**: ni siquiera σ=50 (significativo)
   baja del 99.3 %.
3. **Sobre-exposición duele más que sub-exposición**: illum 2.0× cae a 91 %
   mientras 0.5× se mantiene en 100 %. CLIP está entrenado mayormente con
   imágenes naturales, no con highlights saturados.
4. **Combinación realista** (todas las perturbaciones a niveles típicos
   industriales) → 93.7 %, lo cual es **viable para producción**.

## Decisión

✅ **Merge a `main`**. Validación cuantitativa de que el pipeline
visual del exp 24 sobrevive a condiciones reales.

## Archivos

- `experiments/exp25_clip_image_robustness.py`
- `experiments/results/exp25_robustness/exp25_results.json`
- `experiments/results/exp25_robustness/fig_robustness_curves.png`
