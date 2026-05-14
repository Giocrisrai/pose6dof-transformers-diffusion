# Exploración 6 — VLA-lite multi-atributo (color + forma)

**Estado**: ✅ **ÉXITO** — el modelo selecciona objetos por color, forma o combinación con **99.9 % accuracy global** (n=1200 val). Mergeado a `main`.

**Fecha**: mayo 2026

---

## Hipótesis

> El TextGroundedGate del exp 16 (solo colores) se generaliza a multi-atributo:
> añadiendo forma (cube/sphere/cylinder/box) como atributo adicional con encoder
> one-hot, el modelo puede discriminar por color, forma o ambos sin perder precisión.

## Setup

- **Atributos por objeto**: RGB (3-D) + shape one-hot (4-D) = 7-D total.
- **3 modos de escena** (proporciones equilibradas):
  1. Mismo shape, colores distintos → templates "pick the {color} object"
  2. Mismo color, shapes distintos → templates "grab the {shape}"
  3. Distintos ambos → templates "select the {color} {shape}"
- **Gate**: `MultiAttributeGate(attr_dim=7, clip_dim=512)`.
- **Training**: 6000 escenas, 50 epochs, 1.7 min en M1 Pro.

## Resultados

| Modo | Templates ejemplo | Accuracy | n |
|---|---|---|---|
| **Color** | "pick the red object" | **100.0 %** | 389 |
| **Shape** | "grab the cylinder" | **99.8 %** | 418 |
| **Combinado** | "select the red sphere" | **100.0 %** | 393 |
| **GLOBAL** | | **99.9 %** | 1200 |

| Métrica | Valor | Criterio | Estado |
|---|---|---|---|
| Selection accuracy global | 99.9 % | ≥ 75 % | ✅ |
| Accuracy por modo (mínimo) | 99.8 % | ≥ 65 % | ✅ |
| Latencia DDIM-25 | 1.73 ms/traj | < 200 ms | ✅ |
| Tamaño modelo | 5.6 MB | — | OK |

## Lección importante

El `attr_dim=7` (RGB + shape onehot) es suficiente como inductive bias.
El `MultiAttributeGate` aprende a leer el atributo relevante de la frase
sin necesidad de routing explícito (sin saber a priori si la frase es
sobre color, shape o ambos).

## Decisión

✅ **Merge a main**. Modelo `diffusion_policy_clip_shapes.pth` disponible.

## Limitaciones

- Solo 3 colores × 4 shapes = 12 categorías. Generalización a más atributos
  requiere re-entrenar (esfuerzo trivial: ampliar dimensión de attr_dim).
- Atributos sintéticos declarados (no inferidos visualmente). Para uso real
  habría que pasar por CLIP-image (planeado en roadmap exp 19+).
- Sin tamaño/peso/material como atributos continuos.

## Archivos

- `experiments/exp18_vla_shapes.py`
- `experiments/results/exp18_vla_shapes/exp18_results.json`
- `data/models/diffusion_policy_clip_shapes.pth`
