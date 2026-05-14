# Exploración 9 — VLA-lite con atributo continuo TAMAÑO

**Estado**: ✅ **ÉXITO** — **99.9 %** accuracy global manejando frases con tamaño (small/medium/large). Mergeado a `main`.

**Fecha**: mayo 2026

---

## Hipótesis

> El gate del exp 20 (attrs = color + shape) admite atributos continuos
> ampliando attr_dim sin cambios arquitectónicos. Añadiendo "size" como
> 8º dimensión (normalizada a [-1, 1] desde mm) el modelo discrimina por
> color, forma y tamaño con la misma calidad.

## Setup

- **attr_dim = 8**: RGB (3) + shape onehot (4) + size normalized (1)
- **Tamaños**: small (30 mm), medium (50 mm), large (80 mm) → normalizados a [-1, +1]
- **8 templates** cubriendo distintas combinaciones:
  - "pick the small red cube" (todos los atributos)
  - "grab the small red sphere" (color+size+shape)
  - "pick the {size} {shape}" (size+shape)
  - "select the {size} {color} one" (size+color)
  - "take the {size} object" (solo size)
  - "fetch the {color} {shape}" (sin size)
  - "pick the {color} {shape}" (sin size)
  - "the {size} one" (mínimo)
- **N = 2..5 objetos** por escena (heredado del exp 20)

## Resultados (n=1500 val, 60 epochs, 2.7 min M1 Pro)

### Accuracy por número de objetos

| N | Accuracy | n val |
|:---:|:---:|:---:|
| 2 | 100.0 % | 375 |
| 3 | 100.0 % | 360 |
| 4 | 100.0 % | 386 |
| 5 | **99.7 %** | 379 |
| **Global** | **99.9 %** | 1500 |

### Accuracy por template

| Template | Accuracy | n |
|---|:---:|:---:|
| "fetch the {color} {shape}" | 100.0 % | 193 |
| "grab the {size} {color} {shape}" | 99.5 % | 206 |
| "pick the {color} {shape}" | 100.0 % | 198 |
| "pick the {size} {color} {shape}" | 100.0 % | 172 |
| "pick the {size} {shape}" | 100.0 % | 165 |
| "select the {size} {color} one" | 100.0 % | 197 |
| "take the {size} object" | 100.0 % | 184 |
| "the {size} one" | 100.0 % | 185 |

Todos los templates ≥ 99.5 %.

### Latencia

DDIM-25: **0.05 ms/trayectoria** (medición batched: extremadamente bajo).

## Lecciones aprendidas

1. **Atributos continuos no degradan el modelo** — el gate aprende la métrica
   numérica de "size" sin perder discriminación categórica (color/shape).
2. **Templates extremadamente cortos funcionan** — "the small one" alcanza
   100 % cuando la escena tiene un único objeto del tamaño descrito.
3. **El modelo es **agnóstico al atributo mencionado** — no necesita saber
   a priori si la frase es sobre size, color, shape o combinación.

## Decisión

✅ **Merge a `main`**. Modelo `diffusion_policy_clip_size.pth` guardado (5.6 MB).

## Implicación industrial directa

- **Logística**: "pick the large red box" entre paquetes mixtos
- **Reciclaje**: "grab the small plastic" (separa por tamaño + material)
- **Manufactura**: "select the medium M8 bolt" (tamaño preciso)
- **Médico**: "pick the large vial" (vials de 5/10/20 ml)

## Limitaciones

- Solo 3 niveles discretos (small/medium/large). Para industrial real
  habría que aceptar tamaños arbitrarios en mm.
- Tamaño aún declarado, no inferido visualmente. Para producción
  habría que estimarlo desde la nube de puntos / segmentación.

## Archivos

- `experiments/exp22_vla_size.py`
- `experiments/results/exp22_vla_size/exp22_results.json`
- `data/models/diffusion_policy_clip_size.pth`
