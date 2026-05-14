# Exploración 13 — Razonamiento espacial en VLA-lite

**Estado**: ✅ **ÉXITO** — **98.4 % global** sobre 13 templates de razonamiento espacial ("leftmost", "closest", "highest", "topmost", "the one closest to me"). Mergeado a `main`.

**Fecha**: mayo 2026

---

## Hipótesis

> El gate puede aprender **razonamiento espacial** (referencias a posición
> relativa: izquierda/derecha/arriba/cerca/lejos) si recibe las coordenadas
> XYZ normalizadas de cada objeto como parte de sus atributos.

## Setup

- **`attr_dim = 10`**: RGB (3) + shape onehot (4) + position normalizada (3)
- **13 templates espaciales** con referencias variadas:
  - `pick the leftmost {color} object`
  - `grab the rightmost {shape}`
  - `select the leftmost one`
  - `the rightmost object`
  - `pick the closest {color} {shape}`
  - `the closest one`
  - `the farthest object`
  - `the highest one` / `the topmost {shape}`
  - `the lowest one`
  - `the {color} on the left` / `the {color} on the right`
  - `pick the one closest to me`

- **N = 3-4 objetos** (mín 3 para que las referencias espaciales tengan sentido).
- **8000 train + 1500 val**, 70 epochs, 3.6 min en M1 Pro.

## Resultados (val n=1500)

### Accuracy global: **98.4 %** ✅

### Accuracy por template (ordenado descendente)

| Template | Accuracy | n val |
|---|:---:|:---:|
| `the lowest one` | 100.0 % | 119 |
| `the closest one` | 100.0 % | 138 |
| `the farthest object` | 100.0 % | 105 |
| `select the leftmost one` | 100.0 % | 108 |
| `pick the one closest to me` | 100.0 % | 116 |
| `the rightmost object` | 99.2 % | 120 |
| `the highest one` | 99.1 % | 109 |
| `the {color} on the left` | 98.4 % | 126 |
| `pick the leftmost {color} object` | 98.3 % | 115 |
| `the {color} on the right` | 98.2 % | 114 |
| `grab the rightmost {shape}` | 95.4 % | 109 |
| `the topmost {shape}` | 95.3 % | 85 |
| `pick the closest {color} {shape}` | 94.9 % | 136 |

Todos ≥ 94.9 % — supera ampliamente el criterio de 55 % por template.

## Por qué funciona tan bien

1. **CLIP entiende lenguaje espacial nativamente**: "leftmost", "closest",
   "highest" están en el vocabulario CLIP. El gate solo necesita alinear
   esas palabras con las coordenadas que recibe.
2. **Coordenadas normalizadas a [-1, 1]** facilitan al modelo razonar sobre
   "extremos" (max / min).
3. **El gate aplica scoring por objeto independientemente** (shared weights),
   lo que permite generalizar a cualquier configuración espacial.

## Lección importante

Esto cierra el ciclo VLA-lite a un sistema que **realmente razona**, no solo
filtra por atributos:
- *"pick the leftmost red sphere"* requiere razonar SOBRE las 3 esferas
  rojas presentes y elegir la más a la izquierda.
- *"the one closest to me"* requiere proyectar la noción de "me" como
  Y=0 (centro del workspace) y elegir el objeto con Y mayor.

## Decisión

✅ **Merge a `main`**. Modelo `diffusion_policy_clip_spatial.pth` (5.6 MB).

## Limitaciones

- Referencias absolutas, no relativas a otro objeto ("the one to the right of the red box").
- Sin razonamiento temporal ("the one that came first").
- No probado con escenas dinámicas (objetos en movimiento).

## Implicación industrial

- **Logística**: "pick the leftmost box" cuando hay 5 paquetes en cinta.
- **Reciclaje**: "the highest object" para alcanzar capas superiores.
- **Manufactura**: "the closest screw" para optimizar trayectoria.

## Archivos

- `experiments/exp26_spatial_reasoning.py`
- `experiments/results/exp26_spatial/exp26_results.json`
- `data/models/diffusion_policy_clip_spatial.pth` (5.6 MB)
