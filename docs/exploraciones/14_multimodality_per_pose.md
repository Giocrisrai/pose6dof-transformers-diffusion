# Exploración 14 — Multimodalidad por pose, con la condición de entrenamiento

**Estado**: ✅ **Cerrada** — la política no muestra estrategias de agarre alternativas, y es lo esperable. El modelo destilado es el más preciso y el más rápido.

**Fecha**: noviembre 2026

**Script**: `experiments/exp28_multimodality_per_pose.py` · **Resultado**: `experiments/results/exp28_multimodality/exp28_results.json`

---

## Por qué se rehízo el análisis

La memoria reporta K = 2 modos a partir de `exp8_diffusion_diversity.py`. Al revisarlo aparecieron dos problemas:

1. **La condición no era la de entrenamiento.** `exp8` arma el vector de condición como `[R.flatten(), t]`. El modelo que evalúa (`diffusion_policy_grasp.pth`, notebook 06) se entrenó con `pose[:3, :].flatten()`, es decir las filas de `[R | t]`, y con objetos sobre la mesa (z entre 2 y 8 cm). `exp8` le da además poses de cámara del checkpoint de FoundationPose. El modelo se evaluó fuera de su distribución.
2. **Los modos se buscaron entre escenas.** El K-means de `exp8` agrupa los puntos finales de cinco escenas distintas. Que cinco objetos en cinco sitios den grupos distintos no dice nada sobre multimodalidad, que es tener varias soluciones para *una misma* pose.

## Qué se midió

Para cada modelo, 20 poses tomadas de su propia distribución de entrenamiento y 100 trayectorias por pose (semilla 2026, MPS):

- **Error en el punto de agarre**: distancia entre los waypoints de la fase de agarre y el objeto.
- **Dispersión**: distancia media entre los puntos finales de una misma pose.
- **Modos**: mejor silhouette de K-means (K = 2..5) sobre los puntos finales, comparada con 100 réplicas de una referencia gaussiana de igual media y covarianza, que no tiene modos por construcción. Una pose "muestra modos" si p < 0.05; por azar se espera 1 de cada 20.

## Resultados (medianas sobre 20 poses)

| Modelo | Error de agarre | Dispersión | Silhouette | Referencia sin modos | Poses con modos |
|---|---:|---:|---:|---:|---:|
| `grasp` · DDIM-25 · condición de entrenamiento | 33.8 cm | 56.7 cm | 0.263 | 0.276 | 4 / 20 |
| `grasp` · DDIM-25 · condición de `exp8` | 41.4 cm | 56.3 cm | 0.268 | 0.277 | 3 / 20 |
| `ultra` · DDIM-25 | 3.0 cm | 4.4 cm | 0.407 | 0.288 | 20 / 20 |
| `ultra_fast` · destilado, 1 paso | 1.2 cm | 1.2 cm | 0.277 | 0.284 | 0 / 20 |

## Lectura

- **El modelo de la memoria (`grasp`) no llega al objeto.** Incluso con su condición correcta falla por 33.8 cm, y sus puntos finales se dispersan 56.7 cm. Esa dispersión es la que `exp8` interpretó como diversidad. Con la condición de `exp8` el error sube a 41.4 cm.
- **`ultra` llega al objeto (3.0 cm) y sus muestras sí tienen estructura**, pero no son estrategias alternativas: el grupo mayoritario reúne el 76 % de las trayectorias con 2.4 cm de error, y el resto son muestras peores (4.4 cm). Es una cola de muestras degradadas del muestreador DDIM-25.
- **`ultra_fast` es el mejor en todo**: 1.2 cm de error, 1.2 cm de dispersión y ninguna pose con modos. La destilación eliminó la cola de muestras malas.
- **No hay multimodalidad real, ni podía haberla.** Los datos de entrenamiento salen de un planificador heurístico determinista: para cada pose existe una única trayectoria correcta. Un modelo generativo solo puede aprender varios modos si las demostraciones los contienen.

## Qué implica

- La afirmación "K = 2 modos" de H2 no se sostiene como evidencia de multimodalidad. El criterio de ≥ 3 modos no se puede cumplir con estos datos, sea cual sea el modelo.
- Para estudiar multimodalidad hacen falta demostraciones con alternativas reales: teleoperación, varios planificadores, o agarres top-down y laterales para la misma pieza. Esto encaja con la validación en robot real.
- Para el pipeline actual, el modelo a usar es `ultra_fast`: más preciso y unas 20 veces más rápido que DDIM-25.

## Cambios en el código

- `src/planning/conditioning.py`: única definición de cómo se arma la condición de cada modelo.
- `scripts/api_server.py`: `/plan-grasp` y `/e2e` usan esa definición. Antes `/plan-grasp` daba al modelo `original` la disposición de `ultra`, y `/e2e` pasaba la traslación en milímetros.
- `tests/test_conditioning.py`: fija cada disposición y exige declararla para todo modelo servido por la API.

## Reproducir

```bash
python experiments/exp28_multimodality_per_pose.py     # unos 2 min en un M1
```
