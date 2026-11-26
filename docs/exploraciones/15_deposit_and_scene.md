# Exploración 15 — Por qué fallaba el depósito, y una escena que cambió sin aviso

**Estado**: ✅ **Cerrada** — el pick-and-place con apariencia aleatoria pasa del 36-40 % al 88-92 % sin tocar la política. De paso apareció y se corrigió una regresión silenciosa del agarre.

**Fecha**: noviembre 2026

**Scripts**: `experiments/eval_diffusion_iter9_sim.py --bandeja --release-hold-steps 10` · `experiments/exp29_deposit_by_shape.py`
**Resultados**: `experiments/results/pick_with_diffusion/eval_v9_bandeja.json` · `experiments/results/exp29_deposit_by_shape/exp29_results.json`

---

## El síntoma

Con apariencia aleatoria (Iter 8 e Iter 9) el agarre estaba cerca del 100 %, pero el pick-and-place completo quedaba entre el 36 % y el 48 %. Separado por forma, sobre los 75 agarres de esas corridas:

| Forma | n | Agarre | Depósito | Error mediano de depósito |
|---|---:|---:|---:|---:|
| Cubo | 30 | 100 % | 83 % | 7 cm |
| Esfera | 18 | 100 % | 11 % | 1.24 m |
| Cilindro | 27 | 96 % | 15 % | 2.05 m |

El agarre funciona con las tres formas. Lo que falla es que esferas y cilindros terminan a más de un metro de la zona de depósito.

## La causa no es la política

Se añadió una medida nueva: la posición de la pieza **en el instante de abrir la pinza** (`release_error_m`). En la corrida corregida, 49 de 50 piezas se sueltan dentro de la zona de depósito. La política lleva la pieza donde debe; lo que se medía como fallo ocurre después de soltarla, por dos efectos del simulador:

1. **La pieza recuperaba su física todavía entre los dedos.** Al abrir la pinza, la pieza pasaba a ser dinámica y con colisión en el mismo paso, con los dedos aún cerrados alrededor. La interpenetración la expulsaba (hay errores de 20-34 m en 1.5 s de simulación).
2. **La zona de depósito era una superficie abierta.** Una esfera o un cilindro que caen desde 30 cm ruedan fuera del umbral de 30 cm.

## La corrección

- **Soltado suave** (`release_hold_steps`): la pieza queda quieta 10 pasos mientras la pinza se abre y solo entonces recupera su física.
- **Bandeja de depósito** (`src/simulation/deposit_tray.py`): suelo y cuatro paredes de 16 cm, hueco de 30 × 30 cm. Es lo que hace una celda real: deposita en un contenedor.

Cada medida por separado no basta (10 agarres de prueba, mismas semillas): solo soltado suave, 50 %; solo bandeja, 80 %; ambas, 100 %.

## Resultado (mismas semillas que `eval_v9_clutter.json`, n = 25 por condición)

| Condición | Pick-and-place antes | Pick-and-place corregido | Agarre |
|---|---:|---:|---:|
| Sin distractores | 36 % | **92 %** | 100 % |
| Con 2 distractores | 40 % | **88 %** | 100 % |

Por forma, sobre los 50 agarres: cubo 80 → 95 %, esfera 8 → 83 %, cilindro 11 → 89 %. Los 5 fallos que quedan son piezas que rebotan fuera de la bandeja.

La política y sus pesos no cambian. Las corridas de la memoria se conservan y se siguen pudiendo reproducir: ambas opciones están desactivadas por defecto.

## La escena que cambió sin aviso

Al repetir la evaluación, el agarre salió en 60 % en vez del 100 % versionado. Los pesos estaban bien (con las imágenes guardadas del entrenamiento: 2.0 cm de error, 98 % de agarres plausibles). La diferencia estaba en la imagen: `data/scenes/bin_base.ttt` se volvió a guardar el 29 de junio de 2026 con el brazo en otra postura, después de entrenar v8 y v9. La política ve el brazo en la imagen y, con otra postura, apunta peor:

| Escena | Error de agarre | Agarre |
|---|---:|---:|
| La del entrenamiento (junio) | 1.9 cm | 100 % |
| La re-guardada | 4.5 cm | 72 % |

La fila de la escena re-guardada es la corrida versionada (`eval_v9_escena_reguardada.json`, 25 agarres); en las corridas de diagnóstico previas, no versionadas, el agarre salió entre 56 % y 60 %.

Se restauró la escena del entrenamiento y `tests/test_scene_integrity.py` fija su huella: si el archivo cambia, el test falla.

El mismo efecto apareció con la bandeja: visible para la cámara, el agarre bajó al 80 %; fuera de la imagen, volvió al 100 %.

## Qué significa para el robot real

- **El resultado de pick-and-place ya no depende de la forma de la pieza.**
- **La política visual es sensible a lo que ve fuera de la pieza** (la postura del brazo, un objeto nuevo en la mesa). La aleatorización de Iter 8 y 9 cubrió la apariencia de las piezas, no la del resto de la escena. En un robot real la postura de partida y el entorno cambian: hay que aleatorizarlos en el entrenamiento o recortar la imagen a la zona de agarre. Es el siguiente experimento antes del salto a hardware.
- **La bandeja no se renderiza en la cámara**, así que este resultado mide la contención física, no la robustez visual a la bandeja.
