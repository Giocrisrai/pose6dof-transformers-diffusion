# CoppeliaSim — Smoke test de la infraestructura de simulación

Salidas del script `experiments/run_coppelia_smoke_test.py`. Validación
mínima de que el sistema CoppeliaSim Edu V4.10.0 + cliente Python ZMQ
Remote API está operativo para el pipeline de bin picking del TFM
(cap. 5).

## Requisitos previos

| Componente | Versión validada |
|------------|------------------|
| CoppeliaSim Edu | V4.10.0 rev0 macOS15 arm64 |
| Localización app | `/Applications/CoppeliaSim_Edu.app` |
| Plugin ZMQ Remote API | Cargado por defecto (sim 4.6+) |
| Cliente Python | `coppeliasim_zmqremoteapi_client==2.0.4` |
| Puerto por defecto | `localhost:23000` |
| Hardware probado | MacBook Pro M1 Pro (16 GB), macOS 26.3, ARM64 |

## Procedimiento ejecutado

1. Apertura programática de CoppeliaSim:
   `open -a /Applications/CoppeliaSim_Edu.app`.
2. Conexión al servidor ZMQ (`RemoteAPIClient('localhost', 23000)`).
3. Carga programática de la escena
   `CoppeliaSim_Edu.app/Contents/Resources/scenes/pickAndPlaceDemo.ttt`
   (escena estándar del bundle: cinta transportadora + cajas con
   código de color + zona de detección con vision sensor).
4. Creación de un vision sensor adicional en cenital (1.5 m sobre el
   origen, FOV 60°, resolución 640×480), aliasado como
   `tfm_overview_sensor`, para captura repetible.
5. Ejecución de la simulación en modo stepped, 100 pasos × 50 ms = 5 s
   de tiempo simulado.
6. Renderizado del sensor mediante `sim.handleVisionSensor`,
   captura del buffer RGB y serialización a PNG.
7. Detención y persistencia del JSON con métricas y handles
   descubiertos.

## Resultados (`smoke_test_result.json`)

| Métrica | Valor |
|---------|-------|
| Tiempo de conexión ZMQ | 150 ms |
| Versión del servidor sim | 41000 (= V4.10.0) |
| Escena cargada | `pickAndPlaceDemo.ttt` |
| Handles relevantes encontrados | `/Floor`, `/genericConveyorTypeA`, `/genericDetectionWindow` |
| Latencia por step (media) | 18 ms |
| Tiempo simulado tras 100 pasos | 5.00 s |
| Resolución del render | 640×480 RGB |
| Intensidad media del render | 152.9 (rango [0, 255]) |

## Ficheros

- `coppelia_overview_pickandplace.png` — render cenital tras 5 s sim:
  conveyor en marcha, 7 cajas (2 azul, 3 verde, 2 roja) en la zona de
  detección, plataforma de clasificación a la izquierda.
- `smoke_test_result.json` — métricas + handles + parámetros del sensor.
- `README.md` — este documento.

## Reproducibilidad

```bash
# 1. Lanzar CoppeliaSim (carga el addon ZMQ Remote API por defecto)
open -a /Applications/CoppeliaSim_Edu.app

# 2. Esperar ~5 s y ejecutar el smoke test
.venv/bin/python experiments/run_coppelia_smoke_test.py
```

El script crea su propio vision sensor en cada ejecución, no depende de
modificaciones manuales de la escena. La ejecución es determinista
salvo por el contenido temporal del conveyor (las cajas se generan por
script Lua interno de CoppeliaSim a intervalos regulares; tras 100
pasos se observa siempre una distribución similar).

## Limitaciones reconocidas

1. La escena `pickAndPlaceDemo.ttt` es industrial pero genérica
   (clasificación por color), no incluye objetos BOP (T-LESS / YCB-V)
   ni un brazo manipulador con `/UR5` o `/Franka`. Para una demo
   completa con la API `CoppeliaSimBridge` (`src/simulation/`)
   habría que autorear una escena con esos handles. Este smoke test
   valida la pieza ZMQ + render; la escena dedicada queda como
   trabajo futuro de la fase 2.
2. La addon de cinemática inversa (KineticRobotics IK / MoveIt) no se
   prueba aquí — es ortogonal a la conexión Python ↔ CoppeliaSim.
