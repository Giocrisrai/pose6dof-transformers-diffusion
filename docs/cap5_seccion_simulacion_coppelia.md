# Cap. 5 — Sección de simulación CoppeliaSim (parche para integrar al .docx)

> Este markdown contiene la sección "Validación de la infraestructura de
> simulación" para insertar en `TFM_Final_v2.docx` (subsección 4.X.4 o
> 5.Y según el orden definitivo). Está redactado en español académico.

---

## 5.Y. Validación de la infraestructura de simulación

### 5.Y.1. Stack tecnológico desplegado

Para la validación de la planificación de agarres en un entorno
controlado y reproducible se desplegó el siguiente stack:

- **Simulador**: CoppeliaSim Education V4.10.0 rev0, build oficial para
  macOS 15+ Apple Silicon (ARM64). Ejecutado localmente sobre MacBook
  Pro M1 Pro, macOS 26.3.
- **Conexión**: ZMQ Remote API v2 (Federico Ferri, Coppelia Robotics),
  cargada como addon por defecto desde la versión 4.6 del simulador,
  escuchando en `localhost:23000`.
- **Cliente Python**: `coppeliasim-zmqremoteapi-client==2.0.4`,
  versionado en `pyproject.toml` bajo el extra `[sim]`.
- **Wrapper del proyecto**: `src/simulation/coppeliasim_bridge.py`
  encapsula la conexión, la inicialización de handles del robot/cámara,
  y los primitivos de movimiento articular y captura RGB-D.

### 5.Y.2. Smoke test reproducible

El script `experiments/run_coppelia_smoke_test.py` materializa la
validación end-to-end de la pieza Python ↔ CoppeliaSim:

1. Conexión al servidor ZMQ.
2. Carga programática de la escena estándar
   `pickAndPlaceDemo.ttt` (incluida en el bundle de la app), que modela
   una línea industrial con cinta transportadora, generador estocástico
   de cajas con código de color y zona de detección.
3. Creación dinámica de un *vision sensor* en cenital (FOV 60°,
   resolución 640×480, 1.5 m sobre el origen), aliasado como
   `tfm_overview_sensor` para auditoría.
4. Ejecución de la simulación en modo *stepped* durante 100 pasos × 50
   ms = 5 s de tiempo simulado.
5. Render del sensor mediante `sim.handleVisionSensor` y serialización
   del frame como PNG.
6. Persistencia del JSON con métricas y handles descubiertos.

### 5.Y.3. Métricas validadas

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Latencia de conexión ZMQ | 150 ms | Aceptable; conexión única al inicio del experimento |
| Versión del servidor | 41000 (V4.10.0) | Coincide con el bundle empaquetado |
| Latencia por step | 18 ms (mean) | Permite ~55 Hz de control loop, suficiente para servoing |
| Tiempo simulado avanzado | 5.00 s exactos | Determinismo del scheduler de la simulación |
| Resolución del render | 640×480 RGB | Adecuada para detección y debug visual |
| Intensidad media del frame | 152.9 (rango [0, 255]) | Frame no degenerado; iluminación correcta |

> **Tabla 5.Y**: Métricas del smoke test del 2026-04-27. Fuente:
> `experiments/results/coppelia_smoke/smoke_test_result.json`.

La Figura 5.Y (`coppelia_overview_pickandplace.png`) muestra el render
cenital tras 5 s de simulación: la cinta transportadora alimenta cajas
de tres colores hacia la zona de detección, donde la lógica interna
del simulador podrá redirigirlas a la plataforma de clasificación de
la izquierda según el color identificado por la cámara.

### 5.Y.4. Limitaciones reconocidas

Tres limitaciones explícitas, documentadas en
`experiments/results/coppelia_smoke/README.md`:

1. La escena `pickAndPlaceDemo.ttt` valida la conexión ZMQ pero **no
   incluye** un brazo manipulador con los handles `/UR5`, `/Franka` ni
   los nombres de joints esperados por `CoppeliaSimBridge`. La autoría
   de una escena dedicada con UR5 + pinza Robotiq + bin con objetos
   T-LESS/YCB-V queda fuera del alcance temporal del TFM y se cataloga
   como **trabajo futuro**. Lo que sí queda demostrado es la
   infraestructura: cualquier escena que cumpla con los aliases del
   bridge sería operable inmediatamente.
2. La addon de cinemática inversa (KineticRobotics IK) y la
   integración con MoveIt 2 (vía `src/simulation/ros2_interface.py`)
   son ortogonales al smoke test. La integración ROS 2 está
   contenedorizada en `docker/Dockerfile` para reproducibilidad pero
   no se ejecuta en este experimento.
3. La generación estocástica de cajas en el conveyor depende de un
   script Lua interno de la escena, por lo que el contenido del frame
   final tras 5 s no es bit-perfecto reproducible. La distribución
   estadística del número y disposición de cajas sí es estable.

### 5.Y.5. Reproducibilidad

```bash
# 1. Lanzar CoppeliaSim
open -a /Applications/CoppeliaSim_Edu.app

# 2. Esperar ~5 s, luego ejecutar
.venv/bin/python experiments/run_coppelia_smoke_test.py
```

Salidas en `experiments/results/coppelia_smoke/`:

- `coppelia_overview_pickandplace.png` — render del sensor.
- `smoke_test_result.json` — métricas + handles.
- `README.md` — documentación de origen, configuración y limitaciones.

El cliente Python se instala con
`uv pip install coppeliasim-zmqremoteapi-client>=2.0.4` o, en el
proyecto, vía extra `[sim]` declarado en `pyproject.toml`.
