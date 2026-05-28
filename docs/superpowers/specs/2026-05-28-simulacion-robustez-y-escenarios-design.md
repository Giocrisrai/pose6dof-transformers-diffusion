# Robustez de simulación + librería de escenarios CoppeliaSim

**Fecha:** 2026-05-28
**Estado:** Diseño aprobado, pendiente plan de implementación
**Autor:** ggodoy@mtechsol.com

## Contexto

El pipeline de bin-picking del TFM (FoundationPose → Diffusion Policy → ejecución motora) se valida hoy contra `pickAndPlaceDemo.ttt` (escena genérica que CoppeliaSim trae de fábrica). Esto tiene tres problemas:

1. **Duplicación de integración.** `CoppeliaSimBridge` (502 líneas, `src/simulation/coppeliasim_bridge.py`) está implementado pero solo lo importa `tests/test_grasp_sampler.py`. Los runners (`experiments/run_coppelia_smoke_test.py`, `experiments/run_pipeline_e2e.py`, `experiments/run_e2e_live.py`) instancian `RemoteAPIClient` directamente, cada uno con su propio camino de conexión, cleanup y manejo de errores.
2. **Sin cobertura de tests.** `tests/` no tiene ningún archivo dedicado al módulo de simulación. Cualquier cambio en el bridge o en la API de CoppeliaSim solo se detecta corriendo experimentos a mano.
3. **Bug bloqueante en `run_pipeline_e2e.py`.** El loader del checkpoint en `experiments/run_pipeline_e2e.py:117-120` solo contempla las claves `model` o state-dict plano, pero `data/models/diffusion_policy_grasp.pth` está guardado con la clave `model_state_dict` (formato full-training). El runner E2E aborta antes de tocar la simulación.
4. **Falta de variación de escenarios.** Una sola escena genérica no es evidencia suficiente para la defensa ni para validar generalización. No hay forma estructurada de definir y correr "desafíos" (objetos distintos, colores, iluminación, oclusión).

## Objetivos

- Centralizar la integración con CoppeliaSim en `CoppeliaSimBridge` (un único camino de conexión, cleanup, manejo de errores).
- Habilitar variación visual de escenarios (objetos, colores, luces) mediante una librería de `.ttt` + manifiesto YAML.
- Producir evidencia comparativa entre escenarios (tiempos, success rate, snapshots) en formato pegable en el TFM.
- Cubrir el bridge con tests pytest (unit con mocks + integración opt-in en CI con CoppeliaSim headless).

## No-objetivos (esta iteración)

- Domain randomization automática (texturas, fricción, ruido de cámara, poses aleatorias por run).
- Spawneo programático de mallas YCB-V/T-LESS reales en CoppeliaSim.
- Tocar `experiments/run_e2e_live_ultra.py`, `experiments/record_e2e_video*.py`, `experiments/exp19_visual_simulations.py`, `experiments/exp10_profiling.py`, `dashboard.py`, `scripts/gradio_demo.py`.
- Cambiar los números o JSON ya reportados en cap. 5/6 — los outputs migrados deben ser bit-identical salvo timestamps y ±5% en tiempos.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    experiments/ (runners)                        │
│  run_coppelia_smoke_test.py   ┐                                 │
│  run_pipeline_e2e.py          ├──► with CoppeliaSimBridge() ──► CoppeliaSim
│  run_e2e_live.py              │                                  │
│  run_scenario_battery.py [NEW]┘                                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │      src/simulation/coppeliasim_bridge.py    │
        │  CoppeliaSimBridge                           │
        │   ├── connect(retries) / disconnect          │
        │   ├── __enter__ / __exit__                   │
        │   ├── load_scene / close_scene  [NEW]        │
        │   ├── set_stepping / step                    │
        │   ├── capture_rgbd / get_camera_pose         │
        │   ├── move_joints / actuate_gripper          │
        │   ├── set_object_color    [NEW]              │
        │   ├── set_light_intensity [NEW]              │
        │   ├── set_object_visibility [NEW]            │
        │   └── apply_scenario(dict) [NEW]             │
        └──────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │              data/scenes/                    │
        │   bin_base.ttt    (generada por script)      │
        │   bin_easy.ttt    (vos modelás en GUI)       │
        │   bin_hard.ttt    (vos modelás en GUI)       │
        │   scenarios.yaml  (manifiesto)               │
        └──────────────────────────────────────────────┘
```

## Diseño detallado

### 1. Cambios al bridge

**Archivo:** `src/simulation/coppeliasim_bridge.py`

#### Métodos nuevos

```python
def load_scene(self, scene_path: str | Path, close_current: bool = True) -> None:
    """Carga una escena .ttt. Si close_current, cierra la actual primero.

    Raises:
        FileNotFoundError: si scene_path no existe.
    """

def close_scene(self) -> None:
    """sim.closeScene() — útil para tests/cleanup."""

def set_stepping(self, enabled: bool) -> None:
    """Activa/desactiva modo stepped. Guarda flag para validar step()."""

def get_simulation_state(self) -> int:
    """Wrapper de sim.getSimulationState()."""

def get_simulation_time(self) -> float:
    """sim.getSimulationTime()."""

def set_object_color(self, name: str, rgb: tuple[float, float, float]) -> None:
    """Cambia color shape de un objeto. rgb en [0,1]^3.

    Implementación: sim.getObject(name) → sim.setShapeColor(handle, None,
    sim.colorcomponent_ambient_diffuse, list(rgb)).
    """

def set_light_intensity(self, light_name: str, intensity: float) -> None:
    """Cambia intensidad de luz. intensity ≥ 0 (típicamente 0.0–1.5).

    Implementación:
        handle = sim.getObject(light_name)
        # signature real: setLightParameters(handle, state, diffuse, specular)
        # state=1 (encendida); diffuse = [intensity]*3 (gris escalado);
        # specular = [intensity*0.2]*3 (componente especular tenue)
        sim.setLightParameters(
            handle,
            1,
            None,                          # ambient: usa default
            [intensity, intensity, intensity],
            [intensity*0.2]*3,
        )
    """

def set_object_visibility(self, name: str, visible: bool) -> None:
    """Oculta/muestra un objeto vía sim.setObjectInt32Param(handle,
    sim.objintparam_visibility_layer, layer)."""

def apply_scenario(self, scenario: dict) -> None:
    """Aplica los tweaks de un scenario dict en orden.

    Asume que la escena ya está cargada (el caller debe llamar `load_scene`
    antes). Las claves `id` y `scene` del dict se ignoran acá — solo se itera
    sobre `tweaks`.

    scenario = {
        "id": str,           # ignorado por apply_scenario
        "scene": str,        # ignorado por apply_scenario (caller hace load)
        "tweaks": [
            {"type": "color"|"light"|"visibility", "target": str, ...}
        ]
    }

    Raises:
        ValueError: si un tweak tiene type desconocido o campos requeridos faltantes.
    """
```

#### Hardening del `connect()`

- Nuevos parámetros `retries: int = 3, retry_delay_s: float = 1.0`. Reintenta si el addon ZMQ aún no levantó (caso típico: CoppeliaSim recién abierto).
- Tras conectar, hace ping con `sim.getSimulationTime()` para validar que el server responde (no solo que TCP abrió).
- Loguea versión del servidor en INFO (`sim.getInt32Param(sim.intparam_program_version)`).
- Nuevo parámetro `strict: bool = False`. Si `True`, falta de handles esperados en `_init_handles` levanta `RuntimeError` en vez de loguear warning. Default `False` para no romper consumidores que cargan escenas genéricas (smoke test usa `pickAndPlaceDemo.ttt` que no tiene `/rgb_camera`).

#### Context manager

```python
def __enter__(self):
    self.connect()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    if self._connected and self._sim is not None:
        try:
            if self.get_simulation_state() != self._sim.simulation_stopped:
                self.stop_simulation()
            self.set_stepping(False)
        except Exception as e:
            logger.warning(f"cleanup falló: {e}")
    self.disconnect()
    return False  # no suprime excepciones
```

Usa los wrappers públicos (`get_simulation_state`, `stop_simulation`, `set_stepping`) en vez de hablar directo a `self._sim` — así el camino de cleanup queda cubierto por los mismos tests que cubren la API pública.

#### Otros fixes internos

- `disconnect()`: cerrar el cliente ZMQ explícitamente si la API expone `shutdown()`.

**No cambia:** las firmas públicas existentes (`capture_rgbd`, `move_joints`, `actuate_gripper`, `execute_pick`, `get_joint_positions`, etc.) — solo se agrega.

### 2. Librería de escenarios

#### Estructura en disco

```
data/scenes/
├── bin_base.ttt        # generada por scripts/build_bin_base_scene.py (esta iteración)
├── bin_easy.ttt        # OPCIONAL — usuario la modela en GUI cuando quiera
├── bin_hard.ttt        # OPCIONAL — usuario la modela en GUI cuando quiera
├── scenarios.yaml      # manifiesto (esta iteración: 3 entradas, todas apuntando a bin_base.ttt)
└── README.md           # convenciones de nombres y guía de modelado
```

`bin_easy.ttt` y `bin_hard.ttt` se listan como aspiracionales: hasta que el usuario los modele en la GUI, el `scenarios.yaml` inicial tiene los 3 escenarios apuntando a `bin_base.ttt`, diferenciándose solo por `tweaks`. Cuando el usuario quiera más variedad estructural (no solo color/luz), duplica `bin_base.ttt` en CoppeliaSim, edita la copia, y cambia el campo `scene:` del scenario correspondiente.

#### Convención de nombres (handles)

Para que `_init_handles()` y los tweaks encuentren los objetos, cada `.ttt` debe tener:

| Handle               | Tipo            | Obligatorio |
|----------------------|-----------------|-------------|
| `/UR5e`              | Modelo robot    | Sí (si hay control motor) |
| `/shoulder_pan_joint` ... `/wrist_3_joint` | Joints UR5 | Sí |
| `/rgb_camera`        | Vision sensor RGB | Sí |
| `/depth_camera`      | Vision sensor depth | Sí |
| `/bin`               | Shape           | Sí |
| `/object_1` ... `/object_N` | Shapes  | Sí (N ≥ 1) |
| `/Light`             | Light           | Sí |
| `/gripper`           | Shape o modelo  | Opcional |
| `/tip`               | Dummy (TCP)     | Opcional |

#### `bin_base.ttt` — contenido

Generada por `scripts/build_bin_base_scene.py` (vía ZMQ contra CoppeliaSim corriendo). Contiene:

- Piso por defecto (`/Floor`).
- Robot cargado desde `/Applications/CoppeliaSim_Edu.app/Contents/Resources/models/robots/non-mobile/UR5.ttm` (path verificado: existen `UR3.ttm`, `UR5.ttm`, `UR10.ttm` en esa carpeta — **no hay** `UR5e.ttm`). Tras `loadModelOrPlugin`, se llama `sim.setObjectAlias(root, "UR5e")` para que el handle quede como `/UR5e`, manteniendo consistencia con la `RobotConfig.name` por defecto del bridge. Los joint names de UR5 mecánica son los mismos que la `RobotConfig` espera (`shoulder_pan_joint` ... `wrist_3_joint`).
- `bin` (caja hueca: 4 paredes + piso) en frente del robot, ~30 cm de lado.
- `rgb_camera` cenital sobre el bin (640×480, FOV 60°, near=0.05, far=2.0).
- `depth_camera` co-localizada con la RGB.
- `Light` omnidireccional cenital.
- 5 objetos surtidos en el bin: 3 cubos (rojo, verde, azul, ~5 cm de lado) + 2 cilindros (gris, amarillo, ~5 cm de diámetro × 6 cm de alto), aliasados `/object_1` ... `/object_5`.

#### Formato `scenarios.yaml`

```yaml
scenarios:
  - id: base
    scene: bin_base.ttt
    description: "Escena base, 5 objetos surtidos, iluminación nominal"
    difficulty: easy
    # sin tweaks: la escena tal cual viene

  - id: easy
    scene: bin_base.ttt
    description: "5 objetos con colores acentuados, luz fuerte"
    difficulty: easy
    tweaks:
      - { type: color, target: "/object_1", rgb: [0.9, 0.1, 0.1] }
      - { type: color, target: "/object_2", rgb: [0.1, 0.9, 0.1] }
      - { type: color, target: "/object_3", rgb: [0.1, 0.1, 0.9] }
      - { type: light, target: "/Light", intensity: 1.2 }

  - id: hard
    scene: bin_base.ttt           # cambiará a bin_hard.ttt cuando el usuario lo modele
    description: "Iluminación baja + objetos ocultos (simula oclusión)"
    difficulty: hard
    tweaks:
      - { type: light, target: "/Light", intensity: 0.3 }
      - { type: visibility, target: "/object_4", visible: false }
      - { type: visibility, target: "/object_5", visible: false }
```

#### Tipos de tweak soportados

| Tipo          | Campos                                     | Bridge call                          |
|---------------|--------------------------------------------|--------------------------------------|
| `color`       | `target: str`, `rgb: [r,g,b]` en [0,1]    | `set_object_color`                   |
| `light`       | `target: str`, `intensity: float`         | `set_light_intensity`                |
| `visibility`  | `target: str`, `visible: bool`            | `set_object_visibility`              |

Tweaks con `type` distinto o campos faltantes → `apply_scenario` levanta `ValueError` con el id del scenario.

#### Validación

Schema con `pydantic` (si ya está en deps) o `dataclass` + validación manual. El runner battery falla rápido al inicio si:
- Algún `.ttt` referenciado no existe.
- Algún tweak tiene `type` desconocido o campos requeridos faltantes.

#### Script `scripts/build_bin_base_scene.py`

- Conecta a CoppeliaSim corriendo en `:23000` vía bridge.
- Limpia escena, agrega piso.
- Carga UR5e desde la librería local.
- Crea bin, cámaras, luz, objetos vía `sim.createPrimitiveShape` + `sim.setObjectAlias`.
- Llama `sim.saveScene("data/scenes/bin_base.ttt")`.
- Imprime resumen de handles creados.

### 3. Runner battery — `experiments/run_scenario_battery.py`

```python
def main():
    scenarios = load_scenarios("data/scenes/scenarios.yaml")
    planner, scheduler = load_diffusion_policy()  # MPS
    results = []

    for sc in scenarios:
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(f"data/scenes/{sc.scene}")
            bridge.apply_scenario(sc.to_dict())
            bridge.set_stepping(True)
            bridge.start_simulation()

            rgb, depth = bridge.capture_rgbd()
            snapshot_path = save_snapshot(rgb, sc.id)

            t0 = time.time()
            fp_ms = NOMINAL_FP_MS  # FP no se re-ejecuta sin GPU dedicada
            traj, diff_ms = run_diffusion_planning(rgb, depth, planner, scheduler)
            sim_t0 = time.time()
            for _ in range(SIM_STEPS_PER_INSTANCE):
                bridge.step()
            sim_ms = (time.time() - sim_t0) * 1000

            grasp_success = bridge.is_grasping()
            cycle_total_ms = fp_ms + diff_ms + sim_ms

            bridge.stop_simulation()

        results.append({
            "scenario_id": sc.id,
            "difficulty": sc.difficulty,
            "cycle_total_ms": cycle_total_ms,
            "fp_ms": fp_ms,
            "diff_ms": diff_ms,
            "sim_ms": sim_ms,
            "grasp_success": grasp_success,
            "snapshot": str(snapshot_path),
        })

    save_report(results)
```

**Outputs en `experiments/results/scenario_battery/`:**
- `report.json` — lista de dicts por escenario con todas las métricas.
- `report.md` — tabla Markdown comparativa para pegar en el TFM.
- `snapshots/<scenario_id>.png` — RGB rendereada por escenario.

**Decisión sobre success rate — y mitigación de la fragilidad de `is_grasping()`:**

`is_grasping()` actual (bridge:352-358) lee `sim.getJointForce` del gripper con threshold de 0.5 N. Dos problemas:

1. Falsos negativos si el gripper aún está cerrándose al momento de la lectura.
2. Devuelve `False` silenciosamente si `_gripper_handle is None` — en el battery runner eso se contabiliza como "grasp fallida" cuando en realidad la escena no tiene gripper definido.

**Mitigaciones para esta iteración:**

- En `run_scenario_battery.py`, tras los N steps del pipeline, llamar explícitamente `bridge.actuate_gripper(open=False)` + `for _ in range(5): bridge.step()` (≈ 90 ms reales) para esperar estabilización antes de leer `is_grasping()`.
- Modificar `is_grasping()` para que loguee un warning distinguible (`"is_grasping: gripper no definido en escena"`) cuando `_gripper_handle is None`, en vez de devolver `False` mudo. Sigue devolviendo `False` para mantener compatibilidad con consumidores existentes, pero el log alerta al usuario.
- En `report.md`, además de `grasp_success`, incluir un campo `gripper_present: bool` por escenario — así una fila con `grasp_success=False, gripper_present=False` se lee como "no aplica" en vez de "fallido". El header del `report.md` documenta la convención.

### 4. Fix + migración de runners

#### Fix del checkpoint (solo `run_pipeline_e2e.py`)

**Importante:** `run_e2e_live.py:122-130` ya tiene el loader robusto (prueba `model_state_dict`, `model`, `state_dict` en orden). **No tocar ese archivo.** El fix aplica solo a `run_pipeline_e2e.py:117-120`:

```python
# Antes
if "model" in ckpt:
    planner.load_state_dict(ckpt["model"])
else:
    planner.load_state_dict(ckpt)

# Después (mismo patrón que run_e2e_live.py para consistencia)
if isinstance(ckpt, dict):
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            planner.load_state_dict(ckpt[key])
            break
    else:
        planner.load_state_dict(ckpt)
else:
    planner.load_state_dict(ckpt)
```

#### Bugs adicionales preexistentes a corregir durante la migración

- `run_pipeline_e2e.py:33` usa `client.getObject("sim")`. La API canónica es `client.require("sim")` (lo que ya usa el bridge y el smoke test). La migración elimina este bug implícitamente al reemplazar `RemoteAPIClient` por el bridge, pero queda registrado para que no se omita si la migración se hace parcial.

#### Migración a bridge

Para cada runner (`run_coppelia_smoke_test.py`, `run_pipeline_e2e.py`, `run_e2e_live.py`):

1. Capturar output actual: ejecutar el runner contra `pickAndPlaceDemo.ttt`, guardar JSON en `/tmp/before_<runner>.json`.
2. Reemplazar `RemoteAPIClient` directo por `with CoppeliaSimBridge() as bridge:`.
3. Reemplazar llamadas `sim.loadScene`, `sim.startSimulation`, `sim.stopSimulation`, `sim.setStepping` por sus equivalentes del bridge.
4. **Stepping:** los 3 runners actuales mezclan dos formas — `client.step()` (smoke test) y `sim.step()` (e2e + e2e_live). El bridge unifica a `bridge.step()` que internamente llama `self._client.step()` (que es la forma documentada en el cliente Python para sincronizar el step en stepped mode). Esto puede cambiar tiempos por instancia en `run_pipeline_e2e.py` y `run_e2e_live.py` (que hoy usan `sim.step()`); aceptar la diferencia en la comparación no-regresión.
5. **Escape hatch para operaciones que el bridge no envuelve** (en particular `createVisionSensor`, `handleVisionSensor`, `getVisionSensorImg` aplicados a sensores creados dinámicamente — lo que hace el smoke test): exponer una propiedad pública `bridge.sim` que devuelve `self._sim`, y documentar su uso para casos donde se necesite la API ZMQ cruda. El smoke test queda así:

   ```python
   with CoppeliaSimBridge() as bridge:
       bridge.load_scene(SCENE_PATH)
       # creación dinámica de sensor cenital: API cruda vía escape hatch
       vs = bridge.sim.createVisionSensor(...)
       bridge.sim.setObjectAlias(vs, "tfm_overview_sensor")
       ...
       bridge.start_simulation()
   ```

   Decisión explícita: **no** se agregan wrappers `create_vision_sensor` / `handle_vision_sensor` al bridge en esta iteración (YAGNI — solo el smoke test los usa, y la idea es que las escenas vengan con sus sensores ya definidos).
6. Ejecutar de nuevo, guardar `/tmp/after_<runner>.json`.
7. **Criterio no-regresión refinado** (diff before/after):
   - **Bit-identical exigido en:** estructura del JSON (mismas claves), dimensiones de imagen, número de instancias procesadas, `coppelia_available: true`, `h3_acceptance.criterion`, IDs/nombres de datasets, `scene_loaded` (basename).
   - **Excluidos del diff** (no comparar): handles de CoppeliaSim (asignación dinámica), timestamps, `image_mean_intensity` / `image_std` exactos (CoppeliaSim no es determinista pixel-a-pixel), `step_ms_*` y `cycle_total_ms` exactos por instancia.
   - **Tolerancia ±5%:** sobre estadísticos agregados (`mean`, `median`, `p95`) de tiempos cuando CoppeliaSim está corriendo.
   - **En modo offline de `run_pipeline_e2e.py`** (sin CoppeliaSim): `sim_ms = NOMINAL_STEP_MS * SIM_STEPS_PER_INSTANCE` es constante; ese campo SÍ debe ser bit-identical.
   - Cualquier diff fuera de lo anterior es regresión y bloquea el merge.

### 5. Suite pytest

**Archivo nuevo:** `tests/test_coppeliasim_bridge.py`

```python
class TestBridgeConnect:
    def test_connect_success(mock_remote_api)
    def test_connect_import_error()
    def test_connect_retries_then_fails()
    def test_connect_retries_then_succeeds()
    def test_context_manager_cleans_up_on_exit()
    def test_context_manager_cleans_up_on_exception()

class TestBridgeScene:
    def test_load_scene_file_exists()
    def test_load_scene_file_missing_raises()
    def test_close_scene()
    def test_apply_scenario_color()
    def test_apply_scenario_light()
    def test_apply_scenario_visibility()
    def test_apply_scenario_unknown_tweak_type_raises()
    def test_apply_scenario_missing_required_field_raises()

class TestBridgeRobot:
    def test_move_joints_within_limits()
    def test_move_joints_no_connection_raises()
    def test_actuate_gripper_open()
    def test_actuate_gripper_close()

class TestBridgeCamera:
    def test_capture_rgbd_returns_correct_shapes()
    def test_capture_rgbd_no_handle_returns_zeros()
    def test_get_camera_pose_returns_se3()

class TestBridgeStepping:
    def test_set_stepping_enabled()
    def test_step_advances_simulation()
    def test_get_simulation_state()

@pytest.mark.integration
class TestBridgeIntegration:
    @pytest.fixture
    def live_bridge(self):
        bridge = CoppeliaSimBridge()
        try:
            bridge.connect(retries=1)
        except (ConnectionError, ImportError):
            pytest.skip("CoppeliaSim no disponible en :23000")
        yield bridge
        bridge.disconnect()

    def test_load_bin_base_scene(self, live_bridge)
    def test_apply_scenario_color_changes_handle(self, live_bridge)
    def test_apply_scenario_light_changes_intensity(self, live_bridge)
    def test_capture_rgbd_returns_non_zero_image(self, live_bridge)
```

**`tests/conftest.py`** con fixtures `mock_sim` y `mock_remote_api`:

```python
@pytest.fixture
def mock_sim():
    sim = MagicMock()
    sim.simulation_stopped = 0
    sim.getSimulationState.return_value = 0
    sim.getObject.side_effect = lambda path: _fake_handle(path)
    sim.getVisionSensorImg.return_value = (b"\x00" * (640*480*3), [640, 480])
    sim.getVisionSensorDepth.return_value = (b"\x00" * (640*480*4), [640, 480])
    return sim

@pytest.fixture
def mock_remote_api(monkeypatch, mock_sim):
    client = MagicMock()
    client.require.return_value = mock_sim
    monkeypatch.setattr(
        "coppeliasim_zmqremoteapi_client.RemoteAPIClient",
        lambda host, port: client
    )
    return client
```

**Marker registrado en `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
markers = [
    "integration: requires CoppeliaSim running on localhost:23000",
]
```

Por defecto pytest los corre; el fixture `live_bridge` los skipea si no hay servidor en `:23000`. Excluir explícitamente: `pytest -m "not integration"`.

**Cobertura objetivo:** ≥ 85% del bridge.

### 6. CI: solo unit tests (integration corre local)

**Decisión revisada (post-review):** los tests de integración corren **solo localmente**, no en CI. Razones concretas:

- CoppeliaSim Edu pide aceptar EULA al primer arranque; auto-aceptarlo en `usrset.txt` no es contractualmente seguro para un repo público.
- La descarga del instalador (~1 GB) no tiene URL pública estable ni API automatizable sin cuenta.
- Para la defensa del TFM, ejecutar los integration tests bajo demanda y guardar evidencia (logs + snapshots) en `experiments/results/` es suficiente.

**Archivo nuevo:** `.github/workflows/test.yml` (o extender el existente si lo hay)

- Trigger: `push` a `main` y PRs.
- Runner: `ubuntu-latest`.
- Pasos:
  1. Checkout.
  2. Setup Python 3.12.
  3. Instalar dependencias del repo (`uv pip install -e .` + extras de test).
  4. Correr `pytest -m "not integration" --cov=src/simulation`.
  5. Subir el reporte de coverage como artifact.

**Para correr integration local** (documentado en `tests/README.md`):
```bash
# Con CoppeliaSim Edu abierto y bin_base.ttt disponible:
pytest -m integration tests/test_coppeliasim_bridge.py -v
```

Si en el futuro se quiere migrar a CI headless, queda como trabajo separado fuera del alcance de esta iteración.

## Orden de ejecución

Con checkpoints de revisión del usuario para validar antes de avanzar:

1. **Bridge — agregados** (sin tocar consumidores).
2. **Suite pytest del bridge** (unit + integration opt-in). **← Checkpoint #1:** mostrar cobertura y resultados.
3. **Generar `bin_base.ttt`** vía `scripts/build_bin_base_scene.py` + `data/scenes/README.md`.
4. **`scenarios.yaml` inicial** con 3 escenarios (base / easy / hard, usando solo `bin_base.ttt` + tweaks en una primera versión).
5. **Runner battery** + outputs. **← Checkpoint #2:** mostrar `report.md` y snapshots.
6. **Fix checkpoint** en `run_pipeline_e2e.py`.
7. **Migración de los 3 runners** al bridge con verificación no-regresión. **← Checkpoint #3:** diffs JSON antes/después.
8. **CI** con unit tests (`-m "not integration"`) + coverage; los integration tests quedan documentados para correr local bajo demanda.

## Manejo de errores

| Punto                                  | Falla                                       | Comportamiento                                         |
|----------------------------------------|---------------------------------------------|--------------------------------------------------------|
| `bridge.connect()`                     | CoppeliaSim no responde tras `retries`     | `ConnectionError` con mensaje accionable               |
| `bridge.load_scene()`                  | Archivo no existe                          | `FileNotFoundError`                                    |
| `bridge.apply_scenario()`              | Tweak con `type` desconocido               | `ValueError` con id de scenario y type                 |
| `bridge.apply_scenario()`              | Handle target no existe en escena          | Loguear warning, continuar (no abortar el batch)       |
| `run_scenario_battery`                 | Un scenario falla                          | Loguear traceback, continuar con el siguiente, registrar `error` en report.json |
| `__exit__`                             | Falla `stopSimulation`                     | Loguear warning, no propagar                           |

## Testing del propio diseño

- Unit coverage del bridge ≥ 85%.
- Cada nuevo método tiene al menos 1 test happy-path + 1 test de error.
- Test no-regresión: diff de JSON antes/después de cada migración.
- Integration test que carga `bin_base.ttt` real y verifica que todos los handles convencionales existen.

## Estimación de superficie

- ~150 líneas nuevas en `coppeliasim_bridge.py`
- ~250 líneas en `tests/test_coppeliasim_bridge.py` + `conftest.py`
- ~100 líneas en `scripts/build_bin_base_scene.py`
- ~150 líneas en `experiments/run_scenario_battery.py`
- ~50 líneas en `.github/workflows/sim-integration.yml`
- ~50 líneas modificadas en los 3 runners existentes

Total: ~700 líneas nuevas, ~50 modificadas.

## Decisiones tomadas durante el brainstorming

- **Eje:** robustez del código de simulación + cobertura de escenarios (combinados).
- **Modo de modelar:** escenas `.ttt` a mano + cargar vía bridge.
- **Escena base:** sí, generar `bin_base.ttt` programáticamente.
- **N escenarios iniciales:** 3 (base, easy, hard).
- **Objetos base:** 5 surtidos (cubos + cilindros).
- **Métricas battery:** tiempos + success binario + snapshot RGB.
- **CI inicial:** integration tests también en CI con CoppeliaSim headless.
- **CI revisado (post code-review):** solo unit tests en CI. Integration tests corren local bajo demanda. Decisión basada en: EULA de CoppeliaSim Edu, ausencia de URL pública estable del instalador, y costo/beneficio bajo para defensa del TFM.
