# Simulation Layer

> Documento canónico del subsistema de simulación. Toda referencia o
> mejora futura debe consultar acá primero.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│  CoppeliaSim Edu V4.10 (proceso aparte, GUI o headless)             │
│  - Escena bin_base.ttt: UR5 + RG2 + table + 5 objetos + cámara     │
│  - ZMQ Remote API escucha en localhost:23000                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ ZMQ
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  src/simulation/coppeliasim_bridge.py                               │
│  - Conexión, retries, context manager, escape hatch (bridge.sim)    │
│  - 25+ métodos públicos (move_joints, capture_rgbd, apply_scenario) │
│  - 38 unit tests con mocks + 3 integration tests live              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────────┐
            ▼                                     ▼
┌──────────────────────────┐         ┌──────────────────────────────┐
│ src/simulation/          │         │ src/simulation/              │
│ scenarios.py             │         │ pick_sequence.py             │
│ - YAML manifest loader   │         │ - IK con simIK nativo        │
│ - 5 tests                │         │ - Snap+attach técnica        │
│                          │         │ - Métricas honestas          │
└──────────┬───────────────┘         └────────┬─────────────────────┘
           │                                  │
           └─────────────┬────────────────────┘
                         ▼
         ┌────────────────────────────────────┐
         │  experiments/run_pick_battery.py   │
         │  - Itera scenarios.yaml            │
         │  - Por cada uno: full pick demo    │
         │  - Outputs: MP4 + report.json + md │
         └────────────────────────────────────┘
```

## Componentes y responsabilidad

### `coppeliasim_bridge.py` (502 + 200 líneas)
- **Connect / disconnect** con retries y context manager.
- **Scene**: `load_scene`, `close_scene`.
- **Stepping**: `set_stepping`, `step`, `get_simulation_state`.
- **Camera**: `capture_rgbd`, `get_camera_pose`.
- **Robot**: `move_joints`, `get_joint_positions`, `get_tip_pose`, `actuate_gripper`, `is_grasping`.
- **Object manipulation**: `set_object_color`, `set_light_intensity`, `set_object_visibility`, `apply_scenario`.
- **Escape hatch**: propiedad `bridge.sim` da acceso a la API ZMQ cruda para casos no envueltos.

### `scenarios.py` (90 líneas)
- Dataclass `Scenario`: id, scene, description, difficulty, tweaks.
- `load_scenarios(yaml_path, scenes_dir)`: parse + validación.

### `pick_sequence.py` (244 líneas)
- `setup_robot_control`: pone los joints en dynamic + motor + position-mode + disable script Lua interno del UR5.
- `_setup_ik`: crea simIK environment + group + element. Reusa `bridge._client`.
- `_move_tcp_via_ik`: interpola target dummy → IK con damped least squares → comanda joints como PID target. Trackea convergencia.
- `run_pick_sequence`: ciclo completo home → approach → descend → grasp+attach → lift → deposit → release → home. Outputs PNGs por step.
- `compile_mp4`: usa ffmpeg para producir MP4 de los frames.

### Datos persistidos

| Path | Contenido | Cómo se regenera |
|---|---|---|
| `data/scenes/bin_base.ttt` | Escena con UR5 + RG2 + table + 5 objetos | `python scripts/build_bin_base_scene.py` |
| `data/scenes/scenarios.yaml` | 3 escenarios (base/easy/hard) con tweaks | Editable a mano |
| `data/models/diffusion_policy_grasp.pth` | Diffusion policy entrenada | Notebook de training (no en este subsistema) |
| `experiments/checkpoints/fp_*_checkpoint.json` | Poses estimadas por FoundationPose (Colab) | Re-run FP en GPU |

## Métricas honestas (ver `docs/PICK_LIMITATIONS.md`)

| Métrica | Significado | Threshold |
|---|---|---|
| `tip_grasp_proximity_m` | Distancia tip ↔ cubo PRE-snap | <0.05 m = plausible |
| `deposit_error_m` | Distancia cubo_final ↔ target deposit | <0.30 m = plausible |
| `ik_converged` | Todas las llamadas IK convergieron | True = OK |
| `obj_displaced` | Cubo se desplazó >2 cm (sanity) | True = ciclo corrió |

**NOTA**: el "grasp" usa snap+attach (técnica estándar de sims comerciales).
Ver `docs/PICK_LIMITATIONS.md` para limitaciones declaradas.

## Cómo correr (cheatsheet)

```bash
# Pre: CoppeliaSim Edu V4.10 corriendo, ZMQ en :23000
open -a CoppeliaSim_Edu && sleep 3

# 1. Regenerar escena (si bin_base.ttt no existe o se cambió build script)
python scripts/build_bin_base_scene.py

# 2. Tests
.venv/bin/pytest tests/test_coppeliasim_bridge.py tests/test_scenarios_loader.py
# → 43 passed

# 3. Demo individual con video MP4
python experiments/run_pick_demo.py
# → experiments/results/pick_demo/demo.mp4

# 4. Battery de 3 escenarios (4 min aprox.)
python experiments/run_pick_battery.py
# → experiments/results/pick_battery/{base,easy,hard}/demo.mp4
# → experiments/results/pick_battery/report.{json,md}
```

## Roadmap (mejoras priorizadas)

Ver `docs/INTEGRATION_PIPELINE.md` para el plan de conexión sim↔training.
