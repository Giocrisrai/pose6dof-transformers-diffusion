# Tests

## Correr en CI / local rápido

```bash
.venv/bin/pytest tests/ -m "not integration" -v
```

Esto excluye los integration tests que requieren CoppeliaSim corriendo.

## Correr todos (incluye integration)

```bash
# Pre-requisito: CoppeliaSim Edu V4.10 abierto con ZMQ Remote API en :23000
open -a CoppeliaSim_Edu
sleep 3

# Suite completa
.venv/bin/pytest tests/ -v
```

Los integration tests del bridge están marcados `@pytest.mark.integration` y se skipean automáticamente si CoppeliaSim no responde, así que el comando es seguro de correr sin la app abierta.

## Solo integration tests

```bash
.venv/bin/pytest tests/test_coppeliasim_bridge.py -m integration -v
```

## Coverage del módulo simulation

```bash
.venv/bin/pytest tests/test_coppeliasim_bridge.py tests/test_scenarios_loader.py \
    --cov=src/simulation --cov-report=term-missing
```

Target: ≥ 80% en `src/simulation/coppeliasim_bridge.py` (los métodos `execute_pick`, `randomize_object_poses`, `move_to_home`, `capture_rgbd` full path quedan out-of-scope de la iteración 2026-05-28; ver `docs/superpowers/specs/2026-05-28-simulacion-robustez-y-escenarios-design.md`).
