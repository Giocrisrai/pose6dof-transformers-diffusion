# Escenas para el bridge de CoppeliaSim

Esta carpeta contiene las escenas `.ttt` que el bridge carga y los manifiestos YAML que las describen.

## Archivos

| Archivo            | Origen           | Descripción                                       |
|--------------------|------------------|---------------------------------------------------|
| `bin_base.ttt`     | Generada por `scripts/build_bin_base_scene.py` | Escena base: UR5e + bin + RGB-D + 5 objetos surtidos |
| `bin_easy.ttt`     | Modelada manualmente | (opcional) Variante easy — ver "Crear variantes" abajo |
| `bin_hard.ttt`     | Modelada manualmente | (opcional) Variante hard                          |
| `scenarios.yaml`   | Editable a mano  | Manifiesto que lista los escenarios para `run_scenario_battery.py` |

## Convención de nombres (handles)

Para que el bridge encuentre los objetos correctamente, cada escena debe tener:

| Handle path                                | Tipo               | Obligatorio |
|--------------------------------------------|--------------------|-------------|
| `/UR5e`                                    | Modelo del robot   | Sí (si hay control motor) |
| `/shoulder_pan_joint` ... `/wrist_3_joint` | Joints del UR5     | Sí          |
| `/rgb_camera`                              | Vision sensor RGB  | Sí          |
| `/depth_camera`                            | Vision sensor depth| Sí          |
| `/bin`                                     | Shape (compuesto)  | Recomendado |
| `/object_1` … `/object_N`                  | Shapes             | Sí (N ≥ 1)  |
| `/Light`                                   | Light              | Sí          |
| `/gripper`                                 | Shape o modelo     | Opcional    |
| `/tip`                                     | Dummy (TCP)        | Opcional    |

Si modelás una escena sin alguno de los obligatorios, `_init_handles()` del bridge loguea warning. Si llamás `bridge.connect(strict=True)`, levanta `RuntimeError`.

## Regenerar `bin_base.ttt`

```bash
# Abrir CoppeliaSim Edu (debe estar corriendo el ZMQ Remote API en :23000)
open -a CoppeliaSim_Edu
sleep 3

# Correr el script
.venv/bin/python scripts/build_bin_base_scene.py
```

El script sobreescribe `data/scenes/bin_base.ttt`.

## Crear variantes (`bin_easy.ttt`, `bin_hard.ttt`)

1. Abrir `data/scenes/bin_base.ttt` en CoppeliaSim (File > Open scene).
2. Editar lo que quieras: agregar/quitar objetos, cambiar formas, mover el bin, agregar oclusión, cambiar la luz.
3. **Mantener los nombres de handles obligatorios** (tabla arriba) — si renombrás `/UR5e` o `/rgb_camera`, el bridge no los va a encontrar.
4. File > Save scene as... → `data/scenes/bin_easy.ttt` (o el nombre que quieras).
5. Agregar una entrada en `scenarios.yaml` apuntando a tu archivo nuevo.

## Manifiesto `scenarios.yaml`

Ver `scenarios.yaml` para el formato. Tipos de tweak soportados:

- `color` — `{type: color, target: "/object_1", rgb: [r,g,b]}` (componentes en `[0,1]`).
- `light` — `{type: light, target: "/Light", intensity: 0.5}` (≥ 0).
- `visibility` — `{type: visibility, target: "/object_5", visible: false}`.

Type desconocido o campo faltante → `apply_scenario` levanta `ValueError`.
