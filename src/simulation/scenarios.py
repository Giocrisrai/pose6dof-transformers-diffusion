"""Loader y validador del manifiesto data/scenes/scenarios.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scenario:
    """Un escenario del manifiesto.

    Attributes:
        id: identificador único.
        scene: nombre del archivo .ttt (relativo a data/scenes/).
        description: texto libre para el report.
        difficulty: clasificación libre (easy / medium / hard).
        tweaks: lista de dicts con campo `type` ∈ {color, light, visibility}
            y los campos específicos de cada tipo. Validados al aplicar,
            no acá (el bridge.apply_scenario es la fuente de verdad).
    """
    id: str
    scene: str
    description: str = ""
    difficulty: str = "unknown"
    tweaks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convierte a dict para pasar a bridge.apply_scenario."""
        return {
            "id": self.id,
            "scene": self.scene,
            "description": self.description,
            "difficulty": self.difficulty,
            "tweaks": list(self.tweaks),
        }


def load_scenarios(yaml_path, scenes_dir=None) -> list[Scenario]:
    """Carga y valida un scenarios.yaml.

    Args:
        yaml_path: ruta al manifiesto.
        scenes_dir: directorio donde viven los .ttt referenciados. Si None,
            se asume el mismo dir que yaml_path. Cada `scene:` debe apuntar
            a un archivo existente en este dir.

    Returns:
        Lista de Scenario en el orden del archivo.

    Raises:
        FileNotFoundError: si yaml_path o algún .ttt referenciado no existe.
        ValueError: si un escenario tiene campos requeridos faltantes.
    """
    yaml_path = Path(yaml_path)
    if scenes_dir is None:
        scenes_dir = yaml_path.parent
    scenes_dir = Path(scenes_dir)

    if not yaml_path.exists():
        raise FileNotFoundError(f"scenarios.yaml no encontrado: {yaml_path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    entries = raw.get("scenarios", []) if isinstance(raw, dict) else []
    out: list[Scenario] = []

    for i, entry in enumerate(entries):
        if "id" not in entry:
            raise ValueError(f"scenario[{i}]: missing required field 'id'")
        if "scene" not in entry:
            raise ValueError(f"scenario {entry.get('id', i)}: missing required field 'scene'")

        scene_file = scenes_dir / entry["scene"]
        if not scene_file.exists():
            raise FileNotFoundError(
                f"scenario {entry['id']}: referenced scene '{entry['scene']}' "
                f"not found at {scene_file}"
            )

        out.append(Scenario(
            id=entry["id"],
            scene=entry["scene"],
            description=entry.get("description", ""),
            difficulty=entry.get("difficulty", "unknown"),
            tweaks=entry.get("tweaks", []) or [],
        ))

    return out
