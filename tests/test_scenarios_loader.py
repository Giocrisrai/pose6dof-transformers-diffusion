"""Tests para src/simulation/scenarios.py."""
import pytest

from src.simulation.scenarios import Scenario, load_scenarios


def test_load_scenarios_returns_list(tmp_path):
    yaml_text = """
scenarios:
  - id: base
    scene: bin_base.ttt
    difficulty: easy
"""
    f = tmp_path / "s.yaml"
    f.write_text(yaml_text)
    scenes_dir = tmp_path  # mismo dir para validar 'scene' existe
    (scenes_dir / "bin_base.ttt").write_bytes(b"\x00")

    scenarios = load_scenarios(f, scenes_dir=scenes_dir)

    assert len(scenarios) == 1
    assert isinstance(scenarios[0], Scenario)
    assert scenarios[0].id == "base"
    assert scenarios[0].scene == "bin_base.ttt"
    assert scenarios[0].tweaks == []


def test_load_scenarios_with_tweaks(tmp_path):
    yaml_text = """
scenarios:
  - id: easy
    scene: bin_base.ttt
    difficulty: easy
    tweaks:
      - { type: color, target: "/object_1", rgb: [0.9, 0.1, 0.1] }
      - { type: light, target: "/Light", intensity: 1.2 }
"""
    f = tmp_path / "s.yaml"
    f.write_text(yaml_text)
    (tmp_path / "bin_base.ttt").write_bytes(b"\x00")

    scenarios = load_scenarios(f, scenes_dir=tmp_path)
    sc = scenarios[0]

    assert len(sc.tweaks) == 2
    assert sc.tweaks[0]["type"] == "color"
    assert sc.tweaks[1]["intensity"] == 1.2


def test_load_scenarios_missing_scene_file_raises(tmp_path):
    yaml_text = """
scenarios:
  - id: bad
    scene: no_existe.ttt
    difficulty: easy
"""
    f = tmp_path / "s.yaml"
    f.write_text(yaml_text)

    with pytest.raises(FileNotFoundError, match="no_existe.ttt"):
        load_scenarios(f, scenes_dir=tmp_path)


def test_load_scenarios_missing_id_raises(tmp_path):
    yaml_text = """
scenarios:
  - scene: bin_base.ttt
    difficulty: easy
"""
    f = tmp_path / "s.yaml"
    f.write_text(yaml_text)
    (tmp_path / "bin_base.ttt").write_bytes(b"\x00")

    with pytest.raises(ValueError, match="missing.*id"):
        load_scenarios(f, scenes_dir=tmp_path)


def test_scenario_to_dict():
    sc = Scenario(
        id="easy",
        scene="bin_base.ttt",
        description="test",
        difficulty="easy",
        tweaks=[{"type": "color", "target": "/x", "rgb": [1, 0, 0]}],
    )
    d = sc.to_dict()
    assert d["id"] == "easy"
    assert d["scene"] == "bin_base.ttt"
    assert len(d["tweaks"]) == 1
