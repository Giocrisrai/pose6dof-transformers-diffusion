"""Tests para src/simulation/coppeliasim_bridge.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge


class TestBridgeScene:
    """Tests de gestión de escena."""

    def test_load_scene_file_exists(self, mock_remote_api, tmp_path):
        """load_scene llama sim.loadScene con el path string si el archivo existe."""
        scene = tmp_path / "fake.ttt"
        scene.write_bytes(b"\x00")  # archivo dummy

        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.load_scene(scene)

        bridge._sim.loadScene.assert_called_once_with(str(scene))

    def test_load_scene_file_missing_raises(self, mock_remote_api):
        """load_scene levanta FileNotFoundError si el path no existe."""
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        with pytest.raises(FileNotFoundError):
            bridge.load_scene("/no/existe.ttt")

    def test_load_scene_closes_current_by_default(self, mock_remote_api, tmp_path):
        """load_scene llama closeScene antes de loadScene si close_current=True."""
        scene = tmp_path / "fake.ttt"
        scene.write_bytes(b"\x00")

        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.load_scene(scene)

        bridge._sim.closeScene.assert_called_once()

    def test_load_scene_skip_close_when_false(self, mock_remote_api, tmp_path):
        """load_scene NO llama closeScene si close_current=False."""
        scene = tmp_path / "fake.ttt"
        scene.write_bytes(b"\x00")

        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.load_scene(scene, close_current=False)

        bridge._sim.closeScene.assert_not_called()

    def test_close_scene(self, mock_remote_api):
        """close_scene llama sim.closeScene."""
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.close_scene()

        bridge._sim.closeScene.assert_called_once()


class TestBridgeStepping:
    """Tests de stepping y estado de simulación."""

    def test_set_stepping_enabled(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_stepping(True)

        bridge._sim.setStepping.assert_called_once_with(True)

    def test_set_stepping_disabled(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_stepping(False)

        bridge._sim.setStepping.assert_called_once_with(False)

    def test_get_simulation_state(self, mock_remote_api, mock_sim):
        mock_sim.getSimulationState.return_value = 17
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        assert bridge.get_simulation_state() == 17

    def test_get_simulation_time(self, mock_remote_api, mock_sim):
        mock_sim.getSimulationTime.return_value = 5.42
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        assert bridge.get_simulation_time() == 5.42
