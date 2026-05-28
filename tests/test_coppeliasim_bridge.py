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
        """load_scene llama closeScene ANTES de loadScene si close_current=True."""
        scene = tmp_path / "fake.ttt"
        scene.write_bytes(b"\x00")

        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.load_scene(scene)

        bridge._sim.closeScene.assert_called_once()
        # Validar orden: closeScene debe preceder a loadScene
        call_names = [c[0] for c in bridge._sim.mock_calls if c[0] in ("closeScene", "loadScene")]
        assert call_names == ["closeScene", "loadScene"], (
            f"orden inválido: {call_names} (esperado closeScene antes que loadScene)"
        )

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


class TestBridgeConnect:
    """Tests de connect/disconnect con retries."""

    def test_connect_success(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        assert bridge._connected is True
        assert bridge._sim is mock_sim

    def test_connect_pings_after_connection(self, mock_remote_api, mock_sim):
        """Tras conectar, hace ping con getSimulationTime."""
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        # getSimulationTime debe haberse llamado al menos una vez (ping)
        assert mock_sim.getSimulationTime.called

    def test_connect_retries_then_succeeds(self, monkeypatch, mock_client):
        """Si el primer intento falla, reintenta y eventualmente conecta."""
        attempts = [0]

        def flaky_client_factory(host, port):
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("addon ZMQ aún no listo")
            return mock_client

        monkeypatch.setattr(
            "coppeliasim_zmqremoteapi_client.RemoteAPIClient",
            flaky_client_factory,
        )

        bridge = CoppeliaSimBridge()
        bridge.connect(retries=5, retry_delay_s=0.01)

        assert attempts[0] == 3
        assert bridge._connected is True

    def test_connect_retries_then_fails(self, monkeypatch):
        """Si todos los intentos fallan, levanta ConnectionError."""
        def always_fails(host, port):
            raise ConnectionError("addon ZMQ no responde")

        monkeypatch.setattr(
            "coppeliasim_zmqremoteapi_client.RemoteAPIClient",
            always_fails,
        )

        bridge = CoppeliaSimBridge()
        with pytest.raises(ConnectionError):
            bridge.connect(retries=2, retry_delay_s=0.01)

    def test_connect_import_error(self, monkeypatch):
        """Si el paquete no está instalado, levanta ImportError."""
        # `sys.modules[name] = None` hace que el `from name import X` lance
        # ImportError. monkeypatch.setitem restaura el original automáticamente.
        import sys
        monkeypatch.setitem(sys.modules, "coppeliasim_zmqremoteapi_client", None)

        bridge = CoppeliaSimBridge()
        with pytest.raises(ImportError):
            bridge.connect(retries=1)

    def test_disconnect_clears_state(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.disconnect()

        assert bridge._connected is False
        assert bridge._sim is None
        assert bridge._client is None


class TestBridgeContextManager:
    """Tests del context manager (__enter__ / __exit__)."""

    def test_with_statement_connects_and_disconnects(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        with bridge as b:
            assert b is bridge
            assert b._connected is True

        assert bridge._connected is False

    def test_exit_stops_simulation_if_running(self, mock_remote_api, mock_sim):
        mock_sim.getSimulationState.return_value = 17  # running

        with CoppeliaSimBridge():
            pass

        mock_sim.stopSimulation.assert_called_once()
        mock_sim.setStepping.assert_any_call(False)

    def test_exit_skips_stop_if_already_stopped(self, mock_remote_api, mock_sim):
        mock_sim.getSimulationState.return_value = 0  # stopped

        with CoppeliaSimBridge():
            pass

        mock_sim.stopSimulation.assert_not_called()

    def test_exit_propagates_exceptions(self, mock_remote_api):
        with pytest.raises(ValueError, match="user error"):
            with CoppeliaSimBridge():
                raise ValueError("user error")

    def test_exit_cleans_up_even_on_exception(self, mock_remote_api, mock_sim):
        mock_sim.getSimulationState.return_value = 17

        with pytest.raises(ValueError):
            with CoppeliaSimBridge():
                raise ValueError("boom")

        mock_sim.stopSimulation.assert_called_once()

    def test_exit_does_not_raise_if_cleanup_fails(self, mock_remote_api, mock_sim):
        """Si stopSimulation falla en __exit__, se loguea warning pero no propaga."""
        mock_sim.getSimulationState.return_value = 17
        mock_sim.stopSimulation.side_effect = RuntimeError("sim crashed")

        # No debe levantar
        with CoppeliaSimBridge():
            pass
