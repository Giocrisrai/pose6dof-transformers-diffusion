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


class TestBridgeTweaks:
    """Tests de tweaks visuales: color, light, visibility."""

    def test_set_object_color(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_object_color("/object_1", (0.9, 0.1, 0.1))

        mock_sim.getObject.assert_any_call("/object_1")
        # setShapeColor(handle, None, colorcomponent_ambient_diffuse, rgb_list)
        assert mock_sim.setShapeColor.called
        args, _ = mock_sim.setShapeColor.call_args
        assert args[2] == mock_sim.colorcomponent_ambient_diffuse
        assert list(args[3]) == [0.9, 0.1, 0.1]

    def test_set_light_intensity(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_light_intensity("/Light", 1.2)

        mock_sim.getObject.assert_any_call("/Light")
        assert mock_sim.setLightParameters.called
        args, _ = mock_sim.setLightParameters.call_args
        # signature: (handle, state, ambient_or_None, diffuse, specular)
        assert args[1] == 1  # state on
        assert list(args[3]) == [1.2, 1.2, 1.2]

    def test_set_object_visibility_hidden(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_object_visibility("/object_1", visible=False)

        mock_sim.setObjectInt32Param.assert_called_once()
        args, _ = mock_sim.setObjectInt32Param.call_args
        assert args[1] == mock_sim.objintparam_visibility_layer
        assert args[2] == 0  # layer 0 = hidden

    def test_set_object_visibility_visible(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.set_object_visibility("/object_1", visible=True)

        args, _ = mock_sim.setObjectInt32Param.call_args
        assert args[2] == 1  # layer 1 = visible (default)


class TestBridgeApplyScenario:
    """Tests de apply_scenario."""

    def test_apply_scenario_no_tweaks(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.apply_scenario({"id": "base", "scene": "bin_base.ttt"})  # sin tweaks → no-op

        # No debe haber llamado setShapeColor, setLightParameters, setObjectInt32Param
        bridge._sim.setShapeColor.assert_not_called()
        bridge._sim.setLightParameters.assert_not_called()
        bridge._sim.setObjectInt32Param.assert_not_called()

    def test_apply_scenario_color_tweak(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.apply_scenario({
            "id": "easy",
            "scene": "bin_base.ttt",
            "tweaks": [
                {"type": "color", "target": "/object_1", "rgb": [0.9, 0.1, 0.1]},
            ],
        })

        assert mock_sim.setShapeColor.called

    def test_apply_scenario_light_tweak(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.apply_scenario({
            "id": "hard",
            "scene": "bin_base.ttt",
            "tweaks": [
                {"type": "light", "target": "/Light", "intensity": 0.3},
            ],
        })

        assert mock_sim.setLightParameters.called

    def test_apply_scenario_visibility_tweak(self, mock_remote_api, mock_sim):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge.apply_scenario({
            "id": "hard",
            "scene": "bin_base.ttt",
            "tweaks": [
                {"type": "visibility", "target": "/object_5", "visible": False},
            ],
        })

        assert mock_sim.setObjectInt32Param.called

    def test_apply_scenario_unknown_tweak_type_raises(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        with pytest.raises(ValueError, match="unknown"):
            bridge.apply_scenario({
                "id": "bad",
                "tweaks": [{"type": "unknown_type", "target": "/x"}],
            })

    def test_apply_scenario_missing_field_raises(self, mock_remote_api):
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        with pytest.raises(ValueError, match="missing|required"):
            bridge.apply_scenario({
                "id": "bad",
                "tweaks": [{"type": "color", "target": "/x"}],  # falta rgb
            })

    def test_apply_scenario_ignores_id_and_scene(self, mock_remote_api):
        """Las claves id y scene del dict se ignoran; solo se procesan tweaks."""
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        # No debe intentar cargar bin_base.ttt (apply_scenario no lo hace)
        bridge.apply_scenario({"id": "base", "scene": "bin_base.ttt"})
        bridge._sim.loadScene.assert_not_called()


class TestBridgeMisc:
    """Tests de la propiedad sim (escape hatch) y mejoras a is_grasping."""

    def test_sim_property_exposes_underlying_sim(self, mock_remote_api, mock_sim):
        """bridge.sim devuelve el objeto sim crudo para escape hatch."""
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)

        assert bridge.sim is mock_sim

    def test_sim_property_raises_if_not_connected(self):
        bridge = CoppeliaSimBridge()

        with pytest.raises(RuntimeError, match="not connected|sin conexión"):
            _ = bridge.sim

    def test_is_grasping_warns_if_gripper_missing(self, mock_remote_api, caplog):
        """is_grasping loguea warning si _gripper_handle es None."""
        import logging
        bridge = CoppeliaSimBridge()
        bridge.connect(retries=1)
        bridge._gripper_handle = None

        with caplog.at_level(logging.WARNING):
            result = bridge.is_grasping()

        assert result is False
        assert any("gripper" in r.message.lower() for r in caplog.records)


@pytest.mark.integration
class TestBridgeIntegration:
    """Integration tests — requieren CoppeliaSim corriendo en :23000.

    Se skipean automáticamente si no hay servidor. Para correr:
        pytest -m integration tests/test_coppeliasim_bridge.py -v
    """

    @pytest.fixture
    def live_bridge(self):
        bridge = CoppeliaSimBridge()
        try:
            bridge.connect(retries=1, retry_delay_s=0.1)
        except (ConnectionError, ImportError) as e:
            pytest.skip(f"CoppeliaSim no disponible en :23000 ({e})")
        yield bridge
        bridge.disconnect()

    def test_live_connect_returns_version(self, live_bridge):
        """La conexión live devuelve una versión > 0."""
        version = live_bridge._sim.getInt32Param(
            live_bridge._sim.intparam_program_version
        )
        assert version > 0

    def test_live_get_simulation_state(self, live_bridge):
        """get_simulation_state devuelve un int válido."""
        state = live_bridge.get_simulation_state()
        assert isinstance(state, int)
        assert state in (0, 8, 16, 17)  # stopped, paused, running variants

    def test_live_load_bin_base_if_present(self, live_bridge):
        """Si bin_base.ttt existe, carga sin error y los handles se inicializan."""
        scene = Path("data/scenes/bin_base.ttt")
        if not scene.exists():
            pytest.skip("data/scenes/bin_base.ttt no generada todavía (Task 6)")
        live_bridge.load_scene(scene)
