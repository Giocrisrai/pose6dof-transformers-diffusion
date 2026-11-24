"""Fixtures compartidas para tests."""
import socket
from unittest.mock import MagicMock

import pytest

COPPELIASIM = ("localhost", 23000)


def _simulador_disponible(timeout_s: float = 0.5) -> bool:
    """True si hay algo escuchando en el puerto de la ZMQ Remote API."""
    try:
        with socket.create_connection(COPPELIASIM, timeout=timeout_s):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Omite los tests `integration` cuando CoppeliaSim no está corriendo.

    El cliente ZMQ espera indefinidamente a un servidor ausente, así que sin
    esta comprobación `pytest tests/` se queda colgado en vez de omitirlos.
    """
    integracion = [item for item in items if "integration" in item.keywords]
    if not integracion or _simulador_disponible():
        return
    omitir = pytest.mark.skip(
        reason=f"CoppeliaSim no disponible en {COPPELIASIM[0]}:{COPPELIASIM[1]}"
    )
    for item in integracion:
        item.add_marker(omitir)


@pytest.fixture
def mock_sim():
    """Mock del objeto `sim` que retorna RemoteAPIClient.require('sim')."""
    sim = MagicMock()
    # Constantes que el bridge consulta
    sim.simulation_stopped = 0
    sim.simulation_paused = 8
    sim.simulation_advancing_running = 17
    sim.colorcomponent_ambient_diffuse = 0
    sim.objintparam_visibility_layer = 10

    # Estado por defecto: simulación parada
    sim.getSimulationState.return_value = 0
    sim.getSimulationTime.return_value = 0.0
    sim.getInt32Param.return_value = 41000  # versión CoppeliaSim
    sim.getObject.side_effect = lambda path: hash(path) & 0xFFFF  # handle determinista por path
    sim.getObjectAlias.side_effect = lambda h: f"obj_{h}"

    # Captura de imagen: buffer dummy del tamaño esperado
    sim.getVisionSensorImg.return_value = (b"\x00" * (640 * 480 * 3), [640, 480])
    sim.getVisionSensorDepth.return_value = (b"\x00" * (640 * 480 * 4), [640, 480])

    # Pose / joints
    sim.getObjectPosition.return_value = [0.0, 0.0, 1.0]
    sim.getObjectQuaternion.return_value = [0.0, 0.0, 0.0, 1.0]
    sim.getJointPosition.return_value = 0.0
    sim.getJointForce.return_value = 0.0

    return sim


@pytest.fixture
def mock_client(mock_sim):
    """Mock del RemoteAPIClient."""
    client = MagicMock()
    client.require.return_value = mock_sim
    client.getObject.return_value = mock_sim
    return client


@pytest.fixture
def mock_remote_api(monkeypatch, mock_client):
    """Monkeypatchea RemoteAPIClient para que devuelva mock_client.

    Uso: bridge = CoppeliaSimBridge(); bridge.connect()  → usa el mock.
    """
    monkeypatch.setattr(
        "coppeliasim_zmqremoteapi_client.RemoteAPIClient",
        lambda host, port: mock_client,
    )
    return mock_client
