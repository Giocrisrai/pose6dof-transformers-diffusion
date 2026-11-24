"""La condición de cada modelo se arma como en su entrenamiento."""
import numpy as np
import pytest

from src.planning.conditioning import (
    COND_DIM,
    LAYOUT_BY_MODEL,
    cond_pos3,
    cond_pose12,
    condition_for_model,
)

R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T = np.array([0.3, -0.1, 0.05])


def test_pose12_equivale_a_las_filas_de_la_pose():
    """Notebook 06: cond[:12] = object_pose[:3, :].flatten()."""
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = R, T
    cond = cond_pose12(R, T)
    assert cond.shape == (COND_DIM,) and cond.dtype == np.float32
    np.testing.assert_allclose(cond[:12], pose[:3, :].flatten(), atol=1e-7)
    assert not cond[12:].any()


def test_pose12_no_es_rotacion_aplanada_mas_traslacion():
    """La disposición [R.flatten(), t] que usaba exp8 es otra distinta."""
    assert not np.allclose(cond_pose12(R, T)[:12], np.concatenate([R.flatten(), T]))


def test_pos3_pone_la_posicion_al_principio():
    cond = cond_pos3(T)
    np.testing.assert_allclose(cond[:3], T, atol=1e-7)
    assert not cond[3:].any()
    np.testing.assert_allclose(cond_pos3(T, extra=[0.1, 0.2, 0.3])[3:6], [0.1, 0.2, 0.3], atol=1e-7)


@pytest.mark.parametrize("modelo", sorted(LAYOUT_BY_MODEL))
def test_cada_modelo_servido_tiene_disposicion(modelo):
    cond = condition_for_model(modelo, R, T)
    esperado = cond_pose12(R, T) if LAYOUT_BY_MODEL[modelo] == "pose12" else cond_pos3(T)
    np.testing.assert_array_equal(cond, esperado)


def test_modelo_desconocido_falla_en_vez_de_adivinar():
    with pytest.raises(ValueError, match="disposición"):
        condition_for_model("no_existe", R, T)


def test_los_modelos_no_vla_de_la_api_estan_cubiertos():
    """Si se añade un modelo a la API, debe declararse su disposición."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    pytest.importorskip("fastapi")
    import api_server

    servidos = {n for n, info in api_server.MODELS_INFO.items() if not info.get("is_vla_lite")}
    assert servidos <= set(LAYOUT_BY_MODEL), servidos - set(LAYOUT_BY_MODEL)
