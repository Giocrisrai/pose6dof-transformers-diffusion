"""Tests para scripts/api_server.py.

Usa fastapi.testclient para tests sin levantar servidor.
"""
from pathlib import Path
import sys

import pytest

# Permitir importar scripts/api_server
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    """TestClient compartido para todos los tests."""
    from fastapi.testclient import TestClient
    from scripts.api_server import app
    return TestClient(app)


class TestRootEndpoints:
    def test_root_returns_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "TFM Pose 6-DoF Pipeline API"
        assert "version" in data
        assert "github" in data

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["device"] in ("cuda", "mps", "cpu")
        assert isinstance(data["models_loaded"], list)


class TestModelsEndpoint:
    def test_models_returns_all_registered(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        models = r.json()
        # Modelos definidos en MODELS_INFO (original/extended/ultra + distillados)
        assert len(models) >= 3
        names = {m["name"] for m in models}
        # Los tres modelos del TFM original deben estar siempre
        assert {"original", "extended", "ultra"}.issubset(names)

    def test_models_have_mse_ref(self, client):
        r = client.get("/models")
        models = {m["name"]: m for m in r.json()}
        # MSE de referencia conocidos
        assert models["original"]["mse_val"] == 0.020
        assert abs(models["extended"]["mse_val"] - 0.01288) < 1e-5
        assert abs(models["ultra"]["mse_val"] - 0.00221) < 1e-5


class TestPlanGraspEndpoint:
    def test_plan_grasp_with_ultra(self, client):
        """Solo se ejecuta si el modelo ultra existe."""
        models = {m["name"]: m for m in client.get("/models").json()}
        if not models["ultra"]["exists"]:
            pytest.skip("Modelo ultra no descargado")

        r = client.post("/plan-grasp", json={
            "object_position": [0.0, 0.0, 0.8],
            "model": "ultra",
            "n_samples": 1,
            "n_diffusion_steps": 10,  # rapido para test
        })
        assert r.status_code == 200
        data = r.json()
        assert data["horizon"] == 16
        assert data["action_dim"] == 7
        assert data["n_samples"] == 1
        assert data["model_used"] == "ultra"
        # Trayectoria 3D
        traj = data["trajectory"]
        assert len(traj) == 1
        assert len(traj[0]) == 16
        assert len(traj[0][0]) == 7
        assert data["latency_ms"] > 0

    def test_plan_grasp_with_distilled_model(self, client):
        """El modelo distillado ultra_fast deberia ser >10x mas rapido que ultra."""
        models = {m["name"]: m for m in client.get("/models").json()}
        if "ultra_fast" not in models or not models["ultra_fast"]["exists"]:
            pytest.skip("Modelo ultra_fast (distillado) no presente")
        if not models["ultra"]["exists"]:
            pytest.skip("Modelo ultra no presente para comparar")

        # Calentar modelos (primer call carga pesos)
        warmup = {"object_position": [0.0, 0.0, 0.8], "n_samples": 1}
        client.post("/plan-grasp", json={**warmup, "model": "ultra"})
        client.post("/plan-grasp", json={**warmup, "model": "ultra_fast"})

        n = 5
        req = {"object_position": [0.05, -0.1, 0.85], "n_samples": n, "n_diffusion_steps": 25}

        r_ultra = client.post("/plan-grasp", json={**req, "model": "ultra"})
        r_fast = client.post("/plan-grasp", json={**req, "model": "ultra_fast"})

        assert r_ultra.status_code == 200 and r_fast.status_code == 200
        d_ultra = r_ultra.json()
        d_fast = r_fast.json()

        # Verificar forma y modelo
        assert d_fast["model_used"] == "ultra_fast"
        assert len(d_fast["trajectory"]) == n
        assert len(d_fast["trajectory"][0]) == 16
        assert len(d_fast["trajectory"][0][0]) == 7
        # Latencia distillado debe ser MUCHO menor (>10x)
        assert d_fast["latency_ms"] < d_ultra["latency_ms"] / 10, \
            f"distilled latency {d_fast['latency_ms']:.1f} no es <10x de ultra {d_ultra['latency_ms']:.1f}"

    def test_plan_grasp_unknown_model_400(self, client):
        r = client.post("/plan-grasp", json={
            "object_position": [0.0, 0.0, 0.8],
            "model": "nonexistent_model",
        })
        # 400 (bad model) o 404 (path not found)
        assert r.status_code in (400, 404)

    def test_plan_grasp_invalid_pose_422(self, client):
        # Falta object_position
        r = client.post("/plan-grasp", json={"model": "ultra"})
        assert r.status_code == 422

    def test_plan_grasp_pose_3d_only(self, client):
        # object_position debe ser 3D
        r = client.post("/plan-grasp", json={
            "object_position": [0.0, 0.0],  # 2D, deberia fallar
        })
        assert r.status_code == 422


class TestE2EEndpoint:
    def test_e2e_small_run(self, client):
        models = {m["name"]: m for m in client.get("/models").json()}
        if not models["ultra"]["exists"]:
            pytest.skip("Modelo ultra no descargado")

        ckpt_ycbv = Path(__file__).resolve().parents[1] / "experiments/checkpoints/fp_ycbv_checkpoint.json"
        if not ckpt_ycbv.exists():
            pytest.skip("Checkpoint YCB-V no descargado")

        r = client.post("/e2e", json={
            "dataset": "ycbv",
            "n_instances": 2,
            "use_ultra_model": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["n_instances"] == 2
        assert len(data["cycle_times_ms"]) == 2
        assert data["cycle_p95_ms"] > 0
        # H3 debe pasar para n=2 instancias normales
        assert data["h3_passed"] is True
        assert data["h3_margin_ms"] > 0
        assert data["dataset"] == "ycbv"

    def test_e2e_unknown_dataset_404(self, client):
        r = client.post("/e2e", json={
            "dataset": "fake_dataset",
            "n_instances": 1,
        })
        assert r.status_code == 404


class TestMetricsEndpoint:
    def test_metrics_returns_dict(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # Si los experimentos están commiteados, deben aparecer
        # (no afirmamos que existan porque pueden no estar en CI)


class TestOpenAPIDocs:
    def test_openapi_schema_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert "paths" in schema
        # Endpoints documentados
        assert "/plan-grasp" in schema["paths"]
        assert "/e2e" in schema["paths"]
        assert "/models" in schema["paths"]

    def test_docs_ui_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
        # HTML con Swagger UI
        assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
