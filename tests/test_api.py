"""API endpoint tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.settings import Settings

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert "phase" not in body


def test_capabilities_forecast_api_available(client: TestClient) -> None:
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["endpoints"]["forecast"] is True
    assert body["model"]["trained_state_space"] is True
    assert "forecast_api_cli" in body["implemented"]
    assert "database" in body["not_implemented"]


def test_forecast_route_exists_when_artifact_present() -> None:
    if not ARTIFACT.is_file():
        return
    app = create_app(Settings(artifact_path=ARTIFACT))
    client = TestClient(app)
    # Missing required multipart → 422 from FastAPI, not 404
    assert client.post("/v1/forecast").status_code == 422
