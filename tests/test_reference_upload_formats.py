"""YAML-preferred reference uploads; JSON accepted with equivalent forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.application.reference_upload import (
    parse_batch_references_upload,
    parse_single_reference_upload,
)
from aqua_fouling.settings import Settings
from helpers.response_contract import batch_counts, forecast_core

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF_YAML = REPO / "examples" / "forecast" / "reference.yaml"
DEMO_REF_JSON = REPO / "examples" / "forecast" / "reference.json"
BATCH_CSV = REPO / "examples" / "forecast" / "batch_two_series.csv"
BATCH_REF_YAML = REPO / "examples" / "forecast" / "batch_references.yaml"
BATCH_REF_JSON = REPO / "examples" / "forecast" / "batch_references.json"
BATCH_PARTIAL_YAML = REPO / "examples" / "forecast" / "batch_references_partial.yaml"
BATCH_PARTIAL_JSON = REPO / "examples" / "forecast" / "batch_references_partial.json"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(artifact_path=ARTIFACT)))


def _forecast_core(body: dict) -> dict:
    return forecast_core(body)


def test_parse_single_yaml_and_json_equivalent() -> None:
    y = parse_single_reference_upload(DEMO_REF_YAML.read_text(encoding="utf-8"))
    j = parse_single_reference_upload(DEMO_REF_JSON.read_text(encoding="utf-8"))
    assert y["mode"] == j["mode"] == "documented"
    assert float(y["permeate_flow"]) == float(j["permeate_flow"])
    assert str(y["available_from"])[:10] == str(j["available_from"])[:10]


def test_parse_batch_yaml_and_json_equivalent() -> None:
    y = parse_batch_references_upload(BATCH_REF_YAML.read_text(encoding="utf-8"))
    j = parse_batch_references_upload(BATCH_REF_JSON.read_text(encoding="utf-8"))
    assert set(y) == set(j) == {"DEMO_A", "DEMO_B"}
    assert y["DEMO_A"]["ndp_psi"] == j["DEMO_A"]["ndp_psi"]


def test_single_http_yaml_json_equivalent_forecasts(client: TestClient) -> None:
    ry = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("reference.yaml", DEMO_REF_YAML.read_bytes(), "application/yaml"),
        },
    )
    rj = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("reference.json", DEMO_REF_JSON.read_bytes(), "application/json"),
        },
    )
    assert ry.status_code == 200 and rj.status_code == 200
    assert _forecast_core(ry.json()) == _forecast_core(rj.json())


def test_batch_http_yaml_json_equivalent_forecasts(client: TestClient) -> None:
    ry = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": (
                "batch_references.yaml",
                BATCH_REF_YAML.read_bytes(),
                "application/yaml",
            ),
        },
    )
    rj = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": (
                "batch_references.json",
                BATCH_REF_JSON.read_bytes(),
                "application/json",
            ),
        },
    )
    assert ry.status_code == 200 and rj.status_code == 200
    by, bj = ry.json(), rj.json()
    assert by["overall_status"] == bj["overall_status"] == "ok"
    assert batch_counts(by) == batch_counts(bj)
    assert [_forecast_core(r) for r in by["results"]] == [_forecast_core(r) for r in bj["results"]]


def test_batch_partial_yaml_and_json(client: TestClient) -> None:
    for name, path, ctype in (
        ("partial.yaml", BATCH_PARTIAL_YAML, "application/yaml"),
        ("partial.json", BATCH_PARTIAL_JSON, "application/json"),
    ):
        r = client.post(
            "/v1/forecast/batch",
            files={
                "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
                "references": (name, path.read_bytes(), ctype),
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["overall_status"] == "partial_success"
        assert batch_counts(body)["ok"] == 1
        assert batch_counts(body)["reference_required"] == 1
        by_id = {x["source_series_id"]: x for x in body["results"]}
        assert by_id["DEMO_A"]["status"] == "ok"
        assert by_id["DEMO_B"]["status"] == "reference_required"


def test_malformed_single_reference_json(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("bad.json", b"{not-json", "application/json"),
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] in {
        "malformed_json",
        "malformed_yaml",
        "malformed_reference",
    }


def test_malformed_batch_references_yaml(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": ("bad.yaml", b":\n  -", "application/yaml"),
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] in {
        "malformed_yaml",
        "malformed_json",
        "malformed_reference",
    }


def test_wrong_structure_single_gets_batch_map(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": (
                "batch.yaml",
                BATCH_REF_YAML.read_bytes(),
                "application/yaml",
            ),
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "malformed_reference"
    assert "single reference" in err["message"].lower() or "batch" in err["message"].lower()


def test_wrong_structure_batch_gets_single_ref(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": (
                "reference.yaml",
                DEMO_REF_YAML.read_bytes(),
                "application/yaml",
            ),
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "malformed_reference"
    assert "keyed by series" in err["message"].lower() or "series id" in err["message"].lower()


def test_batch_http_malformed_references_json_code(client: TestClient, tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "X"
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    files = {
        "csv": ("b.csv", path.read_bytes(), "text/csv"),
        "references": ("refs.json", b"{not-json", "application/json"),
    }
    r = client.post("/v1/forecast/batch", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "malformed_json"
