"""CSV-first forecast workflow regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.application.forecast_errors import ForecastValidationError
from aqua_fouling.application.forecast_service import forecast_csv
from aqua_fouling.settings import Settings

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
GENERIC_MAP = REPO / "configs" / "canonical_generic_schema.yaml"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(artifact_path=ARTIFACT)))


def test_csv_only_recognised_schema_returns_reference_required(client: TestClient) -> None:
    files = {"csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv")}
    r = client.post("/v1/forecast", files=files)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "reference_required"
    assert body["error"]["code"] == "reference_required"
    assert body["source_series_id"] == "uploaded_series"
    assert body["mapping"]["resolved"]
    assert body["error"]["details"]["historical_window_reference"]["available"] is False


def test_csv_plus_reference_defaults_forecast(client: TestClient) -> None:
    files = {
        "csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post("/v1/forecast", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    res = body["result"]
    assert res["source_series_id"] == "uploaded_series"
    assert res["horizon_days"] == 14
    assert body["metadata"]["series_id_resolution"]["source"] == "default"
    assert body["metadata"]["mapping"]["resolved"]
    assert abs(float(res["threshold"]["relative_health"]) - 0.85) < 1e-9


def test_infer_series_id_from_column(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "PLANT_A"
    path = tmp_path / "with_id.csv"
    df.to_csv(path, index=False)
    body = forecast_csv(
        path,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
    )
    assert body["result"]["source_series_id"] == "PLANT_A"
    assert body["metadata"]["series_id_resolution"]["source"] == "inferred_column"


def test_mixed_series_rejected(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = ["A"] * 20 + ["B"] * (len(df) - 20)
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ForecastValidationError) as ei:
        forecast_csv(path, reference_path=DEMO_REF, artifact_path=ARTIFACT)
    assert ei.value.code == "mixed_series"
    assert "Split the CSV" in ei.value.actionable_correction


def test_explicit_series_id_mismatch(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "PLANT_A"
    path = tmp_path / "id.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ForecastValidationError) as ei:
        forecast_csv(
            path,
            source_series_id="OTHER",
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert ei.value.code == "series_id_mismatch"


def test_ambiguous_unit_column_returns_mapping_required(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    path = tmp_path / "ambig.csv"
    df.to_csv(path, index=False)
    body = forecast_csv(path, reference_path=DEMO_REF, artifact_path=ARTIFACT)
    assert body["status"] == "mapping_required"
    assert body["error"]["code"] == "mapping_required"
    unresolved = {
        u["canonical_name"]: u for u in body["error"]["details"]["unresolved_required_fields"]
    }
    assert unresolved["stage_3_permeate_flow_source_unit"]["reason"] == "unit_unknown"
    assert "permeate_flow" in unresolved["stage_3_permeate_flow_source_unit"]["suggestions"]


def test_unit_bearing_alias_gpm(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["permeate_flow_gpm"] = df["permeate_flow_gpd"] / 1440.0
    df["feed_flow_gpm"] = df["feed_flow_gpd"] / 1440.0
    df = df.drop(columns=["permeate_flow_gpd", "feed_flow_gpd"])
    path = tmp_path / "gpm.csv"
    df.to_csv(path, index=False)
    body = forecast_csv(
        path,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
    )
    assert body["status"] == "ok"
    units = {r["external_name"]: r["unit"] for r in body["metadata"]["mapping"]["resolved"]}
    assert units["permeate_flow_gpm"] == "gpm"
    assert units["feed_flow_gpm"] == "gpm"


def test_mapping_json_equivalent_to_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(GENERIC_MAP.read_text(encoding="utf-8"))
    mj = {
        "name": raw["name"],
        "schema_id": raw["schema_id"],
        "timestamp_column": raw["timestamp"]["column"],
        "fields": [
            {
                "external_name": f["external_name"],
                "canonical_name": f["canonical_name"],
                "unit": f["unit"]["symbol"],
                "measurement_scope": f["measurement_scope"],
                "kind": f["kind"],
            }
            for f in raw["fields"]
            if f["external_name"] != "days_since_replacement"
            or "days_since_replacement" in pd.read_csv(DEMO_CSV).columns
        ],
    }
    # Keep fields present in demo CSV only
    demo_cols = set(pd.read_csv(DEMO_CSV).columns)
    mj["fields"] = [f for f in mj["fields"] if f["external_name"] in demo_cols]

    via_json = forecast_csv(
        DEMO_CSV,
        mapping_json=mj,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
        horizon_days=14,
    )
    via_yaml = forecast_csv(
        DEMO_CSV,
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
        horizon_days=14,
    )
    assert via_json["result"]["crossing_status"] == via_yaml["result"]["crossing_status"]
    assert (
        via_json["normalised_data"]["n_observations"]
        == via_yaml["normalised_data"]["n_observations"]
    )
    assert len(via_json["result"]["forecast"]) == len(via_yaml["result"]["forecast"])


def test_http_mapping_json_form_field(client: TestClient) -> None:
    raw = yaml.safe_load(GENERIC_MAP.read_text(encoding="utf-8"))
    demo_cols = set(pd.read_csv(DEMO_CSV).columns)
    mj = {
        "fields": [
            {
                "external_name": f["external_name"],
                "canonical_name": f["canonical_name"],
                "unit": f["unit"]["symbol"],
                "measurement_scope": f["measurement_scope"],
            }
            for f in raw["fields"]
            if f["external_name"] in demo_cols
        ],
        "timestamp_column": "observation_date",
    }
    files = {
        "csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post(
        "/v1/forecast",
        files=files,
        data={"mapping_json": json.dumps(mj), "horizon_days": "14"},
    )
    assert r.status_code == 200
    assert r.json()["metadata"]["mapping"]["source"] == "mapping_json"


def test_historical_window_mode_explicitly_unavailable() -> None:
    body = forecast_csv(DEMO_CSV, artifact_path=ARTIFACT)
    assert body["status"] == "reference_required"
    assert body["error"]["details"]["historical_window_reference"]["available"] is False
