"""Downloadable forecast CSV export (presentation; same numbers as JSON)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.application.forecast_csv_export import (
    FORECAST_CSV_COLUMNS,
    export_batch_forecast_csv,
    export_single_forecast_csv,
)
from aqua_fouling.application.forecast_service import forecast_batch_csv, forecast_csv
from aqua_fouling.settings import Settings

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
BATCH_CSV = REPO / "examples" / "forecast" / "batch_two_series.csv"
BATCH_REF = REPO / "examples" / "forecast" / "batch_references.yaml"
BATCH_PARTIAL = REPO / "examples" / "forecast" / "batch_references_partial.yaml"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(artifact_path=ARTIFACT)))


def _parse_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_default_json_unchanged(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
        },
    )
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    body = r.json()
    assert body["status"] == "ok"
    assert "result" in body and "normalised_data" in body
    assert body["normalised_data"]["n_observations"] == 35
    assert len(body["result"]["forecast"]) == 14


def test_single_csv_download_headers_and_rows(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
        },
        data={"response_format": "csv", "horizon_days": "14"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="forecast.csv"' in r.headers["content-disposition"]
    rows = _parse_csv(r.text)
    assert list(rows[0].keys()) == list(FORECAST_CSV_COLUMNS)
    assert len(rows) == 14
    assert rows[0]["days_ahead"] == "1"
    assert rows[0]["crossing_status"] == "no_crossing_in_horizon"
    assert rows[0]["days_to_threshold"] == ""  # null → empty, not 0
    assert rows[0]["model_version"] == "llt_relative_health_v1"
    assert rows[0]["normalisation_version"] == "2.0.0"
    assert rows[0]["error_code"] == ""
    assert rows[0]["relative_health"] != ""


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_csv_row_count_matches_horizon(client: TestClient, horizon: int) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
        },
        data={"response_format": "csv", "horizon_days": str(horizon)},
    )
    assert r.status_code == 200
    assert len(_parse_csv(r.text)) == horizon


def test_json_and_csv_values_agree(client: TestClient) -> None:
    files = {
        "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    data = {"horizon_days": "14", "threshold": "0.85"}
    j = client.post("/v1/forecast", files=files, data={**data, "response_format": "json"})
    c = client.post("/v1/forecast", files=files, data={**data, "response_format": "csv"})
    assert j.status_code == 200 and c.status_code == 200
    body = j.json()
    rows = _parse_csv(c.text)
    points = body["result"]["forecast"]
    assert len(rows) == len(points)
    for row, pt in zip(rows, points, strict=True):
        assert row["forecast_date"] == pt["date"]
        assert float(row["relative_health"]) == pytest.approx(
            float(pt["relative_health_point"]), rel=0, abs=1e-12
        )
        assert float(row["latent_health_low"]) == pytest.approx(
            float(pt["latent_health_interval"]["low"]), rel=0, abs=1e-12
        )
        assert float(row["latent_health_high"]) == pytest.approx(
            float(pt["latent_health_interval"]["high"]), rel=0, abs=1e-12
        )
        assert float(row["future_observation_low"]) == pytest.approx(
            float(pt["future_observation_interval"]["low"]), rel=0, abs=1e-12
        )
        assert float(row["future_observation_high"]) == pytest.approx(
            float(pt["future_observation_interval"]["high"]), rel=0, abs=1e-12
        )
        assert float(row["threshold"]) == pytest.approx(0.85, abs=1e-12)
        assert row["model_version"] == body["metadata"]["model"]["model_version"]
        assert row["normalisation_version"] == body["metadata"]["normalisation"]["version"]
        # days_to_threshold is forecast-level context (from origin), repeated.
        assert row["days_to_threshold"] == ""
        assert body["result"]["days_to_threshold"] is None


def test_crossed_at_origin_zero_preserved(tmp_path: Path, client: TestClient) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["permeate_flow_gpd"] = 4000.0
    path = tmp_path / "low.csv"
    df.to_csv(path, index=False)
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("low.csv", path.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
        },
        data={"response_format": "csv", "horizon_days": "14"},
    )
    assert r.status_code == 200
    rows = _parse_csv(r.text)
    assert rows[0]["crossing_status"] == "crossed_at_origin"
    assert rows[0]["days_to_threshold"] == "0"


def test_batch_csv_independent_series(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": ("refs.yaml", BATCH_REF.read_bytes(), "application/yaml"),
        },
        data={"response_format": "csv"},
    )
    assert r.status_code == 200
    assert 'filename="forecast_batch.csv"' in r.headers["content-disposition"]
    rows = _parse_csv(r.text)
    assert len(rows) == 28  # 2 × 14
    ids = {row["source_series_id"] for row in rows}
    assert ids == {"DEMO_A", "DEMO_B"}
    # Sorted by series then date
    assert [row["source_series_id"] for row in rows[:14]] == ["DEMO_A"] * 14
    assert [row["source_series_id"] for row in rows[14:]] == ["DEMO_B"] * 14


def test_partial_batch_csv_includes_failure_row(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("b.csv", BATCH_CSV.read_bytes(), "text/csv"),
            "references": ("partial.yaml", BATCH_PARTIAL.read_bytes(), "application/yaml"),
        },
        data={"response_format": "csv"},
    )
    assert r.status_code == 200
    rows = _parse_csv(r.text)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_id.setdefault(row["source_series_id"], []).append(row)
    assert "DEMO_A" in by_id and "DEMO_B" in by_id
    assert len(by_id["DEMO_A"]) == 14
    assert all(row["status"] == "ok" for row in by_id["DEMO_A"])
    assert all(row["relative_health"] != "" for row in by_id["DEMO_A"])
    # One explicit failure row for missing reference — distinguishable from no-crossing.
    assert len(by_id["DEMO_B"]) == 1
    fail = by_id["DEMO_B"][0]
    assert fail["status"] == "reference_required"
    assert fail["error_code"] == "reference_required"
    assert fail["error_message"] != ""
    assert fail["forecast_date"] == ""
    assert fail["relative_health"] == ""
    assert fail["crossing_status"] == ""


def test_all_failed_batch_csv(tmp_path: Path, client: TestClient) -> None:
    base = pd.read_csv(DEMO_CSV).iloc[:5]  # insufficient history
    df = pd.concat(
        [base.assign(series_id="A"), base.assign(series_id="B")],
        ignore_index=True,
    )
    path = tmp_path / "short.csv"
    df.to_csv(path, index=False)
    refs = {
        "A": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8")),
        "B": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8")),
    }
    r = client.post(
        "/v1/forecast/batch",
        files={
            "csv": ("short.csv", path.read_bytes(), "text/csv"),
            "references": (
                "refs.yaml",
                yaml.safe_dump(refs).encode(),
                "application/yaml",
            ),
        },
        data={"response_format": "csv"},
    )
    assert r.status_code == 200
    rows = _parse_csv(r.text)
    assert len(rows) == 2
    assert {row["source_series_id"] for row in rows} == {"A", "B"}
    for row in rows:
        assert row["status"] in {"failed", "unavailable"}
        assert row["error_code"] != "" or row["status"] == "unavailable"
        assert row["relative_health"] == ""
        assert row["forecast_date"] == ""


def test_malformed_still_json_when_csv_requested(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
            "mapping": (
                "bad.yaml",
                (REPO / "examples/forecast/postman/malformed_mapping.yaml").read_bytes(),
                "application/yaml",
            ),
        },
        data={"response_format": "csv"},
    )
    assert r.status_code == 400
    assert "application/json" in r.headers["content-type"]
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "malformed_mapping"


def test_mapping_required_stays_json_when_csv_requested(client: TestClient) -> None:
    amb = REPO / "examples/forecast/postman/ambiguous_units.csv"
    r = client.post(
        "/v1/forecast",
        files={"csv": ("a.csv", amb.read_bytes(), "text/csv")},
        data={"response_format": "csv"},
    )
    assert r.status_code == 422
    assert "application/json" in r.headers["content-type"]
    assert r.json()["status"] == "mapping_required"


def test_invalid_response_format(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast",
        files={
            "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
            "reference": ("r.yaml", DEMO_REF.read_bytes(), "application/yaml"),
        },
        data={"response_format": "xlsx"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_response_format"


def test_formula_injection_and_csv_escaping() -> None:
    body = forecast_csv(
        DEMO_CSV,
        source_series_id="=CMD|'/C calc'!A0",
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
        horizon_days=7,
    )
    # Force series id in result for export
    body["result"]["source_series_id"] = '=HYPERLINK("http://evil")'
    body["warnings"] = []
    text = export_single_forecast_csv(body)
    rows = _parse_csv(text)
    assert rows[0]["source_series_id"].startswith("'=")
    # Comma / quote / newline in error message
    fail = {
        "status": "failed",
        "source_series_id": "S1",
        "error": {
            "code": "x",
            "message": 'bad, "quote"\nline',
        },
    }
    batch_text = export_batch_forecast_csv({"results": [fail]})
    # Proper quoting: message field is quoted; newline preserved inside quotes
    assert '"bad, ""quote""\nline"' in batch_text or '""quote""' in batch_text
    parsed = _parse_csv(batch_text)
    assert parsed[0]["error_message"] == 'bad, "quote"\nline'
    # Negative numeric cells are not text-guarded
    ok = forecast_csv(
        DEMO_CSV,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
        horizon_days=7,
    )
    ok_rows = _parse_csv(export_single_forecast_csv(ok))
    # slope lives in JSON only; relative_health should be plain numeric string
    assert not ok_rows[0]["relative_health"].startswith("'")


def test_exporter_matches_service_batch_json() -> None:
    refs = yaml.safe_load(BATCH_REF.read_text(encoding="utf-8"))
    body = forecast_batch_csv(BATCH_CSV, references=refs, artifact_path=ARTIFACT)
    rows = _parse_csv(export_batch_forecast_csv(body))
    assert len(rows) == 28
    a = [r for r in rows if r["source_series_id"] == "DEMO_A"]
    assert float(a[0]["relative_health"]) == pytest.approx(
        float(body["results"][0]["result"]["forecast"][0]["relative_health_point"]),
        abs=1e-12,
    )
