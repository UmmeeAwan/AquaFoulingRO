"""Validation / normalisation service and API tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.application.validate import validate_csv
from aqua_fouling.data.adapters import detect_schema_id, load_adapter_for_schema


def test_detect_schemas() -> None:
    assert detect_schema_id(["date", "feed_psi", "stage_3_flow"]) == "ae_128"
    assert detect_schema_id(["date", "feed_press_stage_1", "perm_flow_stage_3"]) == "fg_117"


def test_missing_units_and_reference_in_validation(fixtures_dir: Path) -> None:
    csv = fixtures_dir / "synthetic" / "series_a.csv"
    # synthetic lacks OCWD schema markers
    result = validate_csv(csv, source_series_id="A01")
    assert result["status"] == "invalid"
    assert result["errors"]


def test_validate_real_ae_sample() -> None:
    path = Path("data/extracted/CSV files/orange_county_ro_A01.csv")
    if not path.is_file():
        pytest.skip("extracted dataset not present")
    result = validate_csv(path, source_series_id="A01")
    assert result["status"] in {"valid", "valid_with_warnings"}
    assert result["metadata"]["schema_id"] == "ae_128"
    assert any(
        m["calculation"].startswith("independent") for m in result["missing_for_calculations"]
    )


def test_normalise_without_concentration_marks_unavailable() -> None:
    path = Path("data/extracted/CSV files/orange_county_ro_A01.csv")
    if not path.is_file():
        pytest.skip("extracted dataset not present")
    result = normalise_csv(path, source_series_id="A01")
    indep = result["normalised"]["derived"]["independent_physical_normalisation"]
    assert indep["available"] is False
    assert indep["blockers"]
    diag = result["normalised"]["derived"]["diagnostic_source_calculated_comparison"]
    assert "DIAGNOSTIC" in diag["label"]
    # No NaN/Inf in JSON-able payload
    dumped = json.dumps(result)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_no_double_correction_of_source_tcf() -> None:
    """Independent TCF must come from temperature, not source tcf_water."""
    path = Path("data/extracted/CSV files/orange_county_ro_A01.csv")
    if not path.is_file():
        pytest.skip("extracted dataset not present")
    result = normalise_csv(path, source_series_id="A01")
    assumptions = result["metadata"]["assumptions"]
    assert any("tcf_water" in a.lower() or "FilmTec" in a for a in assumptions)
    diag_row = result["normalised"]["derived"]["diagnostic_source_calculated_comparison"]["rows"][0]
    assert "double correction" in diag_row["note"].lower() or "Double" in diag_row["note"]


def test_json_safe_nulls_nonfinite() -> None:
    payload = {"a": float("nan"), "b": float("inf"), "c": 1.5}
    cleaned = json_safe(payload)
    assert cleaned["a"] is None
    assert cleaned["b"] is None
    assert cleaned["c"] == 1.5


def test_api_validate_and_normalise(client: TestClient) -> None:
    path = Path("data/extracted/CSV files/orange_county_ro_A01.csv")
    if not path.is_file():
        pytest.skip("extracted dataset not present")
    with path.open("rb") as handle:
        response = client.post(
            "/v1/validate",
            files={"csv": ("A01.csv", handle, "text/csv")},
            data={"source_series_id": "A01"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"valid", "valid_with_warnings", "invalid"}

    with path.open("rb") as handle:
        response2 = client.post(
            "/v1/normalise",
            files={"csv": ("A01.csv", handle, "text/csv")},
            data={"source_series_id": "A01"},
        )
    assert response2.status_code == 200
    nbody = response2.json()
    assert "normalised" in nbody
    assert nbody.get("forecast") is None


def test_api_capabilities_forecast(client: TestClient) -> None:
    body = client.get("/v1/capabilities").json()
    assert body["endpoints"]["forecast"] is True
    assert body["model"]["model_version"] == "llt_relative_health_v1"
    assert "phase" not in body
    assert body["endpoints"]["forecast"] is True
    assert "validate_api_cli" in body["implemented"]
    assert "forecast_api_cli" in body["implemented"]
    assert "trained_state_space_forecast_api" not in body["not_implemented"]


def test_load_both_adapters() -> None:
    ae = load_adapter_for_schema("ae_128")
    fg = load_adapter_for_schema("fg_117")
    assert ae.schema_id == "ae_128"
    assert fg.schema_id == "fg_117"
