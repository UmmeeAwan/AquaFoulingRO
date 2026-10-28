"""Regression: presentation refactor must not change forecast science numbers."""

from __future__ import annotations

from pathlib import Path

from aqua_fouling.application.forecast_service import forecast_batch_csv, forecast_csv

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
BATCH_CSV = REPO / "examples" / "forecast" / "batch_two_series.csv"
BATCH_REF = REPO / "examples" / "forecast" / "batch_references.yaml"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"

# Locked from the trained demo + reference before presentation refactor.
_EXPECTED_ORIGIN = "2020-02-04"
_EXPECTED_FIRST_POINT = 0.9271649865861883
_EXPECTED_FIRST_LATENT = (0.9154191915142537, 0.9389107816581229)
_EXPECTED_FIRST_OBS = (0.890032506268569, 0.9642974669038076)
_EXPECTED_FIRST_NORM = 1.0557382605913386


def test_single_forecast_numeric_regression() -> None:
    body = forecast_csv(
        DEMO_CSV,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
        horizon_days=14,
        threshold=0.85,
    )
    assert body["status"] == "ok"
    assert set(body) >= {"status", "result", "normalised_data", "metadata", "warnings"}
    res = body["result"]
    assert res["crossing_status"] == "no_crossing_in_horizon"
    assert res["days_to_threshold"] is None  # null, never coerced to 0
    assert res["horizon_days"] == 14
    assert res["forecast_origin"]["date"] == _EXPECTED_ORIGIN
    assert abs(res["threshold"]["relative_health"] - 0.85) < 1e-12
    assert len(res["forecast"]) == 14
    p0 = res["forecast"][0]
    assert abs(p0["relative_health_point"] - _EXPECTED_FIRST_POINT) < 1e-12
    assert abs(p0["latent_health_interval"]["low"] - _EXPECTED_FIRST_LATENT[0]) < 1e-12
    assert abs(p0["latent_health_interval"]["high"] - _EXPECTED_FIRST_LATENT[1]) < 1e-12
    assert abs(p0["future_observation_interval"]["low"] - _EXPECTED_FIRST_OBS[0]) < 1e-12
    assert abs(p0["future_observation_interval"]["high"] - _EXPECTED_FIRST_OBS[1]) < 1e-12
    # Interval prose lives once in metadata, not on every point.
    assert "meaning" not in p0["latent_health_interval"]
    assert "latent_health_interval" in body["metadata"]["interval_definitions"]
    obs = body["normalised_data"]["observations"]
    assert body["normalised_data"]["n_observations"] == 35
    assert abs(float(obs[0]["relative_health"]) - _EXPECTED_FIRST_NORM) < 1e-12
    assert "artifact_path" not in body["metadata"]["model"]
    assert "forbidden_practices_not_used" not in (body["metadata"].get("reference") or {})


def test_batch_structure_and_series_identity() -> None:
    import yaml

    refs = yaml.safe_load(BATCH_REF.read_text(encoding="utf-8"))
    body = forecast_batch_csv(BATCH_CSV, references=refs, artifact_path=ARTIFACT)
    assert body["overall_status"] == "ok"
    assert "summary" in body and "counts" in body["summary"]
    assert "status" not in body or body.get("status") == body["overall_status"]
    ids = [r["source_series_id"] for r in body["results"]]
    assert set(ids) == {"DEMO_A", "DEMO_B"}
    for r in body["results"]:
        assert r["status"] == "ok"
        assert r["result"]["crossing_status"] == "no_crossing_in_horizon"
        assert r["result"]["days_to_threshold"] is None
        assert (
            abs(r["result"]["forecast"][0]["relative_health_point"] - _EXPECTED_FIRST_POINT) < 1e-9
        )
