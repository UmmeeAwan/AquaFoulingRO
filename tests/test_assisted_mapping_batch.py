"""Assisted mapping_required and batch forecast regressions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.application.forecast_errors import (
    ForecastUnavailableError,
    ForecastValidationError,
)
from aqua_fouling.application.forecast_service import forecast_batch_csv, forecast_csv
from aqua_fouling.settings import Settings

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
AMBIGUOUS_UNITS_CSV = REPO / "examples" / "forecast" / "postman" / "ambiguous_units.csv"
AMBIGUOUS_UNITS_MAP = REPO / "examples" / "forecast" / "postman" / "ambiguous_units_mapping.yaml"
GENERIC_MAP = REPO / "configs" / "canonical_generic_schema.yaml"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"
S01_REF = REPO / "artifacts" / "synthetic" / "references" / "S01_reference.yaml"


def _complete_permeate_flow_gpd(template: dict) -> dict:
    """Fill the unresolved permeate_flow unit using the source-header unit (gpd)."""
    for field in template["fields"]:
        if field["canonical_name"] == "stage_3_permeate_flow_source_unit":
            field["external_name"] = "permeate_flow"
            field["unit"] = {"symbol": "gpd"}
    return template


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(artifact_path=ARTIFACT)))


def test_recognised_schema_no_mapping_reference_required(client: TestClient) -> None:
    r = client.post("/v1/forecast", files={"csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv")})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "reference_required"
    assert body["error"]["code"] == "reference_required"
    assert body["mapping"]["resolved"]


def test_ambiguous_unit_http_mapping_required(client: TestClient, tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    path = tmp_path / "a.csv"
    df.to_csv(path, index=False)
    r = client.post("/v1/forecast", files={"csv": ("a.csv", path.read_bytes(), "text/csv")})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "mapping_required"
    assert "mapping_template_yaml" in body
    assert "permeate_flow" in body["error"]["details"]["detected_headers"]
    assert body["mapping_template_yaml"].count("external_name: null") >= 1


def test_completed_assisted_template_forecasts(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    path = tmp_path / "ambig.csv"
    df.to_csv(path, index=False)
    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    assert assisted["status"] == "mapping_required"
    template = yaml.safe_load(assisted["mapping_template_yaml"])
    by_canonical = {f["canonical_name"]: f for f in template["fields"]}
    assert by_canonical["cip_flag_0_1"]["external_name"] == "cip"
    assert by_canonical["cip_flag_0_1"]["unit"]["symbol"] == "flag_0_1"
    assert "days_since_replacement_d" not in by_canonical
    _complete_permeate_flow_gpd(template)
    map_path = tmp_path / "fixed.yaml"
    map_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    body = forecast_csv(
        path,
        mapping_path=map_path,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
    )
    assert body["status"] == "ok"
    mapped = {m["canonical_name"] for m in body["metadata"]["mapping"]["resolved"]}
    assert "cip_flag_0_1" in mapped


def test_incomplete_template_submission_rejected(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    path = tmp_path / "ambig.csv"
    df.to_csv(path, index=False)
    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    template = yaml.safe_load(assisted["mapping_template_yaml"])
    # CIP may be confirmed, but unresolved required nulls must still reject.
    assert any(f["canonical_name"] == "cip_flag_0_1" for f in template["fields"])
    assert any(
        f["canonical_name"] == "stage_3_permeate_flow_source_unit" and f["external_name"] is None
        for f in template["fields"]
    )
    map_path = tmp_path / "incomplete.yaml"
    map_path.write_text(assisted["mapping_template_yaml"], encoding="utf-8")
    with pytest.raises(ForecastValidationError) as ei:
        forecast_csv(
            path,
            mapping_path=map_path,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert ei.value.code in {"malformed_mapping", "validation_failed", "invalid_mapping"}


def test_assisted_template_preserves_confirmed_cip_and_replacement(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    df["days_since_replacement"] = list(range(len(df)))
    path = tmp_path / "ambig_optional.csv"
    df.to_csv(path, index=False)

    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    assert assisted["status"] == "mapping_required"
    confirmed = {c["canonical_name"]: c for c in assisted["error"]["details"]["confirmed_mappings"]}
    assert confirmed["cip_flag_0_1"]["external_name"] == "cip"
    assert confirmed["days_since_replacement_d"]["external_name"] == "days_since_replacement"

    template = yaml.safe_load(assisted["mapping_template_yaml"])
    by_canonical = {f["canonical_name"]: f for f in template["fields"]}
    assert by_canonical["cip_flag_0_1"]["external_name"] == "cip"
    assert by_canonical["cip_flag_0_1"]["unit"]["symbol"] == "flag_0_1"
    assert by_canonical["days_since_replacement_d"]["external_name"] == "days_since_replacement"
    assert by_canonical["days_since_replacement_d"]["unit"]["symbol"] == "day"
    # Unresolved required field stays null — not silently filled.
    assert by_canonical["stage_3_permeate_flow_source_unit"]["external_name"] is None
    assert by_canonical["stage_3_permeate_flow_source_unit"]["unit"]["symbol"] is None

    _complete_permeate_flow_gpd(template)
    map_path = tmp_path / "completed_optional.yaml"
    map_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    body = forecast_csv(
        path,
        mapping_path=map_path,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
    )
    assert body["status"] == "ok"
    mapped = {m["canonical_name"]: m for m in body["metadata"]["mapping"]["resolved"]}
    assert mapped["cip_flag_0_1"]["external_name"] == "cip"
    assert mapped["days_since_replacement_d"]["external_name"] == "days_since_replacement"


def test_assisted_template_omits_absent_optional_columns(tmp_path: Path) -> None:
    df = (
        pd.read_csv(DEMO_CSV)
        .rename(columns={"permeate_flow_gpd": "permeate_flow"})
        .drop(columns=["cip"])
    )
    path = tmp_path / "no_cip.csv"
    df.to_csv(path, index=False)
    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    assert assisted["status"] == "mapping_required"
    confirmed_names = {
        c["canonical_name"] for c in assisted["error"]["details"]["confirmed_mappings"]
    }
    assert "cip_flag_0_1" not in confirmed_names
    template = yaml.safe_load(assisted["mapping_template_yaml"])
    assert all(f["canonical_name"] != "cip_flag_0_1" for f in template["fields"])
    assert all(f["canonical_name"] != "days_since_replacement_d" for f in template["fields"])


def test_completed_assisted_cip_boundary_affects_forecast_history(tmp_path: Path) -> None:
    """CIP preserved through assisted completion still splits usable history."""
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    df["cip"] = 0
    # Late CIP leaves only a short post-clean segment (< min_history_days).
    df.loc[df.index[-5], "cip"] = 1
    path = tmp_path / "cip_split.csv"
    df.to_csv(path, index=False)

    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    assert assisted["status"] == "mapping_required"
    template = _complete_permeate_flow_gpd(yaml.safe_load(assisted["mapping_template_yaml"]))
    assert any(
        f["canonical_name"] == "cip_flag_0_1" and f["external_name"] == "cip"
        for f in template["fields"]
    )
    map_path = tmp_path / "cip_map.yaml"
    map_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")

    with pytest.raises(ForecastUnavailableError) as ei:
        forecast_csv(
            path,
            mapping_path=map_path,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert ei.value.code == "forecast_unavailable"
    details = ei.value.details or {}
    assert int(details.get("finite_observations") or 0) <= 5
    assert int(details.get("calendar_span_days") or 0) < 21


def test_completed_assisted_replacement_boundary_affects_forecast_history(
    tmp_path: Path,
) -> None:
    """Replacement counter preserved through assisted completion still splits history."""
    df = pd.read_csv(DEMO_CSV).rename(columns={"permeate_flow_gpd": "permeate_flow"})
    n = len(df)
    df["days_since_replacement"] = list(range(n - 5)) + list(range(5))
    path = tmp_path / "repl_split.csv"
    df.to_csv(path, index=False)

    assisted = forecast_csv(path, artifact_path=ARTIFACT)
    assert assisted["status"] == "mapping_required"
    template = _complete_permeate_flow_gpd(yaml.safe_load(assisted["mapping_template_yaml"]))
    assert any(
        f["canonical_name"] == "days_since_replacement_d"
        and f["external_name"] == "days_since_replacement"
        for f in template["fields"]
    )
    map_path = tmp_path / "repl_map.yaml"
    map_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")

    with pytest.raises(ForecastUnavailableError) as ei:
        forecast_csv(
            path,
            mapping_path=map_path,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert ei.value.code == "forecast_unavailable"
    details = ei.value.details or {}
    assert int(details.get("calendar_span_days") or 0) < 21
    assert int(details.get("finite_observations") or 0) <= 5


def test_ambiguous_units_fixture_unit_is_gpd_from_source_column() -> None:
    """Fixture construction encodes gpd via permeate_flow_gpd rename — not magnitude."""
    demo = pd.read_csv(DEMO_CSV)
    amb = pd.read_csv(AMBIGUOUS_UNITS_CSV)
    renamed = demo.rename(columns={"permeate_flow_gpd": "permeate_flow"})
    assert list(amb.columns) == list(renamed.columns)
    assert renamed.equals(amb)
    assert "permeate_flow_gpd" in demo.columns
    assert "permeate_flow" in amb.columns
    # Sibling feed column retains the explicit gpd token from the same source schema.
    assert "feed_flow_gpd" in amb.columns


def test_ambiguous_units_completed_mapping_file_forecasts(tmp_path: Path) -> None:
    mapping = yaml.safe_load(AMBIGUOUS_UNITS_MAP.read_text(encoding="utf-8"))
    by_canonical = {f["canonical_name"]: f for f in mapping["fields"]}
    assert by_canonical["stage_3_permeate_flow_source_unit"]["external_name"] == "permeate_flow"
    assert by_canonical["stage_3_permeate_flow_source_unit"]["unit"]["symbol"] == "gpd"
    assert by_canonical["cip_flag_0_1"]["external_name"] == "cip"
    assert by_canonical["cip_flag_0_1"]["unit"]["symbol"] == "flag_0_1"

    body = forecast_csv(
        AMBIGUOUS_UNITS_CSV,
        mapping_path=AMBIGUOUS_UNITS_MAP,
        reference_path=DEMO_REF,
        artifact_path=ARTIFACT,
    )
    assert body["status"] == "ok"
    mapped = {m["canonical_name"] for m in body["metadata"]["mapping"]["resolved"]}
    assert "cip_flag_0_1" in mapped


def test_absent_required_field_mapping_required(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV).drop(columns=["feed_concentration_mg_l"])
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    body = forecast_csv(path, artifact_path=ARTIFACT)
    assert body["status"] == "mapping_required"
    unresolved = body["error"]["details"]["unresolved_required_fields"]
    names = {u["canonical_name"] for u in unresolved}
    assert "feed_concentration_mg_l" in names
    absent = next(u for u in unresolved if u["canonical_name"] == "feed_concentration_mg_l")
    assert absent["reason"] == "absent_column"


def test_batch_two_series_partial_success(tmp_path: Path) -> None:
    base = pd.read_csv(DEMO_CSV)
    df = pd.concat(
        [base.assign(series_id="S_OK"), base.assign(series_id="S_REF")],
        ignore_index=True,
    )
    path = tmp_path / "batch.csv"
    df.to_csv(path, index=False)
    refs = {"S_OK": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8"))}
    body = forecast_batch_csv(
        path,
        references=refs,
        artifact_path=ARTIFACT,
    )
    assert body["overall_status"] == "partial_success"
    assert body["summary"]["counts"]["ok"] == 1
    assert body["summary"]["counts"]["reference_required"] == 1
    by_id = {r["source_series_id"]: r for r in body["results"]}
    assert by_id["S_OK"]["status"] == "ok"
    assert by_id["S_REF"]["status"] == "reference_required"


def test_batch_reference_isolation(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "S01"
    path = tmp_path / "one.csv"
    df.to_csv(path, index=False)
    ref_a = yaml.safe_load(DEMO_REF.read_text(encoding="utf-8"))
    ref_b = yaml.safe_load(S01_REF.read_text(encoding="utf-8"))
    body_a = forecast_batch_csv(
        path,
        references={"S01": ref_a},
        artifact_path=ARTIFACT,
    )
    body_b = forecast_batch_csv(
        path,
        references={"S01": ref_b},
        artifact_path=ARTIFACT,
    )
    assert body_a["results"][0]["status"] == "ok"
    assert body_b["results"][0]["status"] == "ok"
    assert (
        body_a["results"][0]["metadata"]["reference"]["permeate_flow"]
        != body_b["results"][0]["metadata"]["reference"]["permeate_flow"]
    )


def test_batch_leading_zero_series_id(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "007"
    path = tmp_path / "lead.csv"
    df.to_csv(path, index=False)
    body = forecast_batch_csv(
        path,
        references={"007": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8"))},
        artifact_path=ARTIFACT,
    )
    assert body["results"][0]["source_series_id"] == "007"
    assert body["results"][0]["status"] == "ok"


def test_batch_blank_series_id_rejected(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = ""
    path = tmp_path / "blank.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ForecastValidationError) as ei:
        forecast_batch_csv(path, artifact_path=ARTIFACT)
    assert ei.value.code == "blank_series_id"


def test_batch_ambiguous_series_columns_rejected(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "A"
    df["unit_id"] = "B"
    path = tmp_path / "ambig_id.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ForecastValidationError) as ei:
        forecast_batch_csv(path, artifact_path=ARTIFACT)
    assert ei.value.code == "ambiguous_series_column"


def test_batch_series_column_explicit(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "A"
    df["unit_id"] = "B"
    path = tmp_path / "two_cols.csv"
    df.to_csv(path, index=False)
    body = forecast_batch_csv(
        path,
        series_column="series_id",
        references={"A": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8"))},
        artifact_path=ARTIFACT,
    )
    assert body["summary"]["series_column"] == "series_id"
    assert body["results"][0]["status"] == "ok"


def test_batch_max_series_limit(tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df = pd.concat([df.assign(series_id=f"S{i}") for i in range(3)], ignore_index=True)
    path = tmp_path / "many.csv"
    df.to_csv(path, index=False)
    limited = Settings(artifact_path=ARTIFACT, max_batch_series=2)
    with (
        patch("aqua_fouling.settings.get_settings", return_value=limited),
        pytest.raises(ForecastValidationError) as ei,
    ):
        forecast_batch_csv(path, artifact_path=ARTIFACT)
    assert ei.value.code == "batch_series_limit"


def test_batch_http_malformed_references_json(client: TestClient, tmp_path: Path) -> None:
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
    assert r.json()["error"]["code"] in {
        "malformed_json",
        "malformed_yaml",
        "malformed_reference",
    }


def test_batch_http_ok_envelope(client: TestClient, tmp_path: Path) -> None:
    df = pd.read_csv(DEMO_CSV)
    df["series_id"] = "PLANT"
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    refs = {"PLANT": yaml.safe_load(DEMO_REF.read_text(encoding="utf-8"))}
    files = {
        "csv": ("b.csv", path.read_bytes(), "text/csv"),
        "references": ("refs.json", json.dumps(refs).encode(), "application/json"),
    }
    r = client.post("/v1/forecast/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["overall_status"] == "ok"
    assert body["results"][0]["source_series_id"] == "PLANT"


def test_malformed_mapping_yaml_client_error(client: TestClient) -> None:
    bad = REPO / "examples" / "forecast" / "postman" / "malformed_mapping.yaml"
    files = {
        "csv": ("d.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "mapping": ("m.yaml", bad.read_bytes(), "application/yaml"),
    }
    r = client.post("/v1/forecast", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["code"] in {"malformed_yaml", "malformed_mapping"}
