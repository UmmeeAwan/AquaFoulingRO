"""Phase 5: CSV forecast API/CLI — equivalence, errors, and isolation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aqua_fouling.api.app import create_app
from aqua_fouling.application.forecast_errors import ForecastServiceError
from aqua_fouling.application.forecast_service import forecast_csv
from aqua_fouling.cli import app as cli_app
from aqua_fouling.settings import Settings

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
GENERIC_MAP = REPO / "configs" / "canonical_generic_schema.yaml"


pytestmark = pytest.mark.skipif(not ARTIFACT.is_file(), reason="trained artifact not present")


@pytest.fixture
def forecast_settings(tmp_path: Path) -> Settings:
    return Settings(
        artifact_path=ARTIFACT,
        default_mapping=GENERIC_MAP,
        project_root=REPO,
    )


@pytest.fixture
def client(forecast_settings: Settings) -> TestClient:
    return TestClient(create_app(forecast_settings))


def _cli_forecast(csv: Path, *, series: str = "DEMO") -> dict:
    runner = CliRunner()
    out = csv.parent / "cli_out.json"
    result = runner.invoke(
        cli_app,
        [
            "forecast",
            "--csv",
            str(csv),
            "--series-id",
            series,
            "--reference",
            str(DEMO_REF),
            "--mapping",
            str(GENERIC_MAP),
            "--artifact",
            str(ARTIFACT),
            "--horizon-days",
            "14",
            "--threshold",
            "0.85",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_api_and_cli_equivalent_predictions(client: TestClient, tmp_path: Path) -> None:
    csv = tmp_path / "demo.csv"
    shutil.copy(DEMO_CSV, csv)
    cli_body = _cli_forecast(csv)

    with DEMO_CSV.open("rb") as cf, DEMO_REF.open("rb") as rf, GENERIC_MAP.open("rb") as mf:
        resp = client.post(
            "/v1/forecast",
            files={
                "csv": ("demo.csv", cf, "text/csv"),
                "reference": ("reference.yaml", rf, "application/yaml"),
                "mapping": ("mapping.yaml", mf, "application/yaml"),
            },
            data={
                "source_series_id": "DEMO",
                "horizon_days": "14",
                "threshold": "0.85",
            },
        )
    assert resp.status_code == 200, resp.text
    api_body = resp.json()
    assert api_body["result"]["crossing_status"] == cli_body["result"]["crossing_status"]
    assert api_body["result"]["days_to_threshold"] == cli_body["result"]["days_to_threshold"]
    assert (
        api_body["result"]["forecast_origin"]["date"]
        == cli_body["result"]["forecast_origin"]["date"]
    )
    api_pts = [p["relative_health_point"] for p in api_body["result"]["forecast"]]
    cli_pts = [p["relative_health_point"] for p in cli_body["result"]["forecast"]]
    assert api_pts == pytest.approx(cli_pts, rel=1e-10, abs=1e-12)
    assert api_body["metadata"]["model"]["model_version"] == "llt_relative_health_v1"
    assert api_body["metadata"]["normalisation"]["method_id"]
    assert api_body["normalised_data"]["n_observations"] == len(
        api_body["normalised_data"]["observations"]
    )


def test_bar_units_equivalent_to_psi(tmp_path: Path) -> None:
    from aqua_fouling.physics.units import PSI_PER_BAR

    psi_csv = tmp_path / "psi.csv"
    bar_csv = tmp_path / "bar.csv"
    shutil.copy(DEMO_CSV, psi_csv)
    # Rewrite pressures as bar with bar mapping
    import pandas as pd

    df = pd.read_csv(DEMO_CSV)
    for col in ("feed_pressure_psi", "permeate_pressure_psi", "device_dp_psi"):
        df[col] = df[col] / PSI_PER_BAR
    df = df.rename(
        columns={
            "feed_pressure_psi": "feed_pressure",
            "permeate_pressure_psi": "permeate_pressure",
            "device_dp_psi": "device_dp",
            "permeate_flow_gpd": "permeate_flow",
            "feed_flow_gpd": "feed_flow",
        }
    )
    df.to_csv(bar_csv, index=False)
    bar_map = tmp_path / "bar_map.yaml"
    bar_map.write_text(
        GENERIC_MAP.read_text(encoding="utf-8")
        .replace("feed_pressure_psi", "feed_pressure")
        .replace("permeate_pressure_psi", "permeate_pressure")
        .replace("device_dp_psi", "device_dp")
        .replace("permeate_flow_gpd", "permeate_flow")
        .replace("feed_flow_gpd", "feed_flow")
        .replace("symbol: psi", "symbol: bar"),
        encoding="utf-8",
    )
    # flows still gpd in rewritten names — keep gpd symbols for flow fields
    text = bar_map.read_text(encoding="utf-8")
    # After blind replace, flow units may have become bar — fix flow back to gpd
    lines = []
    for line in text.splitlines():
        if "flow" in line and "canonical_name" in line:
            pass
        lines.append(line)
    # Simpler: write explicit mapping
    bar_map.write_text(
        """
name: bar_demo
schema_id: canonical_generic_v1
source_series_pattern: ".+"
timestamp: {column: observation_date, format: ISO8601, timezone: UTC}
fields:
  - {external_name: observation_date, canonical_name: observation_date_calendar,
     unit: {symbol: calendar_date}, kind: measured, measurement_scope: whole_unit}
  - {external_name: temperature_c, canonical_name: temperature_c,
     unit: {symbol: deg_C}, kind: measured, measurement_scope: whole_unit}
  - {external_name: feed_pressure, canonical_name: stage_3_feed_pressure_psi,
     unit: {symbol: bar}, kind: measured, measurement_scope: stage_3}
  - {external_name: permeate_pressure, canonical_name: permeate_pressure_psi,
     unit: {symbol: bar}, kind: measured, measurement_scope: whole_unit}
  - {external_name: device_dp, canonical_name: stage_3_device_pressure_drop_psi,
     unit: {symbol: bar}, kind: measured, measurement_scope: stage_3}
  - {external_name: permeate_flow, canonical_name: stage_3_permeate_flow_source_unit,
     unit: {symbol: gpd}, kind: measured, measurement_scope: stage_3}
  - {external_name: feed_flow, canonical_name: stage_3_feed_flow_source_unit,
     unit: {symbol: gpd}, kind: measured, measurement_scope: stage_3}
  - {external_name: feed_concentration_mg_l, canonical_name: feed_concentration_mg_l,
     unit: {symbol: mg_L_NaCl_eq}, kind: measured, measurement_scope: whole_unit}
  - {external_name: cip, canonical_name: cip_flag_0_1,
     unit: {symbol: flag_0_1}, kind: measured, measurement_scope: whole_unit}
""",
        encoding="utf-8",
    )

    r_psi = forecast_csv(
        psi_csv,
        source_series_id="PSI",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    r_bar = forecast_csv(
        bar_csv,
        source_series_id="BAR",
        mapping_path=bar_map,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    assert r_psi["result"]["crossing_status"] == r_bar["result"]["crossing_status"]
    assert r_psi["result"]["days_to_threshold"] == r_bar["result"]["days_to_threshold"]
    pts_a = [p["relative_health_point"] for p in r_psi["result"]["forecast"]]
    pts_b = [p["relative_health_point"] for p in r_bar["result"]["forecast"]]
    assert pts_a == pytest.approx(pts_b, rel=1e-5, abs=1e-5)


def test_repeated_requests_deterministic_artifact_unchanged(
    client: TestClient,
) -> None:
    before = ARTIFACT.read_bytes()
    bodies = []
    for _ in range(2):
        with DEMO_CSV.open("rb") as cf, DEMO_REF.open("rb") as rf, GENERIC_MAP.open("rb") as mf:
            resp = client.post(
                "/v1/forecast",
                files={
                    "csv": ("demo.csv", cf, "text/csv"),
                    "reference": ("reference.yaml", rf, "application/yaml"),
                    "mapping": ("mapping.yaml", mf, "application/yaml"),
                },
                data={"source_series_id": "DEMO", "horizon_days": "14"},
            )
        assert resp.status_code == 200
        bodies.append(resp.json())
    assert bodies[0]["result"]["forecast"] == bodies[1]["result"]["forecast"]
    assert ARTIFACT.read_bytes() == before


def test_different_series_do_not_contaminate(tmp_path: Path) -> None:
    import pandas as pd

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    shutil.copy(DEMO_CSV, a)
    df = pd.read_csv(DEMO_CSV)
    # Flat healthy series B
    df["permeate_flow_gpd"] = 6500.0
    df["feed_flow_gpd"] = 10000.0
    df.to_csv(b, index=False)
    ra = forecast_csv(
        a,
        source_series_id="A",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        artifact_path=ARTIFACT,
    )
    rb = forecast_csv(
        b,
        source_series_id="B",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        artifact_path=ARTIFACT,
    )
    # Declining A should differ from flat B
    assert ra["result"]["forecast"][0]["relative_health_point"] != pytest.approx(
        rb["result"]["forecast"][0]["relative_health_point"], abs=1e-4
    )
    assert ra["result"]["state_at_origin"]["relative_health"] != pytest.approx(
        rb["result"]["state_at_origin"]["relative_health"], abs=1e-4
    )


def test_missing_reference_wrong_units_gaps_short_post_cip(
    tmp_path: Path, client: TestClient
) -> None:
    # Missing reference via service → structured reference_required (not HTTP 500)
    missing_ref = forecast_csv(
        DEMO_CSV,
        source_series_id="DEMO",
        mapping_path=GENERIC_MAP,
        reference_path=None,
        artifact_path=ARTIFACT,
    )
    assert missing_ref["status"] == "reference_required"
    assert missing_ref["error"]["code"] == "reference_required"

    # Wrong units (atm on pressure)
    bad_map = tmp_path / "atm.yaml"
    bad_map.write_text(
        GENERIC_MAP.read_text(encoding="utf-8").replace("symbol: psi", "symbol: atm"),
        encoding="utf-8",
    )
    with pytest.raises(ForecastServiceError) as exc2:
        forecast_csv(
            DEMO_CSV,
            source_series_id="DEMO",
            mapping_path=bad_map,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert exc2.value.http_status in {400, 422}

    # Calendar gap then short post-gap history
    import pandas as pd

    df = pd.read_csv(DEMO_CSV)
    # Keep first 25 days, jump gap, add 5 more days (insufficient post-gap)
    head = df.iloc[:25].copy()
    tail = df.iloc[:5].copy()
    tail["observation_date"] = pd.date_range("2020-03-01", periods=5).strftime("%Y-%m-%d")
    gapped = pd.concat([head, tail], ignore_index=True)
    gap_csv = tmp_path / "gap.csv"
    gapped.to_csv(gap_csv, index=False)
    with pytest.raises(ForecastServiceError) as exc3:
        forecast_csv(
            gap_csv,
            source_series_id="GAP",
            mapping_path=GENERIC_MAP,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert exc3.value.code == "forecast_unavailable"
    blob = (exc3.value.message + " " + exc3.value.actionable_correction).lower()
    assert "insufficient" in blob or "segment" in blob or "cip" in blob

    # Short post-CIP history
    df2 = pd.read_csv(DEMO_CSV)
    df2.loc[20, "cip"] = 1
    cip_csv = tmp_path / "cip.csv"
    df2.to_csv(cip_csv, index=False)
    with pytest.raises(ForecastServiceError) as exc4:
        forecast_csv(
            cip_csv,
            source_series_id="CIP",
            mapping_path=GENERIC_MAP,
            reference_path=DEMO_REF,
            artifact_path=ARTIFACT,
        )
    assert exc4.value.code == "forecast_unavailable"

    # Malformed CSV via API
    bad = tmp_path / "bad.csv"
    bad.write_text('not,a,valid\n"unclosed', encoding="utf-8")
    with bad.open("rb") as cf, DEMO_REF.open("rb") as rf:
        resp = client.post(
            "/v1/forecast",
            files={
                "csv": ("bad.csv", cf, "text/csv"),
                "reference": ("reference.yaml", rf, "application/yaml"),
            },
            data={"source_series_id": "X"},
        )
    assert resp.status_code in {400, 422}
    assert "error" in resp.json()


def test_incompatible_artifact(tmp_path: Path) -> None:
    bad = tmp_path / "bad_art.json"
    bad.write_text('{"model_type": "nope"}', encoding="utf-8")
    with pytest.raises(ForecastServiceError) as exc:
        forecast_csv(
            DEMO_CSV,
            source_series_id="DEMO",
            mapping_path=GENERIC_MAP,
            reference_path=DEMO_REF,
            artifact_path=bad,
        )
    assert exc.value.http_status == 503


def test_already_crossed_and_no_crossing(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(DEMO_CSV)
    # Already crossed: low flows
    low = df.copy()
    low["permeate_flow_gpd"] = 4000.0
    low_csv = tmp_path / "low.csv"
    low.to_csv(low_csv, index=False)
    r_low = forecast_csv(
        low_csv,
        source_series_id="LOW",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    assert r_low["result"]["crossing_status"] == "crossed_at_origin"
    assert r_low["result"]["days_to_threshold"] == 0

    # Stable high — no crossing
    high = df.copy()
    high["permeate_flow_gpd"] = 6500.0
    high["feed_flow_gpd"] = 10000.0
    high_csv = tmp_path / "high.csv"
    high.to_csv(high_csv, index=False)
    r_high = forecast_csv(
        high_csv,
        source_series_id="HIGH",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    assert r_high["result"]["crossing_status"] == "no_crossing_in_horizon"
    assert r_high["result"]["days_to_threshold"] is None


def test_observation_and_forecast_dates_aligned(tmp_path: Path) -> None:
    csv = tmp_path / "demo.csv"
    shutil.copy(DEMO_CSV, csv)
    body = forecast_csv(
        csv,
        source_series_id="DEMO",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        artifact_path=ARTIFACT,
    )
    obs_dates = [o["date"] for o in body["normalised_data"]["observations"]]
    assert obs_dates == sorted(obs_dates)
    assert len(obs_dates) == body["normalised_data"]["n_observations"]
    origin = body["result"]["forecast_origin"]["date"]
    assert origin == obs_dates[-1]
    fc_dates = [p["date"] for p in body["result"]["forecast"]]
    assert len(fc_dates) == body["result"]["horizon_days"]
    assert fc_dates[0] > origin
    # contiguous daily forecast dates
    from datetime import date, timedelta

    d0 = date.fromisoformat(fc_dates[0])
    for i, ds in enumerate(fc_dates):
        assert date.fromisoformat(ds) == d0 + timedelta(days=i)


def test_ready_and_health_distinct(client: TestClient) -> None:
    h = client.get("/health").json()
    r = client.get("/ready").json()
    assert h["status"] == "ok"
    assert h["version"]
    assert "phase" not in h
    assert r["ready"] is True
    assert r["model_version"] == "llt_relative_health_v1"


def test_capabilities_forecast_available(client: TestClient) -> None:
    body = client.get("/v1/capabilities").json()
    assert body["endpoints"]["forecast"] is True
    assert "phase" not in body
    assert "forecast_api_cli" in body["implemented"]
