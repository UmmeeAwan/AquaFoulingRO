"""Narrow pre-training fix regression tests (units, crossing timing, common MAE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.data.adapters import resolve_configs_dir
from aqua_fouling.data.reference import ReferenceConfig, ReferenceMode
from aqua_fouling.forecasting.baselines import PersistenceForecaster
from aqua_fouling.forecasting.evaluation import (
    _observed_crossing_days,
    build_segments_for_series,
    run_rolling_origin,
    summarize_records,
)
from aqua_fouling.physics.units import (
    UnitConversionError,
    bar_to_psi,
    convert_flow,
    convert_pressure,
)


def test_gpd_gpm_equivalent_health(tmp_path: Path) -> None:
    """Equivalent gpd vs gpm inputs + matching references → equivalent health."""
    mapping = resolve_configs_dir() / "canonical_generic_schema.yaml"
    # gpd series
    gpd_csv = tmp_path / "gpd.csv"
    gpd_csv.write_text(
        "observation_date,temperature_c,feed_pressure_psi,permeate_pressure_psi,"
        "device_dp_psi,permeate_flow_gpd,feed_flow_gpd,feed_concentration_mg_l,cip\n"
        "2020-01-01,25.0,200.0,15.0,20.0,14400.0,28800.0,2000.0,0\n"
        "2020-01-02,25.0,200.0,15.0,20.0,14256.0,28512.0,2000.0,0\n",
        encoding="utf-8",
    )
    gpd_ref = tmp_path / "gpd_ref.yaml"
    gpd_ref.write_text(
        yaml.safe_dump(
            {
                "mode": "documented",
                "provenance": "gpd fixture",
                "available_from": "2020-01-01",
                "temperature_c": 25.0,
                "ndp_psi": 150.0,
                "permeate_flow": 14400.0,
                "permeate_flow_unit": "gpd",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    # gpm adapter YAML (same columns renamed conceptually via custom mapping)
    gpm_map = tmp_path / "gpm_map.yaml"
    gpm_map.write_text(
        """
name: gpm_demo
schema_id: canonical_generic_v1
source_series_pattern: ".+"
timestamp: {column: observation_date, format: ISO8601, timezone: UTC}
fields:
  - {external_name: observation_date, canonical_name: observation_date_calendar,
     unit: {symbol: calendar_date}, kind: measured, measurement_scope: whole_unit}
  - {external_name: temperature_c, canonical_name: temperature_c,
     unit: {symbol: deg_C}, kind: measured, measurement_scope: whole_unit}
  - {external_name: feed_pressure_psi, canonical_name: stage_3_feed_pressure_psi,
     unit: {symbol: psi}, kind: measured, measurement_scope: stage_3}
  - {external_name: permeate_pressure_psi, canonical_name: permeate_pressure_psi,
     unit: {symbol: psi}, kind: measured, measurement_scope: whole_unit}
  - {external_name: device_dp_psi, canonical_name: stage_3_device_pressure_drop_psi,
     unit: {symbol: psi}, kind: measured, measurement_scope: stage_3}
  - {external_name: permeate_flow_gpm, canonical_name: stage_3_permeate_flow_source_unit,
     unit: {symbol: gpm}, kind: measured, measurement_scope: stage_3}
  - {external_name: feed_flow_gpm, canonical_name: stage_3_feed_flow_source_unit,
     unit: {symbol: gpm}, kind: measured, measurement_scope: stage_3}
  - {external_name: feed_concentration_mg_l, canonical_name: feed_concentration_mg_l,
     unit: {symbol: mg_L_NaCl_eq}, kind: measured, measurement_scope: whole_unit}
  - {external_name: cip, canonical_name: cip_flag_0_1,
     unit: {symbol: flag_0_1}, kind: measured, measurement_scope: whole_unit}
""",
        encoding="utf-8",
    )
    gpm_csv = tmp_path / "gpm.csv"
    # 14400 gpd = 10 gpm; 28800 gpd = 20 gpm
    gpm_csv.write_text(
        "observation_date,temperature_c,feed_pressure_psi,permeate_pressure_psi,"
        "device_dp_psi,permeate_flow_gpm,feed_flow_gpm,feed_concentration_mg_l,cip\n"
        "2020-01-01,25.0,200.0,15.0,20.0,10.0,20.0,2000.0,0\n"
        "2020-01-02,25.0,200.0,15.0,20.0,9.9,19.8,2000.0,0\n",
        encoding="utf-8",
    )
    gpm_ref = tmp_path / "gpm_ref.yaml"
    gpm_ref.write_text(
        yaml.safe_dump(
            {
                "mode": "documented",
                "provenance": "gpm fixture",
                "available_from": "2020-01-01",
                "temperature_c": 25.0,
                "ndp_psi": 150.0,
                "permeate_flow": 10.0,
                "permeate_flow_unit": "gpm",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    r_gpd = normalise_csv(
        gpd_csv, source_series_id="GPD", mapping_path=mapping, reference_path=gpd_ref
    )
    r_gpm = normalise_csv(
        gpm_csv, source_series_id="GPM", mapping_path=gpm_map, reference_path=gpm_ref
    )
    h_gpd = r_gpd["normalised"]["derived"]["independent_physical_normalisation"]["relative_health"]
    h_gpm = r_gpm["normalised"]["derived"]["independent_physical_normalisation"]["relative_health"]
    assert r_gpd["normalised"]["derived"]["independent_physical_normalisation"]["available"]
    assert r_gpm["normalised"]["derived"]["independent_physical_normalisation"]["available"]
    for a, b in zip(h_gpd, h_gpm, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_bar_to_psi_conversion_and_rejection() -> None:
    assert float(bar_to_psi(1.0)[0]) == pytest.approx(14.503773773)
    converted = convert_pressure(2.0, from_unit="bar", to_unit="psi")
    assert float(converted[0]) == pytest.approx(2.0 * 14.503773773)
    with pytest.raises(UnitConversionError, match="Unsupported"):
        convert_pressure(1.0, from_unit="atm", to_unit="psi")
    with pytest.raises(UnitConversionError, match="unresolved"):
        convert_flow(1.0, from_unit="unresolved_flow", to_unit="gpd")


def test_reference_rejects_nonfinite_and_unresolved_unit() -> None:
    with pytest.raises(ValueError, match="finite"):
        ReferenceConfig(
            mode=ReferenceMode.DOCUMENTED,
            provenance="bad",
            available_from="2020-01-01",
            temperature_c=float("nan"),
            ndp_psi=100.0,
            permeate_flow=10.0,
            permeate_flow_unit="gpd",
        )
    with pytest.raises(ValueError, match="unsupported|unresolved"):
        ReferenceConfig(
            mode=ReferenceMode.DOCUMENTED,
            provenance="bad",
            available_from="2020-01-01",
            temperature_c=25.0,
            ndp_psi=100.0,
            permeate_flow=10.0,
            permeate_flow_unit="unresolved_flow",
        )


def test_crossing_uncertain_timing_gaps_before_crossing() -> None:
    # Gap on day 1, crossing observed on day 3 → occurrence without exact timing
    follow: list[float | None] = [None, 0.9, 0.8]
    out = _observed_crossing_days(1.0, follow, 0.85, complete_followup=False)
    assert out["crossing_occurred"] is True
    assert out["timing_exact"] is False
    assert out["status"] == "observed_crossing_uncertain_timing"
    assert out["days"] is None


def test_crossing_exact_timing_gaps_after_crossing() -> None:
    # Crossing on day 2 with day 1 observed; gap after must not invalidate timing
    follow: list[float | None] = [0.9, 0.8, None, None]
    out = _observed_crossing_days(1.0, follow, 0.85, complete_followup=False)
    assert out["crossing_occurred"] is True
    assert out["timing_exact"] is True
    assert out["status"] == "observed_crossing"
    assert out["days"] == 2


def test_crossing_metrics_denominators_in_summary() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=40, freq="D").strftime("%Y-%m-%d"))
    values = pd.Series([1.0 - 0.01 * i for i in range(40)])
    # Insert a gap before a late crossing window
    values.iloc[25] = np.nan
    _s, _a, seg_ids = build_segments_for_series(dates)
    recs = run_rolling_origin(
        dates,
        values,
        forecaster=PersistenceForecaster(),
        horizons=[14],
        threshold=0.85,
        seg_ids=seg_ids,
        min_history_days=5,
    )
    for r in recs:
        r["series_id"] = "T"
    summary = summarize_records(recs)
    row = summary["by_model_horizon"][0]
    assert "n_scored_origins" in row
    assert "n_pred_unavailable" in row
    assert "n_exact_timing_eligible" in row
    assert "n_observable_crossings_uncertain_timing" in row
    assert "common_origin_mae_mean" in row
    assert "partial_window_mae_mean" in row or row.get("n_partial_window_scored") is not None


def test_resolve_configs_dir_finds_repo_configs() -> None:
    cfg = resolve_configs_dir()
    assert (cfg / "canonical_generic_schema.yaml").is_file()
    assert (cfg / "canonical_synthetic_schema.yaml").is_file()
