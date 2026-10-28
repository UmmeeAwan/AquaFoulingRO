"""Phase 3 corrective-pass regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.application.validate import validate_csv
from aqua_fouling.forecasting.baselines import LinearTrendForecaster, PersistenceForecaster
from aqua_fouling.forecasting.evaluation import (
    build_segments_for_series,
    eligible_origins,
    run_rolling_origin,
    summarize_records,
)
from aqua_fouling.forecasting.interface import ForecastRequest
from aqua_fouling.physics.temperature import filmtec_temperature_correction_factor
from aqua_fouling.simulation.domain import DomainViolationError
from aqua_fouling.simulation.simulator import (
    M2_TO_FT2,
    SeriesSpec,
    flux_gfd_to_flow_gpd,
    load_benchmark_config,
    simulate_series,
)


def test_explicit_mapping_before_ocwd_detection(fixtures_dir: Path) -> None:
    csv = fixtures_dir / "generic" / "canonical_demo.csv"
    mapping = Path("configs/canonical_generic_schema.yaml")
    result = validate_csv(
        csv,
        source_series_id="DEMO01",
        mapping_path=mapping,
    )
    assert result["status"] in {"valid", "valid_with_warnings"}
    assert result["metadata"]["ocwd_autodetect_used"] is False
    assert result["metadata"]["schema_id"] == "canonical_generic_v1"
    assert "feed_concentration_mg_l" in result["metadata"]["canonical_columns"]
    # Measured concentration present — concentration blocker must not be listed.
    conc_blockers = [
        m
        for m in result["missing_for_calculations"]
        if m.get("calculation") == "independent_physical_normalisation_ndp_from_concentration"
    ]
    assert conc_blockers == []


def test_non_ocwd_csv_normalise_integration(fixtures_dir: Path) -> None:
    csv = fixtures_dir / "generic" / "canonical_demo.csv"
    mapping = Path("configs/canonical_generic_schema.yaml")
    ref = fixtures_dir / "generic" / "reference.yaml"
    result = normalise_csv(
        csv,
        source_series_id="DEMO01",
        mapping_path=mapping,
        reference_path=ref,
    )
    assert result["status"] in {"ok", "ok_with_limitations"}
    assert result["metadata"]["schema_id"] == "canonical_generic_v1"
    derived = result["normalised"]["derived"]["independent_physical_normalisation"]
    assert derived["available"] is True
    assert derived["n_normalised"] >= 1
    # Concentration not reported missing when measured column is mapped.
    assert not any(
        m.get("calculation") == "independent_physical_normalisation_ndp_from_concentration"
        for m in result["metadata"].get("missing_for_calculations", [])
    )
    assert "failures" in result["row_failures"]


def test_forecaster_receives_only_history_no_future_leakage() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="D").strftime("%Y-%m-%d").tolist()
    values = [1.0 - 0.002 * i for i in range(40)]
    origin = 20
    hist_dates = dates[: origin + 1]
    hist_values = values[: origin + 1]
    req1 = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=hist_dates,
        values=list(hist_values),
        horizon_days=14,
        threshold=0.85,
    )
    model = LinearTrendForecaster(window_days=14)
    r1 = model.forecast(req1)
    # Corrupt "future" that is NOT in the request
    hist_values2 = list(hist_values)
    req2 = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=hist_dates,
        values=hist_values2,
        horizon_days=14,
        threshold=0.85,
        metadata={"future_corrupted_outside_request": [0.0] * 19},
    )
    r2 = model.forecast(req2)
    assert r1.trajectory == r2.trajectory

    # Changing values after origin inside a full-series request must not be possible
    # because evaluation only passes history; simulate accidental future inclusion:
    bad = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates,
        values=values[: origin + 1] + [0.0] * (40 - origin - 1),
        horizon_days=14,
        threshold=0.85,
    )
    # Forecaster only uses indices ≤ origin_index
    r3 = model.forecast(bad)
    assert r3.trajectory == r1.trajectory


def test_linear_trend_no_silent_outside_window_fallback() -> None:
    # History exists, but only the origin lies inside a 3-day calendar window.
    dates = ["2020-01-01", "2020-01-02", "2020-01-20"]
    values = [1.0, 0.99, 0.94]
    req = ForecastRequest(
        origin_index=2,
        origin_date="2020-01-20",
        dates=dates,
        values=values,
        horizon_days=7,
        threshold=0.85,
    )
    result = LinearTrendForecaster(window_days=3).forecast(req)
    assert result.crossing_status == "unavailable"
    msg = (result.message or "").lower()
    assert "outside-window fallback" in msg or "fewer than two" in msg


def test_incomplete_followup_not_no_crossing() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=30, freq="D").strftime("%Y-%m-%d"))
    values = pd.Series([1.0] * 25 + [np.nan] * 5)
    _segments, _audit, seg_ids = build_segments_for_series(dates)
    # Force a gap mid-series to create incomplete follow-up within a segment?
    # Simpler: drop some follow-up values while keeping segment contiguous.
    elig = eligible_origins(dates, values, seg_ids=seg_ids, horizons=[7], min_history_days=5)
    # Origins near the nan block should be trajectory-eligible but not threshold-eligible
    near = [o for o in elig[7] if o.origin_index == 22]
    assert near
    assert near[0].eligible_trajectory is True
    assert near[0].eligible_threshold_event is False

    recs = run_rolling_origin(
        dates,
        values,
        forecaster=PersistenceForecaster(),
        horizons=[7],
        threshold=0.85,
        seg_ids=seg_ids,
        min_history_days=5,
    )
    r = next(r for r in recs if r["origin_index"] == 22 and r.get("eligible"))
    assert r["obs_crossing_status"] == "censored_incomplete_followup"
    assert r["false_warning"] is None  # cannot claim no-crossing


def test_split_cutoff_blocks_followup_past_boundary() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=60, freq="D").strftime("%Y-%m-%d"))
    values = pd.Series([1.0 - 0.001 * i for i in range(60)])
    _s, _a, seg_ids = build_segments_for_series(dates)
    cutoff = "2020-01-31"
    elig = eligible_origins(
        dates,
        values,
        seg_ids=seg_ids,
        horizons=[7, 14, 30],
        min_history_days=5,
        target_window_end=cutoff,
    )
    for h in (7, 14, 30):
        # Origin such that origin+h > cutoff must be ineligible
        for o in elig[h]:
            if o.eligible_trajectory:
                end = pd.Timestamp(o.origin_date) + pd.Timedelta(days=h)
                assert end <= pd.Timestamp(cutoff)


def test_flux_to_flow_hand_calculation() -> None:
    # Independent fixture: 1 GFD over 1 m² = 10.76391041671 gpd
    assert flux_gfd_to_flow_gpd(1.0, 1.0) == pytest.approx(M2_TO_FT2)
    # A=0.1 GFD/psi, NDP=100 psi, area=37 m² → flux=10 GFD → Q=10*37*M2_TO_FT2
    q = flux_gfd_to_flow_gpd(0.1 * 100.0, 37.0)
    assert q == pytest.approx(10.0 * 37.0 * M2_TO_FT2)


def test_domain_violation_not_silently_repaired() -> None:
    cfg = load_benchmark_config(Path("configs/synthetic_benchmark.yaml"))
    # Force invalid C_fc via high feed + high recovery in domain ranges
    cfg = dict(cfg)
    cfg["domain"] = dict(cfg["domain"])
    cfg["domain"]["feed_concentration_mg_l_range"] = [18000, 18000]
    cfg["domain"]["recovery_range"] = [0.85, 0.85]
    spec = SeriesSpec("BAD", "stable", "train", 1)
    with pytest.raises(DomainViolationError):
        simulate_series(spec, cfg=cfg, base_seed=1)


def test_synthetic_raw_through_canonical_normalise(tmp_path: Path) -> None:
    cfg = load_benchmark_config(Path("configs/synthetic_benchmark.yaml"))
    spec = SeriesSpec("S01", "stable", "train", 1)
    raw, gt, meta = simulate_series(spec, cfg=cfg, base_seed=1)
    raw_path = tmp_path / "S01.csv"
    raw.to_csv(raw_path, index=False)
    ref = meta["reference"]
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(
        yaml.safe_dump(
            {
                "mode": "documented",
                "provenance": ref["provenance"],
                "available_from": ref["available_from"],
                "temperature_c": ref["temperature_c"],
                "ndp_psi": ref["ndp_psi"],
                "permeate_flow": ref["permeate_flow"],
                "permeate_flow_unit": "gpd",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = normalise_csv(
        raw_path,
        source_series_id="S01",
        mapping_path=Path("configs/canonical_synthetic_schema.yaml"),
        reference_path=ref_path,
    )
    assert result["status"] in {"ok", "ok_with_limitations"}
    derived = result["normalised"]["derived"]["independent_physical_normalisation"]
    assert derived["available"] is True
    health = derived["relative_health"]
    # Early clean rows should be near 1.0 (noise aside)
    finite = [h for h in health[:5] if h is not None]
    assert finite
    assert float(np.mean(finite)) == pytest.approx(1.0, abs=0.15)


def test_hand_calculated_normalisation_fixture() -> None:
    """Independent of simulator: Q_norm = Q * (NDP_ref/NDP) * (TCF_ref/TCF)."""
    from aqua_fouling.physics.normalisation import normalised_permeate_flow_from_temperature

    q = 100.0
    ndp_op, ndp_ref = 80.0, 100.0
    t_op, t_ref = 20.0, 25.0
    tcf_op = float(filmtec_temperature_correction_factor(t_op)[0])
    tcf_ref = float(filmtec_temperature_correction_factor(t_ref)[0])
    expected = q * (ndp_ref / ndp_op) * (tcf_ref / tcf_op)
    got = float(normalised_permeate_flow_from_temperature(q, ndp_op, ndp_ref, t_op, t_ref)[0])
    assert got == pytest.approx(expected, rel=1e-12)


def test_metrics_report_denominators_and_timing_n() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=40, freq="D").strftime("%Y-%m-%d"))
    values = pd.Series([1.0 - 0.01 * i for i in range(40)])
    _s, _a, seg_ids = build_segments_for_series(dates)
    recs = run_rolling_origin(
        dates,
        values,
        forecaster=LinearTrendForecaster(window_days=10),
        horizons=[14],
        threshold=0.85,
        seg_ids=seg_ids,
        min_history_days=5,
    )
    summary = summarize_records(recs, group_key="segment_id")
    row = summary["by_model_horizon"][0]
    assert "false_warning_denominator" in row
    assert "missed_warning_denominator" in row
    assert "n_timing_samples" in row
    assert "trajectory_window_mae_mean" in row
    assert "macro_average_by_model_horizon" in summary


def test_intervention_censor_scenario_meta() -> None:
    cfg = load_benchmark_config(Path("configs/synthetic_benchmark.yaml"))
    item = next(s for s in cfg["simulation"]["series"] if s["id"] == "S07")
    spec = SeriesSpec(
        series_id="S07",
        family=item["family"],
        split=item["split"],
        seed_offset=int(item["seed_offset"]),
        health_decay_per_day=float(item["health_decay_per_day"]),
        clean_day=int(item["clean_day"]),
        clean_recovery_fraction=float(item["clean_recovery_fraction"]),
        censor_no_clean_crossing=True,
    )
    _raw, _gt, meta = simulate_series(spec, cfg=cfg, base_seed=int(cfg["seed"]))
    hmeta = meta["health_meta"]
    assert hmeta["intervention_before_crossing"] is True
    assert hmeta["would_cross_without_clean"] is not None
    assert int(item["clean_day"]) < int(hmeta["would_cross_without_clean"])
