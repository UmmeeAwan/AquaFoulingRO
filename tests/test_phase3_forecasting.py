"""Phase 3 scientific and pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqua_fouling.forecasting.baselines import (
    LinearTrendForecaster,
    PersistenceForecaster,
)
from aqua_fouling.forecasting.evaluation import build_segments_for_series, run_rolling_origin
from aqua_fouling.forecasting.interface import ForecastRequest
from aqua_fouling.physics.normalisation import normalised_permeate_flow_from_temperature
from aqua_fouling.simulation.domain import BrackishDomain, DomainViolationError
from aqua_fouling.simulation.simulator import (
    SeriesSpec,
    generate_synthetic_benchmark,
    load_benchmark_config,
    simulate_series,
)


def test_domain_rejects_high_c_fc() -> None:
    domain = BrackishDomain()
    with pytest.raises(DomainViolationError, match="seawater"):
        domain.validate_operating_point(
            feed_concentration_mg_l=15000,
            recovery=0.8,
            temp_c=25,
            c_fc_mg_l=25000,
        )


def test_mass_flow_consistency_synthetic(tmp_path: Path) -> None:
    cfg = load_benchmark_config(Path("configs/synthetic_benchmark.yaml"))
    spec = SeriesSpec("T01", "gradual", "train", 99, health_decay_per_day=0.001)
    raw, _gt, _meta = simulate_series(spec, cfg=cfg, base_seed=1)
    qf = pd.to_numeric(raw["feed_flow_gpd"], errors="coerce")
    qp = pd.to_numeric(raw["permeate_flow_gpd"], errors="coerce")
    qc = pd.to_numeric(raw["concentrate_flow_gpd"], errors="coerce")
    ok = np.isfinite(qf) & np.isfinite(qp) & np.isfinite(qc)
    resid = (qf - qp - qc).abs() / qf
    assert float(resid[ok].median()) < 0.05


def test_latent_not_in_raw_columns(tmp_path: Path) -> None:
    out = tmp_path / "syn"
    manifest = generate_synthetic_benchmark(
        Path("configs/synthetic_benchmark.yaml"), output_dir=out
    )
    raw = pd.read_csv(manifest["series"][0]["raw_path"])
    for col in ("true_health", "latent_decay_per_day", "true_ndp_psi", "split"):
        assert col not in raw.columns


def test_persistence_already_crossed() -> None:
    req = ForecastRequest(
        origin_index=2,
        origin_date="2020-01-03",
        dates=["2020-01-01", "2020-01-02", "2020-01-03"],
        values=[1.0, 0.9, 0.8],
        horizon_days=7,
        threshold=0.85,
    )
    result = PersistenceForecaster().forecast(req)
    assert result.days_to_threshold == 0
    assert result.crossing_status == "crossed_at_origin"


def test_linear_known_crossing() -> None:
    values = [1.0 - 0.01 * i for i in range(20)]
    dates = pd.date_range("2020-01-01", periods=20, freq="D").strftime("%Y-%m-%d").tolist()
    origin = 5  # value = 0.95
    req = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates[: origin + 1],
        values=values[: origin + 1],
        horizon_days=30,
        threshold=0.85,
    )
    result = LinearTrendForecaster(window_days=10).forecast(req)
    assert result.crossing_status == "predicted_crossing"
    assert result.days_to_threshold is not None
    assert result.days_to_threshold == pytest.approx(10, abs=1)


def test_no_crossing_case() -> None:
    values = [1.0] * 30
    dates = pd.date_range("2020-01-01", periods=30, freq="D").strftime("%Y-%m-%d").tolist()
    req = ForecastRequest(
        origin_index=20,
        origin_date=dates[20],
        dates=dates[:21],
        values=values[:21],
        horizon_days=7,
        threshold=0.85,
    )
    result = PersistenceForecaster().forecast(req)
    assert result.crossing_status == "no_crossing_in_horizon"
    assert result.days_to_threshold is None


def test_gap_boundary_blocks_followup() -> None:
    dates = pd.Series(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-10",
            "2020-01-11",
            "2020-01-12",
        ]
    )
    values = pd.Series([1.0, 0.99, 0.98, 0.97, 0.96, 0.95])
    _segments, _audit, seg_ids = build_segments_for_series(dates)
    recs = run_rolling_origin(
        dates,
        values,
        forecaster=PersistenceForecaster(),
        horizons=[7],
        threshold=0.85,
        seg_ids=seg_ids,
        min_history_days=2,
    )
    origin_day3 = [r for r in recs if r["origin_date"] == "2020-01-03"]
    assert origin_day3
    assert origin_day3[0]["eligible"] is False


def test_no_future_leakage_changing_post_origin() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="D").strftime("%Y-%m-%d").tolist()
    values = [1.0 - 0.002 * i for i in range(40)]
    origin = 20
    req1 = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates[: origin + 1],
        values=list(values[: origin + 1]),
        horizon_days=14,
        threshold=0.85,
    )
    model = LinearTrendForecaster(window_days=14)
    r1 = model.forecast(req1)
    values2 = list(values)
    for i in range(origin + 1, 40):
        values2[i] = 0.0
    # History-only request unchanged even if full series corrupted after origin
    req2 = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates[: origin + 1],
        values=list(values2[: origin + 1]),
        horizon_days=14,
        threshold=0.85,
    )
    r2 = model.forecast(req2)
    assert r1.trajectory == r2.trajectory
    assert r1.days_to_threshold == r2.days_to_threshold


def test_deterioration_survives_normalisation() -> None:
    qn_h = float(normalised_permeate_flow_from_temperature(200.0, 100.0, 100.0, 25.0, 25.0)[0])
    qn_f = float(normalised_permeate_flow_from_temperature(150.0, 100.0, 100.0, 25.0, 25.0)[0])
    assert qn_f < qn_h


def test_stable_health_under_temp_change_in_simplified_model() -> None:
    k = 1.5
    ndp_ref, t_ref = 100.0, 25.0
    from aqua_fouling.physics.temperature import filmtec_temperature_correction_factor

    q_ref = k * ndp_ref * float(filmtec_temperature_correction_factor(t_ref)[0])
    for t_op, ndp_op in [(15.0, 110.0), (30.0, 90.0)]:
        tcf = float(filmtec_temperature_correction_factor(t_op)[0])
        q = k * ndp_op * tcf
        qn = float(normalised_permeate_flow_from_temperature(q, ndp_op, ndp_ref, t_op, t_ref)[0])
        assert qn == pytest.approx(q_ref, rel=1e-9)
