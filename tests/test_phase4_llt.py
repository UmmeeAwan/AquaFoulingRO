"""Phase 4: trained local-linear-trend Kalman — behaviour and artifact tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from aqua_fouling.forecasting.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_VERSION,
    ArtifactCompatibilityError,
    LLTArtifact,
    load_artifact,
    save_artifact,
)
from aqua_fouling.forecasting.interface import ForecastRequest
from aqua_fouling.forecasting.llt_kalman import (
    LLTParameters,
    SegmentSeries,
    VarianceConstraintError,
    filter_history_to_origin,
    forecast_from_state,
    total_training_loglik,
)
from aqua_fouling.forecasting.trained_llt import TrainedLLTForecaster
from aqua_fouling.physics.normalisation import NORMALISATION_METHOD_ID, NORMALISATION_VERSION


def _dates(n: int, start: str = "2020-01-01") -> list[str]:
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _artifact(params: LLTParameters | None = None) -> LLTArtifact:
    p = params or LLTParameters(
        r=1e-4,
        q_level=1e-6,
        q_slope=1e-8,
        p0_level=1e-2,
        p0_slope=1e-4,
    )
    return LLTArtifact(
        parameters=p.to_dict(),
        training_series_ids=["S01", "S02", "S03"],
        development_series_ids=["S04", "S05", "S06"],
        held_out_from_training=["S04", "S05", "S06", "S07", "S08", "S09", "S10"],
        random_seed=1,
        training_config_hash="abc",
        training_data_hash="def",
        convergence={"success": True, "final_negative_loglik": 1.0, "n_iterations": 5},
        selected_settings={"min_history_observations": 3},
    )


def test_future_perturbation_cannot_affect_earlier_forecast() -> None:
    art = _artifact()
    model = TrainedLLTForecaster(art, min_history_observations=3)
    dates = _dates(40)
    values = [1.0 - 0.002 * i for i in range(40)]
    origin = 24
    req = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates[: origin + 1],
        values=values[: origin + 1],
        horizon_days=14,
        threshold=0.85,
    )
    r1 = model.forecast(req)
    # Perturb "future" values that must not be in the request
    polluted_values = list(values)
    for i in range(origin + 1, 40):
        polluted_values[i] = -99.0
    req2 = ForecastRequest(
        origin_index=origin,
        origin_date=dates[origin],
        dates=dates[: origin + 1],
        values=polluted_values[: origin + 1],
        horizon_days=14,
        threshold=0.85,
    )
    r2 = model.forecast(req2)
    assert r1.trajectory == r2.trajectory
    assert r1.days_to_threshold == r2.days_to_threshold


def test_segment_boundary_resets_state() -> None:
    params = LLTParameters(r=1e-4, q_level=1e-6, q_slope=1e-8, p0_level=1e-2, p0_slope=1e-4)
    # Segment A: declining; Segment B: restarted near 1.0
    d0 = date.fromisoformat("2020-01-01").toordinal()
    seg_a = SegmentSeries(
        "A",
        0,
        np.asarray([d0 + i for i in range(30)], dtype=np.int64),
        np.asarray([1.0 - 0.01 * i for i in range(30)], dtype=float),
    )
    seg_b = SegmentSeries(
        "A",
        1,
        np.asarray([d0 + 40 + i for i in range(20)], dtype=np.int64),
        np.asarray([1.0 - 0.001 * i for i in range(20)], dtype=float),
    )
    from aqua_fouling.forecasting.llt_kalman import filter_segment_loglik

    _ll_a, n_a, state_a = filter_segment_loglik(seg_a, params)
    _ll_b, n_b, state_b = filter_segment_loglik(seg_b, params)
    assert state_a is not None and state_b is not None
    assert n_a >= 10 and n_b >= 10
    # After reset, segment B health estimate should be near 1, not near A's ~0.7
    assert float(state_b.mean[0]) > 0.9
    assert float(state_a.mean[0]) < 0.85


def test_artifact_save_load_preserves_predictions(tmp_path: Path) -> None:
    art = _artifact()
    path = tmp_path / "model.json"
    save_artifact(art, path)
    loaded = load_artifact(path)
    assert loaded.parameters == art.parameters
    assert loaded.model_version == MODEL_VERSION
    assert loaded.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    m1 = TrainedLLTForecaster(art, min_history_observations=3)
    m2 = TrainedLLTForecaster(loaded, min_history_observations=3)
    dates = _dates(30)
    values = [1.0 - 0.003 * i for i in range(30)]
    req = ForecastRequest(
        origin_index=20,
        origin_date=dates[20],
        dates=dates[:21],
        values=values[:21],
        horizon_days=10,
        threshold=0.85,
    )
    r1 = m1.forecast(req)
    r2 = m2.forecast(req)
    assert r1.trajectory == pytest.approx(r2.trajectory, rel=1e-12, abs=1e-12)
    assert r1.crossing_status == r2.crossing_status


def test_inference_leaves_artifact_unchanged(tmp_path: Path) -> None:
    art = _artifact()
    path = tmp_path / "model.json"
    save_artifact(art, path)
    before = path.read_bytes()
    model = TrainedLLTForecaster.from_path(path, min_history_observations=3)
    dates = _dates(25)
    values = [0.95 - 0.002 * i for i in range(25)]
    req = ForecastRequest(
        origin_index=20,
        origin_date=dates[20],
        dates=dates[:21],
        values=values[:21],
        horizon_days=7,
        threshold=0.85,
    )
    _ = model.forecast(req)
    assert path.read_bytes() == before


def test_invalid_artifact_and_variance_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(VarianceConstraintError):
        LLTParameters(r=-1.0, q_level=1e-5, q_slope=1e-7, p0_level=0.01, p0_slope=0.0001)
    with pytest.raises(VarianceConstraintError):
        LLTParameters(r=0.0, q_level=1e-5, q_slope=1e-7, p0_level=0.01, p0_slope=0.0001)

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="Corrupt|invalid JSON"):
        load_artifact(bad)

    missing = tmp_path / "missing.json"
    with pytest.raises(ArtifactCompatibilityError, match="not found"):
        load_artifact(missing)

    incompatible = tmp_path / "incompat.json"
    art = _artifact()
    payload = art.model_dump(mode="json")
    payload["normalisation_method_id"] = "wrong_method"
    incompatible.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="incompatible"):
        load_artifact(incompatible)

    assert NORMALISATION_METHOD_ID
    assert NORMALISATION_VERSION


def test_stable_declining_insufficient_and_crossed_cases() -> None:
    art = _artifact()
    model = TrainedLLTForecaster(art, min_history_observations=5)

    # Stable above threshold → no_crossing_in_horizon
    dates = _dates(40)
    stable = [0.98] * 40
    req = ForecastRequest(
        origin_index=30,
        origin_date=dates[30],
        dates=dates[:31],
        values=stable[:31],
        horizon_days=14,
        threshold=0.85,
    )
    r = model.forecast(req)
    assert r.crossing_status == "no_crossing_in_horizon"
    assert r.days_to_threshold is None
    assert "no future cleaning" in (r.message or "").lower()

    # Declining → predicted_crossing (still above threshold at origin)
    declining = [1.0 - 0.004 * i for i in range(40)]
    req_d = ForecastRequest(
        origin_index=20,
        origin_date=dates[20],
        dates=dates[:21],
        values=declining[:21],
        horizon_days=30,
        threshold=0.85,
    )
    rd = model.forecast(req_d)
    assert rd.crossing_status == "predicted_crossing"
    assert rd.days_to_threshold is not None and rd.days_to_threshold > 0
    # Slope not forced negative beyond data — improving allowed (no clip flag)
    assert rd.extras.get("no_slope_clipping") is True

    # Insufficient history
    req_i = ForecastRequest(
        origin_index=2,
        origin_date=dates[2],
        dates=dates[:3],
        values=declining[:3],
        horizon_days=7,
        threshold=0.85,
    )
    ri = model.forecast(req_i)
    assert ri.crossing_status == "unavailable"
    assert ri.days_to_threshold is None

    # Already crossed
    crossed = [0.80] * 20
    req_c = ForecastRequest(
        origin_index=15,
        origin_date=dates[15],
        dates=dates[:16],
        values=crossed[:16],
        horizon_days=7,
        threshold=0.85,
    )
    rc = model.forecast(req_c)
    assert rc.crossing_status == "crossed_at_origin"
    assert rc.days_to_threshold == 0


def test_irregular_elapsed_time_uses_calendar_dt() -> None:
    params = LLTParameters(r=1e-4, q_level=1e-6, q_slope=1e-8, p0_level=1e-2, p0_slope=1e-4)
    d0 = date.fromisoformat("2020-01-01").toordinal()
    # Observations at day 0, 1, 5, 6 (gap of 4 between 1 and 5)
    days = np.asarray([d0, d0 + 1, d0 + 5, d0 + 6], dtype=np.int64)
    vals = np.asarray([1.0, 0.99, 0.95, 0.94], dtype=float)
    state = filter_history_to_origin(
        [int(x) for x in days],
        [float(v) for v in vals],
        params,
    )
    assert state is not None
    assert state.n_updates == 4
    fc = forecast_from_state(state, horizon_days=5, params=params)
    assert len(fc["trajectory"]) == 5
    assert "latent_health_interval_low" in fc
    assert "observation_interval_low" in fc
    assert fc["interval_meaning"]


def test_training_excludes_development_and_test(tmp_path: Path) -> None:
    """Unit-level guard: total_training_loglik only sees provided segments."""
    params = LLTParameters(r=1e-3, q_level=1e-5, q_slope=1e-7, p0_level=1e-2, p0_slope=1e-4)
    d0 = date.fromisoformat("2020-01-01").toordinal()
    train_only = [
        SegmentSeries(
            "S01",
            0,
            np.asarray([d0 + i for i in range(20)], dtype=np.int64),
            np.asarray([1.0 - 0.001 * i for i in range(20)], dtype=float),
        )
    ]
    ll, n = total_training_loglik(train_only, params)
    assert n == 20
    assert np.isfinite(ll)

    # Artifact provenance must list held-out IDs
    art = _artifact()
    assert "S07" in art.held_out_from_training
    assert "S01" in art.training_series_ids
    assert "S07" not in art.training_series_ids
