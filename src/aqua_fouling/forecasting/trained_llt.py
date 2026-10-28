"""Trained local-linear-trend Forecaster (frozen artifact; no retrain at inference)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from aqua_fouling.forecasting.artifact import (
    ArtifactCompatibilityError,
    LLTArtifact,
    load_artifact,
)
from aqua_fouling.forecasting.baselines import _crossing_from_trajectory, _future_dates
from aqua_fouling.forecasting.interface import ForecastRequest, ForecastResult
from aqua_fouling.forecasting.llt_kalman import (
    LLTParameters,
    filter_history_to_origin,
    forecast_from_state,
)


class TrainedLLTForecaster:
    """Kalman LLT forecaster loaded from a versioned JSON artifact.

    Inference loads frozen parameters, filters history available at the origin
    within the supplied segment, and open-loop predicts. The artifact file is
    never mutated. Forecasts assume **no future cleaning intervention**.
    Improving (positive) slopes are not clipped.
    """

    def __init__(
        self,
        artifact: LLTArtifact,
        *,
        min_history_observations: int = 5,
        artifact_path: Path | None = None,
    ) -> None:
        self.artifact = artifact
        self.model_id = artifact.model_id
        self.params: LLTParameters = artifact.llt_parameters()
        self.min_history_observations = int(min_history_observations)
        self.artifact_path = artifact_path

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        min_history_observations: int = 5,
    ) -> TrainedLLTForecaster:
        artifact = load_artifact(path)
        return cls(
            artifact,
            min_history_observations=min_history_observations,
            artifact_path=path,
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        traj_dates = _future_dates(request.origin_date, request.horizon_days)
        no_intervention = "Forecast assumes no future cleaning intervention within the horizon."

        # Build calendar ordinals from request dates (segment-restricted history).
        day_ords: list[int] = []
        vals: list[float | None] = []
        origin_d: date | None = None
        try:
            origin_d = date.fromisoformat(str(request.origin_date)[:10])
        except ValueError:
            origin_d = None

        hist_dates = request.dates[: request.origin_index + 1]
        hist_values = request.values[: request.origin_index + 1]
        for d_str, v in zip(hist_dates, hist_values, strict=True):
            try:
                d = date.fromisoformat(str(d_str)[:10])
            except ValueError:
                continue
            if origin_d is not None and d > origin_d:
                continue
            day_ords.append(d.toordinal())
            if v is None or not np.isfinite(float(v)):
                vals.append(None)
            else:
                vals.append(float(v))

        finite_n = sum(1 for v in vals if v is not None)
        if finite_n < self.min_history_observations:
            return ForecastResult(
                model_id=self.model_id,
                origin_index=request.origin_index,
                origin_date=request.origin_date,
                horizon_days=request.horizon_days,
                trajectory=[None] * request.horizon_days,
                trajectory_dates=traj_dates,
                threshold=request.threshold,
                days_to_threshold=None,
                crossing_status="unavailable",
                message=(
                    f"Insufficient usable history for LLT filter "
                    f"(finite_obs={finite_n} < min={self.min_history_observations}). "
                    + no_intervention
                ),
                extras={"prediction_available": False},
            )

        state = filter_history_to_origin(day_ords, vals, self.params)
        if state is None or state.n_updates < self.min_history_observations:
            return ForecastResult(
                model_id=self.model_id,
                origin_index=request.origin_index,
                origin_date=request.origin_date,
                horizon_days=request.horizon_days,
                trajectory=[None] * request.horizon_days,
                trajectory_dates=traj_dates,
                threshold=request.threshold,
                days_to_threshold=None,
                crossing_status="unavailable",
                message=(
                    "LLT filter did not accumulate enough updates at the origin. " + no_intervention
                ),
                extras={"prediction_available": False},
            )

        # Align state to the origin calendar day if the last update was earlier
        # (trailing missing observation): predict forward without inventing y.
        if origin_d is not None and state.last_day_ordinal is not None:
            origin_ord = origin_d.toordinal()
            lag = origin_ord - int(state.last_day_ordinal)
            if lag > 0:
                from aqua_fouling.forecasting.llt_kalman import FilterState, predict_step

                pred = state
                for _ in range(lag):
                    pred = predict_step(pred, dt=1.0, params=self.params)
                state = FilterState(
                    mean=pred.mean,
                    cov=pred.cov,
                    last_day_ordinal=origin_ord,
                    n_updates=state.n_updates,
                )

        current = float(state.mean[0])
        # Prefer last finite observation for "already crossed" decision when available.
        last_obs = next((v for v in reversed(vals) if v is not None), None)
        crossed_ref = float(last_obs) if last_obs is not None else current

        fc = forecast_from_state(
            state,
            horizon_days=request.horizon_days,
            params=self.params,
        )
        traj_f: list[float | None] = [float(x) for x in fc["trajectory"]]
        days, status, msg = _crossing_from_trajectory(
            crossed_ref,
            [float(x) for x in traj_f if x is not None],
            request.threshold,
        )
        notes = [no_intervention]
        if msg:
            notes.insert(0, msg)
        # Irregular spacing note
        if len(day_ords) >= 2:
            gaps = [day_ords[i] - day_ords[i - 1] for i in range(1, len(day_ords))]
            if any(g != 1 for g in gaps):
                notes.append(
                    "History is not strictly daily; filter uses elapsed calendar days "
                    f"(unique_gaps={sorted(set(int(g) for g in gaps if g > 0))})."
                )

        extras: dict[str, Any] = {
            "prediction_available": True,
            "state_health": fc["state_health"],
            "state_slope_per_day": fc["state_slope_per_day"],
            "state_cov": fc["state_cov"],
            "latent_health_interval_low": fc["latent_health_interval_low"],
            "latent_health_interval_high": fc["latent_health_interval_high"],
            "observation_interval_low": fc["observation_interval_low"],
            "observation_interval_high": fc["observation_interval_high"],
            "interval_nominal_coverage": fc["interval_nominal_coverage"],
            "interval_meaning": fc["interval_meaning"],
            "n_filter_updates": state.n_updates,
            "artifact_model_version": self.artifact.model_version,
            "no_slope_clipping": True,
            "assumes_no_future_cleaning": True,
        }
        return ForecastResult(
            model_id=self.model_id,
            origin_index=request.origin_index,
            origin_date=request.origin_date,
            horizon_days=request.horizon_days,
            trajectory=traj_f,
            trajectory_dates=traj_dates,
            threshold=request.threshold,
            days_to_threshold=days,
            crossing_status=status,
            message="; ".join(notes),
            extras=extras,
        )


def load_trained_forecaster(
    path: Path,
    *,
    min_history_observations: int = 5,
) -> TrainedLLTForecaster:
    try:
        return TrainedLLTForecaster.from_path(
            path, min_history_observations=min_history_observations
        )
    except ArtifactCompatibilityError:
        raise
