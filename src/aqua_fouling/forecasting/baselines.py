"""Persistence and recent-linear-trend baselines (no invented uncertainty)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from aqua_fouling.forecasting.interface import Forecaster, ForecastRequest, ForecastResult


def _parse_dates(dates: list[str]) -> list[date | None]:
    out: list[date | None] = []
    for d in dates:
        try:
            out.append(date.fromisoformat(str(d)[:10]))
        except ValueError:
            out.append(None)
    return out


def _elapsed_days(origin: date, earlier: date) -> float:
    return float((origin - earlier).days)


def _finite_history_calendar(
    values: list[float | None],
    dates: list[str],
    origin_index: int,
) -> tuple[list[float], list[date], list[int]]:
    """Finite observations at/before origin (request already segment-restricted)."""
    parsed = _parse_dates(dates)
    origin_d = parsed[origin_index] if 0 <= origin_index < len(parsed) else None
    vals: list[float] = []
    dts: list[date] = []
    idxs: list[int] = []
    for i in range(origin_index + 1):
        v = values[i]
        d = parsed[i]
        if v is not None and np.isfinite(v) and d is not None:
            if origin_d is not None and d > origin_d:
                continue
            vals.append(float(v))
            dts.append(d)
            idxs.append(i)
    return vals, dts, idxs


def _validate_daily_or_use_elapsed(
    dts: list[date],
    *,
    require_daily: bool,
) -> tuple[bool, str | None]:
    if len(dts) < 2:
        return True, None
    gaps = [(dts[i] - dts[i - 1]).days for i in range(1, len(dts))]
    if require_daily and any(g != 1 for g in gaps):
        # Still allow elapsed-calendar modelling, but surface the irregularity.
        return False, (
            "History is not strictly daily; trend uses elapsed calendar days "
            f"(gaps={sorted(set(gaps))})."
        )
    return True, None


def _crossing_from_trajectory(
    current: float,
    trajectory: list[float],
    threshold: float,
) -> tuple[int | None, str, str | None]:
    if current <= threshold:
        return 0, "crossed_at_origin", "Indicator already at/below threshold at origin."
    for step, val in enumerate(trajectory, start=1):
        if val <= threshold:
            return step, "predicted_crossing", None
    return None, "no_crossing_in_horizon", None


def _future_dates(origin_date: str, horizon: int) -> list[str]:
    start = date.fromisoformat(origin_date)
    return [(start + timedelta(days=i)).isoformat() for i in range(1, horizon + 1)]


class PersistenceForecaster:
    """Forecast = last finite observation at origin."""

    model_id = "persistence"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        vals, _dts, _idxs = _finite_history_calendar(
            request.values, request.dates, request.origin_index
        )
        if not vals:
            return ForecastResult(
                model_id=self.model_id,
                origin_index=request.origin_index,
                origin_date=request.origin_date,
                horizon_days=request.horizon_days,
                trajectory=[None] * request.horizon_days,
                trajectory_dates=_future_dates(request.origin_date, request.horizon_days),
                threshold=request.threshold,
                days_to_threshold=None,
                crossing_status="unavailable",
                message="No finite observation available at forecast origin.",
            )
        last = vals[-1]
        traj_f = [last] * request.horizon_days
        days, status, msg = _crossing_from_trajectory(last, traj_f, request.threshold)
        return ForecastResult(
            model_id=self.model_id,
            origin_index=request.origin_index,
            origin_date=request.origin_date,
            horizon_days=request.horizon_days,
            trajectory=list(traj_f),
            trajectory_dates=_future_dates(request.origin_date, request.horizon_days),
            threshold=request.threshold,
            days_to_threshold=days,
            crossing_status=status,
            message=msg,
        )


class LinearTrendForecaster:
    """Recent linear trend in elapsed calendar time (no silent outside-window fallback)."""

    model_id = "recent_linear_trend"

    def __init__(self, window_days: int = 14) -> None:
        self.window_days = window_days

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        vals, dts, _idxs = _finite_history_calendar(
            request.values, request.dates, request.origin_index
        )
        traj_dates = _future_dates(request.origin_date, request.horizon_days)
        if len(vals) < 2 or len(dts) < 2:
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
                message="Need at least two finite history points for linear trend.",
            )
        origin_d = dts[-1]
        # Restrict to calendar window ending at origin — NO fallback outside window.
        pairs = [
            (d, v)
            for d, v in zip(dts, vals, strict=True)
            if _elapsed_days(origin_d, d) <= float(self.window_days)
        ]
        if len(pairs) < 2:
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
                    f"Fewer than two finite points within the recent "
                    f"{self.window_days}-day calendar window; no outside-window fallback."
                ),
            )
        daily_ok, spacing_msg = _validate_daily_or_use_elapsed(
            [p[0] for p in pairs],
            require_daily=request.require_daily_spacing,
        )
        x = np.asarray([_elapsed_days(origin_d, p[0]) * -1.0 for p in pairs], dtype=float)
        # x: days relative to origin (0 at origin, negative earlier)
        y = np.asarray([p[1] for p in pairs], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        current = float(y[-1])
        # Extrapolate to +1..+horizon calendar days from origin (x = +step)
        traj_f = [
            float(intercept + slope * float(step)) for step in range(1, request.horizon_days + 1)
        ]
        days, status, msg = _crossing_from_trajectory(current, traj_f, request.threshold)
        notes = [m for m in (msg, spacing_msg) if m]
        if not daily_ok and request.require_daily_spacing and spacing_msg:
            # Still produce a trajectory using elapsed days; mark irregular spacing.
            notes.append("elapsed_calendar_time_used")
        return ForecastResult(
            model_id=self.model_id,
            origin_index=request.origin_index,
            origin_date=request.origin_date,
            horizon_days=request.horizon_days,
            trajectory=list(traj_f),
            trajectory_dates=traj_dates,
            threshold=request.threshold,
            days_to_threshold=days,
            crossing_status=status,
            message="; ".join(notes) if notes else None,
            extras={
                "slope_per_calendar_day": float(slope),
                "window_calendar_days": self.window_days,
                "n_history_points_in_window": len(pairs),
                "daily_spacing_ok": daily_ok,
            },
        )


def get_baseline(model_id: str, *, trend_window: int = 14) -> Forecaster:
    if model_id == "persistence":
        return PersistenceForecaster()
    if model_id == "recent_linear_trend":
        return LinearTrendForecaster(window_days=trend_window)
    msg = f"Unknown baseline model_id={model_id}"
    raise ValueError(msg)
