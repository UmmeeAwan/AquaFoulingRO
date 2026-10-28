"""Shared forecasting interface (baselines now; state-space later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ForecastRequest:
    """History available at a single forecast origin (no future rows).

    ``dates`` / ``values`` must contain **only** observations at or before the
    origin and within the relevant operational segment. ``origin_index`` is the
    index of the origin within this history list (typically ``len(values) - 1``).
    """

    origin_index: int
    origin_date: str
    dates: list[str]
    values: list[float | None]
    horizon_days: int
    threshold: float
    segment_id: int | None = None
    require_daily_spacing: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Baseline/model forecast output."""

    model_id: str
    origin_index: int
    origin_date: str
    horizon_days: int
    trajectory: list[float | None]
    trajectory_dates: list[str]
    threshold: float
    days_to_threshold: int | None
    crossing_status: str
    # crossing_status values:
    # - crossed_at_origin (0 days)
    # - predicted_crossing
    # - no_crossing_in_horizon
    # - unavailable
    message: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class Forecaster(Protocol):
    """Common interface for baselines and future state-space models."""

    model_id: str

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Produce a trajectory using only data available at the origin."""
        ...
