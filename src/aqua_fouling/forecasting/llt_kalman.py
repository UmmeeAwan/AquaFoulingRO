"""Local linear trend (LLT) Kalman filter for relative normalised health.

State
-----
``x = [health, slope_per_calendar_day]^T``

Transition (elapsed calendar days ``dt > 0``)::

    F(dt) = [[1, dt],
             [0,  1]]

Process noise uses a Harvey-style continuous-time discretisation with
non-negative intensities ``q_level`` (level disturbance) and ``q_slope``
(slope disturbance)::

    Q(dt) = q_level * [[dt, 0], [0, 0]]
          + q_slope * [[dt^3/3, dt^2/2],
                       [dt^2/2, dt]]

Observation::

    y_t = health_t + v_t,   v_t ~ N(0, r)

Training objective
------------------
Sum of Gaussian innovation log-likelihoods across independent series/segments
(reset between segments; series are never concatenated into one sequence).
Missing observations trigger predict-only steps (no invented fill).

Variance parameters are constrained to be strictly positive via ``exp``
reparameterisation. Convergence is reported from the numerical optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

H = np.asarray([[1.0, 0.0]], dtype=float)
I2 = np.eye(2, dtype=float)
LOG_2PI = float(np.log(2.0 * np.pi))
MIN_VARIANCE = 1e-12
MAX_VARIANCE = 1e2


class VarianceConstraintError(ValueError):
    """Raised when variance parameters are non-positive or non-finite."""


@dataclass(frozen=True)
class LLTParameters:
    """Strictly positive LLT noise / initialisation parameters."""

    r: float
    q_level: float
    q_slope: float
    p0_level: float
    p0_slope: float

    def __post_init__(self) -> None:
        for name in ("r", "q_level", "q_slope", "p0_level", "p0_slope"):
            val = float(getattr(self, name))
            if not np.isfinite(val) or val <= 0.0:
                msg = f"LLT parameter '{name}' must be finite and strictly positive; " f"got {val}."
                raise VarianceConstraintError(msg)
            if val < MIN_VARIANCE or val > MAX_VARIANCE:
                msg = (
                    f"LLT parameter '{name}'={val} outside allowed range "
                    f"[{MIN_VARIANCE}, {MAX_VARIANCE}]."
                )
                raise VarianceConstraintError(msg)

    def to_dict(self) -> dict[str, float]:
        return {
            "r": float(self.r),
            "q_level": float(self.q_level),
            "q_slope": float(self.q_slope),
            "p0_level": float(self.p0_level),
            "p0_slope": float(self.p0_slope),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LLTParameters:
        return cls(
            r=float(raw["r"]),
            q_level=float(raw["q_level"]),
            q_slope=float(raw["q_slope"]),
            p0_level=float(raw["p0_level"]),
            p0_slope=float(raw["p0_slope"]),
        )

    def to_unconstrained(self) -> NDArray[np.floating]:
        return np.log(
            np.asarray(
                [self.r, self.q_level, self.q_slope, self.p0_level, self.p0_slope],
                dtype=float,
            )
        )

    @classmethod
    def from_unconstrained(cls, theta: NDArray[np.floating]) -> LLTParameters:
        arr = np.asarray(theta, dtype=float).reshape(-1)
        if arr.shape != (5,):
            msg = f"Expected 5 unconstrained parameters; got shape {arr.shape}."
            raise VarianceConstraintError(msg)
        if not np.all(np.isfinite(arr)):
            msg = "Unconstrained LLT parameters contain non-finite values."
            raise VarianceConstraintError(msg)
        # Soft-clip extreme logs to keep exp in the valid variance band.
        lo = float(np.log(MIN_VARIANCE))
        hi = float(np.log(MAX_VARIANCE))
        clipped = np.clip(arr, lo, hi)
        vals = np.exp(clipped)
        return cls(
            r=float(vals[0]),
            q_level=float(vals[1]),
            q_slope=float(vals[2]),
            p0_level=float(vals[3]),
            p0_slope=float(vals[4]),
        )


def transition_matrices(
    dt: float,
    *,
    q_level: float,
    q_slope: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if not np.isfinite(dt) or dt <= 0.0:
        msg = f"Elapsed calendar days dt must be finite and > 0; got {dt}."
        raise ValueError(msg)
    f = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=float)
    dt2 = dt * dt
    dt3 = dt2 * dt
    q = q_level * np.asarray([[dt, 0.0], [0.0, 0.0]], dtype=float) + q_slope * np.asarray(
        [[dt3 / 3.0, dt2 / 2.0], [dt2 / 2.0, dt]],
        dtype=float,
    )
    return f, q


@dataclass
class FilterState:
    mean: NDArray[np.floating]
    cov: NDArray[np.floating]
    last_day_ordinal: int | None = None
    n_updates: int = 0


def initial_state(
    params: LLTParameters,
    *,
    first_observation: float | None = None,
) -> FilterState:
    mean = np.asarray(
        [
            float(first_observation) if first_observation is not None else 1.0,
            0.0,
        ],
        dtype=float,
    )
    cov = np.diag([float(params.p0_level), float(params.p0_slope)]).astype(float)
    return FilterState(mean=mean, cov=cov, last_day_ordinal=None, n_updates=0)


def predict_step(state: FilterState, *, dt: float, params: LLTParameters) -> FilterState:
    f, q = transition_matrices(dt, q_level=params.q_level, q_slope=params.q_slope)
    mean = f @ state.mean
    cov = f @ state.cov @ f.T + q
    # Symmetrise for numerical stability
    cov = 0.5 * (cov + cov.T)
    return FilterState(
        mean=mean,
        cov=cov,
        last_day_ordinal=state.last_day_ordinal,
        n_updates=state.n_updates,
    )


def update_step(
    state: FilterState,
    observation: float,
    *,
    params: LLTParameters,
    day_ordinal: int,
) -> tuple[FilterState, float, float]:
    """Return updated state, innovation, and innovation variance."""
    y = float(observation)
    if not np.isfinite(y):
        msg = "Observation must be finite for Kalman update."
        raise ValueError(msg)
    innov = y - float((H @ state.mean).item())
    s = float((H @ state.cov @ H.T).item() + params.r)
    if not np.isfinite(s) or s <= 0.0:
        msg = f"Innovation variance must be positive; got {s}."
        raise VarianceConstraintError(msg)
    k = (state.cov @ H.T) / s
    mean = state.mean + (k.reshape(2) * innov)
    cov = (I2 - k @ H) @ state.cov
    cov = 0.5 * (cov + cov.T)
    updated = FilterState(
        mean=mean,
        cov=cov,
        last_day_ordinal=int(day_ordinal),
        n_updates=state.n_updates + 1,
    )
    return updated, innov, s


def innovation_loglik(innovation: float, s: float) -> float:
    return float(-0.5 * (LOG_2PI + np.log(s) + (innovation * innovation) / s))


@dataclass
class SegmentSeries:
    """One contiguous operational segment of measured relative health."""

    series_id: str
    segment_id: int
    day_ordinals: NDArray[np.int64]
    values: NDArray[np.floating]  # NaN allowed for missing


def filter_segment_loglik(
    segment: SegmentSeries,
    params: LLTParameters,
) -> tuple[float, int, FilterState | None]:
    """Run Kalman filter on one segment; return (loglik, n_obs, final_state)."""
    days = np.asarray(segment.day_ordinals, dtype=np.int64)
    vals = np.asarray(segment.values, dtype=float)
    if days.size == 0:
        return 0.0, 0, None
    order = np.argsort(days)
    days = days[order]
    vals = vals[order]

    first_obs: float | None = None
    for v in vals:
        if np.isfinite(v):
            first_obs = float(v)
            break
    if first_obs is None:
        return 0.0, 0, None

    # Initialisation policy: prior mean health = 1.0 (clean-reference scale),
    # prior slope = 0; first finite observation updates from that prior.
    state = initial_state(params, first_observation=1.0)
    total_ll = 0.0
    n_obs = 0
    for day, y in zip(days, vals, strict=True):
        d = int(day)
        if state.last_day_ordinal is None:
            if not np.isfinite(y):
                continue
            state, innov, s = update_step(state, float(y), params=params, day_ordinal=d)
            total_ll += innovation_loglik(innov, s)
            n_obs += 1
            continue
        dt = float(d - int(state.last_day_ordinal))
        if dt <= 0:
            # Same calendar day duplicate — skip rather than invent sub-daily dynamics.
            continue
        state = predict_step(state, dt=dt, params=params)
        if not np.isfinite(y):
            # Gap / missing target: advance the calendar anchor without an update.
            state = FilterState(
                mean=state.mean,
                cov=state.cov,
                last_day_ordinal=d,
                n_updates=state.n_updates,
            )
            continue
        state, innov, s = update_step(state, float(y), params=params, day_ordinal=d)
        total_ll += innovation_loglik(innov, s)
        n_obs += 1
    return total_ll, n_obs, state


def total_training_loglik(
    segments: list[SegmentSeries],
    params: LLTParameters,
) -> tuple[float, int]:
    """Sum innovation log-likelihood across independent segments (series resets)."""
    total = 0.0
    n = 0
    for seg in segments:
        ll, n_obs, _ = filter_segment_loglik(seg, params)
        total += ll
        n += n_obs
    return total, n


def filter_history_to_origin(
    day_ordinals: list[int],
    values: list[float | None],
    params: LLTParameters,
) -> FilterState | None:
    """Filter segment history through the origin; None if insufficient finite obs."""
    days = np.asarray(day_ordinals, dtype=np.int64)
    vals = np.asarray(
        [float(v) if v is not None and np.isfinite(v) else np.nan for v in values],
        dtype=float,
    )
    seg = SegmentSeries(
        series_id="_",
        segment_id=0,
        day_ordinals=days,
        values=vals,
    )
    _ll, n_obs, state = filter_segment_loglik(seg, params)
    if state is None or n_obs < 1:
        return None
    return state


def forecast_from_state(
    state: FilterState,
    *,
    horizon_days: int,
    params: LLTParameters,
    z_critical: float = 1.959963984540054,
) -> dict[str, Any]:
    """Open-loop Kalman prediction for +1 … +horizon calendar days.

    Returns latent-health means/intervals and future-observation intervals.
    Intervals are Gaussian ± z_critical · σ (default ≈ 95%). These are
    **model-based theoretical intervals**, not empirically calibrated confidence.
    """
    means: list[float] = []
    latent_low: list[float] = []
    latent_high: list[float] = []
    obs_low: list[float] = []
    obs_high: list[float] = []
    latent_std: list[float] = []
    obs_std: list[float] = []

    pred = FilterState(
        mean=state.mean.copy(),
        cov=state.cov.copy(),
        last_day_ordinal=state.last_day_ordinal,
        n_updates=state.n_updates,
    )
    for _step in range(1, horizon_days + 1):
        pred = predict_step(pred, dt=1.0, params=params)
        m = float(pred.mean[0])
        var_h = float(pred.cov[0, 0])
        if not np.isfinite(var_h) or var_h < 0.0:
            var_h = 0.0
        std_lat = float(np.sqrt(var_h))
        std_obs = float(np.sqrt(var_h + float(params.r)))
        means.append(m)
        latent_std.append(std_lat)
        obs_std.append(std_obs)
        latent_low.append(m - z_critical * std_lat)
        latent_high.append(m + z_critical * std_lat)
        obs_low.append(m - z_critical * std_obs)
        obs_high.append(m + z_critical * std_obs)

    return {
        "trajectory": means,
        "latent_health_std": latent_std,
        "observation_std": obs_std,
        "latent_health_interval_low": latent_low,
        "latent_health_interval_high": latent_high,
        "observation_interval_low": obs_low,
        "observation_interval_high": obs_high,
        "interval_nominal_coverage": 0.95,
        "interval_z_critical": float(z_critical),
        "interval_meaning": (
            "latent_health_* intervals are for the latent relative-health state; "
            "observation_* intervals add observation noise R and are for a future "
            "measured relative-health observation. Both are Gaussian model-based "
            "intervals (±z·σ), not empirically calibrated confidence intervals."
        ),
        "state_health": float(state.mean[0]),
        "state_slope_per_day": float(state.mean[1]),
        "state_cov": [
            [float(state.cov[0, 0]), float(state.cov[0, 1])],
            [float(state.cov[1, 0]), float(state.cov[1, 1])],
        ],
    }
