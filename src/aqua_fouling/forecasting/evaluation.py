"""Rolling-origin evaluation utilities shared by synthetic and OCWD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from aqua_fouling.application.segments import SegmentRuleConfig, contiguous_segments
from aqua_fouling.forecasting.baselines import get_baseline
from aqua_fouling.forecasting.interface import Forecaster, ForecastRequest, ForecastResult


@dataclass
class OriginEligibility:
    origin_index: int
    origin_date: str
    segment_id: int
    eligible_trajectory: bool
    eligible_threshold_event: bool
    reason: str | None = None
    threshold_reason: str | None = None


def build_segments_for_series(
    dates: pd.Series,
    *,
    cip: pd.Series | None = None,
    days_since_replacement: pd.Series | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Return segments, audit, and per-row segment id array (-1 if none)."""
    segments, audit = contiguous_segments(
        dates,
        cip=cip,
        days_since_replacement=days_since_replacement,
        rules=SegmentRuleConfig(),
    )
    seg_ids = np.full(len(dates), -1, dtype=int)
    for sid, seg in enumerate(segments):
        seg_ids[seg["start_index"] : seg["end_index"] + 1] = sid
    return segments, audit, seg_ids


def _day_ordinals(parsed: pd.Series) -> np.ndarray:
    """Integer day ordinals (NaT → -1) for fast calendar arithmetic."""
    # pandas Timestamp.toordinal via view on datetime64[ns]
    vals = parsed.to_numpy(dtype="datetime64[ns]")
    # days since epoch
    days = vals.astype("datetime64[D]").astype(np.int64)
    out = np.asarray(days, dtype=np.int64)
    out[pd.isna(parsed).to_numpy()] = -1
    return out


def _followup_indices_calendar(
    day_ord: np.ndarray,
    *,
    origin_i: int,
    horizon_days: int,
    seg_ids: np.ndarray,
) -> list[int]:
    """Row indices in the same segment with origin < date <= origin+horizon (calendar)."""
    origin_day = int(day_ord[origin_i])
    if origin_day < 0:
        return []
    end_day = origin_day + int(horizon_days)
    sid = int(seg_ids[origin_i])
    idxs: list[int] = []
    for j in range(origin_i + 1, len(day_ord)):
        if int(seg_ids[j]) != sid:
            break
        dj = int(day_ord[j])
        if dj < 0:
            continue
        if origin_day < dj <= end_day:
            idxs.append(j)
        if dj > end_day:
            break
    return idxs


def _complete_daily_followup(
    day_ord: np.ndarray,
    val_arr: np.ndarray,
    *,
    origin_i: int,
    horizon_days: int,
    follow_idxs: list[int],
) -> tuple[bool, str | None]:
    """True when every calendar day in (origin, origin+h] has a finite observation."""
    origin_day = int(day_ord[origin_i])
    if origin_day < 0:
        return False, "missing_origin_date"
    # Fast path: exactly h contiguous daily finite rows immediately after origin.
    if len(follow_idxs) == horizon_days:
        ok = True
        for step, j in enumerate(follow_idxs, start=1):
            if not np.isfinite(val_arr[j]):
                ok = False
                break
            if int(day_ord[j]) - origin_day != step:
                ok = False
                break
        if ok:
            return True, None
    expected = {origin_day + k for k in range(1, horizon_days + 1)}
    present: set[int] = set()
    for j in follow_idxs:
        if np.isfinite(val_arr[j]) and int(day_ord[j]) >= 0:
            present.add(int(day_ord[j]))
    n_missing = len(expected - present)
    if n_missing:
        return False, f"incomplete_followup_missing_{n_missing}_calendar_days"
    return True, None


def eligible_origins(
    dates: pd.Series,
    values: pd.Series,
    *,
    seg_ids: np.ndarray,
    horizons: list[int],
    min_history_days: int,
    target_window_end: pd.Timestamp | date | str | None = None,
) -> dict[int, list[OriginEligibility]]:
    """Eligibility per horizon with separate trajectory vs threshold-event flags.

    History and follow-up use **elapsed calendar time** within the operational
    segment. Measurement gaps:
    - Split segments via contiguous_segments (calendar gaps / CIP / replacement).
    - Trajectory eligibility requires ≥1 finite follow-up target in the calendar
      horizon window inside the segment.
    - Threshold-event eligibility additionally requires complete daily follow-up
      coverage so missing follow-up cannot establish an observed no-crossing.
    - If ``target_window_end`` is set, follow-up may not extend past that cutoff
      (train/dev/test temporal split integrity).
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    vals = pd.to_numeric(values, errors="coerce")
    val_arr = vals.to_numpy(dtype=float)
    day_ord = _day_ordinals(parsed)
    cutoff_ord: int | None = None
    if target_window_end is not None:
        cutoff_ord = int(
            np.datetime64(pd.Timestamp(target_window_end).normalize().to_datetime64(), "D").astype(
                np.int64
            )
        )
    n = len(dates)
    # Precompute segment starts (avoid O(n²) flatnonzero per row)
    seg_start_of = np.full(n, -1, dtype=int)
    for sid in np.unique(seg_ids):
        if int(sid) < 0:
            continue
        rows = np.flatnonzero(seg_ids == sid)
        seg_start_of[rows] = int(rows[0])

    out: dict[int, list[OriginEligibility]] = {h: [] for h in horizons}
    for i in range(n):
        if seg_ids[i] < 0 or day_ord[i] < 0:
            continue
        sid = int(seg_ids[i])
        seg_start = int(seg_start_of[i])
        if seg_start < 0:
            continue
        span = int(day_ord[i] - day_ord[seg_start]) + 1 if day_ord[seg_start] >= 0 else None
        hist_finite = int(np.isfinite(val_arr[seg_start : i + 1]).sum())
        origin_date = parsed.iloc[i].date().isoformat()
        for h in horizons:
            if (
                span is None
                or span < min_history_days
                or hist_finite < max(2, min_history_days // 2)
            ):
                out[h].append(
                    OriginEligibility(
                        i,
                        origin_date,
                        sid,
                        False,
                        False,
                        "insufficient_calendar_history_in_segment",
                        "insufficient_calendar_history_in_segment",
                    )
                )
                continue
            if not np.isfinite(val_arr[i]):
                out[h].append(
                    OriginEligibility(
                        i,
                        origin_date,
                        sid,
                        False,
                        False,
                        "missing_target_at_origin",
                        "missing_target_at_origin",
                    )
                )
                continue
            if cutoff_ord is not None and int(day_ord[i]) + int(h) > cutoff_ord:
                out[h].append(
                    OriginEligibility(
                        i,
                        origin_date,
                        sid,
                        False,
                        False,
                        "followup_extends_past_split_cutoff",
                        "followup_extends_past_split_cutoff",
                    )
                )
                continue
            follow_idxs = _followup_indices_calendar(
                day_ord, origin_i=i, horizon_days=h, seg_ids=seg_ids
            )
            if not follow_idxs:
                out[h].append(
                    OriginEligibility(
                        i,
                        origin_date,
                        sid,
                        False,
                        False,
                        "insufficient_followup_in_segment",
                        "insufficient_followup_in_segment",
                    )
                )
                continue
            has_finite = any(np.isfinite(val_arr[j]) for j in follow_idxs)
            if not has_finite:
                out[h].append(
                    OriginEligibility(
                        i,
                        origin_date,
                        sid,
                        False,
                        False,
                        "no_finite_followup_targets",
                        "no_finite_followup_targets",
                    )
                )
                continue
            complete, thr_reason = _complete_daily_followup(
                day_ord,
                val_arr,
                origin_i=i,
                horizon_days=h,
                follow_idxs=follow_idxs,
            )
            out[h].append(
                OriginEligibility(
                    i,
                    origin_date,
                    sid,
                    True,
                    complete,
                    None,
                    None if complete else thr_reason,
                )
            )
    return out


def _segment_history_slice(
    dates: list[str],
    values: list[float | None],
    *,
    seg_ids: np.ndarray,
    origin_index: int,
) -> tuple[list[str], list[float | None], int]:
    """Return history restricted to the origin's segment up to the origin."""
    sid = int(seg_ids[origin_index])
    start = int(np.flatnonzero(seg_ids == sid)[0])
    hist_dates = dates[start : origin_index + 1]
    hist_values = values[start : origin_index + 1]
    return hist_dates, hist_values, len(hist_values) - 1


def run_rolling_origin(
    dates: pd.Series,
    values: pd.Series,
    *,
    forecaster: Forecaster,
    horizons: list[int],
    threshold: float,
    seg_ids: np.ndarray,
    min_history_days: int,
    target_window_end: pd.Timestamp | date | str | None = None,
    intervention_censor_day: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate forecaster at eligible origins; each origin counted once per horizon.

    Forecasters receive **only** segment history at/before the origin (no future
    rows). Trajectory metrics use calendar-aligned follow-up. Threshold-event
    metrics require complete daily follow-up; incomplete follow-up is censored
    rather than treated as no-crossing.
    """
    eligibility = eligible_origins(
        dates,
        values,
        seg_ids=seg_ids,
        horizons=horizons,
        min_history_days=min_history_days,
        target_window_end=target_window_end,
    )
    parsed = pd.to_datetime(dates, errors="coerce")
    day_ord = _day_ordinals(parsed)
    date_list = [pd.Timestamp(d).date().isoformat() if pd.notna(d) else "" for d in parsed]
    value_list: list[float | None] = [
        float(v) if pd.notna(v) and np.isfinite(float(v)) else None
        for v in pd.to_numeric(values, errors="coerce")
    ]
    seg_start_of = np.full(len(seg_ids), -1, dtype=int)
    for sid in np.unique(seg_ids):
        if int(sid) < 0:
            continue
        rows = np.flatnonzero(seg_ids == sid)
        seg_start_of[rows] = int(rows[0])
    records: list[dict[str, Any]] = []
    for h, origins in eligibility.items():
        for origin in origins:
            base = {
                "model_id": forecaster.model_id,
                "horizon_days": h,
                "trajectory_window_days": h,
                "origin_index": origin.origin_index,
                "origin_date": origin.origin_date,
                "segment_id": origin.segment_id,
                "eligible_trajectory": origin.eligible_trajectory,
                "eligible_threshold_event": origin.eligible_threshold_event,
                "eligible": origin.eligible_trajectory,
                "exclusion_reason": origin.reason,
                "threshold_exclusion_reason": origin.threshold_reason,
            }
            if not origin.eligible_trajectory:
                records.append(base)
                continue

            start = int(seg_start_of[origin.origin_index])
            hist_dates = date_list[start : origin.origin_index + 1]
            hist_values = value_list[start : origin.origin_index + 1]
            hist_origin = len(hist_values) - 1
            req = ForecastRequest(
                origin_index=hist_origin,
                origin_date=origin.origin_date,
                dates=hist_dates,
                values=hist_values,
                horizon_days=h,
                threshold=threshold,
                segment_id=origin.segment_id,
                require_daily_spacing=True,
            )
            result = forecaster.forecast(req)

            follow_idxs = _followup_indices_calendar(
                day_ord,
                origin_i=origin.origin_index,
                horizon_days=h,
                seg_ids=seg_ids,
            )
            origin_day = int(day_ord[origin.origin_index])
            obs_by_offset: list[float | None] = [None] * h
            for j in follow_idxs:
                offset = int(day_ord[j]) - origin_day
                if 1 <= offset <= h:
                    obs_by_offset[offset - 1] = value_list[j]

            mae, rmse, n_pair = _trajectory_errors(result.trajectory, obs_by_offset)
            current = value_list[origin.origin_index]
            obs_cross = _observed_crossing_days(
                current,
                obs_by_offset,
                threshold,
                complete_followup=origin.eligible_threshold_event,
            )

            # Intervention censoring: cleaning before unobserved no-clean crossing
            if intervention_censor_day is not None:
                try:
                    clean_d = date.fromisoformat(str(intervention_censor_day)[:10])
                    origin_cal = date.fromisoformat(origin.origin_date)
                    in_horizon = origin_cal < clean_d <= origin_cal + timedelta(days=h)
                    censorable = obs_cross["status"] in {
                        "no_crossing_in_horizon",
                        "censored_incomplete_followup",
                    }
                    if in_horizon and censorable:
                        obs_cross = {
                            "days": None,
                            "status": "censored_intervention_before_crossing",
                        }
                except ValueError:
                    pass

            timing = _timing_error(
                result.days_to_threshold,
                obs_cross["days"],
                result.crossing_status,
                obs_cross["status"],
                timing_exact=bool(obs_cross.get("timing_exact", False)),
            )
            complete_horizon_scored = bool(origin.eligible_threshold_event and n_pair == h)
            records.append(
                {
                    **base,
                    "trajectory": result.trajectory,
                    "observed_followup": obs_by_offset,
                    "trajectory_window_mae": mae,
                    "trajectory_window_rmse": rmse,
                    "mae": mae,
                    "rmse": rmse,
                    "n_paired_points": n_pair,
                    "partial_window_scored": bool(mae is not None and not complete_horizon_scored),
                    "complete_horizon_scored": complete_horizon_scored,
                    "pred_days_to_threshold": result.days_to_threshold,
                    "pred_crossing_status": result.crossing_status,
                    "obs_days_to_threshold": obs_cross["days"],
                    "obs_crossing_status": obs_cross["status"],
                    "obs_crossing_occurred": bool(obs_cross.get("crossing_occurred")),
                    "obs_timing_exact": bool(obs_cross.get("timing_exact")),
                    "crossing_timing_error_days": timing,
                    "false_warning": _false_warning(result, obs_cross),
                    "missed_warning": _missed_warning(result, obs_cross),
                    "already_crossed_at_origin": (
                        result.crossing_status == "crossed_at_origin"
                        or obs_cross["status"] == "crossed_at_origin"
                    ),
                    "prediction_unavailable": result.crossing_status == "unavailable"
                    or all(v is None for v in result.trajectory),
                    "message": result.message,
                    "forecast_extras": dict(result.extras),
                }
            )
    return records


def _trajectory_errors(
    pred: list[float | None],
    obs: list[float | None],
) -> tuple[float | None, float | None, int]:
    pairs = [
        (float(p), float(o))
        for p, o in zip(pred, obs, strict=False)
        if p is not None and o is not None and np.isfinite(p) and np.isfinite(o)
    ]
    if not pairs:
        return None, None, 0
    err = np.asarray([p - o for p, o in pairs], dtype=float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), len(pairs)


def _observed_crossing_days(
    current: float | None,
    follow: list[float | None],
    threshold: float,
    *,
    complete_followup: bool,
) -> dict[str, Any]:
    """Separate crossing *occurrence* from exact first-crossing *timing*.

    - ``observed_crossing``: a finite observation at/below threshold exists in the
      horizon (occurrence). Exact timing is available only when every day from
      origin+1 through the first crossing day is observed and finite.
    - Missing days *before* the first observed crossing → occurrence without
      exact timing (``timing_exact`` false; status ``observed_crossing_uncertain_timing``).
    - Missing days *after* a fully observed first crossing do not invalidate that
      crossing or its exact timing.
    - Incomplete follow-up still cannot establish ``no_crossing_in_horizon``.
    """
    if current is None or not np.isfinite(current):
        return {
            "days": None,
            "status": "unavailable",
            "timing_exact": False,
            "crossing_occurred": False,
        }
    if current <= threshold:
        return {
            "days": 0,
            "status": "crossed_at_origin",
            "timing_exact": True,
            "crossing_occurred": True,
        }

    first_cross_step: int | None = None
    for step, v in enumerate(follow, start=1):
        if v is not None and np.isfinite(v) and v <= threshold:
            first_cross_step = step
            break

    if first_cross_step is not None:
        # Exact timing requires contiguous finite observations up to the crossing.
        prefix_complete = all(
            follow[k] is not None and np.isfinite(float(follow[k]))  # type: ignore[arg-type]
            for k in range(first_cross_step)
        )
        if prefix_complete:
            return {
                "days": first_cross_step,
                "status": "observed_crossing",
                "timing_exact": True,
                "crossing_occurred": True,
            }
        return {
            "days": None,
            "status": "observed_crossing_uncertain_timing",
            "timing_exact": False,
            "crossing_occurred": True,
            "first_observed_crossing_day": first_cross_step,
        }

    finite = [v for v in follow if v is not None and np.isfinite(v)]
    if not finite:
        return {
            "days": None,
            "status": "censored_or_unavailable",
            "timing_exact": False,
            "crossing_occurred": False,
        }
    if not complete_followup:
        return {
            "days": None,
            "status": "censored_incomplete_followup",
            "timing_exact": False,
            "crossing_occurred": False,
        }
    return {
        "days": None,
        "status": "no_crossing_in_horizon",
        "timing_exact": False,
        "crossing_occurred": False,
    }


def _timing_error(
    pred_days: int | None,
    obs_days: int | None,
    pred_status: str,
    obs_status: str,
    *,
    timing_exact: bool = True,
) -> int | None:
    """Exact timing error only for future-crossing pairs with exact observed timing."""
    if not timing_exact:
        return None
    if pred_status in {"unavailable", "crossed_at_origin"}:
        return None
    if obs_status in {
        "unavailable",
        "crossed_at_origin",
        "censored_or_unavailable",
        "censored_incomplete_followup",
        "censored_intervention_before_crossing",
        "observed_crossing_uncertain_timing",
    }:
        return None
    if pred_days is None or obs_days is None:
        return None
    return int(pred_days - obs_days)


def _false_warning(result: ForecastResult, obs_cross: dict[str, Any]) -> bool | None:
    """Predicted crossing while observed stays above threshold for full complete horizon."""
    if obs_cross["status"] != "no_crossing_in_horizon":
        return None
    if result.crossing_status == "crossed_at_origin":
        return None
    return result.crossing_status == "predicted_crossing"


def _missed_warning(result: ForecastResult, obs_cross: dict[str, Any]) -> bool | None:
    """Observed future crossing while model predicts no crossing or unavailable.

    Occurrence (including uncertain timing) counts for missed-warning denominators;
    exact timing is not required.
    """
    if not obs_cross.get("crossing_occurred"):
        return None
    if obs_cross["status"] in {"crossed_at_origin"}:
        return None
    return result.crossing_status in {"no_crossing_in_horizon", "unavailable"}


def summarize_records(
    records: list[dict[str, Any]],
    *,
    group_key: str | None = None,
) -> dict[str, Any]:
    """Aggregate metrics with denominators; optional per-series/scenario grouping.

    Trajectory-window MAE/RMSE are mean absolute/RMSE over the horizon window
    (paired calendar offsets), distinct from lead-h point errors or timing errors.
    Overlapping windows are not treated as independent runs.
    """

    def _summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        traj_eligible = [r for r in subset if r.get("eligible_trajectory") or r.get("eligible")]
        by_model_horizon: dict[str, Any] = {}
        # origin_key -> {model_id: mae} for common-origin comparison
        scored_by_origin: dict[tuple[Any, ...], dict[str, float]] = {}
        for r in traj_eligible:
            mid = str(r["model_id"])
            key = f"{mid}::h{r['horizon_days']}"
            by_model_horizon.setdefault(
                key,
                {
                    "model_id": mid,
                    "horizon_days": r["horizon_days"],
                    "maes": [],
                    "rmses": [],
                    "complete_horizon_maes": [],
                    "partial_window_maes": [],
                    "timing_errors": [],
                    "false_warnings": 0,
                    "missed_warnings": 0,
                    "n_false_warning_denominator": 0,
                    "n_missed_warning_denominator": 0,
                    "n_trajectory_eligible_origins": 0,
                    "n_threshold_event_eligible": 0,
                    "n_scored_origins": 0,
                    "n_partial_window_scored": 0,
                    "n_complete_horizon_scored": 0,
                    "n_already_crossed_at_origin": 0,
                    "n_observable_future_crossings": 0,
                    "n_observable_crossings_uncertain_timing": 0,
                    "n_exact_timing_eligible": 0,
                    "n_no_crossing_complete": 0,
                    "n_censored_incomplete_followup": 0,
                    "n_censored_intervention": 0,
                    "n_pred_unavailable": 0,
                    "n_timing_samples": 0,
                },
            )
            bucket = by_model_horizon[key]
            bucket["n_trajectory_eligible_origins"] += 1
            if r.get("eligible_threshold_event"):
                bucket["n_threshold_event_eligible"] += 1
            if r.get("prediction_unavailable") or r.get("pred_crossing_status") == "unavailable":
                bucket["n_pred_unavailable"] += 1
            if r.get("mae") is not None:
                bucket["maes"].append(r["mae"])
                bucket["n_scored_origins"] += 1
                origin_key = (
                    r.get("series_id"),
                    r.get("origin_index"),
                    r.get("origin_date"),
                    r["horizon_days"],
                )
                scored_by_origin.setdefault(origin_key, {})[mid] = float(r["mae"])
                if r.get("complete_horizon_scored"):
                    bucket["complete_horizon_maes"].append(r["mae"])
                    bucket["n_complete_horizon_scored"] += 1
                elif r.get("partial_window_scored"):
                    bucket["partial_window_maes"].append(r["mae"])
                    bucket["n_partial_window_scored"] += 1
            if r.get("rmse") is not None:
                bucket["rmses"].append(r["rmse"])
            if r.get("already_crossed_at_origin"):
                bucket["n_already_crossed_at_origin"] += 1
            if r.get("crossing_timing_error_days") is not None:
                bucket["timing_errors"].append(r["crossing_timing_error_days"])
                bucket["n_timing_samples"] += 1
            if r.get("obs_timing_exact") and r.get("obs_crossing_status") == "observed_crossing":
                bucket["n_exact_timing_eligible"] += 1
            if r.get("false_warning") is not None:
                bucket["n_false_warning_denominator"] += 1
                if r.get("false_warning") is True:
                    bucket["false_warnings"] += 1
            if r.get("missed_warning") is not None:
                bucket["n_missed_warning_denominator"] += 1
                if r.get("missed_warning") is True:
                    bucket["missed_warnings"] += 1
            st = r.get("obs_crossing_status")
            if st == "observed_crossing":
                bucket["n_observable_future_crossings"] += 1
            elif st == "observed_crossing_uncertain_timing":
                bucket["n_observable_crossings_uncertain_timing"] += 1
                bucket["n_observable_future_crossings"] += 1
            elif st == "no_crossing_in_horizon":
                bucket["n_no_crossing_complete"] += 1
            elif st == "censored_incomplete_followup":
                bucket["n_censored_incomplete_followup"] += 1
            elif st == "censored_intervention_before_crossing":
                bucket["n_censored_intervention"] += 1

        # Common-origin MAE: origins scored by every model present for that horizon
        common_mae_by_model_h: dict[str, list[float]] = {}
        for origin_key, model_maes in scored_by_origin.items():
            _sid, _oi, _od, h = origin_key
            models_for_h = {
                str(b["model_id"])
                for b in by_model_horizon.values()
                if int(b["horizon_days"]) == int(h)
            }
            if models_for_h and models_for_h <= set(model_maes.keys()):
                for mid in models_for_h:
                    common_mae_by_model_h.setdefault(f"{mid}::h{h}", []).append(model_maes[mid])

        summary_rows = []
        for bucket in by_model_horizon.values():
            fw_den = bucket["n_false_warning_denominator"]
            mw_den = bucket["n_missed_warning_denominator"]
            key = f"{bucket['model_id']}::h{bucket['horizon_days']}"
            common = common_mae_by_model_h.get(key) or []
            summary_rows.append(
                {
                    "model_id": bucket["model_id"],
                    "horizon_days": bucket["horizon_days"],
                    "n_eligible_origins": bucket["n_trajectory_eligible_origins"],
                    "n_trajectory_eligible_origins": bucket["n_trajectory_eligible_origins"],
                    "n_scored_origins": bucket["n_scored_origins"],
                    "n_pred_unavailable": bucket["n_pred_unavailable"],
                    "n_partial_window_scored": bucket["n_partial_window_scored"],
                    "n_complete_horizon_scored": bucket["n_complete_horizon_scored"],
                    "n_threshold_event_eligible": bucket["n_threshold_event_eligible"],
                    "trajectory_window_mae_mean": (
                        float(np.mean(bucket["maes"])) if bucket["maes"] else None
                    ),
                    "trajectory_window_rmse_mean": (
                        float(np.mean(bucket["rmses"])) if bucket["rmses"] else None
                    ),
                    "partial_window_mae_mean": (
                        float(np.mean(bucket["partial_window_maes"]))
                        if bucket["partial_window_maes"]
                        else None
                    ),
                    "complete_horizon_mae_mean": (
                        float(np.mean(bucket["complete_horizon_maes"]))
                        if bucket["complete_horizon_maes"]
                        else None
                    ),
                    "common_origin_mae_mean": (float(np.mean(common)) if common else None),
                    "n_common_scored_origins": len(common),
                    "mae_mean": float(np.mean(bucket["maes"])) if bucket["maes"] else None,
                    "rmse_mean": float(np.mean(bucket["rmses"])) if bucket["rmses"] else None,
                    "metric_note": (
                        "trajectory_window_* = mean over scored origins (partial or complete). "
                        "common_origin_mae_mean = mean on origins scored by all models at "
                        "that horizon (fair comparison when availability differs). "
                        "partial_window vs complete_horizon distinguish incomplete pairing "
                        "from full-horizon pairing. Not lead-h point error."
                    ),
                    "crossing_timing_error_mae": (
                        float(np.mean(np.abs(bucket["timing_errors"])))
                        if bucket["timing_errors"]
                        else None
                    ),
                    "n_timing_samples": bucket["n_timing_samples"],
                    "n_exact_timing_eligible": bucket["n_exact_timing_eligible"],
                    "timing_note": (
                        "Exact timing requires contiguous finite observations through the "
                        "first crossing; uncertain-timing occurrences are excluded from "
                        "timing MAE but included in occurrence/missed-warning denominators."
                    ),
                    "false_warnings": bucket["false_warnings"],
                    "false_warning_denominator": fw_den,
                    "false_warning_rate": (
                        float(bucket["false_warnings"]) / fw_den if fw_den else None
                    ),
                    "missed_warnings": bucket["missed_warnings"],
                    "missed_warning_denominator": mw_den,
                    "missed_warning_rate": (
                        float(bucket["missed_warnings"]) / mw_den if mw_den else None
                    ),
                    "n_already_crossed_at_origin": bucket["n_already_crossed_at_origin"],
                    "n_observable_future_crossings": bucket["n_observable_future_crossings"],
                    "n_observable_crossings_uncertain_timing": bucket[
                        "n_observable_crossings_uncertain_timing"
                    ],
                    "n_no_crossing_complete": bucket["n_no_crossing_complete"],
                    "n_censored_incomplete_followup": bucket["n_censored_incomplete_followup"],
                    "n_censored_intervention": bucket["n_censored_intervention"],
                    "independence_note": (
                        "Eligible origins counted once per (series, origin, horizon); "
                        "overlapping windows are not treated as independent runs."
                    ),
                }
            )
        n_excluded = sum(
            1 for r in subset if not (r.get("eligible_trajectory") or r.get("eligible"))
        )
        return {
            "n_records": len(subset),
            "n_eligible": len(traj_eligible),
            "n_excluded": n_excluded,
            "by_model_horizon": summary_rows,
        }

    overall = _summarize(records)
    if group_key:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            gk = str(r.get(group_key, "unknown"))
            groups.setdefault(gk, []).append(r)
        per_group = {g: _summarize(rs) for g, rs in sorted(groups.items())}
        # Macro average of MAE/RMSE across groups (equal weight per group)
        macro: dict[str, Any] = {}
        for row in overall.get("by_model_horizon") or []:
            key = f"{row['model_id']}::h{row['horizon_days']}"
            maes = []
            rmses = []
            for gsum in per_group.values():
                for grow in gsum.get("by_model_horizon") or []:
                    if (
                        grow["model_id"] == row["model_id"]
                        and grow["horizon_days"] == row["horizon_days"]
                    ):
                        if grow.get("mae_mean") is not None:
                            maes.append(grow["mae_mean"])
                        if grow.get("rmse_mean") is not None:
                            rmses.append(grow["rmse_mean"])
            macro[key] = {
                "model_id": row["model_id"],
                "horizon_days": row["horizon_days"],
                "macro_mae_mean": float(np.mean(maes)) if maes else None,
                "macro_rmse_mean": float(np.mean(rmses)) if rmses else None,
                "n_groups_with_metric": len(maes),
            }
        overall["per_group"] = per_group
        overall["macro_average_by_model_horizon"] = list(macro.values())
        overall["group_key"] = group_key
    return overall


def make_forecasters(trend_window: int) -> list[Forecaster]:
    return [
        get_baseline("persistence"),
        get_baseline("recent_linear_trend", trend_window=trend_window),
    ]
