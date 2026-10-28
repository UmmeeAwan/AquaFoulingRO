"""Pure audit metrics used by the data-audit service and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DateCoverageResult:
    """Calendar completeness relative to the observed min/max date range."""

    n_rows: int
    n_unique_dates: int
    n_duplicate_date_rows: int
    duplicate_dates: list[str]
    min_date: str | None
    max_date: str | None
    missing_dates: list[str]
    n_missing_dates: int
    is_sorted_ascending: bool


@dataclass(frozen=True)
class MissingRun:
    """A contiguous run of missing measurements for one column."""

    start_index: int
    end_index: int
    length: int


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column to datetime64[ns] (UTC-naive calendar dates)."""
    return pd.to_datetime(series, errors="coerce")


def analyse_date_coverage(dates: pd.Series) -> DateCoverageResult:
    """Detect duplicate dates, ordering, and missing calendar days in the span."""
    parsed = parse_dates(dates)
    n_rows = int(len(parsed))
    valid = parsed.dropna()
    if valid.empty:
        return DateCoverageResult(
            n_rows=n_rows,
            n_unique_dates=0,
            n_duplicate_date_rows=0,
            duplicate_dates=[],
            min_date=None,
            max_date=None,
            missing_dates=[],
            n_missing_dates=0,
            is_sorted_ascending=True,
        )

    date_only = valid.dt.normalize()
    counts = date_only.value_counts()
    dup_mask = counts > 1
    duplicate_dates = sorted(d.date().isoformat() for d in counts[dup_mask].index)
    n_duplicate_date_rows = int((counts[dup_mask] - 1).sum())

    min_ts = date_only.min()
    max_ts = date_only.max()
    expected = pd.date_range(min_ts, max_ts, freq="D")
    present = set(date_only.dt.date)
    missing = [d.date().isoformat() for d in expected if d.date() not in present]

    # Ordering: compare non-null sequence as recorded (including duplicates).
    non_null = parsed.dropna()
    is_sorted = bool(non_null.is_monotonic_increasing)

    return DateCoverageResult(
        n_rows=n_rows,
        n_unique_dates=int(date_only.nunique()),
        n_duplicate_date_rows=n_duplicate_date_rows,
        duplicate_dates=duplicate_dates,
        min_date=min_ts.date().isoformat(),
        max_date=max_ts.date().isoformat(),
        missing_dates=missing,
        n_missing_dates=len(missing),
        is_sorted_ascending=is_sorted,
    )


def missing_measurement_mask(series: pd.Series) -> pd.Series:
    """True where a value is missing (NA) or non-finite for numeric series."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        return values.isna() | ~np.isfinite(values.to_numpy(dtype=float, copy=False))
    return series.isna() | (series.astype(str).str.strip() == "")


def longest_consecutive_missing_runs(
    mask: pd.Series | np.ndarray[Any, Any],
    *,
    top_n: int = 5,
) -> list[MissingRun]:
    """Return the longest consecutive True runs in a boolean missing mask."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []

    runs: list[MissingRun] = []
    start: int | None = None
    for idx, flag in enumerate(arr):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append(MissingRun(start_index=start, end_index=idx - 1, length=idx - start))
            start = None
    if start is not None:
        runs.append(MissingRun(start_index=start, end_index=len(arr) - 1, length=len(arr) - start))

    runs.sort(key=lambda r: (-r.length, r.start_index))
    return runs[:top_n]


def consecutive_flag_runs(flags: pd.Series, *, positive_value: float = 1.0) -> list[MissingRun]:
    """Find consecutive runs where a flag column equals ``positive_value``."""
    numeric = pd.to_numeric(flags, errors="coerce")
    mask = numeric == positive_value
    return longest_consecutive_missing_runs(mask, top_n=50)


def schema_fingerprint(columns: list[str]) -> tuple[str, ...]:
    """Stable schema key from ordered column names."""
    return tuple(columns)


@dataclass(frozen=True)
class _SchemaGroup:
    n_columns: int
    series: list[str]
    columns: list[str]


def compare_schemas(
    schemas: dict[str, list[str]],
) -> dict[str, Any]:
    """Summarise schema differences across series."""
    fingerprints: dict[tuple[str, ...], list[str]] = {}
    for series_id, cols in schemas.items():
        fingerprints.setdefault(schema_fingerprint(cols), []).append(series_id)

    groups: list[_SchemaGroup] = [
        _SchemaGroup(n_columns=len(fp), series=sorted(members), columns=list(fp))
        for fp, members in fingerprints.items()
    ]
    groups.sort(key=lambda g: (-g.n_columns, g.series[0]))

    all_cols: set[str] = set()
    for cols in schemas.values():
        all_cols.update(cols)
    shared = sorted(all_cols.intersection(*map(set, schemas.values()))) if schemas else []
    union = sorted(all_cols)
    only_in: dict[str, dict[str, list[str]]] = {}
    for series_id, cols in schemas.items():
        others = all_cols - set(cols)
        missing_here = sorted(
            c for c in others if any(c in schemas[o] for o in schemas if o != series_id)
        )
        peer_union: set[str] = set()
        for other_id, other_cols in schemas.items():
            if other_id != series_id:
                peer_union.update(other_cols)
        only_here = sorted(set(cols) - peer_union) if len(schemas) > 1 else []
        if missing_here or only_here:
            only_in[series_id] = {
                "missing_vs_union": missing_here,
                "extra_vs_intersection_peers": only_here,
            }

    return {
        "n_distinct_schemas": len(groups),
        "schema_groups": [
            {
                "n_columns": g.n_columns,
                "series": g.series,
                "sample_columns_head": g.columns[:20],
            }
            for g in groups
        ],
        "shared_columns_all_series": shared,
        "n_shared_columns_all_series": len(shared),
        "n_union_columns": len(union),
        "per_series_column_deltas": only_in,
    }


def check_s3sf_relationship(
    frame: pd.DataFrame,
    *,
    s3sf_col: str = "s3sf",
    flux_col: str = "stage_3_flux",
    tcf_col: str = "tcf_water",
    ndp_col: str = "stage_3_net_driving_press",
    date_col: str = "date",
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Check observed relationship: s3sf ≈ stage_3_flux * tcf_water / stage_3_net_driving_press.

    Reports absolute and relative errors. Discrepancies are not described as
    negligible from absolute magnitude alone. This is an empirical residual
    report, not proof of a physical formula.
    """
    required = [s3sf_col, flux_col, tcf_col, ndp_col]
    missing_cols = [c for c in required if c not in frame.columns]
    if missing_cols:
        return {
            "checked": False,
            "missing_columns": missing_cols,
            "note": "Required columns absent; relationship not evaluated.",
            "rtol": rtol,
            "atol": atol,
        }

    s3sf = pd.to_numeric(frame[s3sf_col], errors="coerce")
    flux = pd.to_numeric(frame[flux_col], errors="coerce")
    tcf = pd.to_numeric(frame[tcf_col], errors="coerce")
    ndp = pd.to_numeric(frame[ndp_col], errors="coerce")
    dates = (
        parse_dates(frame[date_col])
        if date_col in frame.columns
        else pd.Series([pd.NaT] * len(frame))
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = flux * tcf / ndp
    residual = s3sf - predicted
    s3sf_arr = s3sf.to_numpy(dtype=float, copy=False)
    pred_arr = predicted.to_numpy(dtype=float, copy=False)
    res_arr = residual.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(s3sf_arr) & np.isfinite(pred_arr) & (ndp.to_numpy(dtype=float) != 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "checked": True,
            "n_valid": 0,
            "note": "No valid rows to evaluate relationship.",
            "rtol": rtol,
            "atol": atol,
        }

    abs_err = np.abs(res_arr[valid])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(s3sf_arr[valid]) > 0,
            abs_err / np.abs(s3sf_arr[valid]),
            np.nan,
        )
    close_ok = np.isclose(
        s3sf_arr[valid],
        pred_arr[valid],
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
    exception_pos = np.flatnonzero(valid)[~close_ok]
    exceptions: list[dict[str, Any]] = []
    for idx in exception_pos:
        obs = float(s3sf_arr[idx])
        calc = float(pred_arr[idx])
        aerr = abs(obs - calc)
        rerr = (aerr / abs(obs)) if obs != 0 else None
        date_val = dates.iloc[int(idx)]
        exceptions.append(
            {
                "row_index": int(idx),
                "date": (date_val.date().isoformat() if pd.notna(date_val) else None),
                "observed_s3sf": obs,
                "calculated_value": calc,
                "absolute_error": aerr,
                "relative_error": rerr,
                "rtol": rtol,
                "atol": atol,
            }
        )

    finite_rel = rel_err[np.isfinite(rel_err)]
    return {
        "checked": True,
        "formula_observed": (f"{s3sf_col} ≈ {flux_col} * {tcf_col} / {ndp_col}"),
        "interpretation": (
            "Observed numerical relationship only; not proof of physical "
            "formula or correction convention. Absolute and relative errors "
            "are both reported; absolute magnitude alone does not justify "
            "calling discrepancies negligible."
        ),
        "n_valid": n_valid,
        "n_exceptions": len(exceptions),
        "rtol": rtol,
        "atol": atol,
        "absolute_error_mean": float(np.mean(abs_err)),
        "absolute_error_median": float(np.median(abs_err)),
        "absolute_error_max": float(np.max(abs_err)),
        "absolute_error_p99": float(np.quantile(abs_err, 0.99)),
        "relative_error_mean": float(np.mean(finite_rel)) if finite_rel.size else None,
        "relative_error_median": float(np.median(finite_rel)) if finite_rel.size else None,
        "relative_error_max": float(np.max(finite_rel)) if finite_rel.size else None,
        "relative_error_p99": (float(np.quantile(finite_rel, 0.99)) if finite_rel.size else None),
        "exceptions": exceptions,
    }


def dss_resets(dss: pd.Series) -> dict[str, Any]:
    """Describe changes where dss drops (candidate resets). Do not infer clean reference."""
    numeric = pd.to_numeric(dss, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=False)
    resets: list[dict[str, Any]] = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        if cur < prev:
            resets.append(
                {
                    "index": i,
                    "previous": float(prev),
                    "value": float(cur),
                    "drop": float(prev - cur),
                }
            )
    zeros = int(np.nansum(values == 0))
    return {
        "n_resets": len(resets),
        "n_zero_values": zeros,
        "resets_sample": resets[:20],
        "caveat": (
            "dss drops are reported as observed. dss=0 is not interpreted "
            "as establishing a clean reference in Phase 1."
        ),
    }


def days_since_replacement_changes(series: pd.Series) -> dict[str, Any]:
    """Summarise changes in days_since_replacement."""
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=False)
    changes: list[dict[str, Any]] = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        delta = cur - prev
        # Expected daily increment is +1; flag other changes.
        if abs(delta - 1.0) > 1e-9:
            changes.append(
                {
                    "index": i,
                    "previous": float(prev),
                    "value": float(cur),
                    "delta": float(delta),
                }
            )
    return {
        "n_non_unit_increments": len(changes),
        "changes_sample": changes[:30],
        "min": float(np.nanmin(values)) if np.isfinite(values).any() else None,
        "max": float(np.nanmax(values)) if np.isfinite(values).any() else None,
    }


def column_numeric_profile(series: pd.Series) -> dict[str, Any]:
    """Profile a column for emptiness, constancy, non-finite values, and range."""
    n = int(len(series))
    miss = missing_measurement_mask(series)
    n_missing = int(miss.sum())
    non_missing = series.loc[~miss]
    profile: dict[str, Any] = {
        "n": n,
        "n_missing": n_missing,
        "missing_fraction": float(n_missing / n) if n else 0.0,
        "fully_empty": n > 0 and n_missing == n,
        "n_unique_non_missing": int(non_missing.nunique(dropna=True)),
        "constant": bool(non_missing.nunique(dropna=True) <= 1) and n_missing < n,
        "dtype": str(series.dtype),
    }

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        arr = numeric.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(arr)
        profile.update(
            {
                "n_non_finite": int((~finite & ~np.isnan(arr)).sum()),
                "min": float(np.nanmin(arr)) if finite.any() else None,
                "max": float(np.nanmax(arr)) if finite.any() else None,
                "mean": float(np.nanmean(arr)) if finite.any() else None,
            }
        )
        longest = longest_consecutive_missing_runs(miss, top_n=1)
        profile["longest_missing_run"] = longest[0].length if longest else 0
        profile["longest_missing_run_start"] = longest[0].start_index if longest else None
    else:
        longest = longest_consecutive_missing_runs(miss, top_n=1)
        profile["longest_missing_run"] = longest[0].length if longest else 0
        profile["longest_missing_run_start"] = longest[0].start_index if longest else None
        profile["n_non_finite"] = 0

    return profile


def calendar_span_days(min_date: str | None, max_date: str | None) -> int | None:
    """Inclusive day count between ISO dates."""
    if not min_date or not max_date:
        return None
    start = date.fromisoformat(min_date)
    end = date.fromisoformat(max_date)
    return (end - start).days + 1


def expected_daily_count(min_date: str | None, max_date: str | None) -> int | None:
    """Expected number of daily records in an inclusive span."""
    span = calendar_span_days(min_date, max_date)
    return span
