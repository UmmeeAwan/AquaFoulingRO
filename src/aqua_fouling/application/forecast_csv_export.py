"""Export forecast results as downloadable CSV (presentation only).

Built from the same JSON forecast payloads produced by ``forecast_csv`` /
``forecast_batch_csv``. Does not re-run forecasting or normalisation.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from typing import Any

FORECAST_CSV_COLUMNS: tuple[str, ...] = (
    "source_series_id",
    "status",
    "forecast_origin",
    "forecast_date",
    "days_ahead",
    "relative_health",
    "latent_health_low",
    "latent_health_high",
    "future_observation_low",
    "future_observation_high",
    "threshold",
    "crossing_status",
    "days_to_threshold",
    "model_version",
    "normalisation_version",
    "error_code",
    "error_message",
)

# Text / identifier cells that may need spreadsheet formula-injection guards.
_TEXT_COLUMNS = frozenset(
    {
        "source_series_id",
        "status",
        "forecast_origin",
        "forecast_date",
        "crossing_status",
        "model_version",
        "normalisation_version",
        "error_code",
        "error_message",
    }
)

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _iso_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return ""
    # Already ISO-like calendar date
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text


def _guard_text(value: Any) -> str:
    """Escape spreadsheet formula injection for user-controlled text cells.

    Numeric cells are never passed through this helper. A leading apostrophe is
    prepended when the first character is a spreadsheet formula marker
    (=, +, -, @, tab, CR). Legitimate negative *numbers* are written via
    ``_num_cell`` and are not altered.
    """
    if value is None:
        return ""
    text = str(value)
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def _num_cell(value: Any) -> str:
    """Serialize a numeric value; empty cell for unavailable (never 'null' or coerced 0)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        # Preserve precision without forcing fixed decimals.
        return format(value, ".16g")
    # Reject accidental string "null"
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return ""
    try:
        num = float(text)
    except ValueError:
        return ""
    if num != num:
        return ""
    if float(int(num)) == num and abs(num) < 1e15:
        return str(int(num))
    return format(num, ".16g")


def _days_ahead(origin: str, forecast_date: str) -> str:
    if not origin or not forecast_date:
        return ""
    try:
        o = date.fromisoformat(origin[:10])
        f = date.fromisoformat(forecast_date[:10])
    except ValueError:
        return ""
    return str((f - o).days)


def _empty_row() -> dict[str, str]:
    return {c: "" for c in FORECAST_CSV_COLUMNS}


def _fill_common(
    row: dict[str, str],
    *,
    source_series_id: str,
    status: str,
    origin: str = "",
    threshold: Any = None,
    crossing_status: str = "",
    days_to_threshold: Any = None,
    model_version: str = "",
    normalisation_version: str = "",
    error_code: str = "",
    error_message: str = "",
) -> dict[str, str]:
    row["source_series_id"] = _guard_text(source_series_id)
    row["status"] = _guard_text(status)
    row["forecast_origin"] = _guard_text(origin)
    row["threshold"] = _num_cell(threshold)
    row["crossing_status"] = _guard_text(crossing_status)
    # Preserve explicit 0 (crossed_at_origin); empty for None / unavailable.
    if days_to_threshold is None:
        row["days_to_threshold"] = ""
    else:
        row["days_to_threshold"] = _num_cell(days_to_threshold)
    row["model_version"] = _guard_text(model_version)
    row["normalisation_version"] = _guard_text(normalisation_version)
    row["error_code"] = _guard_text(error_code)
    row["error_message"] = _guard_text(error_message)
    return row


def _meta_versions(body: dict[str, Any]) -> tuple[str, str]:
    meta = body.get("metadata") or {}
    model = meta.get("model") or {}
    norm = meta.get("normalisation") or {}
    return (
        str(model.get("model_version") or ""),
        str(norm.get("version") or ""),
    )


def rows_from_success_forecast(body: dict[str, Any]) -> list[dict[str, str]]:
    """One CSV row per forecast date from a successful (or unavailable) series payload."""
    status = str(body.get("status") or "ok")
    res = body.get("result") or {}
    sid = str(res.get("source_series_id") or body.get("source_series_id") or "")
    origin = _iso_date((res.get("forecast_origin") or {}).get("date"))
    thr = (res.get("threshold") or {}).get("relative_health")
    crossing = str(res.get("crossing_status") or "")
    dtt = res.get("days_to_threshold")
    model_v, norm_v = _meta_versions(body)
    points = list(res.get("forecast") or [])
    if not points:
        # No daily points: still emit one distinguishable context row.
        row = _empty_row()
        _fill_common(
            row,
            source_series_id=sid,
            status=status,
            origin=origin,
            threshold=thr,
            crossing_status=crossing,
            days_to_threshold=dtt,
            model_version=model_v,
            normalisation_version=norm_v,
        )
        return [row]

    out: list[dict[str, str]] = []
    for point in points:
        row = _empty_row()
        fdate = _iso_date(point.get("date"))
        latent = point.get("latent_health_interval") or {}
        future = point.get("future_observation_interval") or {}
        _fill_common(
            row,
            source_series_id=sid,
            status=status,
            origin=origin,
            threshold=thr,
            crossing_status=crossing,
            days_to_threshold=dtt,
            model_version=model_v,
            normalisation_version=norm_v,
        )
        row["forecast_date"] = _guard_text(fdate)
        row["days_ahead"] = _days_ahead(origin, fdate)
        row["relative_health"] = _num_cell(point.get("relative_health_point"))
        row["latent_health_low"] = _num_cell(latent.get("low"))
        row["latent_health_high"] = _num_cell(latent.get("high"))
        row["future_observation_low"] = _num_cell(future.get("low"))
        row["future_observation_high"] = _num_cell(future.get("high"))
        out.append(row)
    return out


def rows_from_failure_series(item: dict[str, Any]) -> list[dict[str, str]]:
    """One explicit failure row (empty forecast numbers/dates)."""
    status = str(item.get("status") or "failed")
    sid = str(item.get("source_series_id") or "")
    err = item.get("error") or {}
    if not isinstance(err, dict):
        err = {}
    code = str(err.get("code") or item.get("code") or status)
    message = str(err.get("message") or item.get("message") or "")
    row = _empty_row()
    _fill_common(
        row,
        source_series_id=sid,
        status=status,
        error_code=code,
        error_message=message,
    )
    return [row]


def rows_from_batch_item(item: dict[str, Any]) -> list[dict[str, str]]:
    status = str(item.get("status") or "failed")
    if status in {"ok", "unavailable"} and item.get("result") is not None:
        # Batch success items already nest result/metadata at top level.
        return rows_from_success_forecast(item)
    return rows_from_failure_series(item)


def render_forecast_csv(rows: list[dict[str, str]]) -> str:
    """Serialize rows with csv.writer (RFC-aware quoting)."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(
        buf,
        fieldnames=list(FORECAST_CSV_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        safe = {c: row.get(c, "") for c in FORECAST_CSV_COLUMNS}
        # Apply text guards for text columns only (already applied in builders,
        # but ensure empty keys exist).
        for col in _TEXT_COLUMNS:
            if col in safe and safe[col] is not None:
                # Values already guarded when set; leave as-is.
                pass
        writer.writerow(safe)
    return buf.getvalue()


def sort_forecast_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(r: dict[str, str]) -> tuple[str, str]:
        return (r.get("source_series_id") or "", r.get("forecast_date") or "")

    return sorted(rows, key=key)


def export_single_forecast_csv(body: dict[str, Any]) -> str:
    status = str(body.get("status") or "")
    if status in {"ok", "unavailable"}:
        rows = rows_from_success_forecast(body)
    else:
        rows = rows_from_failure_series(body)
    return render_forecast_csv(sort_forecast_rows(rows))


def export_batch_forecast_csv(body: dict[str, Any]) -> str:
    rows: list[dict[str, str]] = []
    for item in body.get("results") or []:
        if not isinstance(item, dict):
            continue
        rows.extend(rows_from_batch_item(item))
    return render_forecast_csv(sort_forecast_rows(rows))
