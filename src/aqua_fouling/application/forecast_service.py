"""Shared CSV → normalise → forecast application service (HTTP + CLI)."""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from aqua_fouling.application.csv_first import (
    SERIES_ID_COLUMNS,
    adapter_from_mapping_json,
    infer_series_id,
    reference_required_payload,
    resolved_fields_from_adapter,
    try_resolve_alias_mapping,
    write_adapter_yaml,
)
from aqua_fouling.application.forecast_errors import (
    ForecastArtifactError,
    ForecastPayloadTooLarge,
    ForecastServiceError,
    ForecastUnavailableError,
    ForecastValidationError,
)
from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.application.response_presentation import (
    INTERVAL_DEFINITIONS,
    concise_threshold_provenance,
    forecast_point_public,
    public_reference,
)
from aqua_fouling.data.adapters import (
    AdapterConfigError,
    MeasurementScope,
    adapt_dataframe,
    load_adapter_config,
    resolve_adapter,
    resolve_configs_dir,
)
from aqua_fouling.forecasting.artifact import ArtifactCompatibilityError, load_artifact
from aqua_fouling.forecasting.evaluation import build_segments_for_series
from aqua_fouling.forecasting.interface import ForecastRequest
from aqua_fouling.forecasting.trained_llt import TrainedLLTForecaster
from aqua_fouling.physics.normalisation import NORMALISATION_METHOD_ID, NORMALISATION_VERSION


def load_serving_defaults(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (resolve_configs_dir() / "forecast_serving.yaml")
    if not cfg_path.is_file():
        return {}
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def resolve_artifact_path(explicit: Path | None = None) -> Path:
    """Resolve artifact from explicit arg / Settings — never from HTTP callers."""
    if explicit is not None:
        return explicit
    from aqua_fouling.settings import get_settings

    return Path(get_settings().artifact_path)


def check_artifact_ready(artifact_path: Path | None = None) -> dict[str, Any]:
    """Readiness probe payload (no filesystem paths in the public response)."""
    path = resolve_artifact_path(artifact_path)
    try:
        art = load_artifact(path)
        return {
            "ready": True,
            "model_version": art.model_version,
            "model_type": art.model_type,
            "normalisation_method_id": art.normalisation_method_id,
            "normalisation_version": art.normalisation_version,
        }
    except (ArtifactCompatibilityError, OSError) as exc:
        return {
            "ready": False,
            "model_version": None,
            "reason": "Configured model artifact is missing or incompatible.",
            "details": {"error_type": type(exc).__name__},
        }


def _validate_limits(
    *,
    horizon_days: int,
    threshold: float,
    n_rows: int,
    upload_bytes: int | None,
    settings: Any,
) -> None:
    if upload_bytes is not None and upload_bytes > int(settings.max_upload_bytes):
        raise ForecastPayloadTooLarge(
            f"CSV upload exceeds max_upload_bytes={settings.max_upload_bytes}.",
            actionable_correction=(
                "Reduce the CSV size or raise AQUA_MAX_UPLOAD_BYTES on the server "
                "(requires restart)."
            ),
            details={
                "upload_bytes": upload_bytes,
                "max_upload_bytes": settings.max_upload_bytes,
            },
        )
    if n_rows > int(settings.max_csv_rows):
        raise ForecastPayloadTooLarge(
            f"CSV has {n_rows} rows; max_csv_rows={settings.max_csv_rows}.",
            actionable_correction=(
                "Trim history or raise AQUA_MAX_CSV_ROWS. Oversized requests are "
                "rejected rather than silently truncated."
            ),
            details={"n_rows": n_rows, "max_csv_rows": settings.max_csv_rows},
        )
    if horizon_days < int(settings.min_horizon_days) or horizon_days > int(
        settings.max_horizon_days
    ):
        raise ForecastValidationError(
            f"horizon_days={horizon_days} outside documented limits "
            f"[{settings.min_horizon_days}, {settings.max_horizon_days}].",
            actionable_correction=(
                f"Use a horizon in [{settings.min_horizon_days}, "
                f"{settings.max_horizon_days}] "
                f"(allowed set: {list(settings.allowed_horizons_days)})."
            ),
            code="invalid_horizon",
            details={"horizon_days": horizon_days},
        )
    allowed = {int(x) for x in settings.allowed_horizons_days}
    if allowed and horizon_days not in allowed:
        raise ForecastValidationError(
            f"horizon_days={horizon_days} is not in the documented allowed set "
            f"{sorted(allowed)}.",
            actionable_correction=f"Choose one of {sorted(allowed)}.",
            code="unsupported_horizon",
            details={"horizon_days": horizon_days, "allowed": sorted(allowed)},
        )
    lo = float(settings.min_threshold_relative_health)
    hi = float(settings.max_threshold_relative_health)
    if not (lo <= float(threshold) <= hi):
        raise ForecastValidationError(
            f"threshold={threshold} outside documented relative-health limits " f"[{lo}, {hi}].",
            actionable_correction=f"Supply a threshold in [{lo}, {hi}].",
            code="invalid_threshold",
            details={"threshold": threshold},
        )


def _sort_and_check_dates(
    frame: pd.DataFrame,
    date_col: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    parsed = pd.to_datetime(frame[date_col], errors="coerce")
    if bool(parsed.isna().any()):
        bad = frame.index[parsed.isna()].tolist()[:10]
        raise ForecastValidationError(
            "One or more observation dates are missing or unparseable.",
            actionable_correction="Provide ISO8601 calendar dates (YYYY-MM-DD) for every row.",
            code="invalid_dates",
            details={"bad_row_indices": bad},
        )
    if bool(parsed.duplicated().any()):
        dups = parsed[parsed.duplicated(keep=False)].dt.strftime("%Y-%m-%d").unique().tolist()[:10]
        raise ForecastValidationError(
            f"Duplicate observation dates are not allowed: {dups}.",
            actionable_correction="Deduplicate rows so each calendar date appears once.",
            code="duplicate_dates",
            details={"duplicate_dates": dups},
        )
    if not bool(parsed.is_monotonic_increasing):
        order = np.argsort(parsed.to_numpy())
        frame = frame.iloc[order].reset_index(drop=True)
        warnings.append(
            {
                "code": "timestamps_sorted",
                "message": (
                    "Input timestamps were not strictly ascending; rows were sorted "
                    "deterministically by calendar date before forecasting."
                ),
                "actionable_correction": (
                    "Prefer pre-sorted CSVs; sorted order was disclosed in warnings."
                ),
            }
        )
    return frame, warnings


def forecast_csv(
    csv_path: Path,
    *,
    source_series_id: str | None = None,
    mapping_path: Path | None = None,
    mapping_json: dict[str, Any] | str | None = None,
    reference_path: Path | None = None,
    horizon_days: int | None = None,
    threshold: float | None = None,
    artifact_path: Path | None = None,
    concentration_mg_l: float | None = None,
    upload_bytes: int | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """End-to-end forecast from a plant operating-history CSV (CSV-first).

    HTTP callers may supply only a CSV. Series id, mapping and reference are
    optional: series id is inferred, mapping may come from aliases /
    ``mapping_json`` / YAML, and a missing documented reference yields a
    structured ``reference_required`` payload (never an invented clean baseline).

    Artifact path comes from server/application configuration (or an explicit
    CLI/server override) — never from an HTTP caller filesystem path.
    """
    from aqua_fouling.settings import get_settings

    settings = get_settings()
    serving = load_serving_defaults()
    h = int(horizon_days if horizon_days is not None else settings.default_horizon_days)
    thr = float(threshold if threshold is not None else settings.default_threshold_relative_health)
    temp_sorted: Path | None = None
    temp_mapping: Path | None = None

    try:
        if not csv_path.is_file():
            raise ForecastValidationError(
                f"CSV file not found: {csv_path}",
                actionable_correction="Provide an existing plant operating-history CSV.",
                code="csv_not_found",
            )
        if reference_path is not None and not reference_path.is_file():
            raise ForecastValidationError(
                f"Reference file not found: {reference_path}",
                actionable_correction="Provide a readable reference YAML path/upload.",
                code="invalid_reference",
            )
        if mapping_path is not None and not mapping_path.is_file():
            raise ForecastValidationError(
                f"Mapping file not found: {mapping_path}",
                actionable_correction="Provide a readable adapter mapping YAML.",
                code="invalid_mapping",
            )
        if mapping_path is not None and mapping_json is not None:
            raise ForecastValidationError(
                "Provide either mapping YAML or mapping_json, not both.",
                code="conflicting_mapping",
                actionable_correction="Omit one of mapping / mapping_json.",
            )

        try:
            header = pd.read_csv(csv_path, nrows=0)
            series_dtypes = {c: str for c in SERIES_ID_COLUMNS if c in header.columns}
            raw = pd.read_csv(
                csv_path,
                dtype=cast(Any, series_dtypes) if series_dtypes else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ForecastValidationError(
                f"Failed to parse CSV: {exc}",
                actionable_correction=("Supply a readable comma-separated CSV with a header row."),
                code="malformed_csv",
            ) from exc

        if raw.empty:
            raise ForecastValidationError(
                "CSV is empty (no data rows).",
                actionable_correction="Include at least one observation row.",
                code="empty_csv",
            )

        _validate_limits(
            horizon_days=h,
            threshold=thr,
            n_rows=len(raw),
            upload_bytes=upload_bytes,
            settings=settings,
        )

        series = infer_series_id(raw, explicit=source_series_id)
        source_series_id = series.source_series_id

        mapping_source = "alias"
        resolved_mapping_fields: list[Any]
        try:
            if mapping_json is not None:
                adapter_for_ts = adapter_from_mapping_json(mapping_json)
                mapping_source = "mapping_json"
                resolved_mapping_fields = resolved_fields_from_adapter(
                    adapter_for_ts, source="mapping_json"
                )
                tmp_fd, tmp_name = tempfile.mkstemp(suffix=".mapping.yaml")
                os.close(tmp_fd)
                temp_mapping = Path(tmp_name)
                write_adapter_yaml(adapter_for_ts, temp_mapping)
                map_path: Path | None = temp_mapping
            elif mapping_path is not None:
                adapter_for_ts = load_adapter_config(mapping_path)
                mapping_source = "mapping_yaml"
                resolved_mapping_fields = resolved_fields_from_adapter(
                    adapter_for_ts, source="mapping_yaml"
                )
                map_path = mapping_path
            else:
                # Prefer exact match to the documented default schema file when
                # headers align; otherwise deterministic aliases (unit in name).
                default_map = Path(settings.default_mapping)
                if not default_map.is_file():
                    default_map = resolve_configs_dir(configs_dir) / "canonical_generic_schema.yaml"
                used_default = False
                if default_map.is_file():
                    try:
                        candidate = load_adapter_config(default_map)
                        needed = {
                            f.external_name
                            for f in candidate.fields
                            if f.canonical_name
                            in {
                                "observation_date_calendar",
                                "temperature_c",
                                "stage_3_feed_pressure_psi",
                                "permeate_pressure_psi",
                                "stage_3_device_pressure_drop_psi",
                                "stage_3_permeate_flow_source_unit",
                                "stage_3_feed_flow_source_unit",
                                "feed_concentration_mg_l",
                            }
                        }
                        if needed.issubset(set(map(str, raw.columns))):
                            adapter_for_ts = candidate
                            map_path = default_map
                            mapping_source = "default_schema_yaml"
                            resolved_mapping_fields = resolved_fields_from_adapter(
                                adapter_for_ts, source="default_schema_yaml"
                            )
                            used_default = True
                    except AdapterConfigError:
                        used_default = False
                if not used_default:
                    alias_result = try_resolve_alias_mapping(
                        [str(c) for c in raw.columns],
                        series=series,
                    )
                    if isinstance(alias_result, dict):
                        return dict(json_safe(alias_result))
                    adapter_for_ts = alias_result.adapter
                    resolved_mapping_fields = alias_result.matched
                    mapping_source = "alias"
                    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".alias.yaml")
                    os.close(tmp_fd)
                    temp_mapping = Path(tmp_name)
                    write_adapter_yaml(adapter_for_ts, temp_mapping)
                    map_path = temp_mapping
        except AdapterConfigError as exc:
            raise ForecastValidationError(
                str(exc),
                code="malformed_mapping",
                actionable_correction=(
                    "Supply a valid adapter mapping YAML with name, schema_id, "
                    "source_series_pattern, timestamp.column, and fields."
                ),
                details={"mapping_path": str(mapping_path) if mapping_path else None},
            ) from exc

        if reference_path is None:
            return dict(
                json_safe(
                    reference_required_payload(
                        series=series,
                        resolved_mapping=resolved_mapping_fields,
                        mapping_source=mapping_source,
                    )
                )
            )

        sort_warnings: list[dict[str, Any]] = []
        date_col = adapter_for_ts.resolve_timestamp_external_column([str(c) for c in raw.columns])
        work_csv = csv_path
        if date_col is not None:
            raw, sort_warnings = _sort_and_check_dates(raw, date_col)
            temp_sorted = csv_path.with_name(csv_path.stem + ".forecast_sorted.csv")
            raw.to_csv(temp_sorted, index=False)
            work_csv = temp_sorted

        art_path = resolve_artifact_path(artifact_path)
        try:
            artifact = load_artifact(art_path)
        except ArtifactCompatibilityError as exc:
            raise ForecastArtifactError(
                str(exc),
                actionable_correction=(
                    "Configure AQUA_ARTIFACT_PATH to a compatible schema-validated "
                    "JSON artifact and restart the service."
                ),
                details={"artifact_path": str(art_path)},
            ) from exc

        art_bytes_before = art_path.read_bytes()

        try:
            norm = normalise_csv(
                work_csv,
                source_series_id=source_series_id,
                mapping_path=map_path,
                reference_path=reference_path,
                configs_dir=configs_dir,
                concentration_mg_l=concentration_mg_l,
            )
        except AdapterConfigError as exc:
            raise ForecastValidationError(
                str(exc),
                code="malformed_mapping",
                actionable_correction=(
                    "Supply a valid adapter mapping YAML with required keys and units."
                ),
            ) from exc
        except yaml.YAMLError as exc:
            raise ForecastValidationError(
                f"Malformed mapping or reference YAML: {exc}",
                code="malformed_yaml",
                actionable_correction=(
                    "Fix YAML syntax in the mapping or reference file "
                    "(indentation, colons, quotes)."
                ),
            ) from exc
        if norm.get("status") == "invalid":
            meta = norm.get("metadata") or {}
            val = norm.get("validation") or {}
            errors = norm.get("errors") or val.get("errors") or meta.get("errors") or []
            raise ForecastValidationError(
                "CSV failed validation/normalisation before forecasting.",
                actionable_correction=(
                    "Fix mapping units, dates, and required columns in details.errors."
                ),
                code="validation_failed",
                details={
                    "validation_status": meta.get("validation_status") or val.get("status"),
                    "errors": errors[:50],
                },
            )

        normalised_block = norm.get("normalised") or {}
        derived_root = normalised_block.get("derived") or {}
        derived = derived_root.get("independent_physical_normalisation") or {}
        if not derived.get("available"):
            blockers = derived.get("blockers") or []
            raise ForecastUnavailableError(
                "Independent physical normalisation is unavailable; cannot forecast "
                "relative health.",
                actionable_correction=(
                    "Ensure concentration and reference are supplied and units are resolved."
                ),
                details={"blockers": blockers[:20]},
            )

        health_raw = list(derived.get("relative_health") or [])
        n_meas = int(normalised_block.get("n_measurements") or len(health_raw))
        health_raw = (health_raw + [None] * n_meas)[:n_meas]

        work_header = pd.read_csv(work_csv, nrows=0)
        work_series_dtypes = {c: str for c in SERIES_ID_COLUMNS if c in work_header.columns}
        frame = pd.read_csv(
            work_csv,
            dtype=cast(Any, work_series_dtypes) if work_series_dtypes else None,
        )
        adapter = resolve_adapter(
            [str(c) for c in frame.columns],
            mapping_path=map_path,
            configs_dir=configs_dir,
        )
        canonical, _ = adapt_dataframe(
            frame,
            source_series_id=source_series_id,
            adapter=adapter,
            stage_focus=MeasurementScope.STAGE_3,
        )
        n = min(len(canonical), n_meas)
        if "observation_date_calendar" in canonical.columns:
            dates_full = [
                str(pd.Timestamp(d).date()) if pd.notna(d) else ""
                for d in pd.to_datetime(canonical["observation_date_calendar"], errors="coerce")
            ][:n]
        else:
            raise ForecastValidationError(
                "Mapped data lack observation_date_calendar after adaptation.",
                actionable_correction="Map a timestamp column to observation_date_calendar.",
                code="missing_dates",
            )

        cip_full = (
            pd.to_numeric(canonical["cip_flag_0_1"], errors="coerce").tolist()[:n]
            if "cip_flag_0_1" in canonical.columns
            else None
        )
        dsr_full = (
            pd.to_numeric(canonical["days_since_replacement_d"], errors="coerce").tolist()[:n]
            if "days_since_replacement_d" in canonical.columns
            else None
        )
        health_full: list[float | None] = []
        for v in health_raw[:n]:
            if v is None:
                health_full.append(None)
                continue
            try:
                fv = float(v)
                health_full.append(fv if np.isfinite(fv) else None)
            except (TypeError, ValueError):
                health_full.append(None)

        row_fail_block = norm.get("row_failures") or {}
        failures = {
            int(f["row"]): f
            for f in (row_fail_block.get("failures") or [])
            if isinstance(f, dict) and "row" in f
        }
        row_quality: list[dict[str, Any]] = []
        for i in range(n):
            rq: dict[str, Any] = {"finite_relative_health": health_full[i] is not None}
            if i in failures:
                rq["normalisation_failure"] = failures[i]
            row_quality.append(rq)

        date_s = pd.Series(dates_full)
        cip_s = pd.Series(cip_full) if cip_full is not None else None
        dsr_s = pd.Series(dsr_full) if dsr_full is not None else None
        segments, _audit, _seg_ids = build_segments_for_series(
            date_s, cip=cip_s, days_since_replacement=dsr_s
        )
        if not segments:
            raise ForecastUnavailableError(
                "No usable operational segment found in the normalised history.",
                actionable_correction="Provide contiguous dated observations within one series.",
            )

        last_seg = segments[-1]
        start_i = int(last_seg["start_index"])
        origin_i = int(last_seg["end_index"])
        origin_date = str(dates_full[origin_i])[:10]
        hist_dates = [str(d)[:10] for d in dates_full[start_i : origin_i + 1]]
        hist_values = list(health_full[start_i : origin_i + 1])
        finite_n = sum(1 for v in hist_values if v is not None)
        try:
            span_days = (
                date.fromisoformat(hist_dates[-1]) - date.fromisoformat(hist_dates[0])
            ).days + 1
        except ValueError:
            span_days = len(hist_dates)

        min_hist_days = int(settings.min_history_days)
        min_hist_obs = int(settings.min_history_observations)
        if span_days < min_hist_days or finite_n < min_hist_obs:
            raise ForecastUnavailableError(
                "Insufficient usable history in the latest operational segment " "for forecasting.",
                actionable_correction=(
                    f"Provide at least {min_hist_days} calendar days and "
                    f"{min_hist_obs} finite normalised observations after the most "
                    "recent CIP/gap/replacement boundary. Pre-cleaning history is "
                    "not borrowed."
                ),
                details={
                    "segment_id": len(segments) - 1,
                    "segment_start": hist_dates[0] if hist_dates else None,
                    "segment_end": hist_dates[-1] if hist_dates else None,
                    "calendar_span_days": span_days,
                    "finite_observations": finite_n,
                    "min_history_days": min_hist_days,
                    "min_history_observations": min_hist_obs,
                    "n_segments": len(segments),
                },
            )
        if hist_values[-1] is None:
            raise ForecastUnavailableError(
                "Relative health at the forecast origin is missing or non-finite.",
                actionable_correction="Ensure the latest observation normalises successfully.",
                details={"origin_date": origin_date},
            )

        # Fresh forecaster per request (frozen params; no cross-request filter state).
        forecaster = TrainedLLTForecaster(
            artifact,
            min_history_observations=min_hist_obs,
            artifact_path=art_path,
        )
        result = forecaster.forecast(
            ForecastRequest(
                origin_index=len(hist_values) - 1,
                origin_date=origin_date,
                dates=hist_dates,
                values=hist_values,
                horizon_days=h,
                threshold=thr,
                segment_id=len(segments) - 1,
                require_daily_spacing=True,
                metadata={"source_series_id": source_series_id},
            )
        )

        if art_path.read_bytes() != art_bytes_before:
            raise ForecastArtifactError(
                "Model artifact was modified during inference — refusing result.",
                actionable_correction="Restore the artifact file and restart the service.",
            )

        observations = []
        for i, (d, hval) in enumerate(zip(dates_full, health_full, strict=True)):
            observations.append(
                {
                    "row": i,
                    "date": str(d)[:10],
                    "relative_health": hval,
                    "unit": "relative_to_documented_reference_1",
                    "in_forecast_history_segment": bool(start_i <= i <= origin_i),
                    "row_quality": row_quality[i],
                }
            )

        extras = dict(result.extras or {})
        traj = result.trajectory
        tdates = result.trajectory_dates
        lat_lo = extras.get("latent_health_interval_low") or [None] * len(traj)
        lat_hi = extras.get("latent_health_interval_high") or [None] * len(traj)
        obs_lo = extras.get("observation_interval_low") or [None] * len(traj)
        obs_hi = extras.get("observation_interval_high") or [None] * len(traj)
        forecast_points = []
        for i, (fd, pv) in enumerate(zip(tdates, traj, strict=False)):
            forecast_points.append(
                forecast_point_public(
                    date=fd,
                    relative_health_point=pv,
                    latent_low=lat_lo[i] if i < len(lat_lo) else None,
                    latent_high=lat_hi[i] if i < len(lat_hi) else None,
                    obs_low=obs_lo[i] if i < len(obs_lo) else None,
                    obs_high=obs_hi[i] if i < len(obs_hi) else None,
                )
            )

        warnings = list(sort_warnings)
        warnings.append(
            {
                "code": "no_future_cleaning",
                "message": (
                    "Forecast assumes no future cleaning within the horizon; "
                    "no_crossing_in_horizon does not guarantee safety or that "
                    "cleaning will not be needed later."
                ),
                "actionable_correction": (
                    "Re-forecast after any cleaning using post-clean history only."
                ),
            }
        )
        if result.message:
            warnings.append(
                {
                    "code": "model_message",
                    "message": result.message,
                    "actionable_correction": "Review gap/history notes before operational use.",
                }
            )

        thr_prov = concise_threshold_provenance(
            settings.threshold_provenance
            or serving.get("threshold_provenance")
            or "Configured relative-health threshold versus the documented reference."
        )
        domain = (
            settings.domain_limits
            or serving.get("domain_limits")
            or "Brackish / NaCl-equivalent demonstration domain; not plant-validated."
        )
        gap_pol = (
            settings.gap_policy
            or serving.get("gap_policy")
            or "Calendar gaps and CIP/replacement resets split segments; "
            "forecasts use the latest applicable segment only."
        )
        norm_meta = norm.get("metadata") or {}
        ref_pub = public_reference(norm_meta.get("reference"))

        status = "unavailable" if result.crossing_status == "unavailable" else "ok"
        payload = {
            "status": status,
            "result": {
                "source_series_id": source_series_id,
                "forecast_origin": {
                    "date": origin_date,
                    "segment_id": len(segments) - 1,
                },
                "horizon_days": h,
                "threshold": {
                    "relative_health": thr,
                    "reference_basis": (
                        "Relative health versus the caller-supplied documented "
                        "clean reference (not an invented first-row or maximum-flow "
                        "baseline). Not a plant-validated cleaning rule."
                    ),
                    "provenance": thr_prov,
                },
                "crossing_status": result.crossing_status,
                "days_to_threshold": result.days_to_threshold,
                "state_at_origin": {
                    "relative_health": extras.get("state_health"),
                    "slope_per_calendar_day": extras.get("state_slope_per_day"),
                },
                "forecast": forecast_points,
            },
            "normalised_data": {
                "n_observations": len(observations),
                "observations": observations,
            },
            "metadata": {
                "model": {
                    "model_id": result.model_id,
                    "model_version": artifact.model_version,
                    "model_type": artifact.model_type,
                    "artifact_schema_version": artifact.artifact_schema_version,
                },
                "normalisation": {
                    "method_id": NORMALISATION_METHOD_ID,
                    "version": NORMALISATION_VERSION,
                    "status": norm.get("status"),
                },
                "reference": ref_pub,
                "mapping": {
                    "source": mapping_source,
                    "schema_id": norm_meta.get("schema_id") or norm.get("schema_id"),
                    "resolved": [r.__dict__ for r in resolved_mapping_fields],
                },
                "series_id_resolution": {
                    "source_series_id": series.source_series_id,
                    "source": series.source,
                    "column": series.column,
                },
                "history": {
                    "date_range": {
                        "start": hist_dates[0] if hist_dates else None,
                        "end": hist_dates[-1] if hist_dates else None,
                    },
                    "n_finite_observations": finite_n,
                    "calendar_span_days": span_days,
                    "segment_id": len(segments) - 1,
                    "n_segments": len(segments),
                },
                "interval_definitions": INTERVAL_DEFINITIONS,
                "gap_policy": gap_pol,
                "assumptions": [
                    "Relative health from the independent normalisation pipeline.",
                    "No future cleaning within the forecast horizon.",
                    "Gaussian intervals are model-based, not calibrated confidence.",
                    "Segments are not concatenated across CIP/gaps.",
                    "Segment prior mean health=1.0 (clean-reference scale); not retuned "
                    "to the first observation.",
                ],
                "scientific_scope": (
                    "Synthetic-trained LLT on a brackish/NaCl-equivalent demonstration "
                    "domain. Not plant-validated; no claim of OCWD transfer. "
                    f"{domain}"
                ),
                "days_to_threshold_meaning": (
                    "Calendar days from forecast origin until relative health reaches "
                    "the configured threshold, conditional on no future cleaning. "
                    "null means no crossing predicted in-horizon or timing unavailable; "
                    "0 means already at/below threshold at the origin. "
                    "Not a manufacturer cleaning date."
                ),
            },
            "warnings": warnings,
        }
        return dict(json_safe(payload))
    finally:
        if temp_sorted is not None:
            temp_sorted.unlink(missing_ok=True)
        if temp_mapping is not None:
            temp_mapping.unlink(missing_ok=True)


def _resolve_batch_series_column(
    columns: list[str],
    *,
    explicit: str | None,
) -> str:
    if explicit is not None and str(explicit).strip():
        col = str(explicit).strip()
        if col not in columns:
            raise ForecastValidationError(
                f"series_column={col!r} is not present in the CSV header.",
                actionable_correction=(
                    "Provide an existing column name, or omit series_column when "
                    "exactly one of source_series_id / series_id / unit_id is present."
                ),
                code="invalid_series_column",
                details={"series_column": col, "detected_headers": columns},
            )
        return col
    present = [c for c in SERIES_ID_COLUMNS if c in columns]
    if not present:
        raise ForecastValidationError(
            "Batch CSV requires a series identifier column.",
            actionable_correction=(
                "Add source_series_id, series_id, or unit_id; or pass series_column "
                "naming an existing column."
            ),
            code="series_column_required",
            details={"expected_any_of": list(SERIES_ID_COLUMNS)},
        )
    if len(present) > 1:
        raise ForecastValidationError(
            "Multiple series identifier columns present; specify series_column.",
            actionable_correction=f"Set series_column to one of: {present}.",
            code="ambiguous_series_column",
            details={"candidates": present},
        )
    return present[0]


def _batch_item_from_error(
    *,
    source_series_id: str,
    exc: ForecastServiceError,
) -> dict[str, Any]:
    from aqua_fouling.application.response_presentation import strip_pathish

    return {
        "status": "failed",
        "source_series_id": source_series_id,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "actionable_correction": exc.actionable_correction,
            "details": strip_pathish(exc.details),
        },
    }


def forecast_batch_csv(
    csv_path: Path,
    *,
    series_column: str | None = None,
    mapping_path: Path | None = None,
    mapping_json: dict[str, Any] | str | None = None,
    references: dict[str, dict[str, Any]] | None = None,
    horizon_days: int | None = None,
    threshold: float | None = None,
    artifact_path: Path | None = None,
    concentration_mg_l: float | None = None,
    upload_bytes: int | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Forecast multiple series from one CSV (shared mapping, per-series references)."""
    from aqua_fouling.settings import get_settings

    settings = get_settings()
    temp_paths: list[Path] = []
    try:
        if not csv_path.is_file():
            raise ForecastValidationError(
                f"CSV file not found: {csv_path}",
                actionable_correction="Provide an existing batch CSV upload.",
                code="csv_not_found",
            )
        if mapping_path is not None and mapping_json is not None:
            raise ForecastValidationError(
                "Provide either mapping YAML or mapping_json, not both.",
                code="conflicting_mapping",
                actionable_correction="Omit one of mapping / mapping_json.",
            )

        header = pd.read_csv(csv_path, nrows=0)
        series_col = _resolve_batch_series_column(
            [str(c) for c in header.columns],
            explicit=series_column,
        )
        dtype: dict[str, type[str]] = {series_col: str}
        try:
            frame = pd.read_csv(csv_path, dtype=cast(Any, dtype))
        except Exception as exc:  # noqa: BLE001
            raise ForecastValidationError(
                f"Failed to parse CSV: {exc}",
                actionable_correction=("Supply a readable comma-separated CSV with a header row."),
                code="malformed_csv",
            ) from exc

        if frame.empty:
            raise ForecastValidationError(
                "CSV is empty (no data rows).",
                actionable_correction="Include at least one observation row.",
                code="empty_csv",
            )

        h = int(horizon_days if horizon_days is not None else settings.default_horizon_days)
        thr = float(
            threshold if threshold is not None else settings.default_threshold_relative_health
        )
        _validate_limits(
            horizon_days=h,
            threshold=thr,
            n_rows=len(frame),
            upload_bytes=upload_bytes,
            settings=settings,
        )

        ids_raw = frame[series_col].astype(str).str.strip()
        blank_mask = ids_raw.eq("") | ids_raw.str.lower().eq("nan")
        if bool(blank_mask.any()):
            raise ForecastValidationError(
                "Blank series identifiers are not allowed in batch CSV.",
                actionable_correction=("Fill every row in the series column or remove blank rows."),
                code="blank_series_id",
                details={
                    "series_column": series_col,
                    "blank_row_indices": frame.index[blank_mask].tolist()[:20],
                },
            )

        series_ids = ids_raw.unique().tolist()
        if len(series_ids) > int(settings.max_batch_series):
            raise ForecastValidationError(
                f"Batch contains {len(series_ids)} series; "
                f"max_batch_series={settings.max_batch_series}.",
                actionable_correction=(
                    "Split the upload or raise AQUA_MAX_BATCH_SERIES on the server."
                ),
                code="batch_series_limit",
                details={
                    "n_series": len(series_ids),
                    "max_batch_series": settings.max_batch_series,
                },
            )

        refs = references or {}
        unknown_ref_keys = sorted(set(refs) - set(series_ids))
        results: list[dict[str, Any]] = []
        counts = {
            "ok": 0,
            "reference_required": 0,
            "mapping_required": 0,
            "failed": 0,
            "total": len(series_ids),
        }

        for sid in series_ids:
            group = frame.loc[ids_raw == sid].copy()
            sub_fd, sub_name = tempfile.mkstemp(suffix=".csv")
            os.close(sub_fd)
            sub_csv = Path(sub_name)
            group.to_csv(sub_csv, index=False)
            temp_paths.append(sub_csv)

            ref_path: Path | None = None
            if sid in refs:
                ref_fd, ref_name = tempfile.mkstemp(suffix=".reference.yaml")
                os.close(ref_fd)
                ref_path = Path(ref_name)
                temp_paths.append(ref_path)
                ref_path.write_text(
                    yaml.safe_dump(refs[sid], sort_keys=False),
                    encoding="utf-8",
                )

            try:
                item = forecast_csv(
                    sub_csv,
                    source_series_id=sid,
                    mapping_path=mapping_path,
                    mapping_json=mapping_json,
                    reference_path=ref_path,
                    horizon_days=h,
                    threshold=thr,
                    artifact_path=artifact_path,
                    concentration_mg_l=concentration_mg_l,
                    upload_bytes=None,
                    configs_dir=configs_dir,
                )
            except ForecastServiceError as exc:
                item = _batch_item_from_error(source_series_id=sid, exc=exc)
            st = str(item.get("status") or "failed")
            if st in {"ok", "unavailable"}:
                item = {
                    "status": st,
                    "source_series_id": sid,
                    "result": item.get("result"),
                    "normalised_data": item.get("normalised_data"),
                    "metadata": item.get("metadata"),
                    "warnings": item.get("warnings") or [],
                }
            else:
                item.setdefault("source_series_id", sid)
            results.append(dict(json_safe(item)))

            if st == "ok" or st == "unavailable":
                counts["ok"] += 1
            elif st == "reference_required":
                counts["reference_required"] += 1
            elif st == "mapping_required":
                counts["mapping_required"] += 1
            else:
                counts["failed"] += 1

        if counts["ok"] == counts["total"]:
            overall = "ok"
        elif counts["ok"] == 0:
            overall = "failed"
        else:
            overall = "partial_success"

        payload: dict[str, Any] = {
            "overall_status": overall,
            "summary": {
                "series_column": series_col,
                "counts": counts,
            },
            "results": results,
        }
        if unknown_ref_keys:
            payload["warnings"] = [
                {
                    "code": "unused_reference_keys",
                    "message": (
                        "References file contained keys with no matching series in the CSV."
                    ),
                    "details": {"unused_keys": unknown_ref_keys},
                }
            ]
        return dict(json_safe(payload))
    finally:
        for p in temp_paths:
            p.unlink(missing_ok=True)
