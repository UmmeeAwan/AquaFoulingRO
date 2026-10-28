"""CSV validation application service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.data.adapters import (
    MeasurementScope,
    adapt_dataframe,
    resolve_adapter,
)
from aqua_fouling.data.reference import ReferenceConfig


def _issue(
    *,
    field: str,
    problem: str,
    expected: str | None = None,
    actionable: str,
    row: int | None = None,
    date: str | None = None,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "field": field,
        "row": row,
        "date": date,
        "problem": problem,
        "expected_unit_or_requirement": expected,
        "actionable_correction": actionable,
    }


def validate_csv(
    csv_path: Path,
    *,
    source_series_id: str,
    mapping_path: Path | None = None,
    reference_path: Path | None = None,
    configs_dir: Path | None = None,
    stage_focus: MeasurementScope = MeasurementScope.STAGE_3,
    concentration_mg_l: float | None = None,
    allow_ocwd_autodetect: bool = True,
) -> dict[str, Any]:
    """Validate a source CSV against an adapter mapping and optional reference.

    When ``mapping_path`` is supplied it is used first; OCWD schema auto-detection
    is only attempted when no mapping is provided.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_for_calculations: list[dict[str, Any]] = []

    if not csv_path.is_file():
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "status": "invalid",
                    "errors": [
                        _issue(
                            field="csv",
                            problem=f"file not found: {csv_path}",
                            actionable="Provide an existing CSV path.",
                        )
                    ],
                    "warnings": [],
                    "missing_for_calculations": [],
                }
            ),
        )

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 — surface parse failures
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "status": "invalid",
                    "errors": [
                        _issue(
                            field="csv",
                            problem=f"failed to parse CSV: {exc}",
                            actionable="Supply a readable comma-separated CSV.",
                        )
                    ],
                    "warnings": [],
                    "missing_for_calculations": [],
                }
            ),
        )

    try:
        adapter = resolve_adapter(
            [str(c) for c in frame.columns],
            mapping_path=mapping_path,
            configs_dir=configs_dir,
            allow_ocwd_autodetect=allow_ocwd_autodetect,
        )
    except (ValueError, OSError) as exc:
        # AdapterConfigError is a ValueError subclass (malformed mapping YAML).
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "status": "invalid",
                    "errors": [
                        _issue(
                            field="schema",
                            problem=str(exc),
                            actionable=(
                                "Supply --mapping with a documented canonical schema YAML "
                                "(e.g. configs/canonical_generic_schema.yaml or "
                                "configs/canonical_synthetic_schema.yaml), or use an OCWD CSV."
                            ),
                        )
                    ],
                    "warnings": [],
                    "missing_for_calculations": [],
                }
            ),
        )

    canonical, adapt_warnings = adapt_dataframe(
        frame,
        source_series_id=source_series_id,
        adapter=adapter,
        stage_focus=stage_focus,
    )
    for w in adapt_warnings:
        sev = str(w.get("severity") or "warning")
        issue = _issue(
            field=str(w.get("field")),
            problem=str(w.get("problem")),
            expected=w.get("expected_unit"),
            actionable=str(w.get("actionable_correction")),
            severity=sev if sev in {"error", "warning"} else "warning",
        )
        if sev == "error":
            errors.append(issue)
        else:
            warnings.append(issue)
    for err in list(canonical.attrs.get("unit_errors") or []):
        errors.append(
            _issue(
                field=str(err.get("field")),
                problem=str(err.get("problem")),
                actionable=str(err.get("actionable_correction")),
                severity="error",
            )
        )

    # --- Timestamp / date validation ---
    if "observation_date_calendar" not in canonical.columns:
        errors.append(
            _issue(
                field="date",
                problem="timestamp column missing after mapping",
                expected="ISO8601 calendar date",
                actionable=(
                    "Map a timestamp column to observation_date_calendar in the adapter YAML."
                ),
            )
        )
        parsed = pd.Series([pd.NaT] * len(canonical))
    else:
        parsed = pd.to_datetime(canonical["observation_date_calendar"], errors="coerce")
        missing_ts = canonical["observation_date_calendar"].isna() | parsed.isna()
        for idx in canonical.index[missing_ts][:20]:
            errors.append(
                _issue(
                    field="observation_date_calendar",
                    problem="missing or unparseable timestamp",
                    expected="ISO8601 date (YYYY-MM-DD)",
                    actionable="Fix or remove the invalid date cell.",
                    row=int(idx),
                    date=str(canonical.at[idx, "observation_date_calendar"]),
                )
            )
        # Duplicate dates among parseable rows
        valid = parsed.dropna()
        dup_mask = valid.duplicated(keep=False)
        if bool(dup_mask.any()):
            for ts in valid[dup_mask].unique()[:10]:
                errors.append(
                    _issue(
                        field="observation_date_calendar",
                        problem=f"duplicate date {pd.Timestamp(ts).date().isoformat()}",
                        expected="unique calendar dates per series",
                        actionable="Deduplicate or correct conflicting rows for the same date.",
                        date=pd.Timestamp(ts).date().isoformat(),
                    )
                )
        # Calendar gaps (informational / warning — not auto-filled)
        if len(valid) >= 2:
            ordered = valid.sort_values()
            deltas = ordered.diff().dropna()
            gap_days = deltas[deltas > pd.Timedelta(days=1)]  # calendar day gaps
            if len(gap_days):
                warnings.append(
                    _issue(
                        field="observation_date_calendar",
                        problem=(
                            f"{len(gap_days)} calendar gap(s) > 1 day between min and max date "
                            f"(largest={int(gap_days.max().days)}d). Not interpolated."
                        ),
                        expected="daily contiguous series or explicit gap handling in forecasting",
                        actionable=(
                            "Confirm missing calendar dates are intentional; forecasting "
                            "eligibility treats gaps as segment boundaries / incomplete follow-up."
                        ),
                        severity="warning",
                    )
                )

    # --- Numeric field validation ---
    numeric_candidates = [
        c
        for c in canonical.columns
        if c
        not in {
            "source_series_id",
            "measurement_scope_focus",
            "schema_id",
            "observation_date_calendar",
        }
    ]
    for col in numeric_candidates:
        series = canonical[col]
        # Skip obvious non-numeric object columns that are not numeric-looking
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = series.notna()
        invalid = non_null & coerced.isna()
        if bool(invalid.any()):
            for idx in canonical.index[invalid][:10]:
                errors.append(
                    _issue(
                        field=col,
                        problem="non-numeric value in numeric field",
                        expected="finite number or empty/missing",
                        actionable=f"Correct row {idx} in column {col} to a number.",
                        row=int(idx),
                    )
                )

    # --- Unit metadata ---
    field_meta = list(canonical.attrs.get("field_metadata", []))
    for meta in field_meta:
        unit = str(meta.get("unit") or "").strip()
        if not unit:
            errors.append(
                _issue(
                    field=str(meta.get("canonical_name")),
                    problem="mapped field has empty unit",
                    expected="explicit unit symbol in adapter YAML",
                    actionable="Add unit.symbol for this field in the mapping file.",
                )
            )

    # --- Concentration for independent NDP ---
    has_conc_column = "feed_concentration_mg_l" in canonical.columns and bool(
        pd.to_numeric(canonical["feed_concentration_mg_l"], errors="coerce").notna().any()
    )
    if concentration_mg_l is not None or has_conc_column:
        # Measured/supplied concentration present — do not report missing concentration.
        if has_conc_column:
            n_miss = int(
                pd.to_numeric(canonical["feed_concentration_mg_l"], errors="coerce").isna().sum()
            )
            if n_miss:
                warnings.append(
                    _issue(
                        field="feed_concentration_mg_l",
                        problem=f"{n_miss} rows missing measured concentration",
                        expected="finite mg/L NaCl-equivalent where independent NDP is required",
                        actionable="Fill missing concentration rows or accept row-level blockers.",
                        severity="warning",
                    )
                )
    else:
        missing_for_calculations.append(
            {
                "calculation": "independent_physical_normalisation_ndp_from_concentration",
                "missing": [
                    "feed_concentration_mg_l_or_conductivity_calibration",
                ],
                "problem": (
                    "No measured feed_concentration_mg_l column and no concentration_mg_l "
                    "argument supplied."
                ),
                "actionable_correction": (
                    "Map feed_concentration_mg_l, pass concentration_mg_l, or supply an "
                    "explicit conductivity→concentration calibration. "
                    "Do not silently choose a universal coefficient."
                ),
            }
        )

    reference_info: dict[str, Any] | None = None
    if reference_path is not None:
        try:
            import yaml

            raw = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
            ref = ReferenceConfig.model_validate(raw)
            reference_info = ref.to_provenance_dict()
        except Exception as exc:  # noqa: BLE001
            errors.append(
                _issue(
                    field="reference",
                    problem=f"invalid reference configuration: {exc}",
                    actionable=(
                        "Provide documented reference values or an explicit window "
                        "with available_from after window_end."
                    ),
                )
            )
    else:
        missing_for_calculations.append(
            {
                "calculation": "reference_dependent_health",
                "missing": ["reference_configuration"],
                "problem": "No reference configuration supplied.",
                "actionable_correction": (
                    "Pass --reference with a documented or explicit_window YAML. "
                    "Never use first row, dataset max, or dss=0 automatically."
                ),
            }
        )

    if errors:
        status = "invalid"
    elif warnings or missing_for_calculations:
        status = "valid_with_warnings"
    else:
        status = "valid"

    sample_cols = [
        c
        for c in canonical.columns
        if c
        in {
            "source_series_id",
            "observation_date_calendar",
            "temperature_c",
            "stage_3_permeate_flow_source_unit",
            "stage_3_ndp_source_calculated",
            "stage_3_specific_flux_source_calculated",
            "tcf_water_source_calculated",
            "feed_concentration_mg_l",
            "cip_flag_0_1",
        }
    ]
    sample = canonical.loc[:, sample_cols].head(5).to_dict(orient="records")

    return cast(
        dict[str, Any],
        json_safe(
            {
                "status": status,
                "source_series_id": source_series_id,
                "errors": errors,
                "warnings": warnings,
                "missing_for_calculations": missing_for_calculations,
                "metadata": {
                    "schema_id": adapter.schema_id,
                    "adapter_name": adapter.name,
                    "ocwd_autodetect_used": mapping_path is None,
                    "n_rows": int(len(canonical)),
                    "canonical_columns": list(canonical.columns),
                    "field_metadata": field_meta,
                    "canonical_sample": sample,
                    "reference": reference_info,
                    "assumptions": [
                        "Explicit mapping is preferred; OCWD auto-detect is optional.",
                        "Conductivity is not TDS.",
                        "Calendar gaps are reported and not interpolated.",
                    ],
                },
            }
        ),
    )
