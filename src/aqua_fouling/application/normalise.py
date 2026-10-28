"""Normalisation application service (independent + diagnostic paths)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from aqua_fouling.application.independent_norm import try_independent_stage3_normalisation
from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.segments import (
    SegmentRuleConfig,
    contiguous_segments,
    flag_abrupt_changes,
)
from aqua_fouling.application.validate import validate_csv
from aqua_fouling.data.adapters import (
    MeasurementScope,
    adapt_dataframe,
    resolve_adapter,
)
from aqua_fouling.data.reference import ReferenceConfig
from aqua_fouling.physics.normalisation import (
    NORMALISATION_METHOD_ID,
    NORMALISATION_VERSION,
    method_metadata,
    normalised_permeate_flow_from_temperature,
    relative_health,
)
from aqua_fouling.physics.temperature import filmtec_temperature_correction_factor


def _load_reference(path: Path | None) -> ReferenceConfig | None:
    if path is None:
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ReferenceConfig.model_validate(raw)


def normalise_csv(
    csv_path: Path,
    *,
    source_series_id: str,
    mapping_path: Path | None = None,
    reference_path: Path | None = None,
    configs_dir: Path | None = None,
    concentration_mg_l: float | None = None,
    conductivity_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalise a CSV using shared adapters and physics services.

    Independent physical normalisation requires concentration (or an explicit
    conductivity calibration). Without it, intermediate calculations that do
    not need concentration may still be returned, and reference-dependent /
    NDP-from-concentration outputs are marked unavailable with diagnostics.

    A separately labelled diagnostic comparison may use source-calculated
    NDP / s3sf / tcf_water — never presented as independent raw-data normalisation.
    """
    # Prefer column-mapped concentration when present (after adapt); also accept caller arg.
    validation = validate_csv(
        csv_path,
        source_series_id=source_series_id,
        mapping_path=mapping_path,
        reference_path=reference_path,
        configs_dir=configs_dir,
        concentration_mg_l=concentration_mg_l,
    )
    if validation["status"] == "invalid":
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "status": "invalid",
                    "source_series_id": source_series_id,
                    "errors": validation.get("errors") or [],
                    "normalised": {
                        "available": False,
                        "n_measurements": 0,
                        "measurements": [],
                        "derived": None,
                        "data_quality_flags": [],
                    },
                    "row_failures": {"n_failures": 0, "failures": []},
                    "warnings": validation.get("warnings") or [],
                    "metadata": {
                        "validation_status": validation["status"],
                        "normalisation_method": method_metadata(),
                        "assumptions": validation.get("assumptions", []),
                    },
                    "validation": validation,
                }
            ),
        )

    frame = pd.read_csv(csv_path)
    adapter = resolve_adapter(
        [str(c) for c in frame.columns],
        mapping_path=mapping_path,
        configs_dir=configs_dir,
    )
    schema_id = adapter.schema_id
    canonical, _ = adapt_dataframe(
        frame,
        source_series_id=source_series_id,
        adapter=adapter,
        stage_focus=MeasurementScope.STAGE_3,
    )

    try:
        reference = _load_reference(reference_path)
    except Exception as exc:  # noqa: BLE001
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "status": "invalid",
                    "source_series_id": source_series_id,
                    "errors": [
                        {
                            "field": "reference",
                            "problem": str(exc),
                            "actionable_correction": "Fix reference YAML.",
                        }
                    ],
                    "normalised": {
                        "available": False,
                        "n_measurements": 0,
                        "measurements": [],
                        "derived": None,
                        "data_quality_flags": [],
                    },
                    "row_failures": {"n_failures": 0, "failures": []},
                    "warnings": [],
                    "metadata": {
                        "validation_status": validation["status"],
                        "schema_id": schema_id,
                        "normalisation_method": method_metadata(),
                        "assumptions": [],
                    },
                    "validation": validation,
                }
            ),
        )

    if "observation_date_calendar" in canonical.columns:
        dates = pd.to_datetime(canonical["observation_date_calendar"], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(canonical))
    cip = canonical.get("cip_flag_0_1")
    dsr = canonical.get("days_since_replacement_d")
    segments, segment_audit = contiguous_segments(
        dates,
        cip=cip,
        days_since_replacement=dsr,
        rules=SegmentRuleConfig(),
    )
    abrupt = []
    flow_col = "stage_3_permeate_flow_source_unit"
    if flow_col in canonical.columns:
        abrupt = flag_abrupt_changes(canonical[flow_col])

    data_quality_flags = segment_audit + abrupt

    # --- Independent TCF from temperature (never from source tcf_water) ---
    independent_tcf: list[float | None] = [None] * len(canonical)
    if "temperature_c" in canonical.columns:
        temp = pd.to_numeric(canonical["temperature_c"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            tcf_vals = filmtec_temperature_correction_factor(temp)
        independent_tcf = [
            float(v) if np.isfinite(v) else None for v in np.asarray(tcf_vals, dtype=float)
        ]

    # --- Concentration / independent NDP ---
    feed_conc: float | pd.Series | None = concentration_mg_l
    if feed_conc is None and "feed_concentration_mg_l" in canonical.columns:
        feed_conc = pd.to_numeric(canonical["feed_concentration_mg_l"], errors="coerce")
        if not np.isfinite(pd.to_numeric(feed_conc, errors="coerce")).any():
            feed_conc = None
    if feed_conc is None and conductivity_calibration is not None:
        # Require explicit slope/intercept — never invent a universal coefficient.
        if "mg_l_per_uS_cm" in conductivity_calibration and "ec_column" in conductivity_calibration:
            ec_col = str(conductivity_calibration["ec_column"])
            # Expect raw frame column still present via original CSV — use conductivity
            # from adapted canonical if mapped, else blocker.
            blockers_cal = []
            slope = float(conductivity_calibration["mg_l_per_uS_cm"])
            intercept = float(conductivity_calibration.get("intercept_mg_l", 0.0))
            # Prefer a caller-supplied constant concentration derived externally.
            if "feed_concentration_mg_l" in conductivity_calibration:
                feed_conc = float(conductivity_calibration["feed_concentration_mg_l"])
            else:
                blockers_cal.append(
                    {
                        "calculation": "conductivity_calibration",
                        "problem": (
                            f"Calibration declared (slope={slope}, intercept={intercept}, "
                            f"ec_column={ec_col}) but per-row EC→mg/L application requires "
                            "the calibrated feed EC series to be mapped; provide "
                            "feed_concentration_mg_l in the calibration object for Phase 2 "
                            "or map feed EC explicitly."
                        ),
                        "actionable_correction": (
                            "Include feed_concentration_mg_l in conductivity_calibration, "
                            "or extend the adapter with a calibrated feed-EC mapping."
                        ),
                    }
                )
            independent_cal_blockers = blockers_cal
        else:
            independent_cal_blockers = [
                {
                    "calculation": "conductivity_calibration",
                    "problem": (
                        "conductivity_calibration must include mg_l_per_uS_cm and ec_column "
                        "(and preferably feed_concentration_mg_l)."
                    ),
                    "actionable_correction": (
                        "Supply an explicit calibration object; no silent default."
                    ),
                }
            ]
    else:
        independent_cal_blockers = []

    reference_provenance = reference.to_provenance_dict() if reference else None
    independent_result: dict[str, Any]
    if feed_conc is None:
        independent_result = {
            "available": False,
            "q_normalised": [None] * len(canonical),
            "ndp_operating_psi": [None] * len(canonical),
            "relative_health": [None] * len(canonical),
            "blockers": independent_cal_blockers
            + [
                {
                    "calculation": "independent_ndp_and_q_normalisation",
                    "problem": (
                        "Feed concentration (mg/L) or an explicit conductivity→concentration "
                        "calibration producing concentration is required for FilmTec osmotic "
                        "pressure and independent NDP."
                    ),
                    "actionable_correction": (
                        "Provide concentration_mg_l or conductivity_calibration with "
                        "feed_concentration_mg_l. Do not silently choose a universal coefficient. "
                        "Source-calculated NDP does not validate our concentration correction."
                    ),
                }
            ],
        }
        if reference is None:
            independent_result["reference_unavailable_reason"] = (
                "No reference configuration supplied. Intermediate FilmTec TCF values may "
                "still be returned; reference-dependent Q_normalised and health are unavailable."
            )
    else:
        independent_result = try_independent_stage3_normalisation(
            canonical,
            reference=reference,
            feed_concentration_mg_l=feed_conc,
        )
        independent_result["blockers"] = independent_cal_blockers + list(
            independent_result.get("blockers") or []
        )

    # Recompute missing_for_calculations: drop concentration blocker when supplied/mapped.
    missing_for_calculations = [
        m
        for m in validation.get("missing_for_calculations", [])
        if not (
            feed_conc is not None
            and m.get("calculation") == "independent_physical_normalisation_ndp_from_concentration"
        )
    ]
    if reference is not None:
        missing_for_calculations = [
            m
            for m in missing_for_calculations
            if m.get("calculation") != "reference_dependent_health"
        ]

    # --- Diagnostic path (source-calculated fields) — clearly labelled ---
    diagnostic_rows: list[dict[str, Any]] = []
    src_ndp = canonical.get("stage_3_ndp_source_calculated")
    src_s3sf = canonical.get("stage_3_specific_flux_source_calculated")
    src_tcf = canonical.get("tcf_water_source_calculated")
    src_flux = canonical.get("stage_3_flux_source_unit")
    for i in range(len(canonical)):
        dval = dates.iloc[i] if len(dates) else pd.NaT
        diagnostic_rows.append(
            {
                "row": i,
                "date": dval.date().isoformat() if pd.notna(dval) else None,
                "source_ndp": (
                    float(src_ndp.iloc[i])
                    if src_ndp is not None and pd.notna(src_ndp.iloc[i])
                    else None
                ),
                "source_s3sf": (
                    float(src_s3sf.iloc[i])
                    if src_s3sf is not None and pd.notna(src_s3sf.iloc[i])
                    else None
                ),
                "source_tcf_water": (
                    float(src_tcf.iloc[i])
                    if src_tcf is not None and pd.notna(src_tcf.iloc[i])
                    else None
                ),
                "source_stage_3_flux": (
                    float(src_flux.iloc[i])
                    if src_flux is not None and pd.notna(src_flux.iloc[i])
                    else None
                ),
                "independent_filmtec_tcf": independent_tcf[i],
                "note": (
                    "DIAGNOSTIC ONLY — uses source-calculated fields. "
                    "Not independent raw-data normalisation. "
                    "Do not combine source tcf_water with FilmTec TCF (no double correction)."
                ),
            }
        )

    # Reference window availability gate for any future independent outputs
    if reference is not None:
        for i, dval in enumerate(dates):
            if pd.isna(dval):
                continue
            obs_day = dval.date() if hasattr(dval, "date") else date.fromisoformat(str(dval)[:10])
            if obs_day < reference.available_from:
                data_quality_flags.append(
                    {
                        "reason": "before_reference_available_from",
                        "at_index": i,
                        "date": obs_day.isoformat(),
                        "action": "reference_dependent_outputs_suppressed",
                    }
                )

    # Build canonical measurement payload (preserve missing as null)
    measurements: list[dict[str, Any]] = []
    for i in range(len(canonical)):
        row: dict[str, Any] = {
            "source_series_id": source_series_id,
            "row": i,
            "date": (dates.iloc[i].date().isoformat() if pd.notna(dates.iloc[i]) else None),
            "measurement_scope_focus": MeasurementScope.STAGE_3.value,
        }
        for col in canonical.columns:
            if col in {"source_series_id", "measurement_scope_focus", "schema_id"}:
                continue
            val = canonical.iloc[i][col]
            if pd.isna(val):
                row[col] = None
            elif isinstance(val, (float, np.floating, int, np.integer)):
                fv = float(val)
                row[col] = fv if np.isfinite(fv) else None
            else:
                row[col] = val if not isinstance(val, float) else None
        measurements.append(row)

    derived = {
        "independent_physical_normalisation": {
            "available": bool(independent_result.get("available")),
            "n_normalised": independent_result.get("n_normalised", 0),
            "q_normalised": independent_result.get("q_normalised"),
            "ndp_operating_psi": independent_result.get("ndp_operating_psi"),
            "relative_health": independent_result.get("relative_health"),
            "independent_filmtec_tcf": independent_tcf,
            "blockers": independent_result.get("blockers", []),
            "reference_unavailable_reason": independent_result.get("reference_unavailable_reason"),
            "assumptions": independent_result.get("assumptions", []),
            "method": method_metadata(),
        },
        "diagnostic_source_calculated_comparison": {
            "label": (
                "DIAGNOSTIC — source-calculated NDP / s3sf / tcf_water. "
                "Not independent physical normalisation from raw measurements."
            ),
            "rows": diagnostic_rows[:50],
            "n_rows_total": len(diagnostic_rows),
        },
        "segments": {
            "count": len(segments),
            "segments": segments,
            "duration_days": [s.get("duration_days") for s in segments],
        },
    }

    # Optional synthetic complete example for API/tests (not OCWD raw path).
    extras = (conductivity_calibration or {}).get("synthetic_complete_example")
    if isinstance(extras, dict):
        try:
            qn = normalised_permeate_flow_from_temperature(
                extras["q_measured"],
                extras["ndp_operating"],
                extras["ndp_reference"],
                extras["temp_operating_c"],
                extras["temp_reference_c"],
            )
            href = relative_health(qn, extras.get("health_reference_qn", float(qn[0])))
            derived["independent_physical_normalisation"]["synthetic_example_result"] = {
                "q_normalised": [float(x) for x in np.asarray(qn, dtype=float)],
                "relative_health": [float(x) for x in np.asarray(href, dtype=float)],
            }
        except Exception as exc:  # noqa: BLE001
            derived["independent_physical_normalisation"]["blockers"].append(
                {
                    "calculation": "synthetic_complete_example",
                    "problem": str(exc),
                    "actionable_correction": "Fix synthetic example inputs.",
                }
            )

    status = (
        "ok"
        if derived["independent_physical_normalisation"]["available"]
        else "ok_with_limitations"
    )

    from aqua_fouling.application.response_presentation import public_reference

    row_failures = independent_result.get("row_level_failures", [])[:200]
    n_row_failures = int(
        independent_result.get("n_row_failures", 0)
        or len(independent_result.get("row_level_failures") or [])
    )
    return cast(
        dict[str, Any],
        json_safe(
            {
                "status": status,
                "source_series_id": source_series_id,
                "normalised": {
                    "available": bool(
                        derived.get("independent_physical_normalisation", {}).get("available")
                    ),
                    "n_measurements": len(measurements),
                    "measurements": measurements[:100],
                    "derived": derived,
                    "data_quality_flags": data_quality_flags[:200],
                },
                "row_failures": {
                    "n_failures": n_row_failures,
                    "failures": row_failures,
                },
                "warnings": [],
                "metadata": {
                    "validation_status": validation["status"],
                    "schema_id": schema_id,
                    "reference": public_reference(reference_provenance),
                    "normalisation_method": {
                        **method_metadata(),
                        "method_id": NORMALISATION_METHOD_ID,
                        "version": NORMALISATION_VERSION,
                    },
                    "missing_for_calculations": missing_for_calculations,
                    "assumptions": [
                        "Independent path uses filmtec_exponential_tcf_v1 from temperature only.",
                        "Source tcf_water is never applied together with FilmTec TCF.",
                        "dss=0 is not treated as proof of cleaning.",
                        "Missing observations are preserved; gaps are not interpolated.",
                        "Row-level normalisation failures are reported.",
                    ],
                },
            }
        ),
    )
