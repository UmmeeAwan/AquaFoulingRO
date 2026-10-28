"""Independent physical normalisation when concentration inputs are available."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aqua_fouling.data.reference import ReferenceConfig, ReferenceMode
from aqua_fouling.physics.ndp import NDPError, net_driving_pressure, require_positive_ndp
from aqua_fouling.physics.normalisation import (
    NormalisationError,
    normalised_permeate_flow_from_temperature,
    relative_health,
)
from aqua_fouling.physics.osmotic import (
    OsmoticPressureError,
    feed_concentrate_concentration_mg_l,
    osmotic_pressure_filmtec_psi,
)
from aqua_fouling.physics.recovery import RecoveryError


def try_independent_stage3_normalisation(
    canonical: pd.DataFrame,
    *,
    reference: ReferenceConfig | None,
    feed_concentration_mg_l: float | pd.Series,
    permeate_osmotic_psi: float = 0.0,
) -> dict[str, Any]:
    """Attempt independent FilmTec-style normalisation for stage-3 rows.

    Returns available arrays and blockers. Does not use source tcf_water or
    source-calculated NDP.
    """
    blockers: list[dict[str, Any]] = []
    n = len(canonical)
    empty = [None] * n

    required = [
        "stage_3_permeate_flow_source_unit",
        "stage_3_feed_flow_source_unit",
        "stage_3_feed_pressure_psi",
        "stage_3_device_pressure_drop_psi",
        "temperature_c",
    ]
    # Permeate pressure: prefer whole-unit perm if stage-3 permeate pressure absent.
    perm_col = None
    if "permeate_pressure_psi" in canonical.columns:
        perm_col = "permeate_pressure_psi"
        blockers.append(
            {
                "severity": "warning",
                "calculation": "stage_3_permeate_pressure",
                "problem": (
                    "Using whole-unit permeate_pressure_psi for stage-3 NDP "
                    "(stage-specific permeate pressure not mapped)."
                ),
                "actionable_correction": (
                    "Map a stage-3 permeate pressure column if available to avoid "
                    "scope mixing concerns."
                ),
            }
        )
    missing = [c for c in required if c not in canonical.columns]
    if missing or perm_col is None:
        blockers.append(
            {
                "calculation": "independent_ndp",
                "problem": f"Missing required columns: {missing or ['permeate_pressure']}",
                "actionable_correction": (
                    "Extend adapter mapping with confirmed stage-3 pressures/flows."
                ),
            }
        )
        return {
            "available": False,
            "q_normalised": empty,
            "ndp_operating_psi": empty,
            "relative_health": empty,
            "blockers": blockers,
            "row_level_failures": [],
            "n_row_failures": 0,
        }

    unresolved_fields = list(canonical.attrs.get("unresolved_independent_norm_fields") or [])
    unit_errors = list(canonical.attrs.get("unit_errors") or [])
    if unresolved_fields or unit_errors:
        blockers.append(
            {
                "calculation": "independent_unit_requirements",
                "problem": (
                    "Independent physical normalisation requires resolved, compatible "
                    f"units. unresolved_fields={unresolved_fields}; "
                    f"unit_errors={[e.get('field') for e in unit_errors]}."
                ),
                "actionable_correction": (
                    "Declare convertible units (psi/bar for pressure, gpd/gpm for flow, "
                    "deg_C for temperature, mg/L for concentration). "
                    "Unresolved OCWD diagnostic fields may remain for comparison only."
                ),
            }
        )
        return {
            "available": False,
            "q_normalised": empty,
            "ndp_operating_psi": empty,
            "relative_health": empty,
            "blockers": blockers,
            "row_level_failures": [],
            "n_row_failures": 0,
        }

    try:
        qp = pd.to_numeric(canonical["stage_3_permeate_flow_source_unit"], errors="coerce")
        qf = pd.to_numeric(canonical["stage_3_feed_flow_source_unit"], errors="coerce")
        # Row-wise recovery where both finite; leave others as NaN without raising globally.
        recovery = np.full(n, np.nan)
        for i in range(n):
            if np.isfinite(qp.iloc[i]) and np.isfinite(qf.iloc[i]) and qf.iloc[i] > 0:
                y = float(qp.iloc[i] / qf.iloc[i])
                if 0 < y < 1:
                    recovery[i] = y
        if isinstance(feed_concentration_mg_l, (int, float)):
            cf = np.full(n, float(feed_concentration_mg_l))
        else:
            cf = pd.to_numeric(feed_concentration_mg_l, errors="coerce").to_numpy(dtype=float)

        row_failures: list[dict[str, Any]] = []
        pi_fc = np.full(n, np.nan)
        temp = pd.to_numeric(canonical["temperature_c"], errors="coerce").to_numpy(dtype=float)
        for i in range(n):
            if not (np.isfinite(recovery[i]) and np.isfinite(cf[i]) and np.isfinite(temp[i])):
                row_failures.append(
                    {
                        "row": i,
                        "stage": "osmotic_inputs",
                        "reason": "missing_recovery_concentration_or_temperature",
                    }
                )
                continue
            try:
                c_fc = float(feed_concentrate_concentration_mg_l(cf[i], recovery[i])[0])
                pi_fc[i] = float(osmotic_pressure_filmtec_psi(c_fc, temp[i])[0])
            except (OsmoticPressureError, RecoveryError) as exc:
                row_failures.append({"row": i, "stage": "osmotic_pressure", "reason": str(exc)})

        pf = pd.to_numeric(canonical["stage_3_feed_pressure_psi"], errors="coerce").to_numpy(
            dtype=float
        )
        pp = pd.to_numeric(canonical[perm_col], errors="coerce").to_numpy(dtype=float)
        dp = pd.to_numeric(canonical["stage_3_device_pressure_drop_psi"], errors="coerce").to_numpy(
            dtype=float
        )

        ndp = np.full(n, np.nan)
        for i in range(n):
            if not all(np.isfinite(x) for x in (pf[i], pp[i], dp[i], pi_fc[i])):
                if np.isfinite(pi_fc[i]):
                    row_failures.append(
                        {
                            "row": i,
                            "stage": "ndp_inputs",
                            "reason": "missing_pressure_inputs_for_ndp",
                        }
                    )
                continue
            try:
                ndp[i] = float(
                    net_driving_pressure(pf[i], pp[i], dp[i], pi_fc[i], permeate_osmotic_psi)[0]
                )
            except NDPError as exc:
                row_failures.append({"row": i, "stage": "ndp", "reason": str(exc)})

        if reference is None:
            blockers.append(
                {
                    "calculation": "q_normalised",
                    "problem": "Reference configuration missing.",
                    "actionable_correction": "Supply documented or explicit_window reference YAML.",
                }
            )
            return {
                "available": False,
                "q_normalised": empty,
                "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
                "relative_health": empty,
                "blockers": blockers,
            }

        if reference.mode != ReferenceMode.DOCUMENTED:
            blockers.append(
                {
                    "calculation": "q_normalised",
                    "problem": (
                        "explicit_window reference averaging is not auto-applied; "
                        "provide documented NDP/flow/temperature or pre-compute window stats."
                    ),
                    "actionable_correction": (
                        "Use mode=documented with explicit NDP/flow/temperature, "
                        "or extend software to aggregate the window after it completes."
                    ),
                }
            )
            return {
                "available": False,
                "q_normalised": empty,
                "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
                "relative_health": empty,
                "blockers": blockers,
            }

        assert reference.ndp_psi is not None
        assert reference.temperature_c is not None
        assert reference.permeate_flow is not None
        assert reference.permeate_flow_unit is not None

        # Enforce matching/converted units for permeate flow vs reference flow.
        from aqua_fouling.physics.units import (
            CANONICAL_FLOW_UNIT,
            UnitConversionError,
            convert_flow,
            is_unresolved_unit,
        )

        field_meta = {
            m.get("canonical_name"): m for m in (canonical.attrs.get("field_metadata") or [])
        }
        qp_meta = field_meta.get("stage_3_permeate_flow_source_unit") or {}
        qf_meta = field_meta.get("stage_3_feed_flow_source_unit") or {}
        qp_unit = str(qp_meta.get("unit") or "")
        qf_unit = str(qf_meta.get("unit") or "")
        if is_unresolved_unit(qp_unit):
            blockers.append(
                {
                    "calculation": "flow_unit_match",
                    "problem": (
                        f"Permeate flow unit '{qp_unit}' is unresolved; cannot match "
                        f"reference unit '{reference.permeate_flow_unit}'."
                    ),
                    "actionable_correction": (
                        "Declare permeate flow as gpd/gpm/m3/d in the adapter, or convert "
                        "the source series before normalisation."
                    ),
                }
            )
            return {
                "available": False,
                "q_normalised": empty,
                "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
                "relative_health": empty,
                "blockers": blockers,
                "row_level_failures": row_failures,
                "n_row_failures": len(row_failures),
            }
        if qf_unit and not is_unresolved_unit(qf_unit) and qf_unit != qp_unit:
            blockers.append(
                {
                    "calculation": "flow_unit_match",
                    "problem": (
                        f"Feed flow unit '{qf_unit}' does not match permeate flow unit "
                        f"'{qp_unit}' after adapter conversion."
                    ),
                    "actionable_correction": (
                        "Ensure feed and permeate flow adapters use convertible units "
                        "that resolve to the same canonical flow unit (gpd)."
                    ),
                }
            )
            return {
                "available": False,
                "q_normalised": empty,
                "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
                "relative_health": empty,
                "blockers": blockers,
                "row_level_failures": row_failures,
                "n_row_failures": len(row_failures),
            }
        try:
            # Measurement columns are canonicalised to gpd when convertible; convert
            # reference into the same unit used for qp (fallback: canonical gpd).
            meas_unit = qp_unit or CANONICAL_FLOW_UNIT
            ref_flow = float(
                convert_flow(
                    reference.permeate_flow,
                    from_unit=reference.permeate_flow_unit,
                    to_unit=meas_unit,
                )[0]
            )
        except UnitConversionError as exc:
            blockers.append(
                {
                    "calculation": "reference_flow_unit_conversion",
                    "problem": str(exc),
                    "actionable_correction": (
                        "Align reference permeate_flow_unit with the adapter flow unit "
                        "(gpd/gpm/m3/d) or convert the reference value."
                    ),
                }
            )
            return {
                "available": False,
                "q_normalised": empty,
                "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
                "relative_health": empty,
                "blockers": blockers,
                "row_level_failures": row_failures,
                "n_row_failures": len(row_failures),
            }

        q_norm = np.full(n, np.nan)
        health = np.full(n, np.nan)
        if "observation_date_calendar" in canonical.columns:
            dates = pd.to_datetime(canonical["observation_date_calendar"], errors="coerce")
        else:
            dates = pd.Series([pd.NaT] * n)
        for i in range(n):
            if not np.isfinite(ndp[i]) or not np.isfinite(qp.iloc[i]) or not np.isfinite(temp[i]):
                if np.isfinite(ndp[i]):
                    row_failures.append(
                        {
                            "row": i,
                            "stage": "normalisation_inputs",
                            "reason": "missing_flow_or_temperature",
                        }
                    )
                continue
            if pd.notna(dates.iloc[i]):
                obs = dates.iloc[i].date()
                if obs < reference.available_from:
                    row_failures.append(
                        {
                            "row": i,
                            "stage": "reference_gate",
                            "reason": "before_reference_available_from",
                        }
                    )
                    continue
            try:
                require_positive_ndp(ndp[i])
                qn = float(
                    normalised_permeate_flow_from_temperature(
                        qp.iloc[i],
                        ndp[i],
                        reference.ndp_psi,
                        temp[i],
                        reference.temperature_c,
                    )[0]
                )
                # Health vs documented reference flow normalised at reference conditions.
                # Q_norm_ref at reference equals permeate_flow when TCF_ref/NDP_ref cancel.
                q_norm[i] = qn
                health[i] = float(relative_health(qn, ref_flow)[0])
            except (NormalisationError, NDPError) as exc:
                row_failures.append({"row": i, "stage": "normalisation", "reason": str(exc)})

        n_ok = int(np.isfinite(q_norm).sum())
        return {
            "available": n_ok > 0,
            "n_normalised": n_ok,
            "q_normalised": [float(v) if np.isfinite(v) else None for v in q_norm],
            "ndp_operating_psi": [float(v) if np.isfinite(v) else None for v in ndp],
            "relative_health": [float(v) if np.isfinite(v) else None for v in health],
            "blockers": blockers,
            "row_level_failures": row_failures,
            "n_row_failures": len(row_failures),
            "assumptions": [
                "stage_2_3_press used as stage-3 feed pressure (inferred) when OCWD mapped.",
                "π_permeate default 0 unless configured.",
                "Osmotic method filmtec_cfc_bar_lt_20000_v1; not for seawater/high-TDS wastewater.",
                "TCF from filmtec_exponential_tcf_v1 only (no source tcf_water).",
                "Row-level failures are listed; values are not silently repaired.",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        blockers.append(
            {
                "calculation": "independent_stage3_normalisation",
                "problem": str(exc),
                "actionable_correction": "Inspect inputs and physical validity ranges.",
            }
        )
        return {
            "available": False,
            "q_normalised": empty,
            "ndp_operating_psi": empty,
            "relative_health": empty,
            "blockers": blockers,
        }
