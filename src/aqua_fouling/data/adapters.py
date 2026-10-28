"""Dataset adapters for the two observed OCWD CSV schemas."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from aqua_fouling.data.contracts import UnitSpec, ValueKind


class MeasurementScope(str, Enum):
    WHOLE_UNIT = "whole_unit"
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"


class AdapterField(BaseModel):
    external_name: str
    canonical_name: str
    unit: UnitSpec
    kind: ValueKind
    measurement_scope: MeasurementScope
    confidence: str = "unresolved"
    notes: str | None = None


class AdapterConfig(BaseModel):
    name: str
    schema_id: str
    source_series_pattern: str
    index_column: str | None = None
    timestamp_column: str = "date"
    timestamp_format: str = "ISO8601"
    timezone: str = "America/Los_Angeles"
    fields: list[AdapterField] = Field(min_length=1)
    notes: str | None = None

    def matches_series(self, series_id: str) -> bool:
        return re.fullmatch(self.source_series_pattern, series_id) is not None

    def resolve_timestamp_external_column(self, columns: list[str] | set[str]) -> str | None:
        """Resolve the raw CSV timestamp column via adapter metadata (not hardcoding)."""
        cols = set(columns)
        if self.timestamp_column and self.timestamp_column in cols:
            return self.timestamp_column
        for field in self.fields:
            if field.canonical_name == "observation_date_calendar" and field.external_name in cols:
                return field.external_name
        return None


class AdapterConfigError(ValueError):
    """Malformed or incomplete adapter mapping YAML."""


def load_adapter_config(path: Path) -> AdapterConfig:
    try:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"Malformed adapter mapping YAML at {path}: {exc}"
        raise AdapterConfigError(msg) from exc
    except OSError as exc:
        msg = f"Unable to read adapter mapping at {path}: {exc}"
        raise AdapterConfigError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Adapter mapping at {path} must be a YAML mapping/object."
        raise AdapterConfigError(msg)
    try:
        fields = []
        for item in raw.get("fields", []):
            fields.append(
                AdapterField(
                    external_name=item["external_name"],
                    canonical_name=item["canonical_name"],
                    unit=UnitSpec(**item["unit"]),
                    kind=ValueKind(item["kind"]),
                    measurement_scope=MeasurementScope(item["measurement_scope"]),
                    confidence=item.get("confidence", "unresolved"),
                    notes=item.get("notes"),
                )
            )
        ts = raw.get("timestamp") or {}
        return AdapterConfig(
            name=raw["name"],
            schema_id=raw["schema_id"],
            source_series_pattern=raw["source_series_pattern"],
            index_column=raw.get("index_column"),
            timestamp_column=ts.get("column", "date"),
            timestamp_format=ts.get("format", "ISO8601"),
            timezone=ts.get("timezone", "America/Los_Angeles"),
            fields=fields,
            notes=raw.get("notes"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Incomplete or invalid adapter mapping at {path}: {exc}"
        raise AdapterConfigError(msg) from exc


def detect_schema_id(columns: list[str]) -> str:
    """Optional OCWD AE/FG auto-detection from column presence."""
    cols = set(columns)
    if "feed_psi" in cols and "stage_3_flow" in cols:
        return "ae_128"
    if "feed_press_stage_1" in cols or "perm_flow_stage_3" in cols:
        return "fg_117"
    msg = (
        "Unable to detect OCWD schema. Expected AE markers "
        "(feed_psi, stage_3_flow) or FG markers "
        "(feed_press_stage_1 / perm_flow_stage_3). "
        "Supply an explicit --mapping YAML for non-OCWD CSVs."
    )
    raise ValueError(msg)


def resolve_configs_dir(explicit: Path | None = None) -> Path:
    """Resolve configs directory for cwd, Docker image, or editable src/ checkout.

    Preference order:
    1. Explicit argument / AQUA_CONFIGS_DIR
    2. ``<repo_root>/configs`` when running from a src/ layout checkout or image
    3. CWD ``configs/``
    """
    import os

    if explicit is not None:
        return explicit
    env = os.environ.get("AQUA_CONFIGS_DIR")
    if env:
        return Path(env)
    # Walk up from this module: works for repo root (…/AquaFoulingRO/configs)
    # and Docker layout (/app/configs with package under /app/src/…).
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    return Path("configs")


def default_adapter_paths(configs_dir: Path | None = None) -> dict[str, Path]:
    root = resolve_configs_dir(configs_dir)
    return {
        "ae_128": root / "ocwd_schema_ae.yaml",
        "fg_117": root / "ocwd_schema_fg.yaml",
        "canonical_synthetic_v1": root / "canonical_synthetic_schema.yaml",
        "canonical_generic_v1": root / "canonical_generic_schema.yaml",
    }


def load_adapter_for_schema(
    schema_id: str,
    configs_dir: Path | None = None,
) -> AdapterConfig:
    paths = default_adapter_paths(configs_dir)
    if schema_id not in paths:
        msg = f"Unknown schema_id={schema_id}"
        raise ValueError(msg)
    return load_adapter_config(paths[schema_id])


def resolve_adapter(
    columns: list[str],
    *,
    mapping_path: Path | None = None,
    configs_dir: Path | None = None,
    allow_ocwd_autodetect: bool = True,
) -> AdapterConfig:
    """Prefer an explicitly supplied mapping; OCWD auto-detect is optional."""
    if mapping_path is not None:
        return load_adapter_config(mapping_path)
    if not allow_ocwd_autodetect:
        msg = (
            "No mapping supplied and OCWD auto-detection is disabled. "
            "Provide --mapping with a documented canonical schema YAML."
        )
        raise ValueError(msg)
    schema_id = detect_schema_id(columns)
    return load_adapter_for_schema(schema_id, configs_dir)


# Canonical fields required (or optionally used) by independent physical normalisation.
INDEPENDENT_NORM_QUANTITY_BY_FIELD: dict[str, str] = {
    "stage_3_permeate_flow_source_unit": "flow",
    "stage_3_feed_flow_source_unit": "flow",
    "unit_permeate_flow_source_unit": "flow",
    "unit_feed_flow_source_unit": "flow",
    "stage_3_feed_pressure_psi": "pressure",
    "stage_3_device_pressure_drop_psi": "pressure",
    "permeate_pressure_psi": "pressure",
    "feed_pressure_psi": "pressure",
    "stage_1_permeate_pressure_psi": "pressure",
    "temperature_c": "temperature",
    "feed_concentration_mg_l": "concentration",
    "permeate_concentration_mg_l": "concentration",
}

# Diagnostic / source-calculated fields may remain unresolved without failing adapt.
DIAGNOSTIC_UNRESOLVED_OK: frozenset[str] = frozenset(
    {
        "stage_3_flux_source_unit",
        "stage_3_ndp_source_calculated",
        "stage_3_specific_flux_source_calculated",
        "tcf_water_source_calculated",
        "permeate_conductivity_source_unit",
        "stage_3_permeate_conductivity_source_unit",
        "ro_feed_toc_avg_mg_per_L",
    }
)


def quantity_for_canonical_field(canonical_name: str) -> str | None:
    """Infer expected physical quantity from the canonical field identity."""
    if canonical_name in INDEPENDENT_NORM_QUANTITY_BY_FIELD:
        return INDEPENDENT_NORM_QUANTITY_BY_FIELD[canonical_name]
    lower = canonical_name.lower()
    if (
        ("temperature" in lower or (lower.endswith("_c") and "toc" not in lower))
        and "conductivity" not in lower
        and "concentration" not in lower
    ):
        return "temperature"
    if (
        "concentration" in lower or lower.endswith("_mg_l") or "mg_l" in lower
    ) and "conductivity" not in lower:
        return "concentration"
    if (
        "pressure" in lower or lower.endswith("_psi") or "_dp_" in lower or "press" in lower
    ) and "unresolved" not in lower:
        return "pressure"
    if "flow" in lower and "flux" not in lower:
        return "flow"
    if "conductivity" in lower:
        return "conductivity"
    return None


def adapt_dataframe(
    frame: pd.DataFrame,
    *,
    source_series_id: str,
    adapter: AdapterConfig,
    stage_focus: MeasurementScope = MeasurementScope.STAGE_3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Map a source CSV frame to canonical columns with quantity-aware unit checks.

    Expected quantity is determined from the canonical field name. Supported
    pressure → psi and flow → gpd conversions are applied. Unsupported or
    incompatible declared units raise actionable errors. Unresolved OCWD
    diagnostic fields are preserved; unresolved units on independent-norm
    fields are flagged so independent calculations are blocked.
    """
    from aqua_fouling.physics.units import (
        CANONICAL_CONCENTRATION_UNIT,
        CANONICAL_FLOW_UNIT,
        CANONICAL_PRESSURE_UNIT,
        CANONICAL_TEMPERATURE_UNIT,
        UnitConversionError,
        convert_quantity,
        is_concentration_unit,
        is_conductivity_unit,
        is_flow_unit,
        is_pressure_unit,
        is_temperature_unit,
        is_unresolved_unit,
        normalize_unit_symbol,
    )

    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not adapter.matches_series(source_series_id):
        warnings.append(
            {
                "field": "source_series_id",
                "problem": (
                    f"series_id {source_series_id} does not match adapter "
                    f"pattern {adapter.source_series_pattern}"
                ),
                "actionable_correction": "Use the adapter matching this series schema.",
            }
        )

    rename: dict[str, str] = {}
    field_by_canonical: dict[str, AdapterField] = {}
    for field in adapter.fields:
        if field.measurement_scope not in {MeasurementScope.WHOLE_UNIT, stage_focus}:
            continue
        if field.external_name not in frame.columns:
            warnings.append(
                {
                    "field": field.external_name,
                    "canonical_name": field.canonical_name,
                    "problem": "column_absent_from_source_schema_or_file",
                    "expected_unit": field.unit.symbol,
                    "actionable_correction": (
                        "Confirm schema adapter; do not treat as a within-column "
                        "missing observation."
                    ),
                }
            )
            continue
        rename[field.external_name] = field.canonical_name
        field_by_canonical[field.canonical_name] = field

    subset = frame.loc[:, list(rename.keys())].rename(columns=rename)
    out = subset.copy()
    meta_rows: list[dict[str, Any]] = []
    conversions: list[dict[str, Any]] = []
    unresolved_independent_fields: list[str] = []

    for canonical_name, field in field_by_canonical.items():
        src_unit = field.unit.symbol
        applied_unit = src_unit
        quantity = quantity_for_canonical_field(canonical_name)
        independent = canonical_name in INDEPENDENT_NORM_QUANTITY_BY_FIELD
        diagnostic_ok = canonical_name in DIAGNOSTIC_UNRESOLVED_OK
        field_name = canonical_name
        external_name = field.external_name
        quantity_label = quantity or "compatible"

        def _reject(
            problem: str,
            expected: str,
            *,
            _field: str = field_name,
            _external: str = external_name,
            _quantity: str = quantity_label,
        ) -> None:
            errors.append(
                {
                    "field": _field,
                    "problem": problem,
                    "expected_unit": expected,
                    "actionable_correction": (
                        f"Update the adapter unit for '{_external}' to a "
                        f"supported {_quantity} unit ({expected}), "
                        "or convert the source column before mapping."
                    ),
                }
            )

        if quantity == "pressure":
            if is_unresolved_unit(src_unit):
                if independent:
                    unresolved_independent_fields.append(canonical_name)
                    warnings.append(
                        {
                            "field": canonical_name,
                            "problem": (
                                f"Pressure unit '{src_unit}' is unresolved; "
                                "independent physical normalisation is blocked."
                            ),
                            "expected_unit": CANONICAL_PRESSURE_UNIT,
                            "actionable_correction": (
                                "Declare psi/bar/Pa/kPa for this pressure field."
                            ),
                        }
                    )
                elif not diagnostic_ok:
                    warnings.append(
                        {
                            "field": canonical_name,
                            "problem": f"unit '{src_unit}' unresolved; values not converted",
                            "expected_unit": CANONICAL_PRESSURE_UNIT,
                            "actionable_correction": "Declare an explicit pressure unit.",
                        }
                    )
            elif not is_pressure_unit(src_unit):
                _reject(
                    (
                        f"Canonical pressure field '{canonical_name}' declared with "
                        f"incompatible/unsupported unit '{src_unit}'. "
                        "Supported: psi, bar, Pa, kPa."
                    ),
                    CANONICAL_PRESSURE_UNIT,
                )
            elif normalize_unit_symbol(src_unit) != CANONICAL_PRESSURE_UNIT:
                try:
                    converted, info = convert_quantity(
                        pd.to_numeric(out[canonical_name], errors="coerce"),
                        from_unit=src_unit,
                        to_unit=CANONICAL_PRESSURE_UNIT,
                        quantity="pressure",
                    )
                    out[canonical_name] = converted
                    applied_unit = CANONICAL_PRESSURE_UNIT
                    conversions.append({"canonical_name": canonical_name, **info})
                except UnitConversionError as exc:
                    _reject(str(exc), CANONICAL_PRESSURE_UNIT)
            else:
                applied_unit = CANONICAL_PRESSURE_UNIT

        elif quantity == "flow":
            if is_unresolved_unit(src_unit):
                if independent:
                    unresolved_independent_fields.append(canonical_name)
                    warnings.append(
                        {
                            "field": canonical_name,
                            "problem": (
                                f"Flow unit '{src_unit}' is unresolved; "
                                "independent physical normalisation is blocked."
                            ),
                            "expected_unit": CANONICAL_FLOW_UNIT,
                            "actionable_correction": (
                                "Declare gpd/gpm/m3/d for this flow field. "
                                "Diagnostic use of unresolved OCWD flows remains allowed."
                            ),
                        }
                    )
                else:
                    warnings.append(
                        {
                            "field": canonical_name,
                            "problem": f"unit '{src_unit}' unresolved; values not converted",
                            "expected_unit": CANONICAL_FLOW_UNIT,
                            "actionable_correction": "Declare an explicit flow unit.",
                        }
                    )
            elif not is_flow_unit(src_unit):
                _reject(
                    (
                        f"Canonical flow field '{canonical_name}' declared with "
                        f"incompatible/unsupported unit '{src_unit}'. "
                        "Supported: gpd, gpm, m3/d."
                    ),
                    CANONICAL_FLOW_UNIT,
                )
            elif normalize_unit_symbol(src_unit) != CANONICAL_FLOW_UNIT:
                try:
                    converted, info = convert_quantity(
                        pd.to_numeric(out[canonical_name], errors="coerce"),
                        from_unit=src_unit,
                        to_unit=CANONICAL_FLOW_UNIT,
                        quantity="flow",
                    )
                    out[canonical_name] = converted
                    applied_unit = CANONICAL_FLOW_UNIT
                    conversions.append({"canonical_name": canonical_name, **info})
                except UnitConversionError as exc:
                    _reject(str(exc), CANONICAL_FLOW_UNIT)
            else:
                applied_unit = CANONICAL_FLOW_UNIT

        elif quantity == "temperature":
            if is_unresolved_unit(src_unit):
                if independent:
                    unresolved_independent_fields.append(canonical_name)
                warnings.append(
                    {
                        "field": canonical_name,
                        "problem": (
                            f"Temperature unit '{src_unit}' is unresolved; "
                            "independent physical normalisation is blocked."
                            if independent
                            else f"temperature unit '{src_unit}' unresolved"
                        ),
                        "expected_unit": CANONICAL_TEMPERATURE_UNIT,
                        "actionable_correction": "Declare deg_C / Celsius.",
                    }
                )
            elif not is_temperature_unit(src_unit):
                _reject(
                    (
                        f"Canonical temperature field '{canonical_name}' declared with "
                        f"incompatible unit '{src_unit}'. "
                        "Accepted aliases: deg_C, C, Celsius. "
                        "No Fahrenheit/Kelvin conversion is implemented; convert externally."
                    ),
                    CANONICAL_TEMPERATURE_UNIT,
                )
            else:
                applied_unit = CANONICAL_TEMPERATURE_UNIT

        elif quantity == "concentration":
            if is_conductivity_unit(src_unit):
                _reject(
                    (
                        f"Canonical concentration field '{canonical_name}' cannot use "
                        f"conductivity unit '{src_unit}'. "
                        "Never silently interpret conductivity as mg/L; supply an "
                        "explicit concentration (mg/L NaCl-equivalent) or a documented "
                        "calibration outside silent conversion."
                    ),
                    CANONICAL_CONCENTRATION_UNIT,
                )
            elif is_unresolved_unit(src_unit):
                if independent:
                    unresolved_independent_fields.append(canonical_name)
                    warnings.append(
                        {
                            "field": canonical_name,
                            "problem": (
                                f"Concentration unit '{src_unit}' unresolved; "
                                "independent normalisation blocked for this field."
                            ),
                            "expected_unit": CANONICAL_CONCENTRATION_UNIT,
                            "actionable_correction": "Declare mg/L (NaCl-equivalent).",
                        }
                    )
            elif not is_concentration_unit(src_unit):
                _reject(
                    (
                        f"Canonical concentration field '{canonical_name}' declared with "
                        f"incompatible unit '{src_unit}'. "
                        "Supported: mg/L, mg_L_NaCl_eq, ppm (as mg/L equivalent)."
                    ),
                    CANONICAL_CONCENTRATION_UNIT,
                )
            else:
                applied_unit = CANONICAL_CONCENTRATION_UNIT

        elif diagnostic_ok and (is_unresolved_unit(src_unit) or is_conductivity_unit(src_unit)):
            warnings.append(
                {
                    "field": canonical_name,
                    "problem": (
                        f"Diagnostic field retained with unit '{src_unit}' "
                        "(not used for independent physical normalisation)."
                    ),
                    "expected_unit": None,
                    "actionable_correction": (
                        "No action required for diagnostic-only fields; resolve units "
                        "before using this column in independent calculations."
                    ),
                }
            )

        meta_rows.append(
            {
                "external_name": field.external_name,
                "canonical_name": canonical_name,
                "source_unit": src_unit,
                "unit": applied_unit,
                "quantity": quantity,
                "kind": field.kind.value,
                "measurement_scope": field.measurement_scope.value,
                "confidence": field.confidence,
                "notes": field.notes,
                "independent_norm_field": independent,
                "unit_unresolved_blocks_independent_norm": (
                    canonical_name in unresolved_independent_fields
                ),
            }
        )

    out.insert(0, "source_series_id", source_series_id)
    out.insert(1, "measurement_scope_focus", stage_focus.value)
    out.insert(2, "schema_id", adapter.schema_id)
    out.attrs["field_metadata"] = meta_rows
    out.attrs["adapter_name"] = adapter.name
    out.attrs["timezone"] = adapter.timezone
    out.attrs["unit_conversions"] = conversions
    out.attrs["unit_errors"] = errors
    out.attrs["unresolved_independent_norm_fields"] = unresolved_independent_fields
    for err in errors:
        warnings.append({**err, "severity": "error"})
    return out, warnings
