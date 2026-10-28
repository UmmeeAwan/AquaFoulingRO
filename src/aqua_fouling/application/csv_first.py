"""CSV-first series-id inference and deterministic column/unit aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yaml

from aqua_fouling.application.forecast_errors import ForecastValidationError
from aqua_fouling.data.adapters import AdapterConfig, AdapterField, MeasurementScope
from aqua_fouling.data.contracts import UnitSpec, ValueKind

SERIES_ID_COLUMNS: tuple[str, ...] = (
    "source_series_id",
    "series_id",
    "unit_id",
)

DEFAULT_SERIES_ID = "uploaded_series"

# Deterministic aliases: exact CSV header (case-sensitive as stored after strip)
# → (canonical_name, unit_symbol, measurement_scope, kind).
# Units come only from explicit unit-bearing names — never from magnitude.
CANONICAL_COLUMN_ALIASES: dict[str, tuple[str, str, MeasurementScope, ValueKind]] = {
    "observation_date": (
        "observation_date_calendar",
        "calendar_date",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "observation_date_calendar": (
        "observation_date_calendar",
        "calendar_date",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "date": (
        "observation_date_calendar",
        "calendar_date",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "temperature_c": (
        "temperature_c",
        "deg_C",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "temp_c": (
        "temperature_c",
        "deg_C",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "feed_pressure_psi": (
        "stage_3_feed_pressure_psi",
        "psi",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "feed_pressure_bar": (
        "stage_3_feed_pressure_psi",
        "bar",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "stage_3_feed_pressure_psi": (
        "stage_3_feed_pressure_psi",
        "psi",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "permeate_pressure_psi": (
        "permeate_pressure_psi",
        "psi",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "permeate_pressure_bar": (
        "permeate_pressure_psi",
        "bar",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "device_dp_psi": (
        "stage_3_device_pressure_drop_psi",
        "psi",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "device_dp_bar": (
        "stage_3_device_pressure_drop_psi",
        "bar",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "stage_3_device_pressure_drop_psi": (
        "stage_3_device_pressure_drop_psi",
        "psi",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "permeate_flow_gpd": (
        "stage_3_permeate_flow_source_unit",
        "gpd",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "permeate_flow_gpm": (
        "stage_3_permeate_flow_source_unit",
        "gpm",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "stage_3_permeate_flow_source_unit": (
        "stage_3_permeate_flow_source_unit",
        "gpd",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "feed_flow_gpd": (
        "stage_3_feed_flow_source_unit",
        "gpd",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "feed_flow_gpm": (
        "stage_3_feed_flow_source_unit",
        "gpm",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "stage_3_feed_flow_source_unit": (
        "stage_3_feed_flow_source_unit",
        "gpd",
        MeasurementScope.STAGE_3,
        ValueKind.MEASURED,
    ),
    "feed_concentration_mg_l": (
        "feed_concentration_mg_l",
        "mg_L_NaCl_eq",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "cip": (
        "cip_flag_0_1",
        "flag_0_1",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "cip_flag_0_1": (
        "cip_flag_0_1",
        "flag_0_1",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "days_since_replacement": (
        "days_since_replacement_d",
        "day",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
    "days_since_replacement_d": (
        "days_since_replacement_d",
        "day",
        MeasurementScope.WHOLE_UNIT,
        ValueKind.MEASURED,
    ),
}

# Headers that are recognised as quantity columns but lack an explicit unit token.
AMBIGUOUS_UNIT_HEADERS: frozenset[str] = frozenset(
    {
        "permeate_flow",
        "feed_flow",
        "feed_pressure",
        "permeate_pressure",
        "device_dp",
        "temperature",
        "temp",
        "concentration",
        "feed_concentration",
    }
)

# Ambiguous headers (no unit token) → canonical quantity they would map to if resolved.
AMBIGUOUS_HEADER_TO_CANONICAL: dict[str, str] = {
    "permeate_flow": "stage_3_permeate_flow_source_unit",
    "feed_flow": "stage_3_feed_flow_source_unit",
    "feed_pressure": "stage_3_feed_pressure_psi",
    "permeate_pressure": "permeate_pressure_psi",
    "device_dp": "stage_3_device_pressure_drop_psi",
    "temperature": "temperature_c",
    "temp": "temperature_c",
    "concentration": "feed_concentration_mg_l",
    "feed_concentration": "feed_concentration_mg_l",
}

REQUIRED_CANONICAL_FOR_FORECAST: frozenset[str] = frozenset(
    {
        "observation_date_calendar",
        "temperature_c",
        "stage_3_feed_pressure_psi",
        "permeate_pressure_psi",
        "stage_3_device_pressure_drop_psi",
        "stage_3_permeate_flow_source_unit",
        "stage_3_feed_flow_source_unit",
        "feed_concentration_mg_l",
    }
)


@dataclass(frozen=True)
class SeriesIdResolution:
    source_series_id: str
    source: str  # explicit | inferred_column | default
    column: str | None = None


@dataclass(frozen=True)
class ResolvedField:
    external_name: str
    canonical_name: str
    unit: str
    source: str  # alias | mapping_yaml | mapping_json


@dataclass(frozen=True)
class AliasMappingSuccess:
    adapter: AdapterConfig
    matched: list[ResolvedField]


def _aliases_for_canonical(canonical_name: str) -> list[str]:
    return sorted(
        header for header, spec in CANONICAL_COLUMN_ALIASES.items() if spec[0] == canonical_name
    )


def _canonical_field_meta(
    canonical_name: str,
) -> tuple[MeasurementScope, ValueKind]:
    for _canonical, _unit, scope, kind in CANONICAL_COLUMN_ALIASES.values():
        if _canonical == canonical_name:
            return scope, kind
    return MeasurementScope.WHOLE_UNIT, ValueKind.MEASURED


def infer_series_id(
    frame: pd.DataFrame,
    *,
    explicit: str | None,
) -> SeriesIdResolution:
    """Infer or validate a single series id for the upload."""
    found_col: str | None = None
    values: list[str] = []
    for col in SERIES_ID_COLUMNS:
        if col not in frame.columns:
            continue
        vals = frame[col].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()
        if not vals:
            continue
        if len(vals) > 1:
            raise ForecastValidationError(
                f"Mixed series detected in column '{col}': {vals}. "
                "This assessment supports one series per request.",
                actionable_correction=(
                    "Split the CSV so each request contains a single series, " "then re-upload."
                ),
                code="mixed_series",
                details={"column": col, "values": vals},
            )
        found_col = col
        values = vals
        break

    if explicit is not None and str(explicit).strip():
        eid = str(explicit).strip()
        if values and values[0] != eid:
            raise ForecastValidationError(
                f"CSV {found_col}={values[0]!r} does not match source_series_id={eid!r}.",
                actionable_correction=(
                    "Align source_series_id with the CSV series column, or omit "
                    "source_series_id to use the inferred value."
                ),
                code="series_id_mismatch",
                details={"column": found_col, "csv_value": values[0], "explicit": eid},
            )
        return SeriesIdResolution(
            source_series_id=eid,
            source="explicit",
            column=found_col,
        )
    if values:
        return SeriesIdResolution(
            source_series_id=values[0],
            source="inferred_column",
            column=found_col,
        )
    return SeriesIdResolution(
        source_series_id=DEFAULT_SERIES_ID,
        source="default",
        column=None,
    )


def _parse_mapping_json(raw: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ForecastValidationError(
                f"Malformed mapping_json: {exc}",
                code="malformed_json",
                actionable_correction=(
                    "Supply mapping_json as a JSON object with a 'fields' array."
                ),
            ) from exc
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ForecastValidationError(
            "mapping_json must be a JSON object.",
            code="malformed_json",
            actionable_correction='Use {"fields":[...], "timestamp_column": "..."}.',
        )
    return parsed


def adapter_from_mapping_json(raw: dict[str, Any] | str) -> AdapterConfig:
    """Build an adapter from a simple Swagger/form mapping_json payload."""
    data = _parse_mapping_json(raw)
    fields_raw = data.get("fields")
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ForecastValidationError(
            "mapping_json.fields must be a non-empty array.",
            code="invalid_mapping_json",
            actionable_correction=(
                "Provide fields like "
                '{"external_name":"permeate_flow_gpd","canonical_name":'
                '"stage_3_permeate_flow_source_unit","unit":"gpd"}.'
            ),
        )
    fields: list[AdapterField] = []
    for item in fields_raw:
        if not isinstance(item, dict):
            raise ForecastValidationError(
                "Each mapping_json field must be an object.",
                code="invalid_mapping_json",
                actionable_correction="Include external_name, canonical_name, and unit.",
            )
        try:
            unit_symbol = item.get("unit") or (item.get("unit_spec") or {}).get("symbol")
            if not unit_symbol:
                raise KeyError("unit")
            scope = item.get("measurement_scope", "stage_3")
            if item.get("canonical_name") in {
                "observation_date_calendar",
                "temperature_c",
                "feed_concentration_mg_l",
                "cip_flag_0_1",
                "days_since_replacement_d",
                "permeate_pressure_psi",
            }:
                scope = item.get("measurement_scope", "whole_unit")
            fields.append(
                AdapterField(
                    external_name=str(item["external_name"]),
                    canonical_name=str(item["canonical_name"]),
                    unit=UnitSpec(symbol=str(unit_symbol)),
                    kind=ValueKind(item.get("kind", "measured")),
                    measurement_scope=MeasurementScope(scope),
                    confidence=str(item.get("confidence", "high")),
                    notes=item.get("notes"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastValidationError(
                f"Invalid mapping_json field entry: {exc}",
                code="invalid_mapping_json",
                actionable_correction=(
                    "Each field needs external_name, canonical_name, and unit symbol."
                ),
                details={"entry": item},
            ) from exc
    ts_col = data.get("timestamp_column") or next(
        (f.external_name for f in fields if f.canonical_name == "observation_date_calendar"),
        "observation_date",
    )
    return AdapterConfig(
        name=str(data.get("name") or "mapping_json_v1"),
        schema_id=str(data.get("schema_id") or "mapping_json_v1"),
        source_series_pattern=str(data.get("source_series_pattern") or ".+"),
        timestamp_column=str(ts_col),
        timestamp_format=str(data.get("timestamp_format") or "ISO8601"),
        timezone=str(data.get("timezone") or "UTC"),
        fields=fields,
        notes=data.get("notes"),
    )


def _adapter_from_alias_matches(matched: list[ResolvedField]) -> AliasMappingSuccess:
    fields: list[AdapterField] = []
    for resolved in matched:
        _c, unit, scope, kind = CANONICAL_COLUMN_ALIASES[resolved.external_name]
        fields.append(
            AdapterField(
                external_name=resolved.external_name,
                canonical_name=resolved.canonical_name,
                unit=UnitSpec(symbol=unit),
                kind=kind,
                measurement_scope=scope,
                confidence="high",
                notes="Resolved from documented CSV-first alias table.",
            )
        )
    ts = next(
        (f.external_name for f in fields if f.canonical_name == "observation_date_calendar"),
        "observation_date",
    )
    adapter = AdapterConfig(
        name="csv_first_alias_v1",
        schema_id="csv_first_alias_v1",
        source_series_pattern=".+",
        timestamp_column=ts,
        timestamp_format="ISO8601",
        timezone="UTC",
        fields=fields,
        notes=(
            "Auto-mapped from documented unit-bearing column aliases. "
            "Units taken from column names only."
        ),
    )
    return AliasMappingSuccess(adapter=adapter, matched=matched)


def _append_template_field(
    lines: list[str],
    *,
    external_name: str | None,
    canonical: str,
    unit_symbol: str | None,
) -> None:
    scope, kind = _canonical_field_meta(canonical)
    ext = "null" if external_name is None else external_name
    unit = "null" if unit_symbol is None else unit_symbol
    lines.extend(
        [
            f"  - external_name: {ext}",
            f"    canonical_name: {canonical}",
            f"    unit: {{ symbol: {unit} }}",
            f"    kind: {kind.value}",
            f"    measurement_scope: {scope.value}",
            "    confidence: high",
        ]
    )


def _build_assisted_mapping_template_yaml(
    *,
    confirmed_by_canonical: dict[str, ResolvedField],
    unresolved: list[dict[str, Any]],
) -> str:
    """Build a completable YAML mapping preserving confirmed required + optional fields.

    Required forecast fields always appear (confirmed values filled; unresolved
    names/units left as null). Confirmed optional fields such as CIP and
    replacement counters are included when present in the CSV — they are never
    invented when absent.
    """
    ts_external: str | None = None
    if "observation_date_calendar" in confirmed_by_canonical:
        ts_external = confirmed_by_canonical["observation_date_calendar"].external_name

    lines: list[str] = [
        "name: assisted_mapping_v1",
        "schema_id: assisted_mapping_v1",
        'source_series_pattern: ".+"',
        "timestamp:",
        f"  column: {ts_external if ts_external is not None else 'null'}",
        "  format: ISO8601",
        "  timezone: UTC",
        "notes: >",
        "  Complete null external_name and unit.symbol entries, save as YAML,",
        "  and resubmit as multipart mapping with your CSV. Confirmed optional",
        "  fields (e.g. cip, days_since_replacement) are preserved when present.",
        "fields:",
    ]
    unresolved_by_name = {u["canonical_name"]: u for u in unresolved}
    emitted: set[str] = set()
    for canonical in sorted(REQUIRED_CANONICAL_FOR_FORECAST):
        unresolved_entry = unresolved_by_name.get(canonical)
        if canonical in confirmed_by_canonical:
            resolved = confirmed_by_canonical[canonical]
            _append_template_field(
                lines,
                external_name=resolved.external_name,
                canonical=canonical,
                unit_symbol=resolved.unit,
            )
            emitted.add(canonical)
            continue
        reason = (unresolved_entry or {}).get("reason", "absent_column")
        if reason == "unit_unknown":
            comment = (
                "  # Column present but unit not in header name — set external_name "
                "and unit.symbol (e.g. gpd, gpm, psi); never infer from magnitude."
            )
        elif reason == "ambiguous":
            comment = (
                "  # Multiple columns match this quantity — keep one external_name "
                "and unit.symbol."
            )
        else:
            comment = (
                "  # Required column missing — set external_name to your CSV header "
                "and unit.symbol."
            )
        lines.append(comment)
        _append_template_field(
            lines,
            external_name=None,
            canonical=canonical,
            unit_symbol=None,
        )
        emitted.add(canonical)

    optional_confirmed = sorted(name for name in confirmed_by_canonical if name not in emitted)
    for canonical in optional_confirmed:
        resolved = confirmed_by_canonical[canonical]
        lines.append("  # Confirmed optional field from CSV — keep when completing the template.")
        _append_template_field(
            lines,
            external_name=resolved.external_name,
            canonical=canonical,
            unit_symbol=resolved.unit,
        )
    return "\n".join(lines) + "\n"


def mapping_required_payload(
    *,
    series: SeriesIdResolution,
    columns: list[str],
    confirmed: list[ResolvedField],
    unresolved_required_fields: list[dict[str, Any]],
    mapping_template_yaml: str,
) -> dict[str, Any]:
    """Structured response when alias mapping cannot resolve all required fields."""
    return {
        "status": "mapping_required",
        "source_series_id": series.source_series_id,
        "error": {
            "code": "mapping_required",
            "message": (
                "CSV headers did not resolve unambiguously to all required canonical "
                "fields via documented aliases."
            ),
            "actionable_correction": (
                "Fill nulls in mapping_template_yaml (units from names or explicit "
                "symbols only), save, and resubmit as multipart mapping. "
                "Units are never inferred from numeric magnitude."
            ),
            "details": {
                "detected_headers": list(columns),
                "confirmed_mappings": [r.__dict__ for r in confirmed],
                "unresolved_required_fields": unresolved_required_fields,
                "series_id_resolution": {
                    "source_series_id": series.source_series_id,
                    "source": series.source,
                    "column": series.column,
                },
            },
        },
        "mapping_template_yaml": mapping_template_yaml,
        "instructions": (
            "1) Copy mapping_template_yaml to a file (e.g. my_mapping.yaml). "
            "2) Replace null external_name and unit.symbol for each required field. "
            "3) Ensure timestamp.column matches the observation_date_calendar "
            "external_name. "
            "4) POST the same CSV with multipart field mapping=<completed YAML>."
        ),
    }


def try_resolve_alias_mapping(
    columns: list[str],
    *,
    series: SeriesIdResolution,
) -> AliasMappingSuccess | dict[str, Any]:
    """Resolve aliases or return an assisted mapping_required payload."""
    cols = [str(c) for c in columns]
    col_set = set(cols)

    ambiguous_present = sorted(col_set & AMBIGUOUS_UNIT_HEADERS)
    ambiguous_by_canonical: dict[str, list[str]] = {}
    for header in ambiguous_present:
        canonical = AMBIGUOUS_HEADER_TO_CANONICAL.get(header)
        if canonical:
            ambiguous_by_canonical.setdefault(canonical, []).append(header)

    matched: list[ResolvedField] = []
    by_canonical: dict[str, list[str]] = {}
    for col in cols:
        if col in SERIES_ID_COLUMNS:
            continue
        spec = CANONICAL_COLUMN_ALIASES.get(col)
        if spec is None:
            continue
        canonical, unit, _scope, _kind = spec
        matched.append(
            ResolvedField(
                external_name=col,
                canonical_name=canonical,
                unit=unit,
                source="alias",
            )
        )
        by_canonical.setdefault(canonical, []).append(col)

    duplicates = {k: v for k, v in by_canonical.items() if len(v) > 1}
    confirmed_by_canonical: dict[str, ResolvedField] = {}
    for resolved in matched:
        if resolved.canonical_name in duplicates:
            continue
        if resolved.canonical_name in ambiguous_by_canonical:
            continue
        confirmed_by_canonical[resolved.canonical_name] = resolved

    unresolved: list[dict[str, Any]] = []
    for canonical in sorted(REQUIRED_CANONICAL_FOR_FORECAST):
        if canonical in confirmed_by_canonical:
            continue
        suggestions = _aliases_for_canonical(canonical)
        if canonical in duplicates:
            unresolved.append(
                {
                    "canonical_name": canonical,
                    "reason": "ambiguous",
                    "explanation": (
                        "Multiple CSV columns map to the same canonical field via "
                        "documented aliases."
                    ),
                    "suggestions": duplicates[canonical],
                }
            )
            continue
        if canonical in ambiguous_by_canonical:
            unresolved.append(
                {
                    "canonical_name": canonical,
                    "reason": "unit_unknown",
                    "explanation": (
                        "A column name is recognised but omits an explicit unit token; "
                        "rename to a unit-bearing alias or set unit.symbol in mapping."
                    ),
                    "suggestions": ambiguous_by_canonical[canonical]
                    + [s for s in suggestions if s not in col_set],
                }
            )
            continue
        unresolved.append(
            {
                "canonical_name": canonical,
                "reason": "absent_column",
                "explanation": (
                    "No CSV header resolves to this required canonical field via "
                    "documented aliases."
                ),
                "suggestions": [s for s in suggestions if s not in col_set],
            }
        )

    if not unresolved:
        return _adapter_from_alias_matches(matched)

    confirmed = list(confirmed_by_canonical.values())
    template = _build_assisted_mapping_template_yaml(
        confirmed_by_canonical=confirmed_by_canonical,
        unresolved=unresolved,
    )
    return mapping_required_payload(
        series=series,
        columns=cols,
        confirmed=confirmed,
        unresolved_required_fields=unresolved,
        mapping_template_yaml=template,
    )


def resolve_alias_mapping(columns: list[str]) -> tuple[AdapterConfig, list[ResolvedField]]:
    """Map CSV headers via aliases; raises when incomplete (non-assisted callers)."""
    series = SeriesIdResolution(
        source_series_id=DEFAULT_SERIES_ID,
        source="default",
        column=None,
    )
    result = try_resolve_alias_mapping(columns, series=series)
    if isinstance(result, dict):
        err = result.get("error") or {}
        details = err.get("details") or {}
        unresolved = details.get("unresolved_required_fields") or []
        missing = [u["canonical_name"] for u in unresolved]
        raise ForecastValidationError(
            err.get("message", "Incomplete alias mapping."),
            code="incomplete_mapping",
            actionable_correction=err.get(
                "actionable_correction",
                "Supply mapping YAML / mapping_json.",
            ),
            details={
                "missing_canonical": missing,
                "unresolved_required_fields": unresolved,
            },
        )
    return result.adapter, result.matched


def adapter_to_mapping_yaml_dict(adapter: AdapterConfig) -> dict[str, Any]:
    return {
        "name": adapter.name,
        "schema_id": adapter.schema_id,
        "source_series_pattern": adapter.source_series_pattern,
        "timestamp": {
            "column": adapter.timestamp_column,
            "format": adapter.timestamp_format,
            "timezone": adapter.timezone,
        },
        "notes": adapter.notes,
        "fields": [
            {
                "external_name": f.external_name,
                "canonical_name": f.canonical_name,
                "unit": {"symbol": f.unit.symbol},
                "kind": f.kind.value,
                "measurement_scope": f.measurement_scope.value,
                "confidence": f.confidence,
                "notes": f.notes,
            }
            for f in adapter.fields
        ],
    }


def write_adapter_yaml(adapter: AdapterConfig, path: Any) -> None:
    path.write_text(
        yaml.safe_dump(adapter_to_mapping_yaml_dict(adapter), sort_keys=False),
        encoding="utf-8",
    )


def resolved_fields_from_adapter(adapter: AdapterConfig, *, source: str) -> list[ResolvedField]:
    return [
        ResolvedField(
            external_name=f.external_name,
            canonical_name=f.canonical_name,
            unit=f.unit.symbol,
            source=source,
        )
        for f in adapter.fields
    ]


def reference_required_payload(
    *,
    series: SeriesIdResolution,
    resolved_mapping: list[ResolvedField],
    mapping_source: str,
) -> dict[str, Any]:
    """Structured response when no usable documented reference was supplied."""
    return {
        "status": "reference_required",
        "source_series_id": series.source_series_id,
        "error": {
            "code": "reference_required",
            "message": (
                "A documented clean reference is required for relative-health "
                "forecasting. No reference was supplied, and the service will not "
                "invent one from the first row, maximum flow, or post-CIP row."
            ),
            "actionable_correction": (
                "Upload a documented reference YAML (preferred; JSON also accepted) "
                "with mode=documented, temperature_c, ndp_psi, permeate_flow, "
                "permeate_flow_unit, available_from, and provenance — see "
                "examples/forecast/reference.yaml. For batch, use "
                "examples/forecast/batch_references.yaml keyed by series id."
            ),
            "details": {
                "required_reference_fields": [
                    "mode=documented",
                    "temperature_c",
                    "ndp_psi",
                    "permeate_flow",
                    "permeate_flow_unit",
                    "available_from",
                    "provenance",
                ],
                "historical_window_reference": {
                    "available": False,
                    "reason": (
                        "Explicit historical-window reference mode is not enabled; "
                        "left unavailable rather than guessing a clean condition."
                    ),
                },
            },
        },
        "mapping": {
            "source": mapping_source,
            "resolved": [r.__dict__ for r in resolved_mapping],
        },
    }
