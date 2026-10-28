"""Validated input contracts for external-to-canonical column mappings.

Phase 1 does not implement RO normalisation. These models capture how source
columns map to unit-explicit canonical fields, and reject ambiguous units or
silent clean-reference assignment.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class ValueKind(str, Enum):
    """Whether a field is measured, derived in-source, or an external assumption."""

    MEASURED = "measured"
    DERIVED = "derived"
    ASSUMPTION = "assumption"


class UnitSpec(BaseModel):
    """Explicit unit declaration. Ambiguous or empty units are rejected."""

    symbol: Annotated[str, Field(min_length=1, description="Unit symbol, e.g. 'gfd' or 'psi'.")]
    description: str | None = None

    @field_validator("symbol")
    @classmethod
    def reject_ambiguous_units(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "Unit symbol must be non-empty and explicit."
            raise ValueError(msg)
        ambiguous = {
            "unitless?",
            "unknown",
            "n/a",
            "na",
            "tbd",
            "?",
            "misc",
            "various",
        }
        if cleaned.lower() in ambiguous:
            msg = (
                f"Ambiguous unit '{value}' is not allowed. "
                "Provide an explicit unit or mark the mapping as unresolved in documentation."
            )
            raise ValueError(msg)
        return cleaned


class TimestampConfig(BaseModel):
    """Timestamp parsing contract for source series."""

    column: Annotated[str, Field(min_length=1)]
    format: Annotated[
        str,
        Field(
            min_length=1,
            description="strftime-compatible format or 'ISO8601' for pandas ISO parsing.",
        ),
    ]
    timezone: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "IANA timezone name or 'UTC'. " "Local-naive sources must still declare intent."
            ),
        ),
    ]

    @field_validator("timezone")
    @classmethod
    def reject_unspecified_tz(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.lower() in {"", "local", "unspecified", "naive", "unknown"}:
            msg = (
                f"Timezone '{value}' is not actionable. "
                "Use an IANA name (e.g. 'America/Los_Angeles') or 'UTC'."
            )
            raise ValueError(msg)
        return cleaned


class ReferenceConditionMetadata(BaseModel):
    """Optional reference-condition metadata. Never silently invent a clean reference."""

    declared: bool = False
    temperature_c: float | None = None
    pressure_psi: float | None = None
    notes: str | None = None
    source: str | None = Field(
        default=None,
        description="Where the reference came from (document, operator, etc.).",
    )

    @model_validator(mode="after")
    def require_provenance_when_declared(self) -> ReferenceConditionMetadata:
        if self.declared and not self.source:
            msg = (
                "Reference conditions are declared but 'source' is missing. "
                "Do not silently assign a clean reference; cite provenance."
            )
            raise ValueError(msg)
        if not self.declared and any(
            v is not None for v in (self.temperature_c, self.pressure_psi)
        ):
            msg = (
                "Numeric reference values were provided but declared=false. "
                "Set declared=true and provide source provenance, or omit the values."
            )
            raise ValueError(msg)
        return self


class ColumnMapping(BaseModel):
    """One external column mapped to a unit-explicit canonical field."""

    external_name: Annotated[str, Field(min_length=1)]
    canonical_name: Annotated[
        str,
        Field(
            min_length=1,
            description="Unit-explicit canonical name, e.g. 'stage_3_flux_gfd'.",
        ),
    ]
    unit: UnitSpec
    kind: ValueKind
    notes: str | None = None

    @field_validator("canonical_name")
    @classmethod
    def require_unit_hint_in_name(cls, value: str) -> str:
        """Canonical names must encode unit intent (suffix) to avoid silent unit loss."""
        cleaned = value.strip()
        if "_" not in cleaned:
            msg = (
                f"Canonical name '{value}' must be unit-explicit "
                "(e.g. 'feed_pressure_psi', 'temperature_c')."
            )
            raise ValueError(msg)
        # Reject names that still look like raw ambiguous source labels without unit tail.
        ambiguous_tails = {"value", "val", "data", "raw", "x", "y"}
        tail = cleaned.rsplit("_", maxsplit=1)[-1].lower()
        if tail in ambiguous_tails:
            msg = (
                f"Canonical name '{value}' ends with ambiguous unit hint '{tail}'. "
                "Use an explicit unit token."
            )
            raise ValueError(msg)
        return cleaned


class MappingValidationError(ValueError):
    """Raised when a mapping configuration is invalid or incomplete."""


class ColumnMappingConfig(BaseModel):
    """External-to-canonical mapping set for one or more source series."""

    name: Annotated[str, Field(min_length=1)]
    mappings: Annotated[list[ColumnMapping], Field(min_length=1)]
    forbid_conductivity_to_tds_guess: bool = True
    notes: str | None = None

    @model_validator(mode="after")
    def validate_unique_and_no_tds_guess(self) -> ColumnMappingConfig:
        externals = [m.external_name for m in self.mappings]
        canonicals = [m.canonical_name for m in self.mappings]
        if len(externals) != len(set(externals)):
            msg = "Duplicate external_name entries in column mapping."
            raise MappingValidationError(msg)
        if len(canonicals) != len(set(canonicals)):
            msg = "Duplicate canonical_name entries in column mapping."
            raise MappingValidationError(msg)

        if self.forbid_conductivity_to_tds_guess:
            for mapping in self.mappings:
                canon = mapping.canonical_name.lower()
                ext = mapping.external_name.lower()
                looks_like_ec_to_tds = "tds" in canon and any(
                    tok in ext for tok in ("ec", "cond", "conductivity")
                )
                missing_coefficient = (
                    mapping.kind == ValueKind.ASSUMPTION
                    and "coefficient" not in (mapping.notes or "").lower()
                )
                if looks_like_ec_to_tds and missing_coefficient:
                    msg = (
                        f"Mapping {mapping.external_name} -> {mapping.canonical_name} "
                        "looks like conductivity-to-TDS conversion without an explicit "
                        "coefficient in notes. Do not guess conversion coefficients."
                    )
                    raise MappingValidationError(msg)
        return self


class SourceSeriesConfig(BaseModel):
    """Configuration for a single source series (e.g. one CSV file)."""

    series_id: str | None = Field(
        default=None,
        description="Optional identifier such as 'A01'. Not assumed to encode stage.",
    )
    path: Annotated[str, Field(min_length=1)]
    timestamp: TimestampConfig
    column_mapping: ColumnMappingConfig
    reference: ReferenceConditionMetadata = Field(default_factory=ReferenceConditionMetadata)
    index_column: str | None = Field(
        default=None,
        description="Optional exported index column name to ignore during mapping.",
    )
