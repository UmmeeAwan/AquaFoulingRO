"""Input contract validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aqua_fouling.data.contracts import (
    ColumnMapping,
    ColumnMappingConfig,
    ReferenceConditionMetadata,
    TimestampConfig,
    UnitSpec,
    ValueKind,
)


def test_ambiguous_unit_rejected() -> None:
    with pytest.raises(ValidationError):
        UnitSpec(symbol="unknown")


def test_unspecified_timezone_rejected() -> None:
    with pytest.raises(ValidationError):
        TimestampConfig(column="date", format="ISO8601", timezone="local")


def test_reference_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        ReferenceConditionMetadata(declared=True, temperature_c=25.0)


def test_canonical_name_must_be_unit_explicit() -> None:
    with pytest.raises(ValidationError):
        ColumnMapping(
            external_name="temp",
            canonical_name="temperature",
            unit=UnitSpec(symbol="deg_C"),
            kind=ValueKind.MEASURED,
        )


def test_conductivity_to_tds_guess_rejected() -> None:
    with pytest.raises(ValidationError, match="conductivity-to-TDS"):
        ColumnMappingConfig(
            name="bad",
            mappings=[
                ColumnMapping(
                    external_name="perm_ec",
                    canonical_name="permeate_tds_mg_per_L",
                    unit=UnitSpec(symbol="mg_per_L"),
                    kind=ValueKind.ASSUMPTION,
                    notes="converted somehow",
                )
            ],
        )


def test_valid_mapping_accepted() -> None:
    cfg = ColumnMappingConfig(
        name="ok",
        mappings=[
            ColumnMapping(
                external_name="temp_c",
                canonical_name="temperature_c",
                unit=UnitSpec(symbol="deg_C"),
                kind=ValueKind.MEASURED,
            )
        ],
    )
    assert cfg.mappings[0].canonical_name == "temperature_c"
