"""Data access helpers: archive extraction, contracts, adapters."""

from aqua_fouling.data.adapters import (
    AdapterConfig,
    MeasurementScope,
    adapt_dataframe,
    detect_schema_id,
    load_adapter_for_schema,
)
from aqua_fouling.data.archive import SafeExtractResult, safe_extract_zip, sha256_file
from aqua_fouling.data.contracts import (
    ColumnMappingConfig,
    MappingValidationError,
    ReferenceConditionMetadata,
    SourceSeriesConfig,
    TimestampConfig,
    UnitSpec,
    ValueKind,
)
from aqua_fouling.data.reference import ReferenceConfig, ReferenceMode

__all__ = [
    "AdapterConfig",
    "ColumnMappingConfig",
    "MappingValidationError",
    "MeasurementScope",
    "ReferenceConditionMetadata",
    "ReferenceConfig",
    "ReferenceMode",
    "SafeExtractResult",
    "SourceSeriesConfig",
    "TimestampConfig",
    "UnitSpec",
    "ValueKind",
    "adapt_dataframe",
    "detect_schema_id",
    "load_adapter_for_schema",
    "safe_extract_zip",
    "sha256_file",
]
