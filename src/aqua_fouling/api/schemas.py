"""Typed response schemas for the API (pre-release presentation contract)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aqua_fouling import __version__


class HealthResponse(BaseModel):
    """Process liveness (distinct from forecast readiness)."""

    status: str = "ok"
    service: str = "aqua-fouling"
    version: str = __version__


class ReadyResponse(BaseModel):
    """Readiness to serve forecasts (artifact loaded/compatible)."""

    ready: bool
    model_version: str | None = None
    model_type: str | None = None
    normalisation_method_id: str | None = None
    normalisation_version: str | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None


class CapabilitiesResponse(BaseModel):
    """Concise capability statement for this build."""

    product: str
    version: str
    model: dict[str, Any] = Field(default_factory=dict)
    normalisation: dict[str, Any] = Field(default_factory=dict)
    endpoints: dict[str, bool] = Field(default_factory=dict)
    implemented: list[str] = Field(default_factory=list)
    not_implemented: list[str] = Field(default_factory=list)
    notes: str


class ValidateResponse(BaseModel):
    status: str
    source_series_id: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    missing_for_calculations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormaliseResponse(BaseModel):
    status: str
    source_series_id: str | None = None
    normalised: dict[str, Any] = Field(default_factory=dict)
    row_failures: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchForecastCounts(BaseModel):
    ok: int = 0
    reference_required: int = 0
    mapping_required: int = 0
    failed: int = 0
    total: int = 0


class BatchForecastSummary(BaseModel):
    series_column: str
    counts: BatchForecastCounts


class BatchForecastResponse(BaseModel):
    """Batch forecast envelope (per-series items may vary by status)."""

    overall_status: str
    summary: BatchForecastSummary
    results: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] | None = None


class ForecastResponse(BaseModel):
    """Grouped forecast success / unavailable response."""

    model_config = ConfigDict(extra="allow")

    status: str
    result: dict[str, Any] = Field(default_factory=dict)
    normalised_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
