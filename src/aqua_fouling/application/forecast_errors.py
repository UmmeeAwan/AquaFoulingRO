"""Structured, actionable forecast errors (no internal tracebacks)."""

from __future__ import annotations

from typing import Any

from aqua_fouling.application.response_presentation import error_envelope


class ForecastServiceError(Exception):
    """Base class for forecast application errors with HTTP-friendly payloads."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        actionable_correction: str,
        http_status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.actionable_correction = actionable_correction
        self.http_status = http_status
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return error_envelope(
            status="error",
            code=self.code,
            message=self.message,
            actionable_correction=self.actionable_correction,
            details=self.details,
        )


class ForecastValidationError(ForecastServiceError):
    def __init__(
        self,
        message: str,
        *,
        actionable_correction: str,
        code: str = "forecast_validation_error",
        details: dict[str, Any] | None = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            actionable_correction=actionable_correction,
            http_status=http_status,
            details=details,
        )


class ForecastConflictError(ForecastServiceError):
    def __init__(
        self,
        message: str,
        *,
        actionable_correction: str,
        code: str = "forecast_conflict",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            actionable_correction=actionable_correction,
            http_status=409,
            details=details,
        )


class ForecastPayloadTooLarge(ForecastServiceError):
    def __init__(
        self,
        message: str,
        *,
        actionable_correction: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="payload_too_large",
            message=message,
            actionable_correction=actionable_correction,
            http_status=413,
            details=details,
        )


class ForecastArtifactError(ForecastServiceError):
    def __init__(
        self,
        message: str,
        *,
        actionable_correction: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="artifact_unavailable",
            message=message,
            actionable_correction=actionable_correction,
            http_status=503,
            details=details,
        )


class ForecastUnavailableError(ForecastServiceError):
    """Inputs validated but forecast cannot be produced (insufficient history, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        actionable_correction: str,
        code: str = "forecast_unavailable",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            actionable_correction=actionable_correction,
            http_status=422,
            details=details,
        )
