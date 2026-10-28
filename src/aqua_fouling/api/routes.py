"""Thin API routes — scientific work runs off the async event loop."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from aqua_fouling.api.schemas import (
    CapabilitiesResponse,
    ForecastResponse,
    HealthResponse,
    NormaliseResponse,
    ReadyResponse,
    ValidateResponse,
)
from aqua_fouling.application.capabilities import get_capabilities
from aqua_fouling.application.forecast_csv_export import (
    export_batch_forecast_csv,
    export_single_forecast_csv,
)
from aqua_fouling.application.forecast_errors import (
    ForecastServiceError,
    ForecastValidationError,
)
from aqua_fouling.application.forecast_service import (
    check_artifact_ready,
    forecast_batch_csv,
    forecast_csv,
)
from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.application.reference_upload import (
    parse_batch_references_upload,
    parse_single_reference_upload,
)
from aqua_fouling.application.validate import validate_csv
from aqua_fouling.data.adapters import AdapterConfigError

_CSV_MEDIA = "text/csv; charset=utf-8"


def _normalize_response_format(value: str | None) -> str:
    fmt = (value or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        raise ForecastValidationError(
            f"Unsupported response_format={value!r}.",
            code="invalid_response_format",
            actionable_correction="Use response_format=json (default) or response_format=csv.",
        )
    return fmt


def _csv_attachment(content: str, *, filename: str) -> Response:
    return Response(
        content=content.encode("utf-8"),
        media_type=_CSV_MEDIA,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _write_parsed_yaml(data: dict[str, Any], *, suffix: str) -> Path:
    """Serialize a mapping with safe_dump for downstream YAML loaders."""
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    path = Path(tmp_name)
    try:
        with open(tmp_fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


router = APIRouter()

_READ_CHUNK = 64 * 1024


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process liveness (does not check model artifact)."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Forecast readiness: configured artifact present and compatible."""
    settings = request.app.state.settings
    payload = check_artifact_ready(Path(settings.artifact_path))
    return ReadyResponse.model_validate(payload)


@router.get("/v1/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    """Describe implemented and non-implemented API/CLI capabilities."""
    payload = get_capabilities()
    return CapabilitiesResponse.model_validate(payload)


def _write_upload(upload: UploadFile, suffix: str, *, max_bytes: int) -> tuple[Path, int]:
    """Stream an upload to a temp file, enforcing ``max_bytes`` before finishing."""
    total = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = upload.file.read(_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ForecastServiceError(
                        code="payload_too_large",
                        message=f"Upload exceeds max_upload_bytes={max_bytes}.",
                        actionable_correction=(
                            "Reduce the file size or raise AQUA_MAX_UPLOAD_BYTES " "on the server."
                        ),
                        http_status=413,
                        details={
                            "upload_bytes_at_least": total,
                            "max_upload_bytes": max_bytes,
                        },
                    )
                tmp.write(chunk)
        return tmp_path, total
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _client_yaml_error(exc: BaseException, *, label: str) -> ForecastValidationError:
    return ForecastValidationError(
        f"Malformed or unreadable {label}: {exc}",
        code="malformed_yaml",
        actionable_correction=(
            f"Supply valid YAML for the {label}. Check indentation, colons, and quotes."
        ),
        details={"label": label, "error_type": type(exc).__name__},
    )


def _client_json_error(exc: BaseException, *, label: str) -> ForecastValidationError:
    return ForecastValidationError(
        f"Malformed JSON for {label}: {exc}",
        code="malformed_json",
        actionable_correction=f"Supply valid JSON for {label}.",
        details={"label": label, "error_type": type(exc).__name__},
    )


@router.post("/v1/validate", response_model=ValidateResponse)
async def validate_endpoint(
    request: Request,
    csv: Annotated[UploadFile, File(description="Plant operating-history CSV")],
    source_series_id: Annotated[str, Form(description="Single series identifier")],
    mapping: Annotated[UploadFile | None, File(description="Optional adapter mapping YAML")] = None,
    reference: Annotated[UploadFile | None, File(description="Optional reference YAML")] = None,
) -> ValidateResponse | JSONResponse:
    """Validate an uploaded CSV against mapping/reference configuration."""
    settings = request.app.state.settings
    paths: list[Path] = []
    try:
        csv_path, _ = _write_upload(csv, ".csv", max_bytes=settings.max_upload_bytes)
        paths.append(csv_path)
        mapping_path = None
        if mapping is not None:
            mapping_path, _ = _write_upload(mapping, ".yaml", max_bytes=settings.max_upload_bytes)
            paths.append(mapping_path)
        reference_path = None
        if reference is not None:
            reference_path, _ = _write_upload(
                reference, ".yaml", max_bytes=settings.max_upload_bytes
            )
            paths.append(reference_path)
        try:
            result = await asyncio.to_thread(
                validate_csv,
                csv_path,
                source_series_id=source_series_id,
                mapping_path=mapping_path,
                reference_path=reference_path,
            )
        except AdapterConfigError as exc:
            raise _client_yaml_error(exc, label="mapping") from exc
        except yaml.YAMLError as exc:
            raise _client_yaml_error(exc, label="mapping or reference") from exc
        return ValidateResponse.model_validate(result)
    except ForecastServiceError as exc:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@router.post("/v1/normalise", response_model=NormaliseResponse)
async def normalise_endpoint(
    request: Request,
    csv: Annotated[UploadFile, File(description="Plant operating-history CSV")],
    source_series_id: Annotated[str, Form()],
    mapping: Annotated[UploadFile | None, File()] = None,
    reference: Annotated[UploadFile | None, File()] = None,
    concentration_mg_l: Annotated[float | None, Form()] = None,
    conductivity_calibration_json: Annotated[str | None, Form()] = None,
) -> NormaliseResponse | JSONResponse:
    """Normalise an uploaded CSV (independent path when inputs allow)."""
    settings = request.app.state.settings
    paths: list[Path] = []
    try:
        csv_path, _ = _write_upload(csv, ".csv", max_bytes=settings.max_upload_bytes)
        paths.append(csv_path)
        mapping_path = None
        if mapping is not None:
            mapping_path, _ = _write_upload(mapping, ".yaml", max_bytes=settings.max_upload_bytes)
            paths.append(mapping_path)
        reference_path = None
        if reference is not None:
            reference_path, _ = _write_upload(
                reference, ".yaml", max_bytes=settings.max_upload_bytes
            )
            paths.append(reference_path)
        try:
            calibration = (
                json.loads(conductivity_calibration_json) if conductivity_calibration_json else None
            )
        except json.JSONDecodeError as exc:
            raise _client_json_error(exc, label="conductivity_calibration_json") from exc
        try:
            result = await asyncio.to_thread(
                normalise_csv,
                csv_path,
                source_series_id=source_series_id,
                mapping_path=mapping_path,
                reference_path=reference_path,
                concentration_mg_l=concentration_mg_l,
                conductivity_calibration=calibration,
            )
        except AdapterConfigError as exc:
            raise _client_yaml_error(exc, label="mapping") from exc
        except yaml.YAMLError as exc:
            raise _client_yaml_error(exc, label="mapping or reference") from exc
        return NormaliseResponse.model_validate(result)
    except ForecastServiceError as exc:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@router.post(
    "/v1/forecast",
    response_model=None,
    responses={
        200: {
            "description": "Forecast JSON (default) or CSV attachment when response_format=csv.",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ForecastResponse"}},
                "text/csv": {
                    "schema": {"type": "string", "format": "binary"},
                    "example": "source_series_id,status,forecast_origin,...\n",
                },
            },
        },
        422: {
            "description": "mapping_required / reference_required / forecast_unavailable (JSON)."
        },
    },
)
async def forecast_endpoint(
    request: Request,
    csv: Annotated[
        UploadFile,
        File(description="Plant operating-history CSV (one series). Only required field."),
    ],
    source_series_id: Annotated[
        str | None,
        Form(
            description=(
                "Optional series id. If omitted, inferred from CSV column "
                "source_series_id/series_id/unit_id, else 'uploaded_series'."
            )
        ),
    ] = None,
    reference: Annotated[
        UploadFile | None,
        File(
            description=(
                "Optional documented reference (YAML preferred; JSON also "
                "accepted). One reference object. If omitted, returns "
                "status=reference_required (never invents a clean baseline). "
                "See examples/forecast/reference.yaml."
            )
        ),
    ] = None,
    mapping: Annotated[
        UploadFile | None,
        File(
            description=(
                "Optional adapter mapping YAML. If omitted, uses documented "
                "unit-bearing column aliases or the default canonical schema."
            )
        ),
    ] = None,
    mapping_json: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON mapping for Swagger corrections "
                '(e.g. {"fields":[{"external_name":"...","canonical_name":"...","unit":"gpd"}]}).'
            )
        ),
    ] = None,
    horizon_days: Annotated[
        int | None,
        Form(description="Forecast horizon in calendar days (default 14; allowed 7, 14, 30)"),
    ] = None,
    threshold: Annotated[
        float | None,
        Form(
            description=(
                "Relative-health threshold override (default from server config, "
                "typically 0.85 vs documented reference)."
            )
        ),
    ] = None,
    concentration_mg_l: Annotated[float | None, Form()] = None,
    response_format: Annotated[
        str | None,
        Form(
            description=(
                "Optional response encoding: json (default) or csv. "
                "CSV returns an attachment of future forecast rows; "
                "request-wide failures still return structured JSON."
            )
        ),
    ] = None,
) -> ForecastResponse | JSONResponse | Response:
    """CSV-first forecast using the server-configured trained artifact.

    Only ``csv`` is required. Series id, mapping and reference are optional.
    Artifact path is selected via AQUA_ARTIFACT_PATH / server config only.
    """
    settings = request.app.state.settings
    paths: list[Path] = []
    try:
        fmt = _normalize_response_format(response_format)
        csv_path, upload_bytes = _write_upload(csv, ".csv", max_bytes=settings.max_upload_bytes)
        paths.append(csv_path)
        ref_path = None
        if reference is not None:
            upload_path, _ = _write_upload(
                reference, ".ref_upload", max_bytes=settings.max_upload_bytes
            )
            paths.append(upload_path)
            ref_obj = parse_single_reference_upload(upload_path.read_text(encoding="utf-8"))
            ref_path = _write_parsed_yaml(ref_obj, suffix=".reference.yaml")
            paths.append(ref_path)
        mapping_path = None
        if mapping is not None:
            mapping_path, _ = _write_upload(mapping, ".yaml", max_bytes=settings.max_upload_bytes)
            paths.append(mapping_path)

        try:
            result = await asyncio.to_thread(
                forecast_csv,
                csv_path,
                source_series_id=source_series_id,
                mapping_path=mapping_path,
                mapping_json=mapping_json,
                reference_path=ref_path,
                horizon_days=horizon_days,
                threshold=threshold,
                artifact_path=Path(settings.artifact_path),
                concentration_mg_l=concentration_mg_l,
                upload_bytes=upload_bytes,
            )
        except AdapterConfigError as exc:
            raise _client_yaml_error(exc, label="mapping") from exc
        except yaml.YAMLError as exc:
            raise _client_yaml_error(exc, label="mapping or reference") from exc
        if result.get("status") in {"reference_required", "mapping_required"}:
            # Assisted / missing-reference outcomes stay JSON even when CSV asked.
            return JSONResponse(status_code=422, content=result)
        if fmt == "csv":
            return _csv_attachment(
                export_single_forecast_csv(result),
                filename="forecast.csv",
            )
        return ForecastResponse.model_validate(result)
    except ForecastServiceError as exc:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@router.post(
    "/v1/forecast/batch",
    response_model=None,
    responses={
        200: {
            "description": (
                "Batch JSON envelope or CSV attachment (response_format=csv). "
                "partial_success / all-failed still HTTP 200 with explicit failure rows in CSV."
            ),
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/BatchForecastResponse"}
                },
                "text/csv": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        }
    },
)
async def forecast_batch_endpoint(
    request: Request,
    csv: Annotated[
        UploadFile,
        File(description="Multi-series plant operating-history CSV."),
    ],
    series_column: Annotated[
        str | None,
        Form(
            description=(
                "Column holding series ids. If omitted, exactly one of "
                "source_series_id / series_id / unit_id must be present."
            )
        ),
    ] = None,
    mapping: Annotated[UploadFile | None, File(description="Shared adapter mapping YAML")] = None,
    mapping_json: Annotated[str | None, Form(description="Shared JSON mapping")] = None,
    references: Annotated[
        UploadFile | None,
        File(
            description=(
                "Optional per-series references (YAML preferred; JSON also "
                "accepted). Top-level keys are series ids; each value is one "
                "reference object. See examples/forecast/batch_references.yaml."
            )
        ),
    ] = None,
    horizon_days: Annotated[int | None, Form()] = None,
    threshold: Annotated[float | None, Form()] = None,
    concentration_mg_l: Annotated[float | None, Form()] = None,
    response_format: Annotated[
        str | None,
        Form(
            description=(
                "Optional response encoding: json (default) or csv. "
                "Partial batches export success rows plus one failure row per "
                "failed series. Request-wide validation failures stay JSON."
            )
        ),
    ] = None,
) -> JSONResponse | Response:
    """Batch forecast: one shared mapping, independent reference per series id."""
    settings = request.app.state.settings
    paths: list[Path] = []
    try:
        fmt = _normalize_response_format(response_format)
        csv_path, upload_bytes = _write_upload(csv, ".csv", max_bytes=settings.max_upload_bytes)
        paths.append(csv_path)
        mapping_path = None
        if mapping is not None:
            mapping_path, _ = _write_upload(mapping, ".yaml", max_bytes=settings.max_upload_bytes)
            paths.append(mapping_path)
        refs_dict: dict[str, dict[str, Any]] | None = None
        if references is not None:
            upload_path, _ = _write_upload(
                references, ".refs_upload", max_bytes=settings.max_upload_bytes
            )
            paths.append(upload_path)
            refs_dict = parse_batch_references_upload(upload_path.read_text(encoding="utf-8"))

        try:
            result = await asyncio.to_thread(
                forecast_batch_csv,
                csv_path,
                series_column=series_column,
                mapping_path=mapping_path,
                mapping_json=mapping_json,
                references=refs_dict,
                horizon_days=horizon_days,
                threshold=threshold,
                artifact_path=Path(settings.artifact_path),
                concentration_mg_l=concentration_mg_l,
                upload_bytes=upload_bytes,
            )
        except AdapterConfigError as exc:
            raise _client_yaml_error(exc, label="mapping") from exc
        except yaml.YAMLError as exc:
            raise _client_yaml_error(exc, label="mapping") from exc
        if fmt == "csv":
            # Valid batch envelopes (ok / partial_success / failed) → CSV attachment.
            return _csv_attachment(
                export_batch_forecast_csv(result),
                filename="forecast_batch.csv",
            )
        return JSONResponse(status_code=200, content=result)
    except ForecastServiceError as exc:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


def register_exception_handlers(app: Any) -> None:
    """Map ForecastServiceError to structured JSON without tracebacks."""

    async def _forecast_error_handler(_request: Request, exc: ForecastServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    async def _http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "actionable_correction": (
                        "Check the request path, method, and multipart fields."
                    ),
                    "details": {},
                },
            },
        )

    app.add_exception_handler(ForecastServiceError, _forecast_error_handler)
    app.add_exception_handler(HTTPException, _http_error_handler)
