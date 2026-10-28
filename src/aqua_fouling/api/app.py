"""FastAPI application factory and lifespan."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aqua_fouling import __version__
from aqua_fouling.api.routes import register_exception_handlers, router
from aqua_fouling.application.forecast_service import check_artifact_ready
from aqua_fouling.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configured artifact at startup (readiness); do not block liveness."""
    settings: Settings = app.state.settings
    status = check_artifact_ready(Path(settings.artifact_path))
    app.state.forecast_ready = bool(status.get("ready"))
    app.state.forecast_readiness = status
    if status.get("ready"):
        logger.info(
            "Forecast artifact ready: %s (%s)",
            status.get("artifact_path"),
            status.get("model_version"),
        )
    else:
        logger.warning(
            "Forecast artifact not ready at %s: %s",
            status.get("artifact_path"),
            status.get("reason"),
        )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    When ``settings`` is omitted, uses ``get_settings()``
    (process env > .env > serving YAML > defaults).

    When ``settings`` is supplied explicitly, it is used as-is (no YAML
    re-application). Callers who want YAML merged should pass ``get_settings()``
    or ``apply_serving_yaml(Settings(...))``.
    """
    cfg = settings if settings is not None else get_settings()
    app = FastAPI(
        title=cfg.app_name,
        version=__version__,
        description=(
            "Assessment software for reverse-osmosis membrane fouling early-warning "
            "forecasts. CSV forecast via POST /v1/forecast using the server-configured "
            "trained local-linear-trend artifact. Artifact path is not caller-supplied. "
            "Brackish / NaCl-equivalent demonstration domain with synthetic validation; "
            "not plant-validated. Interactive docs: /docs"
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.forecast_ready = False
    app.state.forecast_readiness = {"ready": False}
    register_exception_handlers(app)
    app.include_router(router)
    return app
