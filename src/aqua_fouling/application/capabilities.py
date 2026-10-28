"""Capability statements for the forecast API and CLI."""

from __future__ import annotations

from typing import Any

from aqua_fouling import __version__
from aqua_fouling.physics.normalisation import NORMALISATION_METHOD_ID, NORMALISATION_VERSION


def get_capabilities() -> dict[str, Any]:
    return {
        "product": "aqua-fouling",
        "version": __version__,
        "model": {
            "model_version": "llt_relative_health_v1",
            "trained_state_space": True,
        },
        "normalisation": {
            "method_id": NORMALISATION_METHOD_ID,
            "version": NORMALISATION_VERSION,
        },
        "endpoints": {
            "health": True,
            "ready": True,
            "capabilities": True,
            "validate": True,
            "normalise": True,
            "forecast": True,
            "forecast_batch": True,
        },
        "implemented": [
            "validate_api_cli",
            "normalise_api_cli",
            "forecast_api_cli",
            "forecast_batch_api",
            "assisted_mapping",
            "trained_llt_kalman",
            "synthetic_benchmark_cli",
            "ocwd_s3sf_benchmark_cli",
            "train_cli",
            "evaluate_cli",
        ],
        "not_implemented": [
            "database",
            "authentication",
            "frontend",
            "silent_conductivity_to_tds",
            "historical_window_reference_mode",
        ],
        "notes": (
            "CSV forecasts via POST /v1/forecast and /v1/forecast/batch using the "
            "server-configured trained LLT artifact. Synthetic-trained; not "
            "plant-validated; OCWD independent-normalisation blockers remain unresolved."
        ),
    }
