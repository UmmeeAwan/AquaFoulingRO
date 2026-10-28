"""Public API response shaping (presentation only — no science changes)."""

from __future__ import annotations

from typing import Any

_PATHISH_KEYS = frozenset(
    {
        "artifact_path",
        "artifact_path_configured",
        "mapping_path",
        "reference_path",
        "csv_path",
        "path",
    }
)

INTERVAL_DEFINITIONS: dict[str, dict[str, str]] = {
    "latent_health_interval": {
        "label": "latent_relative_health",
        "meaning": (
            "Gaussian ±1.96σ interval for the latent relative-health state "
            "(not empirically calibrated confidence)."
        ),
    },
    "future_observation_interval": {
        "label": "future_measured_relative_health",
        "meaning": (
            "Gaussian ±1.96σ predictive interval for a future measured "
            "relative-health observation (latent variance + R). "
            "Not empirically calibrated confidence."
        ),
    },
}


def strip_pathish(value: Any) -> Any:
    """Recursively drop filesystem-path fields from public payloads."""
    if isinstance(value, dict):
        return {
            k: strip_pathish(v)
            for k, v in value.items()
            if k not in _PATHISH_KEYS and not str(k).endswith("_path")
        }
    if isinstance(value, list):
        return [strip_pathish(v) for v in value]
    return value


def public_reference(ref: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reference values/provenance without internal checklist fields."""
    if not ref:
        return None
    out = {k: v for k, v in ref.items() if k != "forbidden_practices_not_used" and v is not None}
    return out or None


def concise_threshold_provenance(raw: str) -> str:
    """Public threshold wording without phase labels or plant-rule implication."""
    text = " ".join(str(raw).split())
    # Drop chronological phase labels from serving YAML if present.
    for token in ("Phase 3/4 ", "Phase 3 ", "Phase 4 ", "Phase 5 "):
        text = text.replace(token, "")
    if "not a manufacturer" not in text.lower() and "not plant-validated" not in text.lower():
        text = (
            f"{text} Analytical relative-health threshold versus the caller-supplied "
            "documented reference — not a manufacturer cleaning recommendation and "
            "not plant-validated."
        )
    return text.strip()


def error_envelope(
    *,
    code: str,
    message: str,
    actionable_correction: str,
    details: dict[str, Any] | None = None,
    status: str = "error",
) -> dict[str, Any]:
    return {
        "status": status,
        "error": {
            "code": code,
            "message": message,
            "actionable_correction": actionable_correction,
            "details": strip_pathish(details or {}),
        },
    }


def forecast_point_public(
    *,
    date: str,
    relative_health_point: float | None,
    latent_low: float | None,
    latent_high: float | None,
    obs_low: float | None,
    obs_high: float | None,
) -> dict[str, Any]:
    """Daily forecast values with interval bounds only (definitions in metadata)."""
    return {
        "date": date,
        "relative_health_point": relative_health_point,
        "latent_health_interval": {"low": latent_low, "high": latent_high},
        "future_observation_interval": {"low": obs_low, "high": obs_high},
    }
