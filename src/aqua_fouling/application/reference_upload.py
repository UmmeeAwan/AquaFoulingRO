"""Parse documented reference uploads (YAML preferred; JSON also accepted).

Single forecast: one reference object.
Batch forecast: mapping keyed by series id → reference objects.

Uses ``yaml.safe_load`` / ``json.loads`` only — never ``yaml.load``.
Field-level validation remains with ``ReferenceConfig`` on the forecast path.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from aqua_fouling.application.forecast_errors import ForecastValidationError

_SINGLE_HINT_KEYS = frozenset(
    {
        "mode",
        "provenance",
        "available_from",
        "temperature_c",
        "ndp_psi",
        "permeate_flow",
        "permeate_flow_unit",
        "window_start",
        "window_end",
        "prediction_cutoff",
        "notes",
    }
)


def _looks_like_single_reference(raw: dict[str, Any]) -> bool:
    """True when top-level resembles one documented/explicit_window reference."""
    mode = raw.get("mode")
    if isinstance(mode, str):
        return True
    # Scalar-heavy mapping with known reference fields (no nested reference objects).
    if any(isinstance(v, dict) for v in raw.values()):
        return False
    return bool(_SINGLE_HINT_KEYS.intersection(raw.keys()))


def _looks_like_batch_references(raw: dict[str, Any]) -> bool:
    """True when every value is a mapping (series id → reference object)."""
    if not raw:
        return False
    if _looks_like_single_reference(raw):
        return False
    return all(isinstance(v, dict) for v in raw.values())


def parse_reference_document(text: str, *, label: str = "reference") -> dict[str, Any]:
    """Parse YAML or JSON text into a mapping.

    JSON objects are accepted for compatibility; YAML is the documented format.
    """
    if text.strip() == "":
        raise ForecastValidationError(
            f"{label} upload is empty.",
            code="malformed_reference",
            actionable_correction=(
                "Upload a non-empty YAML reference "
                "(see examples/forecast/reference.yaml) or equivalent JSON."
            ),
        )

    stripped = text.lstrip()
    raw: Any
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as json_exc:
            try:
                raw = yaml.safe_load(text)
            except yaml.YAMLError:
                raise ForecastValidationError(
                    f"Malformed JSON for {label}: {json_exc}",
                    code="malformed_json",
                    actionable_correction=(
                        f"Supply valid JSON or YAML for {label}. " "Documented format is YAML."
                    ),
                    details={"label": label, "error_type": type(json_exc).__name__},
                ) from json_exc
    else:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ForecastValidationError(
                f"Malformed or unreadable {label} YAML: {exc}",
                code="malformed_yaml",
                actionable_correction=(
                    f"Supply valid YAML for {label}. Check indentation, colons, "
                    "and quotes. See examples/forecast/reference.yaml."
                ),
                details={"label": label, "error_type": type(exc).__name__},
            ) from exc

    if raw is None:
        raise ForecastValidationError(
            f"{label} document is empty after parsing.",
            code="malformed_reference",
            actionable_correction=(
                "Provide a YAML mapping with reference fields "
                "(mode, provenance, available_from, …)."
            ),
        )
    if not isinstance(raw, dict):
        raise ForecastValidationError(
            f"{label} must be a YAML/JSON object (mapping), not {type(raw).__name__}.",
            code="malformed_reference",
            actionable_correction=(
                "Use a mapping/object at the top level — "
                "a single reference for /v1/forecast, or series-id keys for batch."
            ),
            details={"parsed_type": type(raw).__name__},
        )
    return raw


def parse_single_reference_upload(text: str) -> dict[str, Any]:
    """Parse one documented reference object (YAML preferred; JSON accepted)."""
    raw = parse_reference_document(text, label="reference")
    if _looks_like_batch_references(raw):
        raise ForecastValidationError(
            "Expected a single reference object, not a series-keyed batch map.",
            code="malformed_reference",
            actionable_correction=(
                "For POST /v1/forecast upload one reference "
                "(examples/forecast/reference.yaml). "
                "For multiple series use POST /v1/forecast/batch with "
                "examples/forecast/batch_references.yaml."
            ),
        )
    if not _looks_like_single_reference(raw):
        raise ForecastValidationError(
            "Reference upload is not a recognised single reference object.",
            code="malformed_reference",
            actionable_correction=(
                "Include mode, provenance, available_from, temperature_c, "
                "ndp_psi, permeate_flow, and permeate_flow_unit "
                "(see examples/forecast/reference.yaml)."
            ),
        )
    return raw


def parse_batch_references_upload(text: str) -> dict[str, dict[str, Any]]:
    """Parse series-id → reference mapping (YAML preferred; JSON accepted)."""
    raw = parse_reference_document(text, label="references")
    if _looks_like_single_reference(raw):
        raise ForecastValidationError(
            "Batch references must be keyed by series id, not a single reference object.",
            code="malformed_reference",
            actionable_correction=(
                "Use a YAML mapping like examples/forecast/batch_references.yaml: "
                "each top-level key is a series id; each value is one reference object."
            ),
        )
    if not _looks_like_batch_references(raw):
        raise ForecastValidationError(
            "references must be an object keyed by series id whose values are "
            "reference objects.",
            code="malformed_reference",
            actionable_correction=(
                "Example shape:\n"
                "  DEMO_A:\n"
                "    mode: documented\n"
                "    provenance: …\n"
                "    …\n"
                "See examples/forecast/batch_references.yaml."
            ),
        )
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ForecastValidationError(
                f"references[{key!r}] must be a reference object.",
                code="malformed_reference",
                actionable_correction=(
                    "Each series entry must be a mapping with documented "
                    "reference fields (mode, provenance, temperature_c, …)."
                ),
            )
        sid = str(key)
        if not sid.strip() or sid.lower() == "nan":
            raise ForecastValidationError(
                "Blank series ids are not allowed as reference keys.",
                code="malformed_reference",
                actionable_correction="Use non-empty string series ids matching the CSV.",
            )
        out[sid] = value
    return out
