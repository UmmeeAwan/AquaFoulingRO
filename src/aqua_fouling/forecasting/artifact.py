"""Versioned JSON model artifact schema (no pickle)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from aqua_fouling.forecasting.llt_kalman import LLTParameters, VarianceConstraintError
from aqua_fouling.physics.normalisation import NORMALISATION_METHOD_ID, NORMALISATION_VERSION

ARTIFACT_SCHEMA_VERSION = "1.0.0"
MODEL_TYPE = "local_linear_trend_kalman"
MODEL_VERSION = "llt_relative_health_v1"


class ArtifactCompatibilityError(ValueError):
    """Missing, corrupt, or incompatible model artifact."""


class LLTArtifact(BaseModel):
    """Schema-validated trained LLT artifact."""

    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    model_type: Literal["local_linear_trend_kalman"] = "local_linear_trend_kalman"
    model_version: str = MODEL_VERSION
    model_id: str = MODEL_VERSION
    parameters: dict[str, float]
    initialisation_policy: str = (
        "Prior mean health=1.0 (clean-reference scale), prior slope=0; "
        "P0=diag(p0_level, p0_slope). Filter state resets between independent "
        "series and between intervention/calendar-gap segments. Elapsed "
        "calendar days drive F(dt)/Q(dt); missing observations are predict-only."
    )
    training_objective: str = (
        "Maximize sum of Gaussian Kalman innovation log-likelihoods over "
        "designated training segments only (negative log-likelihood minimized "
        "via L-BFGS-B on log-variance parameters)."
    )
    training_series_ids: list[str]
    development_series_ids: list[str]
    held_out_from_training: list[str]
    normalisation_method_id: str = NORMALISATION_METHOD_ID
    normalisation_version: str = NORMALISATION_VERSION
    target: str = "relative_health"
    random_seed: int
    training_config_hash: str
    training_data_hash: str
    convergence: dict[str, Any]
    selected_settings: dict[str, Any] = Field(default_factory=dict)
    development_selection_note: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, float]) -> dict[str, float]:
        try:
            params = LLTParameters.from_dict(value)
        except (KeyError, TypeError, ValueError, VarianceConstraintError) as exc:
            msg = f"Invalid LLT parameters in artifact: {exc}"
            raise ValueError(msg) from exc
        return params.to_dict()

    def llt_parameters(self) -> LLTParameters:
        return LLTParameters.from_dict(self.parameters)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact(artifact: LLTArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_artifact(path: Path) -> LLTArtifact:
    if not path.is_file():
        msg = f"Model artifact not found: {path}"
        raise ArtifactCompatibilityError(msg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Corrupt model artifact (invalid JSON): {path}: {exc}"
        raise ArtifactCompatibilityError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Corrupt model artifact (expected JSON object): {path}"
        raise ArtifactCompatibilityError(msg)
    try:
        artifact = LLTArtifact.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface schema failures clearly
        msg = f"Incompatible or invalid model artifact {path}: {exc}"
        raise ArtifactCompatibilityError(msg) from exc
    if artifact.normalisation_method_id != NORMALISATION_METHOD_ID:
        msg = (
            f"Artifact normalisation_method_id={artifact.normalisation_method_id!r} "
            f"incompatible with runtime {NORMALISATION_METHOD_ID!r}."
        )
        raise ArtifactCompatibilityError(msg)
    if artifact.normalisation_version != NORMALISATION_VERSION:
        msg = (
            f"Artifact normalisation_version={artifact.normalisation_version!r} "
            f"incompatible with runtime {NORMALISATION_VERSION!r}."
        )
        raise ArtifactCompatibilityError(msg)
    if artifact.model_type != MODEL_TYPE:
        msg = f"Unsupported model_type={artifact.model_type!r}; expected {MODEL_TYPE!r}."
        raise ArtifactCompatibilityError(msg)
    return artifact


def artifact_file_digest(path: Path) -> str:
    return sha256_file(path)
