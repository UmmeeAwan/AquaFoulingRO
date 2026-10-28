"""Application settings: process env > .env > serving YAML > defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for API and CLI (artifact path is server-configured).

    Precedence when resolved via ``get_settings()``:
    process environment (``AQUA_*``) > ``.env`` > ``forecast_serving.yaml`` > field defaults.

    Explicit ``Settings(...)`` constructor keyword arguments follow pydantic-settings
    rules (init kwargs beat env). When passed to ``create_app(settings=...)``, the
    object is used as-is and serving YAML is **not** re-applied — callers own the
    composition (use ``get_settings()`` or ``apply_serving_yaml`` if YAML is needed).
    """

    model_config = SettingsConfigDict(
        env_prefix="AQUA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "aqua-fouling"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    default_archive: Path = Path("data/raw/dataverse_files.zip")
    default_audit_output: Path = Path("reports/data-audit")
    default_extract_dir: Path = Path("data/extracted")

    artifact_path: Path = Path("artifacts/models/llt_relative_health_v1.json")
    default_mapping: Path = Path("configs/canonical_generic_schema.yaml")
    default_horizon_days: int = 14
    min_horizon_days: int = 1
    max_horizon_days: int = 30
    allowed_horizons_days: list[int] = Field(default_factory=lambda: [7, 14, 30])
    default_threshold_relative_health: float = 0.85
    min_threshold_relative_health: float = 0.50
    max_threshold_relative_health: float = 0.99
    min_history_days: int = 21
    min_history_observations: int = 5
    max_upload_bytes: int = 5_242_880
    max_csv_rows: int = 5000
    max_batch_series: int = 50
    threshold_provenance: str = (
        "Relative-health threshold on independently normalised permeate flow versus "
        "the caller-supplied documented reference. Default 0.85 matches the synthetic "
        "analytical threshold (15% decline from clean reference). Not a manufacturer "
        "cleaning recommendation; not plant-validated."
    )
    domain_limits: str = (
        "Brackish / NaCl-equivalent demonstration domain compatible with "
        "filmtec_cfc_bar_lt_20000_v1 (C_fc < 20000 mg/L). Not a validated seawater "
        "or real-plant model."
    )
    gap_policy: str = (
        "Calendar gaps > 1 day and CIP/replacement resets split operational segments. "
        "Forecasts use only the latest applicable segment. Missing observations are "
        "not interpolated. Pre-cleaning history is not borrowed when post-cleaning "
        "history is insufficient."
    )
    serving_config_path: Path | None = None


_SERVING_YAML_KEYS: dict[str, str] = {
    "artifact_path": "artifact_path",
    "default_mapping": "default_mapping",
    "default_horizon_days": "default_horizon_days",
    "min_horizon_days": "min_horizon_days",
    "max_horizon_days": "max_horizon_days",
    "allowed_horizons_days": "allowed_horizons_days",
    "default_threshold_relative_health": "default_threshold_relative_health",
    "min_threshold_relative_health": "min_threshold_relative_health",
    "max_threshold_relative_health": "max_threshold_relative_health",
    "min_history_days": "min_history_days",
    "min_history_observations": "min_history_observations",
    "max_upload_bytes": "max_upload_bytes",
    "max_csv_rows": "max_csv_rows",
    "max_batch_series": "max_batch_series",
    "threshold_provenance": "threshold_provenance",
    "domain_limits": "domain_limits",
    "gap_policy": "gap_policy",
}


def _serving_yaml_path(explicit: Path | None = None) -> Path | None:
    from aqua_fouling.data.adapters import resolve_configs_dir

    if explicit is not None:
        return explicit if explicit.is_file() else None
    env = os.environ.get("AQUA_SERVING_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    candidate = resolve_configs_dir() / "forecast_serving.yaml"
    return candidate if candidate.is_file() else None


def _field_names_from_aqua_keys(keys: set[str]) -> set[str]:
    """Map AQUA_* suffix keys to Settings field names (case-insensitive)."""
    fields = {name.lower(): name for name in Settings.model_fields}
    out: set[str] = set()
    for key in keys:
        lk = key.lower()
        if lk in fields:
            out.add(fields[lk])
    return out


def aqua_keys_from_process_env() -> set[str]:
    return {k[len("AQUA_") :] for k in os.environ if k.startswith("AQUA_")}


def aqua_keys_from_dotenv(path: Path | None = None) -> set[str]:
    """Keys present in ``.env`` (even when not exported into ``os.environ``)."""
    env_path = path or Path(".env")
    if not env_path.is_file():
        return set()
    values = dotenv_values(env_path)
    out: set[str] = set()
    for key, value in values.items():
        if not key or not key.startswith("AQUA_") or value is None:
            continue
        out.add(key[len("AQUA_") :])
    return out


def apply_serving_yaml(
    settings: Settings,
    path: Path | None = None,
    *,
    env_file: Path | None = None,
) -> Settings:
    """Overlay serving YAML under process env and ``.env`` (never above them).

    Protected fields (not overwritten by YAML):
    - any ``AQUA_*`` key present in the process environment
    - any ``AQUA_*`` key present in ``.env`` (parsed via python-dotenv)
    """
    yaml_path = _serving_yaml_path(path)
    if yaml_path is None or not yaml_path.is_file():
        return settings
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return settings

    protected = _field_names_from_aqua_keys(
        aqua_keys_from_process_env() | aqua_keys_from_dotenv(env_file)
    )
    updates: dict[str, Any] = {"serving_config_path": yaml_path}
    for yaml_key, field_name in _SERVING_YAML_KEYS.items():
        if yaml_key not in raw:
            continue
        if field_name in protected:
            continue
        updates[field_name] = raw[yaml_key]
    data = settings.model_dump()
    data.update(updates)
    # Re-validate with env/.env still applied by Settings (env beats init for
    # fields we did not put in updates). Init values for YAML-updated fields
    # must not beat process env: those fields were skipped when protected.
    return Settings.model_validate(data)


def get_settings(
    *,
    serving_yaml: Path | None = None,
    env_file: Path | None = None,
) -> Settings:
    """Resolve settings: process env > .env > serving YAML > defaults."""
    # Settings() already applies: defaults < env_file < process environment.
    # YAML is then layered underneath protected env/.env keys.
    if env_file is None:
        base = Settings()
    else:

        class _SettingsWithEnvFile(Settings):
            model_config = SettingsConfigDict(
                env_prefix="AQUA_",
                env_file=str(env_file),
                env_file_encoding="utf-8",
                extra="ignore",
            )

        base = _SettingsWithEnvFile()
    return apply_serving_yaml(base, path=serving_yaml, env_file=env_file)
