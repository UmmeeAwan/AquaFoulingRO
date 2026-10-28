"""Assessment-delivery integration regressions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from aqua_fouling.api.app import create_app
from aqua_fouling.application.forecast_errors import ForecastUnavailableError
from aqua_fouling.application.forecast_service import forecast_csv
from aqua_fouling.settings import Settings, apply_serving_yaml, get_settings

REPO = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
GENERIC_MAP = REPO / "configs" / "canonical_generic_schema.yaml"
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"
SERVING_YAML = REPO / "configs" / "forecast_serving.yaml"


def _base_demo_frame() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


def test_post_replacement_history_does_not_borrow_pre_replacement(tmp_path: Path) -> None:
    """Mapped days_since_replacement reset must split segments for history checks."""
    df = _base_demo_frame()
    # Long pre-replacement history, then a reset with only 5 post-replacement days.
    n = len(df)
    dsr = list(range(n - 5)) + list(range(5))
    assert len(dsr) == n
    df["days_since_replacement"] = dsr
    path = tmp_path / "post_repl.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ForecastUnavailableError) as ei:
        forecast_csv(
            path,
            source_series_id="DEMO",
            mapping_path=GENERIC_MAP,
            reference_path=DEMO_REF,
            horizon_days=14,
            threshold=0.85,
            artifact_path=ARTIFACT,
        )
    err = ei.value
    assert err.code == "forecast_unavailable"
    details = err.details or {}
    assert int(details.get("calendar_span_days") or 0) < 21
    assert int(details.get("finite_observations") or 0) <= 5


def test_timestamp_column_resolved_via_adapter_not_hardcoded_name(tmp_path: Path) -> None:
    df = _base_demo_frame().rename(columns={"observation_date": "plant_ts"})
    # Shuffle so sorting is required via adapter timestamp.column
    df = df.iloc[::-1].reset_index(drop=True)
    csv_path = tmp_path / "renamed_ts.csv"
    df.to_csv(csv_path, index=False)

    mapping = yaml.safe_load(GENERIC_MAP.read_text(encoding="utf-8"))
    mapping["timestamp"] = {"column": "plant_ts", "format": "ISO8601", "timezone": "UTC"}
    for field in mapping["fields"]:
        if field["external_name"] == "observation_date":
            field["external_name"] = "plant_ts"
    map_path = tmp_path / "map_plant_ts.yaml"
    map_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    body = forecast_csv(
        csv_path,
        source_series_id="DEMO",
        mapping_path=map_path,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    assert body["status"] == "ok"
    assert body["normalised_data"]["observations"][0]["date"] == "2020-01-01"
    assert any(w.get("code") == "timestamps_sorted" for w in body.get("warnings") or [])


def test_settings_precedence_env_over_dotenv_over_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("AQUA_MIN_HISTORY_DAYS=55\n", encoding="utf-8")
    yaml_path = tmp_path / "serving.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "min_history_days: 33",
                "default_horizon_days: 7",
                "artifact_path: artifacts/models/llt_relative_health_v1.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("AQUA_MIN_HISTORY_DAYS", raising=False)
    monkeypatch.delenv("AQUA_DEFAULT_HORIZON_DAYS", raising=False)
    s = get_settings(serving_yaml=yaml_path, env_file=dotenv)
    assert int(s.min_history_days) == 55  # .env > YAML
    assert int(s.default_horizon_days) == 7  # YAML > default 14

    monkeypatch.setenv("AQUA_MIN_HISTORY_DAYS", "88")
    s2 = get_settings(serving_yaml=yaml_path, env_file=dotenv)
    assert int(s2.min_history_days) == 88  # process env > .env


def test_explicit_settings_to_create_app_not_overwritten_by_yaml() -> None:
    explicit = Settings(min_history_days=99, artifact_path=ARTIFACT)
    app = create_app(settings=explicit)
    assert int(app.state.settings.min_history_days) == 99


def test_malformed_mapping_yaml_is_client_error_not_500(tmp_path: Path) -> None:
    app = create_app(Settings(artifact_path=ARTIFACT))
    client = TestClient(app)
    bad_map = tmp_path / "bad.yaml"
    bad_map.write_text(":\n  - not: valid: yaml: [[[\n", encoding="utf-8")
    files = {
        "csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "mapping": ("bad.yaml", bad_map.read_bytes(), "application/yaml"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post(
        "/v1/forecast",
        files=files,
        data={"source_series_id": "DEMO", "horizon_days": "14"},
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] in {"malformed_yaml", "malformed_mapping"}


def test_malformed_calibration_json_is_client_error_not_500() -> None:
    app = create_app(Settings(artifact_path=ARTIFACT))
    client = TestClient(app)
    files = {
        "csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "mapping": ("map.yaml", GENERIC_MAP.read_bytes(), "application/yaml"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post(
        "/v1/normalise",
        files=files,
        data={
            "source_series_id": "DEMO",
            "conductivity_calibration_json": "{not-json",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "malformed_json"


def test_upload_temps_cleaned_when_later_upload_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """CSV temp must be removed if a later multipart part fails size checks."""
    from aqua_fouling.api import routes as routes_mod

    created: list[Path] = []
    real_write = routes_mod._write_upload

    def tracking_write(upload, suffix, *, max_bytes):  # type: ignore[no-untyped-def]
        # Tiny limit on the reference upload so it fails after CSV was accepted.
        limit = 32 if suffix in {".yaml", ".ref_upload", ".refs_upload"} else max_bytes
        path, nbytes = real_write(upload, suffix, max_bytes=limit)
        created.append(path)
        return path, nbytes

    monkeypatch.setattr(routes_mod, "_write_upload", tracking_write)
    app = create_app(Settings(artifact_path=ARTIFACT))
    client = TestClient(app)
    files = {
        "csv": ("demo.csv", DEMO_CSV.read_bytes(), "text/csv"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post(
        "/v1/forecast",
        files=files,
        data={"source_series_id": "DEMO"},
    )
    assert r.status_code == 413
    for p in created:
        assert not p.exists(), f"temp upload not cleaned: {p}"


def test_bounded_upload_rejects_without_buffering_entire_oversize() -> None:
    app = create_app(Settings(artifact_path=ARTIFACT, max_upload_bytes=1024))
    client = TestClient(app)
    big = b"x" * 5000
    files = {
        "csv": ("big.csv", big, "text/csv"),
        "reference": ("reference.yaml", DEMO_REF.read_bytes(), "application/yaml"),
    }
    r = client.post(
        "/v1/forecast",
        files=files,
        data={"source_series_id": "DEMO"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_apply_serving_yaml_respects_documented_keys() -> None:
    s = apply_serving_yaml(Settings(), SERVING_YAML)
    raw = yaml.safe_load(SERVING_YAML.read_text(encoding="utf-8"))
    assert int(s.min_history_days) == int(raw["min_history_days"])
    assert float(s.default_threshold_relative_health) == float(
        raw["default_threshold_relative_health"]
    )
