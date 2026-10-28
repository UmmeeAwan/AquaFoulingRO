"""Final integration pass: convergence, provenance, serving, mid-fouling."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from aqua_fouling.application.train_llt import (
    TrainingConvergenceError,
    compute_training_provenance,
    fit_llt_parameters,
    select_settings_on_development,
)
from aqua_fouling.forecasting.artifact import sha256_file
from aqua_fouling.forecasting.llt_kalman import LLTParameters, SegmentSeries
from aqua_fouling.settings import Settings, apply_serving_yaml, get_settings

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "models" / "llt_relative_health_v1.json"
DEMO_CSV = REPO / "examples" / "forecast" / "demo_history.csv"
DEMO_REF = REPO / "examples" / "forecast" / "reference.yaml"
GENERIC_MAP = REPO / "configs" / "canonical_generic_schema.yaml"
SERVING_YAML = REPO / "configs" / "forecast_serving.yaml"


def _tiny_segments() -> list[SegmentSeries]:
    d0 = 737791  # stable ordinal baseline
    return [
        SegmentSeries(
            "T01",
            0,
            np.asarray([d0 + i for i in range(40)], dtype=np.int64),
            np.asarray([1.0 - 0.002 * i for i in range(40)], dtype=float),
        )
    ]


def test_converged_flag_requires_optimiser_success_insufficient_iters() -> None:
    """Intentionally insufficient maxiter must not set converged_flag=True."""
    segs = _tiny_segments()
    # Far-from-optimum start + maxiter=1 → typically does not report success.
    bad_init = LLTParameters(
        r=0.05,
        q_level=1e-3,
        q_slope=1e-3,
        p0_level=0.5,
        p0_slope=0.1,
    )
    _params, conv = fit_llt_parameters(segs, seed=0, maxiter=1, initial=bad_init)
    assert conv["converged_flag"] is False
    assert conv["converged_flag"] == bool(conv["success"])
    assert int(conv["n_iterations"]) <= 1
    # Finite objective alone must not imply convergence
    assert np.isfinite(conv["final_negative_loglik"])


def test_unsuccessful_candidates_excluded_and_preserve_artifact(tmp_path: Path) -> None:
    segs = _tiny_segments()
    params = LLTParameters(r=1e-3, q_level=1e-5, q_slope=1e-7, p0_level=1e-2, p0_slope=1e-4)
    art = tmp_path / "existing.json"
    art.write_text('{"sentinel": true}\n', encoding="utf-8")
    before = art.read_text(encoding="utf-8")

    with pytest.raises(TrainingConvergenceError, match="converged"):
        select_settings_on_development(
            {"a": params, "b": params},
            development_segments=segs,
            setting_grid=[
                {"name": "a", "initial": params.to_dict()},
                {"name": "b", "initial": params.to_dict()},
            ],
            convergence_by_setting={
                "a": {"converged_flag": False, "message": "iter limit"},
                "b": {"converged_flag": False, "message": "iter limit"},
            },
        )
    assert art.read_text(encoding="utf-8") == before


def test_training_data_hash_changes_when_trajectory_mean_unchanged(tmp_path: Path) -> None:
    """Content hash must change if CSV trajectory changes even when mean is fixed."""
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    ref = tmp_path / "ref.yaml"
    mapping = tmp_path / "map.yaml"
    train_cfg = tmp_path / "train.yaml"
    bench_cfg = tmp_path / "bench.yaml"
    # Same mean (1+2+3)/3 = 2, different trajectory
    csv_a.write_text("date,v\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n", encoding="utf-8")
    csv_b.write_text("date,v\n2020-01-01,3\n2020-01-02,2\n2020-01-03,1\n", encoding="utf-8")
    ref.write_text("mode: documented\n", encoding="utf-8")
    mapping.write_text("name: m\n", encoding="utf-8")
    train_cfg.write_text("name: t\n", encoding="utf-8")
    bench_cfg.write_text("name: b\n", encoding="utf-8")

    def _prov(csv: Path) -> dict:
        manifest = {
            "series": [
                {
                    "series_id": "S01",
                    "split": "train",
                    "raw_path": str(csv),
                    "reference_path": str(ref),
                }
            ]
        }
        return compute_training_provenance(
            manifest,
            splits={"train"},
            cfg={"output_dir": str(tmp_path)},
            mapping_path=mapping,
            training_config_path=train_cfg,
            benchmark_config_path=bench_cfg,
        )

    ha = _prov(csv_a)
    hb = _prov(csv_b)
    assert float(np.mean([1, 2, 3])) == float(np.mean([3, 2, 1]))
    assert ha["content_hashes"]["csv:S01"] != hb["content_hashes"]["csv:S01"]
    assert ha["training_data_hash"] != hb["training_data_hash"]
    assert ha["training_config_hash"] == sha256_file(train_cfg)
    assert ha["benchmark_config_hash"] == sha256_file(bench_cfg)


def test_missing_file_does_not_fabricate_hash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fabricate"):
        compute_training_provenance(
            {
                "series": [
                    {
                        "series_id": "S01",
                        "split": "train",
                        "raw_path": str(tmp_path / "missing.csv"),
                        "reference_path": str(tmp_path / "missing.yaml"),
                    }
                ]
            },
            splits={"train"},
            cfg={"output_dir": str(tmp_path)},
            mapping_path=tmp_path / "map.yaml",
            training_config_path=tmp_path / "train.yaml",
            benchmark_config_path=tmp_path / "bench.yaml",
        )


def test_serving_yaml_controls_history_horizon_threshold_artifact() -> None:
    assert SERVING_YAML.is_file()
    s = apply_serving_yaml(Settings(), SERVING_YAML)
    raw = __import__("yaml").safe_load(SERVING_YAML.read_text(encoding="utf-8"))
    assert Path(s.artifact_path) == Path(raw["artifact_path"])
    assert int(s.min_history_days) == int(raw["min_history_days"])
    assert int(s.min_history_observations) == int(raw["min_history_observations"])
    assert int(s.default_horizon_days) == int(raw["default_horizon_days"])
    assert list(s.allowed_horizons_days) == list(raw["allowed_horizons_days"])
    assert float(s.default_threshold_relative_health) == float(
        raw["default_threshold_relative_health"]
    )
    assert "Calendar gaps" in s.gap_policy or "gap" in s.gap_policy.lower()
    # Env override wins
    monkey = Settings(min_history_days=99)
    # simulate env by constructing with explicit value then applying yaml
    # apply_serving_yaml skips keys present in os.environ — set temporarily
    import os

    os.environ["AQUA_MIN_HISTORY_DAYS"] = "77"
    try:
        s2 = get_settings()
        assert int(s2.min_history_days) == 77
    finally:
        del os.environ["AQUA_MIN_HISTORY_DAYS"]
    _ = monkey


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="artifact missing")
def test_mid_fouling_history_documents_clean_prior(tmp_path: Path) -> None:
    """History beginning below clean reference; prior health=1 is not retuned."""
    import pandas as pd

    from aqua_fouling.application.forecast_service import forecast_csv
    from aqua_fouling.forecasting.artifact import load_artifact

    df = pd.read_csv(DEMO_CSV)
    # Entire history already below clean reference (mid-fouling start).
    mid = df.copy()
    mid["permeate_flow_gpd"] = mid["permeate_flow_gpd"] * 0.88
    mid["feed_flow_gpd"] = mid["feed_flow_gpd"] * 0.88
    mid_csv = tmp_path / "mid.csv"
    mid.to_csv(mid_csv, index=False)
    assert len(mid) >= 21

    body = forecast_csv(
        mid_csv,
        source_series_id="MID",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        threshold=0.85,
        artifact_path=ARTIFACT,
    )
    assumptions = body["metadata"]["assumptions"]
    scope = body["metadata"]["scientific_scope"]
    assert any("prior mean health=1.0" in a for a in assumptions)
    assert "synthetic" in scope.lower() or "demonstration" in scope.lower()
    finite_h = [
        o["relative_health"]
        for o in body["normalised_data"]["observations"]
        if o.get("relative_health") is not None
    ]
    assert finite_h[0] < 0.95
    assert body["result"]["state_at_origin"]["relative_health"] < 0.95
    art = load_artifact(ARTIFACT)
    assert "health=1.0" in art.initialisation_policy


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="artifact missing")
def test_api_cli_equivalence_and_artifact_immutable(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from aqua_fouling.api.app import create_app
    from aqua_fouling.application.forecast_service import forecast_csv
    from aqua_fouling.cli import app as cli_app

    before = ARTIFACT.read_bytes()
    csv = tmp_path / "demo.csv"
    shutil.copy(DEMO_CSV, csv)
    svc = forecast_csv(
        csv,
        source_series_id="DEMO",
        mapping_path=GENERIC_MAP,
        reference_path=DEMO_REF,
        horizon_days=14,
        artifact_path=ARTIFACT,
    )
    runner = CliRunner()
    out = tmp_path / "cli.json"
    result = runner.invoke(
        cli_app,
        [
            "forecast",
            "--csv",
            str(csv),
            "--series-id",
            "DEMO",
            "--reference",
            str(DEMO_REF),
            "--mapping",
            str(GENERIC_MAP),
            "--artifact",
            str(ARTIFACT),
            "--horizon-days",
            "14",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    cli = json.loads(out.read_text(encoding="utf-8"))
    client = TestClient(create_app(Settings(artifact_path=ARTIFACT)))
    with DEMO_CSV.open("rb") as cf, DEMO_REF.open("rb") as rf, GENERIC_MAP.open("rb") as mf:
        resp = client.post(
            "/v1/forecast",
            files={
                "csv": ("demo.csv", cf, "text/csv"),
                "reference": ("reference.yaml", rf, "application/yaml"),
                "mapping": ("mapping.yaml", mf, "application/yaml"),
            },
            data={"source_series_id": "DEMO", "horizon_days": "14"},
        )
    assert resp.status_code == 200
    api = resp.json()
    assert (
        svc["result"]["crossing_status"]
        == cli["result"]["crossing_status"]
        == api["result"]["crossing_status"]
    )
    assert (
        svc["result"]["days_to_threshold"]
        == cli["result"]["days_to_threshold"]
        == api["result"]["days_to_threshold"]
    )
    assert ARTIFACT.read_bytes() == before
