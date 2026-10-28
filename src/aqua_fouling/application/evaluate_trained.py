"""Evaluate trained LLT against persistence and recent linear trend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.synthetic_benchmark import (
    LATENT_FORBIDDEN,
    _health_via_canonical_normalise,
)
from aqua_fouling.forecasting.artifact import load_artifact
from aqua_fouling.forecasting.baselines import get_baseline
from aqua_fouling.forecasting.evaluation import (
    build_segments_for_series,
    run_rolling_origin,
    summarize_records,
)
from aqua_fouling.forecasting.trained_llt import TrainedLLTForecaster
from aqua_fouling.simulation.simulator import generate_synthetic_benchmark, load_benchmark_config


def load_eval_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Evaluation config must be a mapping: {path}"
        raise ValueError(msg)
    return raw


def _attach_interval_metrics(records: list[dict[str, Any]]) -> None:
    """Add observation-interval coverage / width using forecast extras when present."""
    for r in records:
        extras = r.get("forecast_extras") or {}
        obs_lo = extras.get("observation_interval_low")
        obs_hi = extras.get("observation_interval_high")
        lat_lo = extras.get("latent_health_interval_low")
        lat_hi = extras.get("latent_health_interval_high")
        follow = r.get("observed_followup") or []
        traj = r.get("trajectory") or []
        if not obs_lo or not obs_hi or not follow:
            r["obs_interval_coverage"] = None
            r["obs_interval_width_mean"] = None
            r["n_interval_points"] = 0
            r["n_interval_points_covered"] = 0
            r["latent_interval_width_mean"] = None
            continue
        covered = 0
        widths: list[float] = []
        lat_widths: list[float] = []
        n = 0
        for i, o in enumerate(follow):
            if o is None or not np.isfinite(float(o)):
                continue
            if i >= len(obs_lo) or i >= len(obs_hi):
                continue
            lo = float(obs_lo[i])
            hi = float(obs_hi[i])
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            n += 1
            widths.append(hi - lo)
            if lo <= float(o) <= hi:
                covered += 1
            if lat_lo and lat_hi and i < len(lat_lo) and i < len(lat_hi):
                lat_widths.append(float(lat_hi[i]) - float(lat_lo[i]))
        r["obs_interval_coverage"] = (covered / n) if n else None
        r["obs_interval_width_mean"] = float(np.mean(widths)) if widths else None
        r["n_interval_points"] = n
        r["n_interval_points_covered"] = covered
        r["latent_interval_width_mean"] = float(np.mean(lat_widths)) if lat_widths else None
        r["n_trajectory_points"] = sum(1 for t in traj if t is not None)


def _extend_summarize_with_intervals(
    summary: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    for row in summary.get("by_model_horizon") or []:
        mid = row["model_id"]
        h = row["horizon_days"]
        subset = [
            r
            for r in records
            if r.get("model_id") == mid
            and r.get("horizon_days") == h
            and (r.get("eligible_trajectory") or r.get("eligible"))
            and r.get("n_interval_points", 0) > 0
        ]
        if not subset:
            row["average_per_origin_observation_interval_coverage"] = None
            row["observation_interval_coverage_mean"] = None  # alias
            row["observation_interval_width_mean"] = None
            row["latent_interval_width_mean"] = None
            row["n_interval_coverage_origins"] = 0
            row["n_scored_forecast_observation_pairs"] = 0
            row["pooled_pointwise_observation_interval_coverage"] = None
            row["interval_note"] = (
                "No model-based intervals for this model "
                "(baselines do not emit intervals). Latent-state and "
                "future-observation intervals remain distinct when present."
            )
            continue
        covs = [
            r["obs_interval_coverage"] for r in subset if r.get("obs_interval_coverage") is not None
        ]
        widths = [
            r["obs_interval_width_mean"]
            for r in subset
            if r.get("obs_interval_width_mean") is not None
        ]
        lat_w = [
            r["latent_interval_width_mean"]
            for r in subset
            if r.get("latent_interval_width_mean") is not None
        ]
        n_pairs = int(sum(int(r.get("n_interval_points") or 0) for r in subset))
        n_covered = int(sum(int(r.get("n_interval_points_covered") or 0) for r in subset))
        avg_cov = float(np.mean(covs)) if covs else None
        pooled = (n_covered / n_pairs) if n_pairs else None
        row["average_per_origin_observation_interval_coverage"] = avg_cov
        row["observation_interval_coverage_mean"] = avg_cov  # backward-compatible alias
        row["observation_interval_width_mean"] = float(np.mean(widths)) if widths else None
        row["latent_interval_width_mean"] = float(np.mean(lat_w)) if lat_w else None
        row["n_interval_coverage_origins"] = len(subset)
        row["n_scored_forecast_observation_pairs"] = n_pairs
        row["pooled_pointwise_observation_interval_coverage"] = pooled
        row["interval_note"] = (
            "average_per_origin_observation_interval_coverage = mean over origins of "
            "each origin's future-observation-interval hit rate. "
            "pooled_pointwise_observation_interval_coverage = covered pairs / "
            f"scored forecast–observation pairs (n_pairs={n_pairs}). "
            "Latent-state intervals (latent_interval_width_mean) are distinct from "
            "future-observation intervals (observation_interval_width_mean = latent P + R). "
            "Overlapping rolling-origin windows are not independent observations; "
            "do not interpret coverage as calibrated confidence. "
            "Gaussian ±1.96σ model-based intervals only."
        )


def _run_with_extras(
    dates: pd.Series,
    health: pd.Series,
    *,
    forecaster: Any,
    horizons: list[int],
    threshold: float,
    seg_ids: np.ndarray,
    min_history_days: int,
    intervention_censor_day: str | None,
) -> list[dict[str, Any]]:
    """Run shared rolling-origin scoring and attach interval metrics from extras."""
    records = run_rolling_origin(
        dates,
        health,
        forecaster=forecaster,
        horizons=horizons,
        threshold=threshold,
        seg_ids=seg_ids,
        min_history_days=min_history_days,
        intervention_censor_day=intervention_censor_day,
    )
    _attach_interval_metrics(records)
    return records


def _failure_cases(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for r in records:
        if not (r.get("eligible_trajectory") or r.get("eligible")):
            continue
        tag = None
        if r.get("false_warning") is True:
            tag = "false_warning"
        elif r.get("missed_warning") is True:
            tag = "missed_warning"
        elif r.get("prediction_unavailable"):
            tag = "prediction_unavailable"
        elif (
            r.get("crossing_timing_error_days") is not None
            and abs(int(r["crossing_timing_error_days"])) >= 7
        ):
            tag = "large_timing_error"
        elif r.get("mae") is not None and float(r["mae"]) > 0.15:
            tag = "large_trajectory_mae"
        if tag is None:
            continue
        cases.append(
            {
                "tag": tag,
                "model_id": r.get("model_id"),
                "series_id": r.get("series_id"),
                "split": r.get("split"),
                "family": r.get("family"),
                "origin_date": r.get("origin_date"),
                "horizon_days": r.get("horizon_days"),
                "mae": r.get("mae"),
                "pred_crossing_status": r.get("pred_crossing_status"),
                "obs_crossing_status": r.get("obs_crossing_status"),
                "pred_days_to_threshold": r.get("pred_days_to_threshold"),
                "obs_days_to_threshold": r.get("obs_days_to_threshold"),
                "crossing_timing_error_days": r.get("crossing_timing_error_days"),
                "message": r.get("message"),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def run_evaluation(config_path: Path) -> dict[str, Any]:
    eval_cfg = load_eval_config(config_path)
    bench_path = Path(eval_cfg["synthetic_benchmark_config"])
    bench = load_benchmark_config(bench_path)
    regenerate = bool(eval_cfg.get("regenerate_synthetic", False))
    if regenerate:
        manifest = generate_synthetic_benchmark(bench_path)
    else:
        out = Path(bench["output_dir"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    artifact_path = Path(eval_cfg["artifact_path"])
    artifact = load_artifact(artifact_path)
    # Ensure artifact was not trained on test
    test_ids = {str(item["series_id"]) for item in manifest["series"] if item["split"] == "test"}
    if set(artifact.training_series_ids) & test_ids:
        msg = "Refusing evaluation: artifact training_series_ids overlap test split."
        raise RuntimeError(msg)

    min_hist_obs = int(
        eval_cfg.get(
            "min_history_observations",
            artifact.selected_settings.get("min_history_observations", 5),
        )
    )
    trained = TrainedLLTForecaster.from_path(artifact_path, min_history_observations=min_hist_obs)
    # Snapshot digest before inference to prove no mutation
    before = artifact_path.read_bytes()

    horizons = list(bench["forecasting"]["horizons_days"])
    threshold = float(bench["forecasting"]["threshold_relative_health"])
    min_hist = int(bench["forecasting"]["min_history_days"])
    trend_window = int(
        eval_cfg.get(
            "recent_trend_window_days",
            bench["forecasting"]["recent_trend_window_days"],
        )
    )
    forecasters = [
        get_baseline("persistence"),
        get_baseline("recent_linear_trend", trend_window=trend_window),
        trained,
    ]

    all_records: list[dict[str, Any]] = []
    per_series_info: list[dict[str, Any]] = []

    for item in manifest["series"]:
        raw = pd.read_csv(item["raw_path"])
        for latent in LATENT_FORBIDDEN:
            if latent in raw.columns:
                msg = f"Latent column {latent} must not appear in forecasting inputs"
                raise RuntimeError(msg)
        ref_path = Path(item.get("reference_path") or "")
        if not ref_path.is_file():
            ref_path = (
                Path(bench["output_dir"]) / "references" / f"{item['series_id']}_reference.yaml"
            )
        health, norm_meta = _health_via_canonical_normalise(
            Path(item["raw_path"]),
            series_id=item["series_id"],
            reference_path=ref_path,
        )
        dates = raw["date"]
        cip = raw["cip_flag"]
        dsr = raw["days_since_replacement"]
        segments, _audit, seg_ids = build_segments_for_series(
            dates, cip=cip, days_since_replacement=dsr
        )
        intervention_day = None
        hmeta = item["meta"].get("health_meta") or {}
        if hmeta.get("intervention_before_crossing") and item.get("family") == "partial_cleaning":
            cip_rows = raw.index[raw["cip_flag"] == 1].tolist()
            if cip_rows:
                intervention_day = str(raw.loc[cip_rows[0], "date"])

        series_records: list[dict[str, Any]] = []
        for model in forecasters:
            recs = _run_with_extras(
                dates,
                health,
                forecaster=model,
                horizons=horizons,
                threshold=threshold,
                seg_ids=seg_ids,
                min_history_days=min_hist,
                intervention_censor_day=intervention_day,
            )
            for r in recs:
                r["series_id"] = item["series_id"]
                r["split"] = item["split"]
                r["family"] = item["family"]
            series_records.extend(recs)
        all_records.extend(series_records)
        per_series_info.append(
            {
                "series_id": item["series_id"],
                "split": item["split"],
                "family": item["family"],
                "n_segments": len(segments),
                "canonical_normalisation": norm_meta,
                "metrics": summarize_records(series_records),
            }
        )

    after = artifact_path.read_bytes()
    if after != before:
        msg = "Artifact file mutated during evaluation inference — aborting."
        raise RuntimeError(msg)

    def _split_summary(split: str) -> dict[str, Any]:
        subset = [r for r in all_records if r.get("split") == split]
        summary = summarize_records(subset, group_key="series_id")
        _extend_summarize_with_intervals(summary, subset)
        return summary

    development_summary = _split_summary("development")
    test_summary = _split_summary("test")
    # Development was used only for setting selection; report it separately.
    mid = trained.model_id
    failure_dev = _failure_cases(
        [r for r in all_records if r.get("split") == "development" and r.get("model_id") == mid]
    )
    failure_test = _failure_cases(
        [r for r in all_records if r.get("split") == "test" and r.get("model_id") == mid]
    )

    payload = {
        "experiment": "synthetic_trained_llt_phase4",
        "evaluation_config": str(config_path),
        "artifact_path": str(artifact_path),
        "artifact_model_version": artifact.model_version,
        "artifact_schema_version": artifact.artifact_schema_version,
        "trained_parameters": artifact.parameters,
        "threshold_relative_health": threshold,
        "horizons_days": horizons,
        "models": [f.model_id for f in forecasters],
        "train_series_excluded_from_test_claim": artifact.training_series_ids,
        "development_used_for": "small initialisation-grid selection only (frozen before test)",
        "test_set_label": (
            "inspected_benchmark — previously inspected in Phase 3; not a pristine "
            "unseen holdout. No post-test retuning."
        ),
        "disclaimer": (
            "Synthetic trained-model metrics do not establish plant accuracy or "
            "validated transfer to OCWD / real plants. OCWD remains a separate "
            "experiment and is not pooled here."
        ),
        "interval_disclaimer": (
            "Reported interval coverage uses theoretical Gaussian observation "
            "intervals; do not interpret as calibrated confidence."
        ),
        "no_future_cleaning_assumption": True,
        "artifact_unchanged_during_inference": True,
        "development_split_summary": development_summary,
        "test_split_summary": test_summary,
        "per_series": per_series_info,
        "failure_cases_development": failure_dev,
        "failure_cases_test": failure_test,
        "remaining_limitations": [
            "Gaussian LLT may under-represent abrupt fouling acceleration.",
            "No days-until-cleaning uncertainty intervals in this phase.",
            "OCWD independent normalisation blockers remain unresolved; no OCWD "
            "transfer claim for the synthetic-trained artifact.",
            "Test set was previously inspected during Phase 3 baseline work.",
        ],
    }

    report_dir = Path(eval_cfg.get("report_dir", "reports/phase4"))
    report_dir.mkdir(parents=True, exist_ok=True)
    out_json = report_dir / "evaluation_metrics.json"
    out_json.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    md = _write_markdown_summary(payload, report_dir / "evaluation_summary.md")
    payload["markdown_summary"] = str(md)
    return payload


def _fmt_row(row: dict[str, Any]) -> str:
    mae = row.get("common_origin_mae_mean")
    mae_s = f"{mae:.5f}" if isinstance(mae, (int, float)) else "n/a"
    rmse = row.get("trajectory_window_rmse_mean")
    rmse_s = f"{rmse:.5f}" if isinstance(rmse, (int, float)) else "n/a"
    cov = row.get("average_per_origin_observation_interval_coverage")
    if cov is None:
        cov = row.get("observation_interval_coverage_mean")
    cov_s = f"{cov:.3f}" if isinstance(cov, (int, float)) else "n/a"
    pooled = row.get("pooled_pointwise_observation_interval_coverage")
    pooled_s = f"{pooled:.3f}" if isinstance(pooled, (int, float)) else "n/a"
    tw = row.get("crossing_timing_error_mae")
    tw_s = f"{tw:.3f}" if isinstance(tw, (int, float)) else "n/a"
    n_origins = row.get("n_interval_coverage_origins", 0)
    n_pairs = row.get("n_scored_forecast_observation_pairs", 0)
    obs_w = row.get("observation_interval_width_mean")
    lat_w = row.get("latent_interval_width_mean")
    obs_w_s = f"{obs_w:.5f}" if isinstance(obs_w, (int, float)) else "n/a"
    lat_w_s = f"{lat_w:.5f}" if isinstance(lat_w, (int, float)) else "n/a"
    return (
        f"| {row.get('model_id')} | {row.get('horizon_days')} | {mae_s} | {rmse_s} | "
        f"{row.get('n_common_scored_origins')} | {row.get('n_pred_unavailable')} | "
        f"{row.get('false_warnings')}/{row.get('false_warning_denominator')} | "
        f"{row.get('missed_warnings')}/{row.get('missed_warning_denominator')} | "
        f"{tw_s} (n={row.get('n_timing_samples')}) | {cov_s} "
        f"(origins={n_origins}; pairs={n_pairs}; pooled={pooled_s}; "
        f"obs_w={obs_w_s}; lat_w={lat_w_s}) |"
    )


def _write_markdown_summary(payload: dict[str, Any], path: Path) -> Path:
    interval_legend = (
        "Coverage column = **average per-origin observation-interval coverage** "
        "(mean over origins of each origin's future-observation-interval hit rate). "
        "Also reports origin count, scored forecast–observation pair count, "
        "pooled pointwise coverage, mean observation-interval width, and mean "
        "latent-state interval width (distinct: obs width = latent P + R). "
        "Overlapping rolling-origin windows are not independent observations; "
        "do not interpret coverage as calibrated confidence."
    )
    lines = [
        "# Phase 4 — Trained LLT evaluation summary",
        "",
        f"- Artifact: `{payload['artifact_path']}` (`{payload['artifact_model_version']}`)",
        f"- Test set label: {payload['test_set_label']}",
        f"- Disclaimer: {payload['disclaimer']}",
        f"- Interval metrics: {interval_legend}",
        "",
        "## Development split (selection freeze; not used for claims of success)",
        "",
        (
            "| model | h | common-origin MAE | RMSE | n_common | n_unavail | "
            "false warn | missed warn | timing MAE | avg per-origin obs-int cov |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in (payload.get("development_split_summary") or {}).get("by_model_horizon") or []:
        lines.append(_fmt_row(row))
    lines += [
        "",
        "## Test split (inspected benchmark)",
        "",
        (
            "| model | h | common-origin MAE | RMSE | n_common | n_unavail | "
            "false warn | missed warn | timing MAE | avg per-origin obs-int cov |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in (payload.get("test_split_summary") or {}).get("by_model_horizon") or []:
        lines.append(_fmt_row(row))
    lines += ["", "## Failure cases (trained model, test)", ""]
    for c in payload.get("failure_cases_test") or []:
        lines.append(
            f"- `{c.get('tag')}` {c.get('series_id')} origin={c.get('origin_date')} "
            f"h={c.get('horizon_days')} pred={c.get('pred_crossing_status')} "
            f"obs={c.get('obs_crossing_status')} mae={c.get('mae')}"
        )
    lines += ["", "## Remaining limitations", ""]
    for lim in payload.get("remaining_limitations") or []:
        lines.append(f"- {lim}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
