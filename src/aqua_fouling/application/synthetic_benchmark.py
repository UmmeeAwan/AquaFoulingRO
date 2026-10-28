"""Run synthetic baseline benchmark and write Phase 3 artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.data.adapters import resolve_configs_dir
from aqua_fouling.forecasting.evaluation import (
    build_segments_for_series,
    make_forecasters,
    run_rolling_origin,
    summarize_records,
)
from aqua_fouling.simulation.simulator import generate_synthetic_benchmark, load_benchmark_config

SYNTHETIC_MAPPING = resolve_configs_dir() / "canonical_synthetic_schema.yaml"


def _health_via_canonical_normalise(
    raw_path: Path,
    *,
    series_id: str,
    reference_path: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    """Shared pipeline: synthetic raw → adapt → independent normalisation."""
    result = normalise_csv(
        raw_path,
        source_series_id=series_id,
        mapping_path=SYNTHETIC_MAPPING,
        reference_path=reference_path,
    )
    derived = (result.get("derived_normalised_values") or {}).get(
        "independent_physical_normalisation"
    ) or {}
    health_list = derived.get("relative_health") or []
    n = int(result.get("n_measurements") or len(health_list))
    health = pd.Series(
        [float(v) if v is not None else np.nan for v in health_list]
        + [np.nan] * max(0, n - len(health_list))
    )[:n]
    meta = {
        "normalise_status": result.get("status"),
        "available": derived.get("available"),
        "n_normalised": derived.get("n_normalised"),
        "n_row_level_failures": result.get("n_row_level_normalisation_failures", 0),
        "row_level_failures_sample": (result.get("row_level_normalisation_failures") or [])[:20],
        "missing_for_calculations": result.get("missing_for_calculations"),
        "blockers": derived.get("blockers"),
    }
    return health, meta


def run_synthetic_baseline_benchmark(
    config_path: Path,
    report_dir: Path,
    *,
    regenerate: bool = True,
) -> dict[str, Any]:
    cfg = load_benchmark_config(config_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if regenerate:
        manifest = generate_synthetic_benchmark(config_path)
    else:
        out = Path(cfg["output_dir"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    horizons = list(cfg["forecasting"]["horizons_days"])
    threshold = float(cfg["forecasting"]["threshold_relative_health"])
    min_hist = int(cfg["forecasting"]["min_history_days"])
    trend_window = int(cfg["forecasting"]["recent_trend_window_days"])
    forecasters = make_forecasters(trend_window)

    all_records: list[dict[str, Any]] = []
    per_series_info: list[dict[str, Any]] = []

    for item in manifest["series"]:
        raw = pd.read_csv(item["raw_path"])
        gt = pd.read_csv(item["ground_truth_path"])
        for latent in LATENT_FORBIDDEN:
            if latent in raw.columns:
                msg = f"Latent column {latent} must not appear in raw forecasting inputs"
                raise RuntimeError(msg)

        ref_path = Path(item.get("reference_path") or "")
        if not ref_path.is_file():
            # Rebuild reference YAML from meta if needed
            ref_path = (
                Path(cfg["output_dir"]) / "references" / f"{item['series_id']}_reference.yaml"
            )
            ref = item["meta"]["reference"]
            import yaml

            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(
                yaml.safe_dump(
                    {
                        "mode": "documented",
                        "provenance": ref["provenance"],
                        "available_from": ref["available_from"],
                        "temperature_c": ref["temperature_c"],
                        "ndp_psi": ref["ndp_psi"],
                        "permeate_flow": ref["permeate_flow"],
                        "permeate_flow_unit": ref.get("permeate_flow_unit", "gpd"),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
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
            # CIP date from raw
            cip_rows = raw.index[raw["cip_flag"] == 1].tolist()
            if cip_rows:
                intervention_day = str(raw.loc[cip_rows[0], "date"])

        series_records: list[dict[str, Any]] = []
        for model in forecasters:
            recs = run_rolling_origin(
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

        per_series_summary = summarize_records(
            [r for r in series_records if r.get("split") == item["split"]]
        )
        per_series_info.append(
            {
                "series_id": item["series_id"],
                "split": item["split"],
                "family": item["family"],
                "n_segments": len(segments),
                "reference": item["meta"]["reference"],
                "health_meta": hmeta,
                "canonical_normalisation": norm_meta,
                "metrics": per_series_summary,
            }
        )

        if item["series_id"] in {"S02", "S08"}:
            _fig_raw_vs_norm(raw, health, gt, fig_dir / f"{item['series_id']}_raw_vs_norm.png")
        if item["series_id"] == "S08":
            _fig_forecast_example(
                dates, health, series_records, fig_dir / "S08_forecast_example.png", threshold
            )
        if item["series_id"] == "S07":
            _fig_failure_case(
                dates, health, series_records, fig_dir / "S07_failure_case.png", threshold
            )

    test_records = [r for r in all_records if r.get("split") == "test"]
    train_dev_note = (
        "Scenario groups (train/development/test) were defined in "
        "configs/synthetic_benchmark.yaml before baseline evaluation. "
        "Baselines are not fitted; trend window taken from config (development-facing). "
        "No tuning on the final test set. Prior inspected Phase 3 scores were not used to "
        "retune the simulator for a preferred model."
    )
    summary = summarize_records(test_records, group_key="series_id")
    full_summary = summarize_records(all_records, group_key="series_id")

    payload = {
        "experiment": "synthetic_primary_benchmark",
        "config": str(config_path),
        "mapping": str(SYNTHETIC_MAPPING),
        "threshold_relative_health": threshold,
        "threshold_justification": cfg["forecasting"].get("threshold_justification"),
        "horizons_days": horizons,
        "train_dev_test_note": train_dev_note,
        "corrective_pass_note": (
            "Phase 3 corrective pass regenerated these baselines after evaluation/"
            "physics/CSV fixes. Not a trained-model submission. "
            "Does not claim real-plant validation."
        ),
        "disclaimer": ("Synthetic benchmark performance does not establish plant accuracy."),
        "ocwd_blockers_resolved": False,
        "per_series": per_series_info,
        "test_split_summary": summary,
        "all_splits_summary": full_summary,
        "n_test_eligible_records": sum(1 for r in test_records if r.get("eligible")),
        "false_warning_definition": cfg["evaluation"]["false_warning_definition"],
        "missed_warning_definition": cfg["evaluation"]["missed_warning_definition"],
        "figures": [str(p) for p in sorted(fig_dir.glob("S*.png"))],
    }
    out_json = report_dir / "synthetic_metrics.json"
    out_json.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return payload


LATENT_FORBIDDEN = (
    "true_health",
    "latent_decay_per_day",
    "true_ndp_psi",
    "true_permeate_flow_clean_gpd",
    "clean_permeability_gfd_per_psi",
)


def _fig_raw_vs_norm(
    raw: pd.DataFrame,
    health: pd.Series,
    gt: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(raw["date"])
    flow_col = "permeate_flow_gpd" if "permeate_flow_gpd" in raw.columns else "permeate_flow"
    ax.plot(x, raw[flow_col], label="raw permeate_flow_gpd", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, health, color="C1", label="normalised relative health")
    ax2.plot(x, gt["true_health"], color="C2", linestyle="--", label="true health (GT only)")
    ax.set_xlabel("date")
    ax.set_ylabel("permeate flow (gpd)")
    ax2.set_ylabel("health")
    ax.set_title("Raw flow vs normalised health (GT dashed — not a forecast input)")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_forecast_example(
    dates: pd.Series,
    health: pd.Series,
    records: list[dict[str, Any]],
    path: Path,
    threshold: float,
) -> None:
    cand = [
        r
        for r in records
        if r.get("eligible") and r.get("model_id") == "persistence" and r.get("horizon_days") == 14
    ]
    if not cand:
        return
    r = cand[len(cand) // 2]
    origin = r["origin_index"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(dates)
    ax.plot(x, health, label="observed health", color="C0")
    ax.axhline(threshold, color="k", linestyle=":", label="threshold")
    fut_x = pd.date_range(
        x.iloc[origin] + pd.Timedelta(days=1), periods=len(r["trajectory"]), freq="D"
    )
    ax.plot(fut_x, r["trajectory"], label="persistence forecast", color="C3")
    ax.axvline(x.iloc[origin], color="gray", linestyle="--", label="origin")
    ax.set_title(f"Forecast example — origin {r['origin_date']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_failure_case(
    dates: pd.Series,
    health: pd.Series,
    records: list[dict[str, Any]],
    path: Path,
    threshold: float,
) -> None:
    fails = [
        r
        for r in records
        if r.get("eligible") and (r.get("missed_warning") is True or r.get("false_warning") is True)
    ]
    if not fails:
        fails = [r for r in records if r.get("eligible") and r.get("horizon_days") == 30]
    if not fails:
        return
    r = fails[0]
    origin = r["origin_index"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(dates)
    ax.plot(x, health, label="observed health")
    ax.axhline(threshold, color="k", linestyle=":")
    traj = r.get("trajectory") or []
    if traj:
        fut_x = pd.date_range(x.iloc[origin] + pd.Timedelta(days=1), periods=len(traj), freq="D")
        ax.plot(fut_x, traj, label=f"{r['model_id']} forecast")
    ax.axvline(x.iloc[origin], color="gray", linestyle="--")
    ax.set_title(
        f"Failure/edge case — {r.get('model_id')} "
        f"missed={r.get('missed_warning')} false={r.get('false_warning')}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
