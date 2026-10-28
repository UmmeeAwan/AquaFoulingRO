"""Separate OCWD s3sf baseline forecasting experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.forecasting.evaluation import (
    build_segments_for_series,
    make_forecasters,
    run_rolling_origin,
    summarize_records,
)

SERIES_RE = re.compile(r"orange_county_ro_([A-G]\d{2})\.csv$", re.IGNORECASE)


def run_ocwd_baseline_benchmark(
    config_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(cfg["extracted_csv_dir"])
    target = cfg["target_column"]
    horizons = list(cfg["forecasting"]["horizons_days"])
    min_hist = int(cfg["forecasting"]["min_history_days"])
    trend_window = int(cfg["forecasting"]["recent_trend_window_days"])
    ref_win = cfg["forecasting"]["reference_window"]
    rel_thr = float(cfg["forecasting"]["threshold_relative_to_reference"])
    train_end = pd.Timestamp(cfg["chronology"]["train_end"])
    dev_end = pd.Timestamp(cfg["chronology"]["development_end"])
    available_from = pd.Timestamp(ref_win["available_from"])

    forecasters = make_forecasters(trend_window)
    all_records: list[dict[str, Any]] = []
    per_series: list[dict[str, Any]] = []

    paths = sorted(csv_dir.glob("orange_county_ro_*.csv"))
    for path in paths:
        m = SERIES_RE.search(path.name)
        series_id = m.group(1).upper() if m else path.stem
        frame = pd.read_csv(path)
        if target not in frame.columns or "date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce")
        s3sf = pd.to_numeric(frame[target], errors="coerce")

        ref_mask = (dates >= pd.Timestamp(ref_win["start"])) & (
            dates <= pd.Timestamp(ref_win["end"])
        )
        assert pd.Timestamp(ref_win["end"]) < available_from
        ref_vals = s3sf[ref_mask & np.isfinite(s3sf)]
        if ref_vals.empty:
            continue
        ref_mean = float(ref_vals.mean())
        threshold = rel_thr * ref_mean
        indicator = s3sf / ref_mean

        cip = frame["cip"] if "cip" in frame.columns else None
        dsr = frame["days_since_replacement"] if "days_since_replacement" in frame.columns else None
        segments, audit, seg_ids = build_segments_for_series(
            frame["date"], cip=cip, days_since_replacement=dsr
        )

        def _split_for_date(ts: pd.Timestamp) -> str:
            if ts <= train_end:
                return "train"
            if ts <= dev_end:
                return "development"
            return "test"

        # Single rolling-origin pass; enforce train/dev cutoffs by excluding
        # origins whose calendar follow-up would extend past the split end.
        series_recs: list[dict[str, Any]] = []
        for model in forecasters:
            recs = run_rolling_origin(
                frame["date"],
                indicator,
                forecaster=model,
                horizons=horizons,
                threshold=rel_thr,
                seg_ids=seg_ids,
                min_history_days=min_hist,
            )
            for r in recs:
                origin_ts = pd.Timestamp(r["origin_date"])
                split_name = _split_for_date(origin_ts)
                r["series_id"] = series_id
                r["split"] = split_name
                if origin_ts < available_from:
                    r["eligible"] = False
                    r["eligible_trajectory"] = False
                    r["eligible_threshold_event"] = False
                    r["exclusion_reason"] = "before_analytical_reference_available_from"
                else:
                    cutoff = (
                        train_end
                        if split_name == "train"
                        else (dev_end if split_name == "development" else None)
                    )
                    if cutoff is not None:
                        horizon_end = origin_ts + pd.Timedelta(days=int(r["horizon_days"]))
                        if horizon_end > cutoff:
                            r["eligible"] = False
                            r["eligible_trajectory"] = False
                            r["eligible_threshold_event"] = False
                            r["exclusion_reason"] = "followup_extends_past_split_cutoff"
                series_recs.append(r)
        all_records.extend(series_recs)
        series_metrics = summarize_records(
            [r for r in series_recs if r.get("split") == "test"],
        )
        per_series.append(
            {
                "source_series_id": series_id,
                "n_segments": len(segments),
                "reference": {
                    "label": ref_win["label"],
                    "window": [ref_win["start"], ref_win["end"]],
                    "available_from": ref_win["available_from"],
                    "s3sf_mean": ref_mean,
                    "threshold_s3sf": threshold,
                    "note": (
                        "Analytical reference only; not proven clean. "
                        "Relative decline ≠ manufacturer cleaning recommendation."
                    ),
                },
                "n_cip_boundaries": sum(1 for a in audit if a.get("reason") == "cip_flag_present"),
                "test_metrics": series_metrics,
            }
        )

        if series_id == "A01":
            _fig_ocwd_example(
                dates, indicator, series_recs, fig_dir / "ocwd_A01_forecast.png", rel_thr
            )

    test_recs = [r for r in all_records if r.get("split") == "test"]
    summary = summarize_records(test_recs, group_key="series_id")
    payload = {
        "experiment": "ocwd_s3sf_separate_benchmark",
        "config": str(config_path),
        "target": target,
        "temperature_correction_reapplied": False,
        "pooled_with_synthetic": False,
        "claims_cleaning_time_validity": False,
        "claims_synthetic_transfer": False,
        "claims_real_plant_validation": False,
        "chronology": cfg["chronology"],
        "chronology_enforcement": (
            "Train/development origins require origin+horizon ≤ split cutoff. "
            "Prior Phase 3 test metrics were inspected; this corrective pass regenerates "
            "baselines without claiming plant validation."
        ),
        "reference_policy": ref_win,
        "disclaimer": (
            "OCWD experiment uses source-calculated s3sf. Normalisation provenance "
            "remains external and incompletely verified. Independent-normalisation "
            "blockers from Phase 2 are NOT resolved."
        ),
        "per_series": per_series,
        "test_split_summary": summary,
        "n_test_records": len(test_recs),
        "figures": [str(p) for p in sorted(fig_dir.glob("ocwd_*.png"))],
        "notes": cfg.get("notes", []),
    }
    (report_dir / "ocwd_metrics.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _fig_ocwd_example(
    dates: pd.Series,
    indicator: pd.Series,
    records: list[dict[str, Any]],
    path: Path,
    threshold: float,
) -> None:
    cand = [
        r
        for r in records
        if r.get("eligible")
        and r.get("split") == "test"
        and r.get("model_id") == "recent_linear_trend"
        and r.get("horizon_days") == 14
    ]
    if not cand:
        cand = [r for r in records if r.get("eligible") and r.get("horizon_days") == 14]
    if not cand:
        return
    r = cand[0]
    origin = r["origin_index"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(dates), indicator, label="s3sf / ref_mean", alpha=0.8)
    ax.axhline(threshold, color="k", linestyle=":", label="analytical relative threshold")
    traj = r.get("trajectory") or []
    fut = pd.date_range(
        pd.to_datetime(dates).iloc[origin] + pd.Timedelta(days=1),
        periods=len(traj),
        freq="D",
    )
    ax.plot(fut, traj, label="linear-trend forecast")
    ax.axvline(pd.to_datetime(dates).iloc[origin], color="gray", linestyle="--")
    ax.set_title(f"OCWD separate experiment — {r['series_id']} origin {r['origin_date']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
