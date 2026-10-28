"""Quantitative per-series readiness audit for Phase 3."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqua_fouling.application.audit_metrics import (
    analyse_date_coverage,
    longest_consecutive_missing_runs,
    missing_measurement_mask,
)
from aqua_fouling.forecasting.evaluation import build_segments_for_series, eligible_origins

SERIES_RE = re.compile(r"orange_county_ro_([A-G]\d{2})\.csv$", re.IGNORECASE)


def _series_id(path: Path) -> str | None:
    m = SERIES_RE.search(path.name)
    return m.group(1).upper() if m else None


def audit_series_quantitative(
    path: Path,
    *,
    target_col: str = "s3sf",
    physical_input_cols: list[str] | None = None,
    horizons: list[int] | None = None,
    min_history_days: int = 21,
) -> dict[str, Any]:
    """Per-series quantitative readiness metrics."""
    horizons = horizons or [7, 14, 30]
    physical_input_cols = physical_input_cols or [
        "temp_c",
        "stage_3_flux",
        "stage_3_net_driving_press",
        "tcf_water",
    ]
    series_id = _series_id(path) or path.stem
    frame = pd.read_csv(path)
    dates = frame["date"] if "date" in frame.columns else pd.Series(dtype=object)
    coverage = analyse_date_coverage(dates)

    # Missing measurement runs for target
    target_missing_runs: list[dict[str, Any]] = []
    n_missing_target = 0
    if target_col in frame.columns:
        mask = missing_measurement_mask(frame[target_col])
        n_missing_target = int(mask.sum())
        runs = longest_consecutive_missing_runs(mask, top_n=10)
        target_missing_runs = [
            {"start_index": r.start_index, "end_index": r.end_index, "length": r.length}
            for r in runs
        ]

    # Present dates with missing physical inputs
    phys_present = [c for c in physical_input_cols if c in frame.columns]
    phys_missing_any = np.zeros(len(frame), dtype=bool)
    for col in phys_present:
        phys_missing_any |= missing_measurement_mask(frame[col]).to_numpy()
    n_present_date_missing_physical = int(phys_missing_any.sum())

    cip = frame["cip"] if "cip" in frame.columns else None
    dsr = frame["days_since_replacement"] if "days_since_replacement" in frame.columns else None
    segments, audit, seg_ids = build_segments_for_series(dates, cip=cip, days_since_replacement=dsr)
    cip_boundaries = [a for a in audit if a.get("reason") == "cip_flag_present"]
    replacement_resets = [
        a for a in audit if a.get("reason") == "replacement_counter_reset_candidate"
    ]

    target = frame[target_col] if target_col in frame.columns else pd.Series([np.nan] * len(frame))
    elig = eligible_origins(
        dates,
        target,
        seg_ids=seg_ids,
        horizons=horizons,
        min_history_days=min_history_days,
    )
    eligible_counts = {
        str(h): sum(1 for o in origins if o.eligible_trajectory) for h, origins in elig.items()
    }

    durations: list[float] = [
        float(s["duration_days"]) for s in segments if s.get("duration_days") is not None
    ]
    return {
        "source_series_id": series_id,
        "path": str(path),
        "n_rows": int(len(frame)),
        "missing_calendar_dates": {
            "n": coverage.n_missing_dates,
            "sample": coverage.missing_dates[:20],
            "note": "Calendar days absent between min and max date.",
        },
        "present_dates_missing_target": {
            "column": target_col,
            "n": n_missing_target,
            "longest_runs": target_missing_runs,
            "note": "Rows present with missing/non-finite target values. Not labelled shutdown.",
        },
        "present_dates_missing_physical_inputs": {
            "columns_checked": phys_present,
            "n_rows_with_any_missing": n_present_date_missing_physical,
            "note": "Present dates where at least one mapped physical input is missing.",
        },
        "segments": {
            "count": len(segments),
            "lengths_rows": [s["n_rows"] for s in segments],
            "duration_days": durations,
            "duration_days_mean": float(np.mean(durations)) if durations else None,
            "duration_days_median": float(np.median(durations)) if durations else None,
            "segments": segments,
        },
        "cip_boundaries": {"count": len(cip_boundaries), "events_sample": cip_boundaries[:20]},
        "replacement_reset_candidates": {
            "count": len(replacement_resets),
            "events_sample": replacement_resets[:20],
        },
        "eligible_forecast_origins": {
            "horizons_days": horizons,
            "min_history_days": min_history_days,
            "counts_by_horizon": eligible_counts,
        },
        "caveats": [
            "Long gaps are not interpolated.",
            "Missingness is not interpreted as a confirmed shutdown.",
            "Filename/stage semantics remain unresolved; series_id only.",
        ],
    }


def run_quantitative_readiness(
    csv_dir: Path,
    output_dir: Path,
    *,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """Audit all OCWD CSVs and write JSON + markdown appendix."""
    horizons = horizons or [7, 14, 30]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(csv_dir.glob("orange_county_ro_*.csv"))
    per_series = [audit_series_quantitative(p, horizons=horizons) for p in paths]
    summary = {
        "n_series": len(per_series),
        "horizons_days": horizons,
        "total_eligible_origins_by_horizon": {
            str(h): int(
                sum(
                    s["eligible_forecast_origins"]["counts_by_horizon"].get(str(h), 0)
                    for s in per_series
                )
            )
            for h in horizons
        },
        "total_segments": int(sum(s["segments"]["count"] for s in per_series)),
        "total_cip_boundaries": int(sum(s["cip_boundaries"]["count"] for s in per_series)),
        "total_replacement_reset_candidates": int(
            sum(s["replacement_reset_candidates"]["count"] for s in per_series)
        ),
        "series": per_series,
        "ocwd_independent_normalisation_blockers_resolved": False,
        "note": (
            "OCWD independent-normalisation blockers from Phase 2 are NOT resolved. "
            "This audit quantifies forecast eligibility on source-calculated targets."
        ),
    }
    (output_dir / "readiness_quantitative.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Quantitative readiness audit (Phase 3)",
        "",
        f"Series: **{summary['n_series']}**",
        f"Total segments: **{summary['total_segments']}**",
        f"CIP boundary events (sum): **{summary['total_cip_boundaries']}**",
        f"Replacement reset candidates (sum): **{summary['total_replacement_reset_candidates']}**",
        "",
        "Trajectory-eligible forecast origins by horizon (sum across series):",
        "",
        "(≥1 finite follow-up in the calendar window; threshold-event eligibility is stricter.)",
        "",
    ]
    eligible_map = summary["total_eligible_origins_by_horizon"]
    assert isinstance(eligible_map, dict)
    for h, n in eligible_map.items():
        lines.append(f"- {h}d: **{n}**")
    lines.extend(
        [
            "",
            "## Per-series snapshot",
            "",
            "| series | segments | CIP bounds | repl. resets | miss target | elig 7/14/30 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for s in per_series:
        e = s["eligible_forecast_origins"]["counts_by_horizon"]
        lines.append(
            f"| {s['source_series_id']} | {s['segments']['count']} | "
            f"{s['cip_boundaries']['count']} | {s['replacement_reset_candidates']['count']} | "
            f"{s['present_dates_missing_target']['n']} | "
            f"{e.get('7', 0)}/{e.get('14', 0)}/{e.get('30', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Distinctions",
            "",
            "- Missing calendar dates ≠ present dates with missing target "
            "≠ missing physical inputs.",
            "- Missingness is **not** labelled as confirmed shutdown.",
            "- OCWD independent-normalisation blockers remain unresolved.",
            "",
        ]
    )
    (output_dir / "readiness_quantitative.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
