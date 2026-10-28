"""Assemble reports/phase3/benchmark_summary.md from metric JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_benchmark_summary(
    report_dir: Path,
    *,
    synthetic: dict[str, Any],
    ocwd: dict[str, Any],
    readiness: dict[str, Any] | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 benchmark summary",
        "",
        "This is **not** the final trained-model submission. Baselines only; "
        "a fitted state-space candidate will follow after baseline inspection.",
        "",
        "## Experiment separation",
        "",
        "| Experiment | Target | Metrics file |",
        "| --- | --- | --- |",
        "| Primary synthetic | Independent relative health from raw | `synthetic_metrics.json` |",
        "| Separate OCWD | Source-calculated `s3sf` (no extra TCF) | `ocwd_metrics.json` |",
        "",
        "**Do not pool OCWD and synthetic metrics.**",
        "",
        "Synthetic benchmark performance does **not** establish plant accuracy.",
        "",
        "OCWD independent-normalisation blockers from Phase 2 are **not** resolved.",
        "",
        "## Quantitative readiness (OCWD)",
        "",
    ]
    if readiness:
        elig = readiness.get("total_eligible_origins_by_horizon") or {}
        elig_txt = (
            ", ".join(f"{h}d={n}" for h, n in elig.items()) if isinstance(elig, dict) else str(elig)
        )
        lines.extend(
            [
                f"- Series: {readiness.get('n_series')}",
                f"- Total segments: {readiness.get('total_segments')}",
                f"- Eligible origins by horizon: {elig_txt}",
                f"- CIP boundaries (sum): {readiness.get('total_cip_boundaries')}",
                f"- Replacement reset candidates (sum): "
                f"{readiness.get('total_replacement_reset_candidates')}",
                "",
            ]
        )
    else:
        lines.append("- (readiness_quantitative.json not supplied)")
        lines.append("")

    lines.extend(
        [
            "## Synthetic baseline results (test split)",
            "",
            f"- Threshold (relative health): {synthetic.get('threshold_relative_health')}",
            f"- Horizons: {synthetic.get('horizons_days')}",
            f"- Disclaimer: {synthetic.get('disclaimer')}",
            "",
            "Trajectory-window MAE/RMSE = mean error over calendar offsets in the "
            "horizon window (not single lead-h point errors).",
            "",
            "Common-origin MAE uses origins scored by all models; unavailable "
            "predictions are counted separately. Partial-window ≠ complete-horizon.",
            "",
            "| model | h | n | commonMAE(n) | winMAE | part/comp | na | time(n) | false | miss |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (synthetic.get("test_split_summary") or {}).get("by_model_horizon") or []:
        lines.append(
            f"| {row['model_id']} | {row['horizon_days']} | "
            f"{row.get('n_scored_origins', row.get('n_eligible_origins'))} | "
            f"{_fmt(row.get('common_origin_mae_mean'))} "
            f"({row.get('n_common_scored_origins', '—')}) | "
            f"{_fmt(row.get('trajectory_window_mae_mean', row.get('mae_mean')))} | "
            f"{row.get('n_partial_window_scored', '—')}/"
            f"{row.get('n_complete_horizon_scored', '—')} | "
            f"{row.get('n_pred_unavailable', '—')} | "
            f"{_fmt(row.get('crossing_timing_error_mae'))} "
            f"({row.get('n_timing_samples', '—')}) | "
            f"{row.get('false_warnings')}/"
            f"{row.get('false_warning_denominator', '—')} | "
            f"{row.get('missed_warnings')}/"
            f"{row.get('missed_warning_denominator', '—')} |"
        )

    lines.extend(
        [
            "",
            "### Synthetic figures",
            "",
        ]
    )
    for fig in synthetic.get("figures") or []:
        lines.append(f"- `{fig}`")

    lines.extend(
        [
            "",
            "## OCWD separate s3sf baseline results (test split)",
            "",
            f"- TCF re-applied: {ocwd.get('temperature_correction_reapplied')}",
            f"- Claims cleaning-time validity: {ocwd.get('claims_cleaning_time_validity')}",
            f"- Pooled with synthetic: {ocwd.get('pooled_with_synthetic')}",
            f"- Disclaimer: {ocwd.get('disclaimer')}",
            "",
            "| model | h | n_scored | commonMAE (n) | winMAE | unavail | false | miss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (ocwd.get("test_split_summary") or {}).get("by_model_horizon") or []:
        lines.append(
            f"| {row['model_id']} | {row['horizon_days']} | "
            f"{row.get('n_scored_origins', row.get('n_eligible_origins'))} | "
            f"{_fmt(row.get('common_origin_mae_mean'))} "
            f"({row.get('n_common_scored_origins', '—')}) | "
            f"{_fmt(row.get('trajectory_window_mae_mean', row.get('mae_mean')))} | "
            f"{row.get('n_pred_unavailable', '—')} | "
            f"{row.get('false_warnings')}/"
            f"{row.get('false_warning_denominator', '—')} | "
            f"{row.get('missed_warnings')}/"
            f"{row.get('missed_warning_denominator', '—')} |"
        )

    lines.extend(
        [
            "",
            "### OCWD figures",
            "",
        ]
    )
    for fig in ocwd.get("figures") or []:
        lines.append(f"- `{fig}`")

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "- Persistence",
            "- Recent linear trend",
            "",
            "No uncertainty intervals. Shared `Forecaster` interface ready for a "
            "state-space model in a later phase.",
            "",
            "## Next (not this phase)",
            "",
            "- Fitted state-space candidate",
            "- Versioned model artefact",
            "- Final forecast API",
            "",
        ]
    )
    path = report_dir / "benchmark_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    # Also ensure metric JSON copies exist if callers already wrote them
    return path


def load_json(path: Path) -> dict[str, Any]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return loaded


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)
