"""Gap / cleaning / replacement segment utilities with audit trail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentRuleConfig:
    """Documented rules for contiguous usable segments."""

    max_internal_gap_days: int = 0
    split_on_cip_flag: bool = True
    split_on_replacement_counter_reset: bool = True
    flag_abrupt_flow_relative_change: float = 0.35


def contiguous_segments(
    dates: pd.Series,
    *,
    cip: pd.Series | None = None,
    days_since_replacement: pd.Series | None = None,
    rules: SegmentRuleConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return usable segments and an exclusion/event audit trail.

    Missing observations are preserved in the source frame; this function only
    reports segment boundaries. Long gaps are not interpolated. CIP flags are
    preserved without inventing cleaning duration. Abrupt changes are flagged
    for inspection, not automatically labelled as cleanings. Replacement-counter
    resets are candidate events. Low-flow rows are not auto-discarded.
    """
    cfg = rules or SegmentRuleConfig()
    audit: list[dict[str, Any]] = []
    parsed = pd.to_datetime(dates, errors="coerce")
    n = len(parsed)
    if n == 0:
        return [], audit

    split_after = np.zeros(n, dtype=bool)
    # Calendar gaps between successive present rows
    for i in range(1, n):
        if pd.isna(parsed.iloc[i]) or pd.isna(parsed.iloc[i - 1]):
            continue
        gap = (parsed.iloc[i] - parsed.iloc[i - 1]).days
        if gap > cfg.max_internal_gap_days + 1:
            split_after[i] = True
            audit.append(
                {
                    "reason": "calendar_gap",
                    "at_index": i,
                    "gap_days": int(gap),
                    "action": "split_segment",
                }
            )

    if cfg.split_on_cip_flag and cip is not None:
        cip_num = pd.to_numeric(cip, errors="coerce")
        for i in range(n):
            if cip_num.iloc[i] == 1:
                # Boundary at flagged day; do not invent duration.
                split_after[i] = True
                audit.append(
                    {
                        "reason": "cip_flag_present",
                        "at_index": i,
                        "action": "split_segment_boundary",
                        "note": "CIP duration not invented from flag alone.",
                    }
                )

    if cfg.split_on_replacement_counter_reset and days_since_replacement is not None:
        dsr = pd.to_numeric(days_since_replacement, errors="coerce").to_numpy(dtype=float)
        for i in range(1, n):
            prev, cur = dsr[i - 1], dsr[i]
            if np.isfinite(prev) and np.isfinite(cur) and cur + 1e-9 < prev:
                split_after[i] = True
                audit.append(
                    {
                        "reason": "replacement_counter_reset_candidate",
                        "at_index": i,
                        "previous": float(prev),
                        "value": float(cur),
                        "action": "split_segment_boundary_for_inspection",
                    }
                )

    segments: list[dict[str, Any]] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or split_after[i]:
            end = i - 1
            if start <= end:
                d0 = parsed.iloc[start]
                d1 = parsed.iloc[end]
                duration = int((d1 - d0).days) + 1 if pd.notna(d0) and pd.notna(d1) else None
                segments.append(
                    {
                        "start_index": start,
                        "end_index": end,
                        "n_rows": end - start + 1,
                        "start_date": d0.date().isoformat() if pd.notna(d0) else None,
                        "end_date": d1.date().isoformat() if pd.notna(d1) else None,
                        "duration_days": duration,
                    }
                )
            start = i

    return segments, audit


def flag_abrupt_changes(
    values: pd.Series,
    *,
    relative_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Flag abrupt relative jumps for inspection (not auto-labelled as CIP)."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    flags: list[dict[str, Any]] = []
    for i in range(1, len(numeric)):
        prev, cur = numeric[i - 1], numeric[i]
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        if prev == 0:
            continue
        rel = abs(cur - prev) / abs(prev)
        if rel >= relative_threshold:
            flags.append(
                {
                    "reason": "abrupt_change_for_inspection",
                    "at_index": i,
                    "previous": float(prev),
                    "value": float(cur),
                    "relative_change": float(rel),
                    "action": "flag_only_not_auto_cleaning",
                }
            )
    return flags
