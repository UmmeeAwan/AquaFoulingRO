"""Data audit application service: extract, analyse, write reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aqua_fouling.application.audit_metrics import (
    analyse_date_coverage,
    check_s3sf_relationship,
    column_numeric_profile,
    compare_schemas,
    consecutive_flag_runs,
    days_since_replacement_changes,
    dss_resets,
    missing_measurement_mask,
)
from aqua_fouling.data.archive import SafeExtractResult, safe_extract_zip, sha256_file

SERIES_PATTERN = re.compile(r"orange_county_ro_([A-G]\d{2})\.csv$", re.IGNORECASE)

# Candidate normalisation-related inputs observed in source headers (availability only).
NORMALISATION_CANDIDATE_COLUMNS = [
    "tcf_water",
    "tcf_salt",
    "tcf",
    "temp_c",
    "temp",
    "temp_2",
    "stage_1_norm_removal",
    "stage_2_norm_removal",
    "stage_3_norm_removal",
    "stage_1_norm_pressure",
    "stage_2_norm_pressure",
    "stage_3_norm_pressure",
    "unit_norm_percent_removal",
    "unit_afbc_norm_percent_removal",
    "s3sf",
    "stage_1_sf",
    "stage_2_sf",
    "stage_1_flux",
    "stage_2_flux",
    "stage_3_flux",
    "stage_1_net_driving_press",
    "stage_2_net_driving_press",
    "stage_3_net_driving_press",
]

# Plant-wide / water-quality fields often shared (availability comparison only).
WATER_QUALITY_CANDIDATES = [
    "ph",
    "temp",
    "ec",
    "turb",
    "c12_free",
    "ti_flow",
    "cart_filter_dp",
    "cl2_tot",
    "rop_turb_avg",
    "rop_turb_max",
    "rof_toc_avg",
    "rof_toc_min",
    "rof_toc_max",
    "rop_toc_avg",
    "rop_toc_min",
    "rop_toc_max",
    "rof_ec_min",
    "rof_ec_max",
    "rop_ec_avg",
    "percent_toc_removal",
    "percent_ec_removal",
]

DOC_CLAIMS: dict[str, str | int | bool] = {
    "doc_date_end_stated": "2020-01-13",
    "doc_date_start_stated": "2019-01-01",
    "doc_source_years_stated": "2019-21",
    "doc_n_files": 21,
    "doc_n_rows_per_file": 744,
    "doc_columns_a_d": 127,
    "doc_columns_f_g": 116,
    "doc_implies_filename_is_stage": True,
    "doc_licence_in_archive": False,
}


@dataclass
class DataAuditService:
    """Orchestrates safe extraction and dataset audit report generation."""

    archive_path: Path
    output_dir: Path
    extract_dir: Path

    def run(self) -> dict[str, Any]:
        """Execute the full audit and write artefacts under ``output_dir``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        extract_result = safe_extract_zip(self.archive_path, self.extract_dir)
        audit = build_audit_report(extract_result)
        write_audit_artefacts(audit, self.output_dir)
        return audit


def run_data_audit(
    archive: Path,
    output: Path,
    *,
    extract_dir: Path | None = None,
) -> dict[str, Any]:
    """CLI/API-facing entry point for the data audit."""
    extract = extract_dir or (output.parent.parent / "data" / "extracted")
    service = DataAuditService(
        archive_path=archive,
        output_dir=output,
        extract_dir=extract,
    )
    return service.run()


def _series_id_from_path(path: Path) -> str | None:
    match = SERIES_PATTERN.search(path.name)
    return match.group(1).upper() if match else None


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_documentation(extract_root: Path) -> dict[str, Any]:
    rd_candidates = list(extract_root.rglob("*.Rd"))
    licence_candidates = [
        p
        for p in extract_root.rglob("*")
        if p.is_file()
        and p.name.lower()
        in {
            "license",
            "license.txt",
            "licence",
            "licence.txt",
            "copying",
            "copyright",
        }
    ]
    rd_text = (
        rd_candidates[0].read_text(encoding="utf-8", errors="replace") if rd_candidates else ""
    )
    return {
        "rd_files": [str(p) for p in rd_candidates],
        "licence_files_found": [str(p) for p in licence_candidates],
        "rd_excerpt_chars": len(rd_text),
        "rd_mentions_2020": "2020" in rd_text,
        "rd_mentions_2021": "2021" in rd_text or "2019-21" in rd_text,
        "rd_mentions_stage_in_filename": "stage 1" in rd_text.lower() and "A01" in rd_text,
    }


def _stage_columns_present(columns: list[str]) -> dict[str, bool]:
    joined = set(columns)
    return {
        "has_stage_1_signals": any(
            c.startswith("stage_1_") or c in {"stage_1_flow", "perm_ec_stage_1"} for c in joined
        ),
        "has_stage_2_signals": any(
            c.startswith("stage_2_") or c in {"stage_2_flow", "perm_ec_stage_2"} for c in joined
        ),
        "has_stage_3_signals": any(
            c.startswith("stage_3_") or c in {"stage_3_flow", "perm_ec_stage_3", "s3sf"}
            for c in joined
        ),
    }


def audit_one_series(series_id: str, path: Path, file_sha256: str) -> dict[str, Any]:
    """Audit a single CSV series."""
    frame = _load_csv(path)
    columns = [str(c) for c in frame.columns]
    date_cov = analyse_date_coverage(
        frame["date"] if "date" in frame.columns else pd.Series(dtype=object)
    )

    column_profiles: dict[str, Any] = {}
    empty_cols: list[str] = []
    constant_cols: list[str] = []
    for col in columns:
        profile = column_numeric_profile(frame[col])
        column_profiles[col] = profile
        if profile["fully_empty"]:
            empty_cols.append(col)
        if profile["constant"]:
            constant_cols.append(col)

    # Measurement missingness excluding structural date completeness.
    measurement_cols = [c for c in columns if c not in {"", "date", "date_2"}]
    n_meas_missing_cells = 0
    for col in measurement_cols:
        n_meas_missing_cells += int(missing_measurement_mask(frame[col]).sum())

    cip_summary: dict[str, Any] | None = None
    if "cip" in frame.columns:
        cip_numeric = pd.to_numeric(frame["cip"], errors="coerce")
        cip_runs = consecutive_flag_runs(frame["cip"])
        cip_summary = {
            "n_cip_flagged_rows": int((cip_numeric == 1).sum()),
            "n_consecutive_runs": len(cip_runs),
            "runs": [
                {"start_index": r.start_index, "end_index": r.end_index, "length": r.length}
                for r in cip_runs
            ],
            "caveat": (
                "CIP flags are per-series observations. Do not interpret each flag "
                "as an independent plant-wide event without cross-series analysis."
            ),
        }

    dss_summary = dss_resets(frame["dss"]) if "dss" in frame.columns else None
    dsr_summary = (
        days_since_replacement_changes(frame["days_since_replacement"])
        if "days_since_replacement" in frame.columns
        else None
    )
    s3sf_check = check_s3sf_relationship(frame)

    norm_availability = {col: (col in frame.columns) for col in NORMALISATION_CANDIDATE_COLUMNS}
    wq_availability = {col: (col in frame.columns) for col in WATER_QUALITY_CANDIDATES}

    dtypes = {col: str(frame[col].dtype) for col in columns}

    return {
        "series_id": series_id,
        "path": str(path),
        "sha256": file_sha256,
        "n_rows": int(len(frame)),
        "n_columns": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "date_coverage": {
            "n_rows": date_cov.n_rows,
            "n_unique_dates": date_cov.n_unique_dates,
            "n_duplicate_date_rows": date_cov.n_duplicate_date_rows,
            "duplicate_dates": date_cov.duplicate_dates,
            "min_date": date_cov.min_date,
            "max_date": date_cov.max_date,
            "n_missing_dates": date_cov.n_missing_dates,
            "missing_dates_sample": date_cov.missing_dates[:30],
            "is_sorted_ascending": date_cov.is_sorted_ascending,
        },
        "missing_measurements": {
            "n_measurement_missing_cells": n_meas_missing_cells,
            "note": (
                "Missing measurements are distinct from missing calendar dates. "
                "Missing periods are not interpreted as plant shutdowns in Phase 1."
            ),
        },
        "empty_columns": empty_cols,
        "constant_columns": constant_cols,
        "column_profiles": column_profiles,
        "cip": cip_summary,
        "dss": dss_summary,
        "days_since_replacement": dsr_summary,
        "s3sf_relationship": s3sf_check,
        "stage_signals": _stage_columns_present(columns),
        "normalisation_candidate_availability": norm_availability,
        "water_quality_candidate_availability": wq_availability,
        "filename_stage_semantics": {
            "source_series_id": series_id,
            "suffix": series_id[1:] if len(series_id) >= 3 else None,
            "status": "unresolved",
            "note": (
                "Filename/stage semantics are unresolved. "
                "`source_series_id` is preserved unchanged. "
                "Each file contains multi-stage columns; stage analysis must "
                "select stage-scoped fields, not infer stage from the filename suffix."
            ),
        },
    }


def _shared_water_quality_equality(
    series_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Check whether candidate plant-wide fields are identical across series (where present)."""
    results: dict[str, Any] = {}
    for col in WATER_QUALITY_CANDIDATES:
        present = {sid: df[col] for sid, df in series_frames.items() if col in df.columns}
        if len(present) < 2:
            results[col] = {"compared_series": list(present), "identical_where_aligned": None}
            continue
        # Align on date when possible.
        aligned: dict[str, pd.Series] = {}
        for sid, series in present.items():
            df = series_frames[sid]
            if "date" in df.columns:
                aligned[sid] = pd.Series(
                    pd.to_numeric(series, errors="coerce").to_numpy(),
                    index=pd.to_datetime(df["date"], errors="coerce"),
                )
            else:
                aligned[sid] = pd.to_numeric(series, errors="coerce")
        ids = sorted(aligned)
        base = aligned[ids[0]]
        identical = True
        for other_id in ids[1:]:
            other = aligned[other_id]
            joined = pd.concat([base, other], axis=1, join="inner")
            if joined.empty:
                identical = False
                break
            a = joined.iloc[:, 0]
            b = joined.iloc[:, 1]
            # Treat pairwise NA equality as matching for "shared field" check.
            both_na = a.isna() & b.isna()
            both_eq = (a == b) | both_na
            if not bool(both_eq.all()):
                identical = False
                break
        results[col] = {
            "compared_series": ids,
            "identical_where_aligned": identical,
        }
    return results


def build_audit_report(extract: SafeExtractResult) -> dict[str, Any]:
    """Build the full audit dictionary from an extraction result."""
    csv_records = [
        f
        for f in extract.files
        if f.destination.suffix.lower() == ".csv" and _series_id_from_path(f.destination)
    ]
    csv_records.sort(key=lambda r: _series_id_from_path(r.destination) or "")

    series_audits: list[dict[str, Any]] = []
    schemas: dict[str, list[str]] = {}
    frames: dict[str, pd.DataFrame] = {}

    for record in csv_records:
        sid = _series_id_from_path(record.destination)
        assert sid is not None
        series = audit_one_series(sid, record.destination, record.sha256)
        series_audits.append(series)
        schemas[sid] = series["columns"]
        frames[sid] = _load_csv(record.destination)

    schema_comparison = compare_schemas(schemas)
    docs = _read_documentation(extract.destination_root)

    # Aggregate CIP
    total_cip = sum((s["cip"]["n_cip_flagged_rows"] if s.get("cip") else 0) for s in series_audits)

    global_min = min(
        (s["date_coverage"]["min_date"] for s in series_audits if s["date_coverage"]["min_date"]),
        default=None,
    )
    global_max = max(
        (s["date_coverage"]["max_date"] for s in series_audits if s["date_coverage"]["max_date"]),
        default=None,
    )

    # Documentation inconsistencies (observed vs Rd claims).
    col_counts = {s["series_id"]: s["n_columns"] for s in series_audits}
    a_e = {k: v for k, v in col_counts.items() if k[0] in "ABCDE"}
    f_g = {k: v for k, v in col_counts.items() if k[0] in "FG"}
    doc_inconsistencies: list[dict[str, Any]] = []

    if docs["rd_mentions_2020"] and global_max and global_max > "2020-01-13":
        doc_inconsistencies.append(
            {
                "topic": "date_range",
                "documentation": (
                    f"Rd description states end date {DOC_CLAIMS['doc_date_end_stated']}"
                ),
                "observed": f"max date across series is {global_max}",
                "also_noted": "Rd \\source mentions 2019-21",
            }
        )

    # Exported index contributes +1 to CSV column counts versus Rd data-frame counts.
    rd_cols_ad = 127
    rd_cols_fg = 116
    assert DOC_CLAIMS["doc_columns_a_d"] == rd_cols_ad
    assert DOC_CLAIMS["doc_columns_f_g"] == rd_cols_fg
    index_note = (
        "CSV exports include an unnamed exported index column "
        "(typically `Unnamed: 0`), which increases the observed column count by 1 "
        "relative to the Rd-stated data-frame width. "
        f"Rd A–D={rd_cols_ad} → CSV often {rd_cols_ad + 1}; "
        f"Rd F–G={rd_cols_fg} → CSV often {rd_cols_fg + 1}."
    )
    if a_e:
        observed_ae = sorted(set(a_e.values()))
        doc_inconsistencies.append(
            {
                "topic": "column_counts_a_e",
                "documentation": (
                    f"Rd states banks A–D have {DOC_CLAIMS['doc_columns_a_d']} columns "
                    "(E not called out separately)."
                ),
                "observed_column_counts_including_exported_index": observed_ae,
                "exported_index_contribution": index_note,
                "status": (
                    "reconciled_if_index_excluded"
                    if observed_ae == [rd_cols_ad + 1]
                    else "unresolved"
                ),
            }
        )
    if f_g:
        observed_fg = sorted(set(f_g.values()))
        doc_inconsistencies.append(
            {
                "topic": "column_counts_f_g",
                "documentation": (
                    f"Rd states banks F–G have {DOC_CLAIMS['doc_columns_f_g']} columns"
                ),
                "observed_column_counts_including_exported_index": observed_fg,
                "exported_index_contribution": index_note,
                "status": (
                    "reconciled_if_index_excluded"
                    if observed_fg == [rd_cols_fg + 1]
                    else "unresolved"
                ),
            }
        )

    multi_stage = [s["series_id"] for s in series_audits if all(s["stage_signals"].values())]
    doc_inconsistencies.append(
        {
            "topic": "filename_vs_stage",
            "status": "unresolved",
            "documentation": ("Rd text describes A01/A02/A03 as bank A stages 1/2/3."),
            "observed": (
                f"{len(multi_stage)} series contain stage_1, stage_2 and stage_3 "
                "signal columns in the same file."
            ),
            "resolution": (
                "Filename/stage semantics remain unresolved. "
                "`source_series_id` values (e.g. A01) are preserved unchanged as "
                "series identifiers only. Stage-level analysis must select "
                "stage-scoped columns inside each file, not rely on the filename suffix."
            ),
        }
    )

    if not docs["licence_files_found"]:
        doc_inconsistencies.append(
            {
                "topic": "licence",
                "documentation": "No licence/copyright file located in the archive.",
                "observed": "licence_files_found=[]",
            }
        )

    unresolved_units = [
        {
            "column": "s3sf",
            "issue": "Unit not stated in Rd; specific-flux unit remains unresolved.",
            "confidence": "unresolved",
        },
        {
            "column": "stage_3_flux",
            "issue": "Unit not stated in Rd; flux unit remains unresolved.",
            "confidence": "unresolved",
        },
        {
            "column": "stage_3_net_driving_press",
            "issue": (
                "Unit not stated in Rd; pressure unit unresolved " "(psi inferred only elsewhere)."
            ),
            "confidence": "unresolved",
        },
        {
            "column": "tcf_water",
            "issue": ("Dimensionless factor by naming, but equation/convention not stated in Rd."),
            "confidence": "inferred",
        },
        {
            "column": "feed_psi",
            "issue": (
                "Column name ends in `_psi` — direct naming evidence for psi; "
                "not restated in Rd prose."
            ),
            "confidence": "inferred",
        },
        {
            "column": "perm_ec",
            "issue": (
                "Rd glosses ec as conductivity; unit symbol not stated. " "Conductivity is not TDS."
            ),
            "confidence": "unresolved",
        },
        {
            "column": "rof_toc_avg",
            "issue": "Rd documents unit as mg/L.",
            "confidence": "documented",
            "documented_unit": "mg/L",
            "evidence": "orange_county_ro.Rd item rof_toc_avg",
        },
    ]

    archive_listing = [
        {
            "archive_member": f.archive_member,
            "destination": str(f.destination),
            "sha256": f.sha256,
            "size_bytes": f.size_bytes,
        }
        for f in extract.files
    ]

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": 2,
        "archive": {
            "path": str(extract.archive_path),
            "sha256": extract.archive_sha256,
            "size_bytes": extract.archive_path.stat().st_size,
        },
        "extraction": {
            "destination_root": str(extract.destination_root),
            "n_files_extracted": len(extract.files),
            "files": archive_listing,
        },
        "summary": {
            "n_csv_series": len(series_audits),
            "series_ids": [s["series_id"] for s in series_audits],
            "rows_per_series": {s["series_id"]: s["n_rows"] for s in series_audits},
            "columns_per_series": col_counts,
            "global_min_date": global_min,
            "global_max_date": global_max,
            "total_cip_flagged_rows": total_cip,
            "all_series_have_three_stage_signals": all(
                all(s["stage_signals"].values()) for s in series_audits
            ),
        },
        "schema_comparison": schema_comparison,
        "missingness_definitions": {
            "missing_columns_schema_difference": (
                "A column absent from a series because that series uses a different "
                "CSV schema (see schema_comparison.per_series_column_deltas). "
                "This is not a missing observation."
            ),
            "missing_observations_within_column": (
                "A cell that is NA/non-finite inside a column that exists on that "
                "series. Distinct from missing calendar dates."
            ),
            "missing_calendar_dates": (
                "A calendar day between min and max date with no row present."
            ),
        },
        "shared_water_quality_fields": _shared_water_quality_equality(frames),
        "documentation": docs,
        "documentation_claims": DOC_CLAIMS,
        "documentation_inconsistencies": doc_inconsistencies,
        "unit_notes": unresolved_units,
        "caveats": [
            "Do not interpret every missing period as shutdown.",
            "Do not interpret every CIP flag as an independent plant-wide event.",
            "Do not infer that dss=0 establishes a clean reference.",
            "s3sf relationship check is observational only.",
            "No source data were filled, filtered, or modified.",
            "Forecasting is not implemented in Phase 1.",
        ],
        "series": series_audits,
    }
    return report


def _per_series_summary_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in audit["series"]:
        dc = s["date_coverage"]
        cip = s.get("cip") or {}
        s3 = s.get("s3sf_relationship") or {}
        rows.append(
            {
                "series_id": s["series_id"],
                "sha256": s["sha256"],
                "n_rows": s["n_rows"],
                "n_columns": s["n_columns"],
                "min_date": dc.get("min_date"),
                "max_date": dc.get("max_date"),
                "n_unique_dates": dc.get("n_unique_dates"),
                "n_duplicate_date_rows": dc.get("n_duplicate_date_rows"),
                "n_missing_dates": dc.get("n_missing_dates"),
                "is_sorted_ascending": dc.get("is_sorted_ascending"),
                "n_measurement_missing_cells": s["missing_measurements"][
                    "n_measurement_missing_cells"
                ],
                "n_empty_columns": len(s["empty_columns"]),
                "n_constant_columns": len(s["constant_columns"]),
                "n_cip_flagged_rows": cip.get("n_cip_flagged_rows"),
                "n_cip_runs": cip.get("n_consecutive_runs"),
                "dss_n_resets": (s.get("dss") or {}).get("n_resets"),
                "dsr_n_non_unit_increments": (s.get("days_since_replacement") or {}).get(
                    "n_non_unit_increments"
                ),
                "s3sf_n_valid": s3.get("n_valid"),
                "s3sf_n_exceptions": s3.get("n_exceptions"),
                "s3sf_absolute_error_max": s3.get("absolute_error_max"),
                "s3sf_relative_error_max": s3.get("relative_error_max"),
                "s3sf_rtol": s3.get("rtol"),
                "s3sf_atol": s3.get("atol"),
                "has_stage_1": s["stage_signals"]["has_stage_1_signals"],
                "has_stage_2": s["stage_signals"]["has_stage_2_signals"],
                "has_stage_3": s["stage_signals"]["has_stage_3_signals"],
            }
        )
    return rows


def _column_inventory_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    # Union of columns with presence and basic stats per series group.
    rows: list[dict[str, Any]] = []
    for s in audit["series"]:
        for col, profile in s["column_profiles"].items():
            rows.append(
                {
                    "series_id": s["series_id"],
                    "column": col,
                    "dtype": profile.get("dtype"),
                    "n_missing": profile.get("n_missing"),
                    "missing_fraction": profile.get("missing_fraction"),
                    "fully_empty": profile.get("fully_empty"),
                    "constant": profile.get("constant"),
                    "n_unique_non_missing": profile.get("n_unique_non_missing"),
                    "n_non_finite": profile.get("n_non_finite"),
                    "min": profile.get("min"),
                    "max": profile.get("max"),
                    "mean": profile.get("mean"),
                    "longest_missing_run": profile.get("longest_missing_run"),
                }
            )
    return rows


def render_audit_markdown(audit: dict[str, Any]) -> str:
    """Render a human-readable audit report."""
    summary = audit["summary"]
    lines: list[str] = [
        "# Data audit report (Phase 1)",
        "",
        f"Generated (UTC): `{audit['generated_at_utc']}`",
        "",
        "## Archive",
        "",
        f"- Path: `{audit['archive']['path']}`",
        f"- SHA-256: `{audit['archive']['sha256']}`",
        f"- Size (bytes): {audit['archive']['size_bytes']}",
        f"- Extracted files: {audit['extraction']['n_files_extracted']}",
        "",
        "## Summary findings",
        "",
        f"- CSV series: **{summary['n_csv_series']}** (`{', '.join(summary['series_ids'])}`)",
        f"- Global date span: **{summary['global_min_date']}** → **{summary['global_max_date']}**",
        f"- Total CIP flagged rows (sum across series): **{summary['total_cip_flagged_rows']}**",
        (
            "- All series contain stage 1/2/3 signals: "
            f"**{summary['all_series_have_three_stage_signals']}**"
        ),
        "",
        "### Rows and columns per series",
        "",
        "| series | rows | columns |",
        "| --- | ---: | ---: |",
    ]
    for sid in summary["series_ids"]:
        lines.append(
            f"| {sid} | {summary['rows_per_series'][sid]} | "
            f"{summary['columns_per_series'][sid]} |"
        )

    lines.extend(
        [
            "",
            "## Schema comparison",
            "",
            f"- Distinct schemas: **{audit['schema_comparison']['n_distinct_schemas']}**",
            f"- Columns shared by all series: "
            f"**{audit['schema_comparison']['n_shared_columns_all_series']}**",
            "",
        ]
    )
    for group in audit["schema_comparison"]["schema_groups"]:
        lines.append(f"- {group['n_columns']} columns: {', '.join(group['series'])}")

    lines.extend(["", "## Documentation inconsistencies", ""])
    if not audit["documentation_inconsistencies"]:
        lines.append("- None recorded.")
    else:
        for item in audit["documentation_inconsistencies"]:
            status = item.get("status", "")
            status_bit = f" [{status}]" if status else ""
            lines.append(f"- **{item['topic']}**{status_bit}: {item.get('documentation', '')}")
            if "observed" in item:
                lines.append(f"  - Observed: {item['observed']}")
            if "observed_column_counts_including_exported_index" in item:
                lines.append(
                    "  - Observed CSV counts (incl. exported index): "
                    f"{item['observed_column_counts_including_exported_index']}"
                )
            if "exported_index_contribution" in item:
                lines.append(f"  - Index note: {item['exported_index_contribution']}")
            if "resolution" in item:
                lines.append(f"  - Resolution: {item['resolution']}")

    lines.extend(
        [
            "",
            "## Missingness definitions",
            "",
            "- Schema-missing columns: "
            f"{audit['missingness_definitions']['missing_columns_schema_difference']}",
            "- Missing observations: "
            f"{audit['missingness_definitions']['missing_observations_within_column']}",
            "- Missing calendar dates: "
            f"{audit['missingness_definitions']['missing_calendar_dates']}",
            "",
            "## Unit notes",
            "",
            "Canonical mappings must declare units explicitly; no conductivity-to-TDS "
            "coefficient is assumed. `rof_toc_avg` is documented as mg/L in the Rd.",
            "",
        ]
    )
    for item in audit["unit_notes"]:
        conf = item.get("confidence", "unresolved")
        extra = ""
        if item.get("documented_unit"):
            extra = f" Documented unit: {item['documented_unit']}."
        lines.append(f"- `{item['column']}` [{conf}]: {item['issue']}{extra}")

    lines.extend(
        [
            "",
            "## s3sf observational check",
            "",
            "Relationship tested per series: "
            "`s3sf ≈ stage_3_flux * tcf_water / stage_3_net_driving_press`.",
            "Absolute and relative errors are both reported. Absolute magnitude alone "
            "does not justify calling discrepancies negligible.",
            "This is an observed numerical relationship, not proof of a physical formula.",
            "",
            "| series | n_valid | n_exceptions | abs err max | rel err max | rtol | atol |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in audit["series"]:
        rel = s["s3sf_relationship"]
        lines.append(
            f"| {s['series_id']} | {rel.get('n_valid', '')} | "
            f"{rel.get('n_exceptions', '')} | {rel.get('absolute_error_max', '')} | "
            f"{rel.get('relative_error_max', '')} | {rel.get('rtol', '')} | "
            f"{rel.get('atol', '')} |"
        )
    lines.extend(["", "### Exception detail (all series)", ""])
    for s in audit["series"]:
        for exc in s["s3sf_relationship"].get("exceptions") or []:
            lines.append(
                f"- {s['series_id']} date={exc.get('date')} row={exc.get('row_index')}: "
                f"observed={exc.get('observed_s3sf')} calculated={exc.get('calculated_value')} "
                f"abs_err={exc.get('absolute_error')} rel_err={exc.get('relative_error')} "
                f"(rtol={exc.get('rtol')}, atol={exc.get('atol')})"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in audit["caveats"]:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Artefacts",
            "",
            "- `audit.json` — machine-readable full report",
            "- `audit.md` — this document",
            "- `per_series_summary.csv` — one row per series",
            "- `column_inventory.csv` — per series × column profiles",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_artefacts(audit: dict[str, Any], output_dir: Path) -> None:
    """Write audit.json, audit.md, and summary CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "audit.json"
    # Drop bulky per-column profiles from a slim copy? Keep full for machine use.
    # column_profiles make JSON large; retain them for completeness as required.
    json_path.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")

    md_path = output_dir / "audit.md"
    md_path.write_text(render_audit_markdown(audit), encoding="utf-8")

    per_series = pd.DataFrame(_per_series_summary_rows(audit))
    per_series.to_csv(output_dir / "per_series_summary.csv", index=False)

    inventory = pd.DataFrame(_column_inventory_rows(audit))
    inventory.to_csv(output_dir / "column_inventory.csv", index=False)

    # Also write a lightweight hashes sidecar for convenience.
    hashes = {
        "archive_sha256": audit["archive"]["sha256"],
        "extracted": [
            {"member": f["archive_member"], "sha256": f["sha256"]}
            for f in audit["extraction"]["files"]
        ],
    }
    (output_dir / "checksums.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )


# Re-export for typing convenience in tests.
__all__ = [
    "DataAuditService",
    "build_audit_report",
    "run_data_audit",
    "sha256_file",
]
