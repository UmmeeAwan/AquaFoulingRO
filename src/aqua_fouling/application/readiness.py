"""Generate Phase 2 normalisation readiness report from audit artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_readiness_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    series = audit.get("series", [])
    s3_exc_total = sum((s.get("s3sf_relationship") or {}).get("n_exceptions") or 0 for s in series)
    rel_maxes: list[float] = [
        float((s.get("s3sf_relationship") or {})["relative_error_max"])
        for s in series
        if (s.get("s3sf_relationship") or {}).get("relative_error_max") is not None
    ]
    abs_maxes: list[float] = [
        float((s.get("s3sf_relationship") or {})["absolute_error_max"])
        for s in series
        if (s.get("s3sf_relationship") or {}).get("absolute_error_max") is not None
    ]

    lines = [
        "# Phase 2 normalisation readiness",
        "",
        "This report does **not** declare the dataset fully suitable for training "
        "merely because software runs.",
        "",
        "## Documented vs assumed vs unresolved",
        "",
        "| Topic | Status |",
        "| --- | --- |",
        "| `rof_toc_avg` unit mg/L (Rd) | Documented |",
        "| `feed_psi` / `temp_c` naming units | Inferred from column names |",
        "| Conductivity ≠ TDS | Documented policy |",
        "| `tcf_water` convention vs FilmTec TCF | Unresolved (do not equate) |",
        "| Filename = stage | Unresolved |",
        "| Stage-3 feed pressure = `stage_2_3_press` | Assumed/inferred for independent NDP |",
        "| Flux / specific-flux / NDP units | Unresolved in Rd |",
        "| Licence | Unresolved (none in archive) |",
        "",
        "## Units and mapping decisions",
        "",
        "- Two adapters: `configs/ocwd_schema_ae.yaml` (A–E) and "
        "`configs/ocwd_schema_fg.yaml` (F–G).",
        "- `source_series_id` values such as A01 are preserved unchanged.",
        "- Stage-3 analysis selects stage-3 columns inside each file (not `*03` files only).",
        "- Exported index column explains CSV width = Rd width + 1.",
        "- Source-calculated `stage_3_net_driving_press`, `s3sf`, `tcf_water` are preserved "
        "and labelled distinct from physics-module outputs.",
        "",
        "## s3sf formula exceptions (absolute and relative)",
        "",
        f"- Series: {summary.get('n_csv_series')}",
        f"- Total exception rows (rtol/atol as recorded in audit): **{s3_exc_total}**",
        ("- Max absolute error across series: " f"**{max(abs_maxes) if abs_maxes else 'n/a'}**"),
        ("- Max relative error across series: " f"**{max(rel_maxes) if rel_maxes else 'n/a'}**"),
        "",
        "Absolute magnitude alone is not used to call discrepancies negligible. "
        "See `reports/data-audit/audit.md` exception detail for dates, observed, "
        "calculated, rtol and atol.",
        "",
        "## Source-calculated vs independently calculated",
        "",
        "- **Diagnostic path:** compares/preserves source `s3sf`, source NDP, source "
        "`tcf_water` — labelled diagnostic only.",
        "- **Independent path:** FilmTec exponential TCF (`filmtec_exponential_tcf_v1`) "
        "+ FilmTec C_fc osmotic pressure (`filmtec_cfc_bar_lt_20000_v1`) + ASTM/FilmTec-style NDP "
        "+ Q_norm = Q * NDP_ref/NDP_op * TCF_ref/TCF_op.",
        "- Independent path on OCWD requires explicit feed concentration (mg/L) or "
        "calibration; conductivity is preserved and never silently converted.",
        "- Source-calculated NDP is **not** treated as validation of our concentration correction.",
        "",
        "## Usable segments",
        "",
        "Segmentisation rules (documented in `application/segments.py`): split on calendar "
        "gaps (>0 internal gap days by default), CIP flag boundaries (duration not invented), "
        "and replacement-counter reset candidates. Abrupt flow jumps are flagged for "
        "inspection only. Per-series segment counts/durations are produced at normalise time; "
        "run `poetry run aqua normalise` per series to populate operational segment stats.",
        "",
        "## Reference availability",
        "",
        "- Reference must be `documented` or `explicit_window` with provenance.",
        "- Forbidden: first row, dataset maximum, dss=0 as clean proof, post-cutoff "
        "reference establishment, automatic reset after every CIP.",
        "- Without reference: intermediate FilmTec TCF may still be returned; "
        "Q_normalised / health marked unavailable.",
        "",
        "## Which records support which analysis",
        "",
        "| Analysis | Currently supported on OCWD CSVs |",
        "| --- | --- |",
        "| Schema adapt + validate | Yes (all 21 series with correct adapter) |",
        "| Diagnostic source-field comparison | Yes |",
        "| Independent FilmTec TCF from `temp_c` | Yes where `temp_c` finite |",
        "| Independent NDP + Q_norm | Only when concentration/calibration supplied "
        "and stage pressure/flow mappings accepted as inferred |",
        "| Forecasting labels / training set | **Not ready** |",
        "",
        "## Remaining blockers to training (smallest practical resolutions)",
        "",
        "1. **Confirm stage-3 hydraulic column identities and units** (pressures, flows, DP) "
        "against P&ID/diagrams — remove inferred `stage_2_3_press` assumption.",
        "2. **Provide conductivity→concentration calibration** (or measured concentration) "
        "with documented scope; do not invent a coefficient.",
        "3. **Confirm flux / specific-flux units** for any model that consumes them.",
        "4. **Document a plant reference** (start-up or design) with provenance and "
        "`available_from` — do not auto-pick healthy rows.",
        "5. **Resolve licence/citation** before publishing derived training artefacts.",
        "6. **Resolve filename/stage semantics** if any labelling scheme depends on them "
        "(currently unresolved; series ids unchanged).",
        "",
        "## Verdict",
        "",
        "**Not ready to begin forecasting.** Phase 2 physics and APIs are in place for "
        "validation/normalisation workflows, but independent physical normalisation of the "
        "OCWD archive remains blocked on concentration calibration and confirmed stage "
        "hydraulics. Diagnostic analysis using source-calculated fields is available and "
        "must remain clearly labelled as such.",
        "",
    ]
    return "\n".join(lines)


def write_normalisation_readiness_report(
    *,
    audit_json: Path,
    output: Path,
) -> Path:
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_readiness_markdown(audit), encoding="utf-8")
    return output
