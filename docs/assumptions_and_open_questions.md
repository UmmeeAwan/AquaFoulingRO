# Assumptions and open questions

Phase 2 adds an independent physics path and adapters. Phase 3 adds a synthetic primary benchmark and a **separate** OCWD `s3sf` baseline experiment. Items below remain **unresolved** unless cited as documented.

## Working constraints (not scientific claims)

- Filename/stage semantics are **unresolved**. `source_series_id` values (e.g. A01) are preserved unchanged; stage analysis selects stage-scoped columns inside each file.
- Missing calendar dates, schema-missing columns, and missing measurements are distinct (see `reports/phase3/readiness_quantitative.md`).
- CIP flags are preserved without inventing duration; not auto plant-wide events.
- `dss=0` is **not** a clean-reference proof; reference is never first-row / dataset-max / auto-reset after every CIP.
- `s3sf ≈ flux * tcf_water / ndp` is observational (absolute **and** relative errors reported).
- Conductivity is **not** TDS; no silent coefficient.
- Source `tcf_water` is **not** assumed equal to `filmtec_exponential_tcf_v1`; no double correction.
- Independent osmotic pressure uses FilmTec C_fc&lt;20000 mg/L method only — **not** labelled valid for seawater/high-TDS wastewater without further validation.
- `stage_2_3_press` → stage-3 feed pressure is an **inferred assumption** pending P&ID confirmation.

## Phase 3 synthetic domain

- Primary end-to-end benchmark uses a documented **brackish / NaCl-equivalent** demonstration domain (`brackish_nacl_demo_v1` in `configs/synthetic_benchmark.yaml`).
- Validity is checked at effective concentrate-side **C_fc**, not only at feed concentration; invalid scenarios raise (not silently repaired).
- Flow conversion: `Q_gpd = flux_GFD × area_m² × 10.76391041671` (explicit gpd columns).
- Cleaning threshold 0.85 is a synthetic-only analytical choice (15% decline from documented clean reference), not a manufacturer trigger.
- This is **not** a validated seawater model; unsupported conditions raise `DomainViolationError` with actionable messages.
- Literature-supported relationships (FilmTec TCF, C_fc&lt;20000 osmotic, ASTM-style NDP) are separated from chosen experimental scenario ranges (feed concentration, recovery, clean permeability).
- Temperature treatment uses FilmTec TCF once; viscosity is not double-counted via an extra empirical TCF.
- Synthetic clean reference is noise-free matching T/NDP/flow by construction; latent ground truth is never exposed as forecasting inputs.
- Synthetic raw CSVs use `configs/canonical_synthetic_schema.yaml` through the shared `normalise_csv` path.
- **Synthetic benchmark performance does not establish plant accuracy.**

## Phase 3 OCWD experiment (separate)

- Target is source-calculated `s3sf` without re-applying our TCF.
- Reference window is labelled `analytical_reference_not_proven_clean` (`available_from` after the window end); not equated with a manufacturer cleaning recommendation.
- OCWD independent-normalisation blockers from Phase 2 are **not** claimed resolved.
- Metrics are **not** pooled with synthetic results; this experiment does **not** claim synthetic→plant model transfer.

## Documentation inconsistencies

- Rd end date 2020-01-13 vs source 2019–21 vs observed through 2021-01-13.
- Rd column counts vs CSV (+1 exported index `Unnamed: 0`) — reconciled if index excluded.
- Rd filename=stage narrative vs multi-stage columns in every file — **unresolved**.
- Bank E not separately described in Rd; CSV groups with A–D (128 cols).

## Open questions / blockers (still open)

1. Confirm stage-3 hydraulic column identities and units.
2. Provide conductivity→concentration calibration (or measured concentration).
3. Confirm flux / specific-flux / NDP units.
4. Document a plant reference with proven clean status (current OCWD window is analytical only).
5. Resolve licence/citation.
6. Confirm timezone (adapter currently declares `America/Los_Angeles` as intent).
7. Confirm whether FilmTec TCF matches plant `tcf_water` (currently must not be equated).

## Explicitly out of scope until a later phase

- Trained state-space fouling forecast model / versioned artefact / forecast HTTP API
- Automatic healthy-reference selection / auto-reset after every CIP
- Silent unit invention or EC→TDS conversion
- Claiming absolute cleaning-time validity on OCWD without a justified reference and threshold
