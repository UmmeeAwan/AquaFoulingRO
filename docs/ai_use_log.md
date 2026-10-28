# AI use log

This log records actual assistance used while implementing Phases 1–3 and
subsequent fix passes. It does not invent verification results, prompts, or
source licences.

## Phase 1 (summary)

Scaffolded Poetry project, audit CLI, health/capabilities API, synthetic tests, Docker/CI. Python **3.12** used because 3.13 was not installed. Click pinned `<8.2` after Typer broke on Click 8.5 (`make_metavar` TypeError). First Docker build failed on SSL fetching NumPy; retry succeeded.

## Phase 2 assistance

An AI coding agent (Cursor / Composer) implemented Phase 2 on top of the Phase 1 foundation:

- Corrected audit outputs (absolute + relative s3sf errors with dates; Rd `rof_toc_avg` mg/L; exported-index column-count explanation; filename/stage marked **unresolved**; schema-missing vs observation-missing).
- Evidence-based data dictionary (`docs/data_dictionary.md` + `.json`).
- Dual YAML adapters (`ocwd_schema_ae.yaml`, `ocwd_schema_fg.yaml`).
- Physics module using **FilmTec Form 609-02084-0905** (URL fetched successfully during Phase 2).
- Reference handling, segment/CIP audit trail, validate/normalise API+CLI, readiness report.
- Independent scientific tests including the FilmTec worked example (gpm figures used after verifying that the PDF’s 127 m³/h label does not reproduce the published 636 gpm arithmetic with the stated NDP/TCF ratios).

## Phase 2 commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff check src tests` | Passed |
| `poetry run mypy` | Passed (31 files) |
| `poetry run pytest -q` | **41 passed** |
| `poetry run aqua audit-data ...` | Succeeded (21 series, 71 CIP rows) |
| `poetry run aqua readiness-report` | Wrote `reports/phase2/normalisation_readiness.md` |
| `poetry run aqua validate ... A01` | Wrote `reports/phase2/validate_A01.json` |
| `poetry run aqua normalise ... A01` | Wrote `reports/phase2/normalise_A01.json` |
| `docker compose up --build` | One SSL failure installing pandas; retry built successfully |

## Phase 3 assistance

Same agent continued Phase 3 (synthetic primary benchmark + separate OCWD `s3sf` baselines; stopped before trained state-space).

## Phase 3 corrective pass

- Explicit mapping before optional OCWD auto-detect; canonical synthetic/generic schemas.
- Forecast history segment-restricted; calendar-time windows; no silent outside-window fallback.
- Trajectory vs threshold-event eligibility; incomplete follow-up censored (not no-crossing).
- Train/dev follow-up cutoffs; GFD→gpd conversion; coherent clean reference.
- Regenerated `reports/phase3/*`. Pytest: **63 passed**.

## Pre-training fix pass (this session)

### Prompts / review assistance (actual)

User prompt (summarised): narrowly scoped pre-training fix — (1) enforce units in adapters/normalisation with gpd/gpm and bar/psi tests plus finite reference validation; (2) separate crossing occurrence from exact timing when gaps precede first crossing; (3) package `configs/` in Docker and smoke-test generic CSV normalisation inside the container; (4) common-origin trajectory MAE when models differ in availability, with partial vs complete horizon distinction; (5) preserve tests, regenerate reports, update this AI-use log without inventing source research.

ChatGPT / review assistance was used to critique prior Phase 3 artefacts (unit silence, crossing-time gaps, Docker missing configs, unfair MAE denominators). Corrections implemented in Cursor against those findings; no new scientific literature was invented for this pass.

### Reproduced errors and verified corrections

| Reproduced issue | Correction verified |
| --- | --- |
| Adapter units were labels only; gpm/gpd and bar/psi not converted | `physics/units.py` convert helpers; `adapt_dataframe` converts to gpd/psi or errors; reference/measurement flow units matched in `independent_norm` |
| Documented reference allowed `unresolved_flow` / non-finite values | `ReferenceConfig` rejects unresolved flow units and non-finite / out-of-range T, NDP, flow |
| Exact timing scored when gaps preceded first observed crossing | `_observed_crossing_days` → `observed_crossing_uncertain_timing` when prefix incomplete; gaps after a fully observed first crossing do not invalidate timing |
| Docker image omitted `configs/` (`.dockerignore` excluded it); cwd-relative paths failed in `/app` | Removed `configs` from `.dockerignore`; `COPY configs`; `resolve_configs_dir()` walks to `/app/configs` |
| Trajectory MAE compared models with different scored-origin sets without disclosure | `common_origin_mae_mean` + `n_pred_unavailable`; partial vs complete horizon counts in summary |
| First `docker compose build` after Dockerfile change: TLS `bad record MAC` | Retried build succeeded (`aquafoulingro-api Built`) |

### Commands actually run (this pass)

| Command | Outcome |
| --- | --- |
| `poetry run ruff check src tests` | Passed |
| `poetry run mypy` | Passed (41 files) |
| `poetry run pytest -q` | **70 passed** |
| `docker compose build` | Succeeded after TLS retry; configs present in image |
| `docker compose up -d` | `/health` → `{"status":"ok","phase":3}` |
| `curl POST /v1/normalise` with `tests/fixtures/generic/canonical_demo.csv` + `canonical_generic_schema.yaml` + reference | `status=ok`, independent normalisation `available=True`, `n_norm=5` |
| Regenerated synthetic + OCWD baselines + `benchmark_summary.md` | Wrote `reports/phase3/*` |

## Scientific source access

- FilmTec record-keeping PDF fetched successfully from  
  `https://www.bigbrandwater.com/assets/library/filmtec/filmtec-record-keeping.pdf`  
  during Phase 2. No additional source documents were fetched for the pre-training fix pass or Phase 4.

## Phase 4 — trained LLT + versioned artifact (this session)

### Assistance

User prompt (summarised): implement Phase 4 local-linear-trend Kalman candidate with reproducible training, versioned JSON artifact, Forecaster interface, uncertainty intervals, evaluation vs persistence/linear trend, CLI `train`/`evaluate`, tests, README + AI-use log; preserve normalisation/baselines/separate OCWD; stop before forecast HTTP.

Implemented in Cursor: Harvey-style LLT Kalman (NumPy + SciPy L-BFGS-B on log-variances), schema-validated artifact, train/evaluate services, synthetic-only training (S01–S03), development grid selection, inspected-benchmark test reporting.

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry add 'scipy>=1.14'` | Installed |
| `poetry run aqua train --config configs/training.yaml` | Artifact `artifacts/models/llt_relative_health_v1.json`; success=True; nit=34; final_nll=−1342.86; n_obs=527 |
| `poetry run aqua evaluate --config configs/evaluation.yaml` | Wrote `reports/phase4/evaluation_*.{json,md}`; artifact unchanged during inference |
| `poetry run ruff check src tests` | Passed |
| `poetry run mypy` | Passed (46 files) |
| `poetry run pytest -q` | **83 passed** |

### Honest evaluation notes (synthetic inspected test)

- Trajectory common-origin MAE: LLT better than both baselines at h=7/14/30.
- Exact crossing-time MAE at h=30: recent linear trend slightly better than LLT (6.21 vs 6.59 days); LLT is not uniformly best.
- Observation-interval empirical coverage ≈0.96 (nominal 0.95); theoretical intervals — not claimed as calibrated confidence.
- Failure modes include missed warnings near S07 CIP / S08 threshold approach.
- No OCWD pooling or transfer claim.

## Phase 5 — CSV forecast API/CLI (this session)

### Assistance

User prompt (summarised): Phase 5 shared forecast service for FastAPI + CLI using the existing trained artifact; no retrain/tune/benchmark changes; server-configured artifact; structured outputs/errors; Docker packaging; readiness vs liveness; meaningful tests; in-container forecast verification; README + AI-use log.

Implemented in Cursor: `forecast_service.py`, `POST /v1/forecast`, `GET /ready`, `aqua forecast`, settings/serving YAML, Docker COPY of artifact + examples, Phase 5 tests.

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run aqua forecast --csv examples/forecast/demo_history.csv ...` | `crossing_status=no_crossing_in_horizon`; model `llt_relative_health_v1` |
| `poetry run ruff check src tests` / `mypy` | Passed (48 source files) |
| `poetry run pytest -q` | **93 passed** |
| Full `docker compose build` | Failed twice on PyPI TLS/read timeout installing numpy/pytz (retried; network) |
| Overlay image `aquafoulingro-api:phase5` (COPY src/configs/artifact/examples onto prior image) + `curl POST /v1/forecast` | HTTP 200; model `llt_relative_health_v1`; norm `independent_q_norm_filmtec_tcf_v1` v2.0.0; 35 obs; 14 forecast steps; `crossing_status=no_crossing_in_horizon`; `/ready` true |
| In-container `forecast_csv(...)` | Same structured fields confirmed |

### Notes

- Artifact path is server/env only (`AQUA_ARTIFACT_PATH`); not accepted from HTTP callers.
- `/health` = liveness; `/ready` = artifact compatibility.
- No retraining or benchmark retuning in Phase 5.

## Phase 5 final integration pass (this session)

### Assistance

User prompt (summarised): preserve Phase 5; fix training `converged_flag` to optimiser success only and exclude failed candidates; content-hash provenance (no fabricated hashes); clarify average per-origin observation-interval coverage vs pooled/latent; verify serving YAML/env; mid-fouling documents health=1 prior without retune; clean Docker with TLS preserved; distinguish clean-build vs overlay; update README + this log. Assessment software with synthetic validation only.

### Artifact / provenance

- Regenerated `artifacts/models/llt_relative_health_v1.json` with same seed `20261027` and same settings/grid.
- **Parameters unchanged** (bit-identical to pre-regen freeze).
- `training_data_hash` changed from summary-stat style
  `e0269937…` → content-digest aggregate
  `b165a1b0…` (CSVs S01–S03 + references + mapping).
- Added `benchmark_config_hash`, per-file `content_hashes`, and `hash_policy` in provenance.
- `training_config_hash` unchanged (`8c34000f…`).

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run aqua train --config configs/training.yaml` | success=True; nit=34; params identical; new content hashes |
| `poetry run aqua evaluate --config configs/evaluation.yaml` | Interval fields: avg per-origin cov, pooled, pair counts, latent vs obs widths |
| `poetry run pytest -q` | **100 passed** |
| README `aqua forecast` demo | `crossing_status=no_crossing_in_horizon`; model `llt_relative_health_v1`; 35 normalised obs (2020-01-01…2020-02-04); 14 forecast dates; distinct latent + future-observation intervals |
| Insufficient history (10-day CSV) | `forecast_unavailable` with min_history_days=21 actionable correction |
| Invalid unit (`atm` on pressure) | `validation_failed` unsupported unit; supported psi/bar/Pa/kPa |
| Local `aqua serve` + `/health` `/ready` + HTTP forecast | Liveness ok; readiness true; API matches CLI fields |
| `docker build --no-cache -t aquafoulingro-api:cleanbuild .` | **Blocked** — hung on `apt-get update` fetching `deb.debian.org` bookworm Packages (~8.8 MB) with TLS/HTTP verification intact (no `--trusted-host` / insecure flags). Host curl diagnosis: debian Packages.gz timed out at 30s after ~400 KB / ~12 MB; `pypi.org/simple/numpy` timed out at 30s after ~470 KB. Prior Phase 5 build logs: `HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out` / SSLError during `poetry install`. Overlay smoke is **not** counted as clean-build verification. |
| Local uvicorn `/health` `/ready` + README HTTP forecast | `/health` ok phase=5; `/ready` true artifact `llt_relative_health_v1`; HTTP 200 forecast matches CLI (35 obs, 14 steps, `no_crossing_in_horizon`) — **local smoke**, not a clean build |

### Clean-build vs overlay (historical — Phase 5 attempt)

At that time, clean-build verification of the then-current Dockerfile was
**blocked** by network/TLS toward Debian mirrors and/or PyPI. Overlay images
used for Phase 5 API smoke were documented only as overlay smoke tests — not as
a successful clean build. **Current status:** see the usability delivery
clean-build section below (succeeded).

## Assessment delivery preparation (this session)

### Assistance

User prompt (summarised): prepare for assessment delivery — fix forecast replacement-counter segments, settings precedence (env > .env > YAML > defaults), malformed YAML/JSON as client errors, temp upload cleanup, adapter timestamp resolution, bounded uploads; simplify Docker (pip Poetry, no apt compilers if wheels suffice); clean Phase wording; `.env.example`; Postman; reviewer README; preserve trained model.

### Code removals / wording

- Removed public `phase` fields from `/health` and `/v1/capabilities` (replaced with `version` / model+normalisation versions). Updated tests accordingly.
- Rewrote stale Phase 1 package/docstrings (`__init__.py`, `training/__init__.py`, `pyproject.toml` description, API description, CLI help). Chronological Phase history retained only in this AI-use log.
- No large dead-code deletions beyond those wording/fields; `AdapterConfig.timestamp_column` is now used for forecast sorting (was previously loaded but unused for that path).

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff check src tests` | Passed |
| `poetry run mypy` | Passed (48 files) |
| `poetry run pytest -q` | **109 passed** |
| `docker build --no-cache -t aquafoulingro-api:assessment .` | **Blocked** after successful `pip install poetry==2.1.3` (TLS OK). Hung/stalled during `poetry install --only main --no-root` while installing large wheels (`scipy`/`pandas`/`matplotlib` et al. from the lockfile). No overlay used. TLS verification preserved. |

### Docker simplification

- Dockerfile no longer runs `apt-get` / `curl` / `build-essential`.
- Pinned `poetry==2.1.3` installed with `pip` (TLS verified, retries).
- `poetry install --only main --no-root` then `poetry install --only-root`.
- At that assessment-prep attempt, fresh clean-build verification was blocked at
  the dependency-install step by slow/stalling PyPI wheel downloads — not by
  missing compilers. **Later usability delivery clean-build succeeded** (see below).

## What was not invented

- No silent conductivity→TDS coefficient.
- No claim that OCWD independent normalisation blockers are resolved.
- No pooling of OCWD and synthetic metrics; no synthetic→plant transfer claim.
- No claim that synthetic benchmark performance establishes plant accuracy.
- No hand-picked parameters described as trained; learned variances come from innovation MLE.
- No invented literature for unit constants beyond standard engineering conversions already used in-repo (psi/bar; gpd/gpm = 1440).
- No claim that an overlay Docker image constitutes a successful clean build.

## Usability delivery — assisted mapping + batch forecast (this session)

### Assistance

User prompt (summarised): assessment-ready usability — (1) assisted `mapping_required` + YAML template when aliases are incomplete/ambiguous; (2) `POST /v1/forecast/batch` with per-series isolation and references JSON; (3) meaningful tests; (4) README/Postman/AI-log; (5) clean Docker build + runtime verification; no retrain; no commit/push; no UI/auth/DB/queue.

### Implemented

- `try_resolve_alias_mapping` / `mapping_required_payload` / assisted YAML template in `csv_first.py`.
- Single-series `POST /v1/forecast` returns HTTP 422 `mapping_required` or `reference_required` without inventing clean baselines.
- `forecast_batch_csv` + `POST /v1/forecast/batch`: shared mapping, per-series references, sequential processing, `max_batch_series` (default 50).
- Fixtures: `examples/forecast/batch_two_series.csv`, `batch_references.json`, `batch_references_partial.json`, `postman/ambiguous_units.csv`.
- README sections: quickstart, CSV-only limits, assisted mapping, reference meaning, single/batch examples, `partial_success`, Poetry/Docker, tests, scientific limits. Postman examples with assertions.
- Phase-named evidence under `reports/` retained; obsolete phase wording kept off public API surfaces.

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff check src tests` | Passed |
| `poetry run mypy` | Passed (49 source files) |
| `poetry run pytest -q` | **134 passed** |
| `poetry run aqua --help` + `create_app()` import | Installable package path OK in existing Poetry env |
| `docker compose config` | Validated (service `api`, port 8000, artifact path set) |
| `docker build --no-cache -t aquafoulingro-api:usability .` | **Succeeded** (~141s; TLS verified; Poetry via pip; lockfile wheels) |
| `docker run … aquafoulingro-api:usability` | `/health` ok; `/ready` true `llt_relative_health_v1`; single forecast 35 obs / 14 steps / `no_crossing_in_horizon`; batch `DEMO_A`+`DEMO_B` overall ok; malformed mapping HTTP 400 `malformed_mapping`; ambiguous CSV HTTP 422 `mapping_required` |

### Clean-build vs overlay

This usability delivery’s clean `--no-cache` build and runtime curls against that image **succeeded**. Prior assessment deliveries that hung on PyPI remain historical network failures; they are not re-used as overlay evidence here.

## Forecast CSV download (response_format)

### Assistance

User prompt (summarised): optional `response_format=json|csv` on single and batch
forecast; CSV attachment from the same computed results; partial-batch failure
rows; formula-injection-safe text; keep JSON default and historical
`normalised_data`; tests/docs/Postman; Docker verify; no commit/push.

### Implemented

- `application/forecast_csv_export.py` + route wiring.
- Documented columns; empty cells for unavailable; `days_to_threshold` context
  from origin; batch CSV filename `forecast_batch.csv`.

### Commands actually run

| Command | Outcome |
| --- | --- |
| quality gates + pytest | See final report |
| Docker rebuild + CSV/JSON curls | See final report |

## API response presentation cleanup (pre-release contract)

### Assistance

User prompt (summarised): reshape API responses for clarity — forecast results
near the top; group related fields; remove redundant prose/paths/unused fields;
consistent errors; update schemas/CLI/README/Postman/tests; regression-lock
forecast numbers; Docker rebuild; no model/science changes; no commit/push.

### Implemented

- Success forecast envelope: `status` → `result` → `normalised_data` →
  `metadata` → `warnings`.
- Interval definitions once in metadata; per-point bounds only.
- Batch: `overall_status` + `summary.counts` + per-series results.
- Errors: `status` + `error` envelope; mapping/reference required retain
  templates / required fields / resolved mappings.
- `/ready` omits filesystem paths; capabilities condensed.
- Validate/normalise: outcome first; mapping/reference under `metadata`.

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff format/check` + `mypy` + `pytest -q` | See final report |
| Docker rebuild + runtime curls | See final report |

## Reference upload standardisation (YAML preferred)

### Assistance

User prompt (summarised): document YAML as the reference format for single and
batch forecast; accept YAML and JSON on both; same validation rules; actionable
malformed/structure errors; add `batch_references.yaml` + partial YAML; update
README/Postman; tests for YAML/JSON equivalence; no model/science changes.

### Implemented

- `application/reference_upload.py` — safe YAML/JSON parse for one reference or
  series-keyed batch map; structure checks with `malformed_reference` /
  `malformed_json` / `malformed_yaml`.
- Both forecast endpoints parse uploads then normalise single refs to YAML for
  the existing `ReferenceConfig` path; per-series isolation unchanged.
- Fixtures: `batch_references.yaml`, `batch_references_partial.yaml`,
  `reference.json` (JSON twin for equivalence tests).

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff format src tests` | Passed |
| `poetry run ruff check src tests` | Passed |
| `poetry run ruff format --check src tests` | Passed |
| `poetry run mypy` | Passed (50 source files) |
| `poetry run pytest -q` | **144 passed** |

## Assessment delivery hygiene (this session)

### Assistance

User prompt (summarised): prepare for assessment delivery without new features —
format `src`/`tests`, re-run quality gates, update README Docker status to the
successful `aquafoulingro-api:usability` clean build, keep historical Docker
failures only as chronology in this log, verify README examples, ensure batch
references JSON has no placeholders, tighten `.gitignore` while retaining
`.env.example`, `poetry.lock`, serving artifact, configs, examples, Postman,
and phase-named evaluation evidence; no commit/push.

### Commands actually run

| Command | Outcome |
| --- | --- |
| `poetry run ruff format src tests` | 10 files reformatted |
| `poetry run ruff check src tests` | Passed |
| `poetry run ruff format --check src tests` | 67 files already formatted |
| `poetry run mypy` | Passed (49 source files) |
| `poetry run pytest -q` | **134 passed** |

### Hygiene notes

- README Docker section now states **current** clean-build verified; historical
  PyPI/Debian stalls remain documented above as past attempts only.
- `.gitignore` excludes secrets/`.env`/venvs/caches/raw archives/temps while
  keeping `reports/phase3/**` and `reports/phase4/**` evaluation evidence and
  `artifacts/models/llt_relative_health_v1.json`.
- No phase-named tests or scientific evidence deleted.

### What was not invented

- No clean-reference invention; no unit inference from magnitude.
- No silent confirmation of suggestion headers as mappings.
- No cross-series reference reuse; no background queue.
- No claim that overlay Docker constitutes a clean build.
