# Aqua Fouling RO

Assessment software for early-warning forecasting of reverse-osmosis membrane
fouling. Validation is **synthetic** (brackish / NaCl-equivalent demonstration
domain). This is **not** a plant-validated operational product. OCWD independent
normalisation blockers remain unresolved.

Default trained model: `llt_relative_health_v1`  
Normalisation: `independent_q_norm_filmtec_tcf_v1` v`2.0.0`

## Quickstart

```bash
poetry install
poetry run aqua serve
# Interactive OpenAPI / Swagger UI: http://127.0.0.1:8000/docs
# Liveness:  GET http://127.0.0.1:8000/health
# Readiness: GET http://127.0.0.1:8000/ready
```

Docker:

```bash
docker compose up --build -d
```

No `.env` file is required for the default demo. Optional knobs are listed in
[`.env.example`](.env.example). Precedence: **process environment > `.env` >
`configs/forecast_serving.yaml` > defaults**.

Compose note: a host `.env` is used for *compose-file substitution* (`${VAR}`).
Values under `environment:` in `docker-compose.yml` become **container process
environment** and override image `ENV` / serving YAML. Host `.env` is not mounted
into the container unless you add `env_file:`.

## CSV-only behaviour and its limits

Only the **CSV file** is required at the request schema for `POST /v1/forecast`.
Series id, mapping and reference are optional. Horizon defaults to **14** days;
threshold defaults to the server-configured relative-health value (typically
**0.85** vs a documented reference). The trained model and threshold are not
retuned for scores.

| CSV-only case | Result |
| --- | --- |
| Headers match documented unit-bearing aliases | `status=reference_required` (HTTP 422) with `resolved_mapping` — **not** a forecast |
| Same + documented `reference` YAML | `status=ok` forecast (`uploaded_series` if no series column) |
| Ambiguous / unit-less recognised names (`permeate_flow`) | `status=mapping_required` (HTTP 422) with assisted YAML template |
| Absent required measurement | `mapping_required` with `reason=absent_column` — editing YAML alone cannot invent a missing column |
| Mixed `series_id` values in a single-series request | `mixed_series` — use `POST /v1/forecast/batch` or split the file |
| Arbitrary unrelated CSV | `mapping_required` / incomplete mapping — not every CSV can forecast |

**Honest limit:** CSV-only never invents a clean reference. A forecast requires a
caller-supplied documented reference. Historical-window “analytical clean” mode
is **explicitly unavailable** in this build. Units are never inferred from
numeric magnitude.

## Completing an assisted mapping

When aliases cannot resolve all required fields unambiguously, the API returns
HTTP **422** with `status=mapping_required` and:

- `detected_headers`
- `confirmed_mappings` (only unambiguous alias hits)
- `unresolved_required_fields` (with `absent_column` vs `unit_unknown` vs `ambiguous`)
- `mapping_template_yaml` — a YAML string using the same adapter schema as
  `configs/canonical_generic_schema.yaml`
- `instructions` for saving and resubmitting

**How to use the template**

1. Copy `mapping_template_yaml` from the JSON response into a file, e.g.
   `my_mapping.yaml` (`python -c "import json,sys; print(json.load(sys.stdin)['mapping_template_yaml'])" > my_mapping.yaml`).
2. Fill `null` `external_name` / `unit.symbol` only where the column **exists** in
   the CSV. Suggestions in the JSON must not be treated as confirmed.
3. Keep `timestamp.column` identical to the `observation_date_calendar`
   field’s `external_name`.
4. Resubmit the same CSV with multipart `mapping=@my_mapping.yaml` (and a
   documented `reference` when ready).

An incomplete template (nulls left in) fails validation and does **not** forecast.
Swagger UI shows the YAML string in the JSON body; it does **not** provide a
dedicated download button for `mapping_template_yaml`.

```bash
# Trigger assisted mapping (ambiguous permeate_flow header)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/postman/ambiguous_units.csv" | python3 -m json.tool
```

## What a reference means (and why it can still be required)

A **documented reference** is the clean-condition baseline for relative health
(temperature, NDP, permeate flow + unit, provenance). It is plant-/series-
specific. The service will **not** silently use the first row, maximum flow, or
post-CIP row as clean.

**Documented upload format is YAML** for both single and batch forecast. JSON
with the same fields is also accepted for compatibility. Validation rules are
identical regardless of encoding.

Use only the reference that matches that series’ plant/conditions. Do **not**
reuse `examples/forecast/reference.yaml` for a different plant.

Documented fields (`examples/forecast/reference.yaml`):

- `mode: documented`
- `temperature_c`, `ndp_psi`, `permeate_flow`, `permeate_flow_unit`, `available_from`, `provenance`

If mapping succeeds but no reference is uploaded → `reference_required` (422).

## Single-series example

```bash
# CSV only → reference_required
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/demo_history.csv" | python3 -m json.tool

# CSV + documented reference → forecast JSON (defaults: horizon 14, threshold ~0.85)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/demo_history.csv" \
  -F "reference=@examples/forecast/reference.yaml" | python3 -m json.tool

# Same forecast as a downloadable CSV attachment (future forecast rows only)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/demo_history.csv" \
  -F "reference=@examples/forecast/reference.yaml" \
  -F "response_format=csv" \
  -o forecast.csv
```

Expected for the committed demo + its matching reference: `status=ok`,
`result.crossing_status=no_crossing_in_horizon`,
`result.days_to_threshold=null` (not `0`), model `llt_relative_health_v1` under
`metadata.model`, 35 normalised observations, 14 forecast dates with distinct
**latent-health** and **future-observation** interval bounds (definitions once
in `metadata.interval_definitions`).

CLI equivalent:

```bash
poetry run aqua forecast \
  --csv examples/forecast/demo_history.csv \
  --series-id DEMO \
  --mapping configs/canonical_generic_schema.yaml \
  --reference examples/forecast/reference.yaml \
  --horizon-days 14 \
  --threshold 0.85 \
  --output /tmp/forecast_demo.json
```

Artifact path is **server-configured** (`AQUA_ARTIFACT_PATH` /
`configs/forecast_serving.yaml`). HTTP callers cannot supply a filesystem path.

Optional `mapping_json` (Swagger-friendly) or `mapping` YAML uploads use the
same adapter contract. Prefer unit-bearing aliases (`permeate_flow_gpd`, …).

## Batch forecasting (`POST /v1/forecast/batch`)

One homogeneous CSV (shared schema/units), optional shared mapping, optional
`series_column` (else exactly one of `source_series_id` / `series_id` /
`unit_id`), and optional **references** upload keyed by series id (YAML
preferred; JSON also accepted).

Each series is forecast **independently** via the same application service.
Histories, CIP segments, replacements, references and filter state are never
combined across series. A missing reference for one series yields that series’
`reference_required` result — another series’ reference is never reused.

**HTTP convention:** request-wide validation failures → **422** (or **400** for
malformed reference/mapping uploads). A valid batch that was processed →
**200** with `overall_status` of `ok`, `partial_success`, or `failed` (all
series failed) and per-series outcomes. Default `max_batch_series=50`;
byte/row limits match single forecast. Input is never silently truncated.
Processing is bounded and sequential (no queue).

```bash
# Successful batch JSON (two interleaved synthetic series + matching YAML refs)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast/batch" \
  -F "csv=@examples/forecast/batch_two_series.csv" \
  -F "references=@examples/forecast/batch_references.yaml" | python3 -m json.tool

# Batch CSV download (filename forecast_batch.csv)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast/batch" \
  -F "csv=@examples/forecast/batch_two_series.csv" \
  -F "references=@examples/forecast/batch_references.yaml" \
  -F "response_format=csv" \
  -o forecast_batch.csv

# Partial success JSON (only DEMO_A has a reference)
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast/batch" \
  -F "csv=@examples/forecast/batch_two_series.csv" \
  -F "references=@examples/forecast/batch_references_partial.yaml" | python3 -m json.tool

# Partial batch as CSV: daily rows for successes + one failure row per failed series
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast/batch" \
  -F "csv=@examples/forecast/batch_two_series.csv" \
  -F "references=@examples/forecast/batch_references_partial.yaml" \
  -F "response_format=csv" \
  -o forecast_batch_partial.csv
```

### Batch references YAML format

```yaml
DEMO_A:
  mode: documented
  provenance: Documented clean reference for the forecast demo (brackish NaCl-equivalent)
  available_from: "2020-01-01"
  temperature_c: 25.0
  ndp_psi: 150.0
  permeate_flow: 6500.0
  permeate_flow_unit: gpd
DEMO_B:
  mode: documented
  provenance: Documented clean reference for the forecast demo (brackish NaCl-equivalent)
  available_from: "2020-01-01"
  temperature_c: 25.0
  ndp_psi: 150.0
  permeate_flow: 6500.0
  permeate_flow_unit: gpd
```

Committed fixtures `batch_two_series.csv` + `batch_references.yaml` use the same
synthetic demo plant for both series IDs — not a template for other plants.
Equivalent JSON (`batch_references.json`) is kept for compatibility tests.

### Interpreting `partial_success`

`overall_status=partial_success` means at least one series returned `ok` (or
unavailable-but-handled forecast payload counted as ok) and at least one did
not. Inspect `summary.counts` and each `results[].status` (`ok`,
`reference_required`, `mapping_required`, or structured `error`). Every result
includes `source_series_id` (string; leading zeros preserved).

With `response_format=csv`, successful series contribute one row per forecast
date; each failed series contributes **one** explicit failure row with
`status` / `error_code` / `error_message` populated and forecast number/date
columns empty. Failed series are never silently omitted. An all-failed but
request-valid batch still returns HTTP **200** with only failure rows (JSON
or CSV). Request-wide validation failures (malformed mapping, bad
`response_format`, blank series ids, …) remain structured **JSON** errors even
when CSV was requested.

## Forecast CSV download

Optional multipart field `response_format=json|csv` (default **json**) on
`POST /v1/forecast` and `POST /v1/forecast/batch`.

CSV responses use `Content-Type: text/csv; charset=utf-8` and
`Content-Disposition: attachment; filename="forecast.csv"` (batch:
`forecast_batch.csv`). Rows are generated from the **same** computed result as
JSON — not a second forecasting path. Normalized historical observations remain
in the JSON `normalised_data` block and are **not** included in the CSV export
(CSV is future-forecast rows only).

Stable columns (one row per forecast date for successes):

`source_series_id`, `status`, `forecast_origin`, `forecast_date`, `days_ahead`,
`relative_health`, `latent_health_low`, `latent_health_high`,
`future_observation_low`, `future_observation_high`, `threshold`,
`crossing_status`, `days_to_threshold`, `model_version`,
`normalisation_version`, `error_code`, `error_message`.

`days_to_threshold` is measured from `forecast_origin` and repeated as
forecast-level context on each daily row — it is **not** recalculated relative
to each `forecast_date`. Unavailable values are empty cells (never the string
`null`); `crossed_at_origin` still uses `0`. Text identifiers that start with
spreadsheet formula markers (`=`, `+`, `-`, `@`, …) are prefixed with `'` for
safe opening in Excel/Sheets; legitimate negative **numeric** predictions are
unchanged.

## Response contract (pre-release)

Successful forecast responses use a fixed top-level order:

1. `status`
2. `result` — series id, origin, horizon, threshold (+ reference basis),
   `crossing_status`, `days_to_threshold` (**null** when no crossing / timing
   unavailable; **0** only when already crossed at origin), daily forecast
   points with latent and future-observation interval **bounds**
3. `normalised_data` — observation count, dated relative-health history, row
   quality / history-inclusion flags
4. `metadata` — model/normalisation versions, reference, mapping, history
   segment summary, interval definitions (once), assumptions / scientific scope
5. `warnings`

Batch responses use `overall_status`, `summary.counts`, and per-series
`results` (same success grouping or `error` on failure). Errors use
`status` + `error.{code,message,actionable_correction,details}` (plus
`mapping_template_yaml` for `mapping_required`). Public responses do not
include server filesystem paths.

**Field moves (this cleanup):** `forecast` → `result.forecast`;
`normalised_observations` → `normalised_data.observations`;
`n_normalised_observations` → `normalised_data.n_observations`;
`model` / normalisation / mapping → `metadata.*`; batch `counts` →
`summary.counts`; `reference_required` / `mapping_required` codes live under
`error`.

## CSV schema, units, mapping and reference

Documented aliases / default mapping: [`configs/canonical_generic_schema.yaml`](configs/canonical_generic_schema.yaml)

| Column | Canonical | Unit |
| --- | --- | --- |
| `observation_date` (or `timestamp.column`) | `observation_date_calendar` | calendar date |
| `temperature_c` | `temperature_c` | deg_C |
| `feed_pressure_psi` / `_bar` | `stage_3_feed_pressure_psi` | psi or bar |
| `permeate_pressure_psi` | `permeate_pressure_psi` | psi |
| `device_dp_psi` | `stage_3_device_pressure_drop_psi` | psi |
| `permeate_flow_gpd` / `_gpm` | `stage_3_permeate_flow_source_unit` | gpd or gpm |
| `feed_flow_gpd` / `_gpm` | `stage_3_feed_flow_source_unit` | gpd or gpm |
| `feed_concentration_mg_l` | `feed_concentration_mg_l` | mg/L NaCl-eq |
| `cip` | `cip_flag_0_1` | flag 0/1 |
| `days_since_replacement` (optional) | `days_since_replacement_d` | day |

Series id (single request): inferred from `source_series_id` / `series_id` /
`unit_id`, else `uploaded_series`. Mixed values are rejected on
`POST /v1/forecast`.

## Output interpretation and errors

| `result.crossing_status` | `result.days_to_threshold` | Meaning |
| --- | --- | --- |
| `crossed_at_origin` | `0` | Already at/below threshold at origin |
| `predicted_crossing` | positive int | Crossing forecast within horizon |
| `no_crossing_in_horizon` | `null` | No crossing predicted in-horizon (not a safety guarantee) |
| `unavailable` | `null` | Insufficient/unusable history |

Structured HTTP errors include `status` and `error.{code,message,actionable_correction,details}`
(for example `validation_failed`, `malformed_yaml`, `malformed_json`,
`forecast_unavailable`, `payload_too_large`, `batch_series_limit`,
`blank_series_id`).

`days_to_threshold` is calendar days from the forecast origin until relative
health reaches the configured **analytical** threshold versus the documented
reference, **conditional on no future cleaning**. It is not a manufacturer or
plant-validated cleaning date.

## Local installation and Docker startup

**Poetry (installable package):**

```bash
poetry install
poetry run aqua serve
# or: poetry run uvicorn aqua_fouling.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

**Docker Compose:**

```bash
docker compose config   # validate compose file
docker compose up --build -d
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
```

The image copies `src/`, `configs/`, `examples/forecast/`, and
`artifacts/models/llt_relative_health_v1.json`. Raw dataset directories are not
included. TLS verification is not disabled.

## Postman

Import:

- [`postman/AquaFoulingRO.postman_collection.json`](postman/AquaFoulingRO.postman_collection.json)
- [`postman/local.postman_environment.json`](postman/local.postman_environment.json) (`base_url=http://localhost:8000`)

**File selection:** each request that uploads CSV/YAML/JSON uses Postman
form-data *File* fields. Choose files from this repository after import (for
example `examples/forecast/demo_history.csv`). Do **not** hard-code
machine-specific absolute paths in the collection.

Examples include CSV-only `reference_required`, `mapping_required`, completed
mapping, successful batch, and `partial_success` batch, with response assertions.

## Training / evaluation and model replacement

```bash
poetry run aqua train --config configs/training.yaml
poetry run aqua evaluate --config configs/evaluation.yaml

export AQUA_ARTIFACT_PATH=/path/to/llt_relative_health_v1.json
poetry run aqua serve   # restart required after artifact changes
```

Docker: rebuild after replacing `artifacts/models/…`, or mount a file and set
`AQUA_ARTIFACT_PATH` in Compose `environment:`.

## Validation results and known scientific limitations

- Synthetic LLT evaluation summaries live under `reports/phase4/` (inspected
  benchmark — not a pristine unseen holdout). Chronological phase folder names
  are retained as evidence locations.
- Observation-interval coverage is reported as **average per-origin** coverage;
  overlapping windows are not independent observations; latent vs future-
  observation intervals are distinct.
- Synthetic-trained performance does **not** establish OCWD or plant transfer.
- Segment prior uses mean health=1.0 (clean-reference scale); mid-fouling starts
  are documented, not silently retuned.
- No invented clean references; historical-window reference mode unavailable.

## Docker verification status

The current `Dockerfile` installs pinned Poetry via `pip` (no `apt-get` /
`build-essential` / `curl`), keeps TLS verification, uses `poetry.lock`, installs
dependencies once (`--no-root`) then the project (`--only-root`), and copies
`src/`, `configs/`, `examples/forecast/`, and `artifacts/models/`.

**Current clean-build status:** verified (including response-contract rebuild).

```bash
docker build --no-cache -t aquafoulingro-api:response-contract .
docker run --rm -p 8000:8000 aquafoulingro-api:response-contract
```

Against that image (not an overlay): `/health` → `status=ok`; `/ready` →
`ready=true`, `model_version=llt_relative_health_v1` (no filesystem path);
documented single-series forecast → grouped `result` /
`normalised_data` / `metadata`, 35 observations, 14 steps,
`crossing_status=no_crossing_in_horizon`, `days_to_threshold=null`; batch YAML
refs → `DEMO_A` / `DEMO_B`, `overall_status=ok`; malformed mapping → HTTP 400
`status=error`; ambiguous units → HTTP 422 `mapping_required`.

Earlier network/PyPI stalls during prior delivery attempts are recorded as
history in [`docs/ai_use_log.md`](docs/ai_use_log.md). They are **not** current
build blockers. Overlay images are never counted as a clean build.

## Quality checks / test commands

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy
poetry run pytest -q
```

## Other CLI entry points

```bash
poetry run aqua validate --csv ... --series-id ...
poetry run aqua normalise --csv ... --series-id ... --reference ...
poetry run aqua benchmark-synthetic --config configs/synthetic_benchmark.yaml
poetry run aqua benchmark-ocwd --config configs/ocwd_forecast_benchmark.yaml
```
