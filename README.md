# Aqua Fouling RO

Early-warning software for reverse-osmosis membrane fouling assessment.

Upload plant operating CSV → normalised relative health → fouling forecast →
estimated days until a documented cleaning threshold.

| | |
| --- | --- |
| **Model** | `llt_relative_health_v1` (local linear trend / Kalman) |
| **Normalisation** | `independent_q_norm_filmtec_tcf_v1` v`2.0.0` |
| **Domain** | Synthetic brackish / NaCl-equivalent demonstration |
| **Not claimed** | Plant-validated operational product; OCWD transfer |

Repository: [github.com/UmmeeAwan/AquaFoulingRO](https://github.com/UmmeeAwan/AquaFoulingRO)

---

## 1. Clone and start (Docker — recommended)

```bash
git clone https://github.com/UmmeeAwan/AquaFoulingRO.git
cd AquaFoulingRO

docker compose up --build -d

curl -sS http://127.0.0.1:8000/health   # liveness
curl -sS http://127.0.0.1:8000/ready    # model artefact ready
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Stop:

```bash
docker compose down
```

No `.env` is required for the demo. Optional knobs: [`.env.example`](.env.example).
Precedence: **process environment > `.env` > `configs/forecast_serving.yaml` > defaults**.

---

## 2. Local install (Poetry)

Requires Python **3.12+**.

```bash
git clone https://github.com/UmmeeAwan/AquaFoulingRO.git
cd AquaFoulingRO

poetry install
poetry run aqua serve
# OpenAPI UI: http://127.0.0.1:8000/docs
```

---

## 3. Run a forecast

### Successful demo (JSON)

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/demo_history.csv" \
  -F "reference=@examples/forecast/reference.yaml" | python3 -m json.tool
```

Expected: `status=ok`, `crossing_status=no_crossing_in_horizon`,
`days_to_threshold=null`, 35 normalised observations, 14 forecast steps.

### Download forecast CSV

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/demo_history.csv" \
  -F "reference=@examples/forecast/reference.yaml" \
  -F "response_format=csv" \
  -o forecast.csv
```

### CLI

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

### Batch (multiple series in one CSV)

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast/batch" \
  -F "csv=@examples/forecast/batch_two_series.csv" \
  -F "references=@examples/forecast/batch_references.yaml" | python3 -m json.tool
```

---

## 4. What you must supply

| Input | Required? | Notes |
| --- | --- | --- |
| `csv` | **Yes** | Daily operating history |
| `reference` | For a forecast | Documented clean baseline (YAML preferred; JSON accepted) |
| `mapping` | Sometimes | Only when headers are ambiguous or non-standard |
| `horizon_days` | No | Default **14** |
| `threshold` | No | Default **0.85** relative health |
| `response_format` | No | `json` (default) or `csv` |

**CSV alone never invents a clean reference.** If mapping succeeds but no
reference is uploaded → HTTP 422 `reference_required`.

Documented reference fields (`examples/forecast/reference.yaml`):

- `mode: documented`
- `temperature_c`, `ndp_psi`, `permeate_flow`, `permeate_flow_unit`
- `available_from`, `provenance`

Use a reference that matches **that** series’ plant. Do not reuse the demo
reference for a different plant.

---

## 5. CSV columns (demo schema)

Default mapping: [`configs/canonical_generic_schema.yaml`](configs/canonical_generic_schema.yaml)

| Column | Unit | Required |
| --- | --- | --- |
| `observation_date` | calendar date | yes |
| `temperature_c` | °C | yes |
| `feed_pressure_psi` | psi | yes |
| `permeate_pressure_psi` | psi | yes |
| `device_dp_psi` | psi | yes |
| `permeate_flow_gpd` (or `_gpm`) | gpd / gpm | yes |
| `feed_flow_gpd` (or `_gpm`) | gpd / gpm | yes |
| `feed_concentration_mg_l` | mg/L NaCl-eq | yes |
| `cip` | 0/1 flag | optional |
| `days_since_replacement` | day | optional |
| `series_id` / `source_series_id` / `unit_id` | — | for batch / multi-series |

Prefer **unit-bearing** headers (`permeate_flow_gpd`). Units are never inferred
from numeric magnitude.

### Assisted mapping

If a header is recognised but unit-less (e.g. `permeate_flow`), the API returns
HTTP 422 `mapping_required` with `mapping_template_yaml`. Confirmed fields
(including optional CIP / replacement when present) are preserved; unresolved
required names/units stay `null`.

```bash
# Trigger assisted mapping
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/postman/ambiguous_units.csv" | python3 -m json.tool

# Complete with committed mapping + reference
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -F "csv=@examples/forecast/postman/ambiguous_units.csv" \
  -F "mapping=@examples/forecast/postman/ambiguous_units_mapping.yaml" \
  -F "reference=@examples/forecast/reference.yaml" | python3 -m json.tool
```

---

## 6. Response shape

Successful JSON (top-level order):

1. **`status`** — `ok`
2. **`result`** — origin, horizon, threshold, `crossing_status`,
   `days_to_threshold`, daily forecast points + intervals
3. **`normalised_data`** — relative-health history used for forecasting
4. **`metadata`** — model / normalisation versions, reference, mapping, scope
5. **`warnings`**

| `crossing_status` | `days_to_threshold` | Meaning |
| --- | --- | --- |
| `crossed_at_origin` | `0` | Already at/below threshold |
| `predicted_crossing` | positive int | Crossing within horizon |
| `no_crossing_in_horizon` | `null` | No in-horizon crossing (not a safety guarantee) |
| `unavailable` | `null` | Insufficient usable history |

`days_to_threshold` is calendar days from forecast origin until relative health
reaches the analytical threshold, **conditional on no future cleaning**. It is
not a manufacturer or plant-validated cleaning date.

CSV download (`response_format=csv`) exports **future forecast rows only**
(`forecast.csv` / `forecast_batch.csv`). Errors and request-wide failures remain
structured JSON even when CSV was requested.

---

## 7. Postman

Import:

- [`postman/AquaFoulingRO.postman_collection.json`](postman/AquaFoulingRO.postman_collection.json)
- [`postman/local.postman_environment.json`](postman/local.postman_environment.json)

Select files from this repo for each upload field (no machine-specific paths).

---

## 8. Quality checks

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy
poetry run pytest -q
```

---

## 9. Retrain / swap the model

```bash
poetry run aqua train --config configs/training.yaml
poetry run aqua evaluate --config configs/evaluation.yaml

export AQUA_ARTIFACT_PATH=/path/to/llt_relative_health_v1.json
poetry run aqua serve   # restart after changing the artefact
```

Docker: rebuild after replacing `artifacts/models/…`, or mount a file and set
`AQUA_ARTIFACT_PATH` in Compose `environment:`.

Other CLI: `aqua validate`, `aqua normalise`, `aqua benchmark-synthetic`,
`aqua benchmark-ocwd`.

---

## 10. Known limitations

- Validation is **synthetic** (brackish / NaCl-equivalent). Not seawater-validated.
- OCWD public archive was audited; independent physical normalisation remains
  blocked (concentration / stage hydraulics). No OCWD transfer claim.
- Default threshold **0.85** is an analytical demo choice (≈15% decline from a
  documented clean reference), not a FilmTec or plant cleaning rule.
- The service will not invent a clean reference from the first row, max flow, or
  post-CIP row.
- Gaussian LLT can miss abrupt fouling acceleration; missed warnings appear near
  some CIP / threshold approaches in the inspected synthetic benchmark.
- Observation-interval coverage is theoretical / average per-origin — not claimed
  as calibrated confidence.

Further reading: [`docs/data_sources.md`](docs/data_sources.md),
[`docs/ai_use_log.md`](docs/ai_use_log.md),
[`reports/technical_report.pdf`](reports/technical_report.pdf).
