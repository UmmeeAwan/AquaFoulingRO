# Phase 3 benchmark summary

This is **not** the final trained-model submission. Baselines only; a fitted state-space candidate will follow after baseline inspection.

## Experiment separation

| Experiment | Target | Metrics file |
| --- | --- | --- |
| Primary synthetic | Independent relative health from raw | `synthetic_metrics.json` |
| Separate OCWD | Source-calculated `s3sf` (no extra TCF) | `ocwd_metrics.json` |

**Do not pool OCWD and synthetic metrics.**

Synthetic benchmark performance does **not** establish plant accuracy.

OCWD independent-normalisation blockers from Phase 2 are **not** resolved.

## Quantitative readiness (OCWD)

- Series: 21
- Total segments: 94
- Eligible origins by horizon: 7d=12465, 14d=12469, 30d=12490
- CIP boundaries (sum): 71
- Replacement reset candidates (sum): 2

## Synthetic baseline results (test split)

- Threshold (relative health): 0.85
- Horizons: [7, 14, 30]
- Disclaimer: Synthetic benchmark performance does not establish plant accuracy.

Trajectory-window MAE/RMSE = mean error over calendar offsets in the horizon window (not single lead-h point errors).

Common-origin MAE uses origins scored by all models; unavailable predictions are counted separately. Partial-window ≠ complete-horizon.

| model | h | n | commonMAE(n) | winMAE | part/comp | na | time(n) | false | miss |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| persistence | 7 | 602 | 0.0218 (602) | 0.0218 | 119/483 | 0 | — (0) | 0/199 | 45/45 |
| persistence | 14 | 602 | 0.0263 (602) | 0.0263 | 214/388 | 0 | — (0) | 0/143 | 65/65 |
| persistence | 30 | 602 | 0.0384 (602) | 0.0384 | 389/213 | 0 | — (0) | 0/52 | 97/97 |
| recent_linear_trend | 7 | 602 | 0.0172 (602) | 0.0172 | 119/483 | 0 | 0.9375 (16) | 0/199 | 29/45 |
| recent_linear_trend | 14 | 602 | 0.0188 (602) | 0.0188 | 214/388 | 0 | 3.8667 (45) | 1/143 | 20/65 |
| recent_linear_trend | 30 | 602 | 0.0224 (602) | 0.0224 | 389/213 | 0 | 6.2105 (76) | 0/52 | 21/97 |

### Synthetic figures

- `reports/phase3/figures/S02_raw_vs_norm.png`
- `reports/phase3/figures/S07_failure_case.png`
- `reports/phase3/figures/S08_forecast_example.png`
- `reports/phase3/figures/S08_raw_vs_norm.png`

## OCWD separate s3sf baseline results (test split)

- TCF re-applied: False
- Claims cleaning-time validity: False
- Pooled with synthetic: False
- Disclaimer: OCWD experiment uses source-calculated s3sf. Normalisation provenance remains external and incompletely verified. Independent-normalisation blockers from Phase 2 are NOT resolved.

| model | h | n_scored | commonMAE (n) | winMAE | unavail | false | miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 7 | 4582 | 0.0549 (4560) | 0.0549 | 0 | 0/2606 | 224/224 |
| persistence | 14 | 4583 | 0.0785 (4561) | 0.0785 | 0 | 0/2057 | 395/395 |
| persistence | 30 | 4604 | 0.1137 (4582) | 0.1138 | 0 | 0/1249 | 646/646 |
| recent_linear_trend | 7 | 4560 | 0.0727 (4560) | 0.0727 | 22 | 96/2606 | 100/224 |
| recent_linear_trend | 14 | 4561 | 0.0978 (4561) | 0.0978 | 22 | 147/2057 | 155/395 |
| recent_linear_trend | 30 | 4582 | 0.1451 (4582) | 0.1451 | 22 | 144/1249 | 213/646 |

### OCWD figures

- `reports/phase3/figures/ocwd_A01_forecast.png`

## Baselines

- Persistence
- Recent linear trend

No uncertainty intervals. Shared `Forecaster` interface ready for a state-space model in a later phase.

## Next (not this phase)

- Fitted state-space candidate
- Versioned model artefact
- Final forecast API
