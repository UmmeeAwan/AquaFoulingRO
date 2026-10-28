# Phase 4 — Trained LLT evaluation summary

- Artifact: `artifacts/models/llt_relative_health_v1.json` (`llt_relative_health_v1`)
- Test set label: inspected_benchmark — previously inspected in Phase 3; not a pristine unseen holdout. No post-test retuning.
- Disclaimer: Synthetic trained-model metrics do not establish plant accuracy or validated transfer to OCWD / real plants. OCWD remains a separate experiment and is not pooled here.
- Interval metrics: Coverage column = **average per-origin observation-interval coverage** (mean over origins of each origin's future-observation-interval hit rate). Also reports origin count, scored forecast–observation pair count, pooled pointwise coverage, mean observation-interval width, and mean latent-state interval width (distinct: obs width = latent P + R). Overlapping rolling-origin windows are not independent observations; do not interpret coverage as calibrated confidence.

## Development split (selection freeze; not used for claims of success)

| model | h | common-origin MAE | RMSE | n_common | n_unavail | false warn | missed warn | timing MAE | avg per-origin obs-int cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 7 | 0.02084 | 0.02415 | 468 | 0 | 0/163 | 41/41 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| persistence | 14 | 0.02324 | 0.02709 | 468 | 0 | 0/128 | 62/62 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| persistence | 30 | 0.03075 | 0.03540 | 468 | 0 | 0/63 | 109/109 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 7 | 0.01809 | 0.02134 | 468 | 0 | 9/163 | 13/41 | 1.077 (n=26) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 14 | 0.02047 | 0.02436 | 468 | 0 | 5/128 | 22/62 | 2.618 (n=34) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 30 | 0.02536 | 0.02973 | 468 | 0 | 7/63 | 43/109 | 4.193 (n=57) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| llt_relative_health_v1 | 7 | 0.01514 | 0.01827 | 468 | 0 | 0/163 | 19/41 | 1.364 (n=22) | 0.960 (origins=468; pairs=3157; pooled=0.959; obs_w=0.07525; lat_w=0.02632) |
| llt_relative_health_v1 | 14 | 0.01548 | 0.01888 | 468 | 0 | 0/128 | 21/62 | 3.175 (n=40) | 0.961 (origins=468; pairs=6175; pooled=0.960; obs_w=0.07706; lat_w=0.03079) |
| llt_relative_health_v1 | 30 | 0.01589 | 0.01935 | 468 | 0 | 0/63 | 24/109 | 4.183 (n=60) | 0.970 (origins=468; pairs=12530; pooled=0.968; obs_w=0.08282; lat_w=0.04184) |

## Test split (inspected benchmark)

| model | h | common-origin MAE | RMSE | n_common | n_unavail | false warn | missed warn | timing MAE | avg per-origin obs-int cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 7 | 0.02183 | 0.02526 | 602 | 0 | 0/199 | 45/45 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| persistence | 14 | 0.02629 | 0.03047 | 602 | 0 | 0/143 | 65/65 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| persistence | 30 | 0.03837 | 0.04427 | 602 | 0 | 0/52 | 97/97 | n/a (n=0) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 7 | 0.01717 | 0.02018 | 602 | 0 | 0/199 | 29/45 | 0.938 (n=16) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 14 | 0.01878 | 0.02227 | 602 | 0 | 1/143 | 20/65 | 3.867 (n=45) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| recent_linear_trend | 30 | 0.02242 | 0.02639 | 602 | 0 | 0/52 | 21/97 | 6.211 (n=76) | n/a (origins=0; pairs=0; pooled=n/a; obs_w=n/a; lat_w=n/a) |
| llt_relative_health_v1 | 7 | 0.01477 | 0.01767 | 602 | 0 | 0/199 | 23/45 | 1.591 (n=22) | 0.959 (origins=602; pairs=4019; pooled=0.959; obs_w=0.07529; lat_w=0.02643) |
| llt_relative_health_v1 | 14 | 0.01498 | 0.01825 | 602 | 0 | 0/143 | 23/65 | 3.548 (n=42) | 0.963 (origins=602; pairs=7807; pooled=0.962; obs_w=0.07708; lat_w=0.03082) |
| llt_relative_health_v1 | 30 | 0.01632 | 0.01993 | 602 | 0 | 0/52 | 29/97 | 6.588 (n=68) | 0.959 (origins=602; pairs=15617; pooled=0.957; obs_w=0.08262; lat_w=0.04144) |

## Failure cases (trained model, test)

- `missed_warning` S07 origin=2020-03-07 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.015308025646424623
- `missed_warning` S07 origin=2020-03-08 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.018036412562263675
- `missed_warning` S07 origin=2020-03-09 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.015172116671538569
- `missed_warning` S07 origin=2020-03-10 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.014242771704413792
- `missed_warning` S07 origin=2020-03-01 h=14 pred=no_crossing_in_horizon obs=observed_crossing mae=0.019146692415168125
- `missed_warning` S07 origin=2020-03-02 h=14 pred=no_crossing_in_horizon obs=observed_crossing mae=0.015289582030681132
- `missed_warning` S08 origin=2020-02-24 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.01435467361876692
- `missed_warning` S08 origin=2020-02-25 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.013968209939009062
- `missed_warning` S08 origin=2020-02-26 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.01280845500655013
- `missed_warning` S08 origin=2020-02-27 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.018447268618420925
- `missed_warning` S08 origin=2020-02-28 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.017543842016652995
- `missed_warning` S08 origin=2020-02-29 h=7 pred=no_crossing_in_horizon obs=observed_crossing mae=0.022137304460747416

## Remaining limitations

- Gaussian LLT may under-represent abrupt fouling acceleration.
- No days-until-cleaning uncertainty intervals in this phase.
- OCWD independent normalisation blockers remain unresolved; no OCWD transfer claim for the synthetic-trained artifact.
- Test set was previously inspected during Phase 3 baseline work.
