# Data dictionary (Phase 2)

Machine-readable twin: `docs/data_dictionary.json`.

Evidence hierarchy: Rd documentation > column-name evidence > diagram support for investigation > unresolved.

**Diagrams** (`orange_county_ro_diagram1.png`, `diagram2.png`) illustrate bank/stage layout and support investigation; they do **not** automatically establish units or semantics of every CSV column.

**Scientific equations** (FilmTec Form 609-02084-0905):  
https://www.bigbrandwater.com/assets/library/filmtec/filmtec-record-keeping.pdf  
- TCF exponential convention (T in °C; TCF(25 °C)=1)  
- π (bar) = C_fc*(T+320)/491000 for C_fc < 20000 mg/L  
- Normalised flow structure Q_s = Q_o * NDP_s/NDP_o * TCF_s/TCF_o  

AFBC is **not** reinterpreted beyond the Rd token list (Rd lists “afbc” without a reliable expansion we can defend).

---

## Adapter fields (union of AE + FG mappings)

| Original | Schema | Canonical | Unit | Conversion | Provenance kind | Evidence | Confidence | Assumptions / restrictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| date | AE, FG | observation_date_calendar | calendar_date | none | measured | Rd `date` | documented | Timezone declared America/Los_Angeles in adapter (intent); plant TZ not proven in Rd |
| temp_c | AE, FG | temperature_c | deg_C | none | measured | Column suffix `_c` | inferred | Naming evidence only |
| feed_psi | AE | feed_pressure_psi | psi | none | measured | Column suffix `_psi` | inferred | Whole-unit; not mixed into stage-3 NDP |
| feed_press_stage_1 | FG | stage_1_feed_pressure_psi | psi | none | measured | Naming | inferred | FG schema substitute; stage-1 scope |
| stage_2_3_press | AE | stage_3_feed_pressure_psi | psi | none | measured | Naming + diagram investigation | inferred | **Assumption:** entering stage 3; confirm on P&ID |
| perm_psi / perm_press_unit | AE / FG | permeate_pressure_psi | psi | none | measured | Naming | inferred | Whole-unit permeate pressure |
| stage_3_flow / perm_flow_stage_3 | AE / FG | stage_3_permeate_flow_source_unit | unresolved_flow | none | measured | Rd stage context | unresolved | Unit not in Rd |
| stage_2_3_feed_flow | AE | stage_3_feed_flow_source_unit | unresolved_flow | none | measured | Naming | unresolved | Used for recovery when present |
| stage_3_flux | AE, FG | stage_3_flux_source_unit | unresolved_flux | none | source-calculated | Present in CSV; Rd silent on unit | unresolved | Distinct from our physics outputs |
| stage_3_net_driving_press | AE, FG | stage_3_ndp_source_calculated | unresolved_pressure | none | source-calculated | CSV | unresolved | **Not** our NDP; not used to validate concentration correction |
| s3sf | AE, FG | stage_3_specific_flux_source_calculated | unresolved_specific_flux | none | source-calculated | Rd interest variable | unresolved | Observationally ≈ flux*tcf_water/ndp |
| tcf_water | AE, FG | tcf_water_source_calculated | 1 (dimensionless) | none | source-calculated | Naming “tcf” | inferred | Convention ≠ assumed FilmTec; **no double correction** |
| perm_ec / perm_ec_stage_3 | AE, FG | *_conductivity_source_unit | unresolved_conductivity | none | measured | Rd: ec = conductivity | unresolved | **Not TDS**; no silent coefficient |
| rof_toc_avg | AE, FG | ro_feed_toc_avg_mg_per_L | mg_per_L | none | measured | Rd “(mg/L)” | documented | — |
| cip | AE, FG | cip_flag_0_1 | flag_0_1 | none | measured | Rd | documented | Duration not invented |
| dss | AE, FG | days_since_cleaning_d | d | none | source-calculated | Rd | documented | dss=0 ≠ clean reference proof |
| days_since_replacement | AE, FG | days_since_replacement_d | d | none | source-calculated | Rd | documented | Resets = inspection candidates |
| x3rd_pass_dp / stage_3_d_p | AE / FG | stage_3_device_pressure_drop_psi | psi | none | measured | Naming | inferred | Confirm unit/identity |
| pf / ff | AE, FG | unit_*_flow_source_unit | unresolved_flow | none | measured | Rd pf/ff gloss | unresolved | Whole-unit scope |

## Derived by our software (not source columns)

| Canonical / quantity | Unit | Kind | Evidence | Confidence | Restrictions |
| --- | --- | --- | --- | --- | --- |
| independent FilmTec TCF | 1 | derived by us | Form 609-02084 | documented equation | From `temperature_c` only |
| C_fc | mg/L | derived by us | Form 609-02084 | documented equation | Needs C_f and 0&lt;Y&lt;1 |
| π_fc (FilmTec) | psi (via bar) | derived by us | Form 609-02084 | documented for C_fc&lt;20000 | Not for seawater/high-TDS wastewater without validation |
| independent NDP | psi | derived by us | FilmTec/ASTM-style | documented structure | Needs confirmed stage hydraulics |
| Q_normalised | same as Q | derived by us | Form 609-02084 / ASTM D4516 structure | documented structure | Needs reference + positive NDP |

## Schema-missing vs observation-missing

- **Schema-missing:** column not in that series’ CSV schema (AE vs FG) — see adapters / audit `missingness_definitions`.
- **Observation-missing:** NA/non-finite inside an existing column.
- **Missing calendar date:** day absent between min/max date.
