"""Reproducible synthetic RO raw-measurement simulator.

Permeability / flux / flow formulation (documented)
--------------------------------------------------
At temperature T:

    A(T) = A_25 * TCF_filmtec(T)          # GFD/psi ; filmtec_exponential_tcf_v1 only
    NDP  = P_feed - ΔP/2 - P_perm - π_fc + π_perm   # psi
    flux = A(T) * health * NDP            # GFD (gal / ft² / day)
    Area_ft2 = Area_m2 * M2_TO_FT2
    Q_p  = flux * Area_ft2                # gpd (gallons per day)

Temperature is corrected **only** via FilmTec TCF. No additional empirical
viscosity TCF is applied (avoids double-counting).

Ground-truth (health, clean performance, latents) is written to a separate
file and must never be used as forecasting inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aqua_fouling.physics.ndp import net_driving_pressure
from aqua_fouling.physics.osmotic import (
    feed_concentrate_concentration_mg_l,
    osmotic_pressure_filmtec_psi,
)
from aqua_fouling.physics.temperature import filmtec_temperature_correction_factor
from aqua_fouling.simulation.domain import BrackishDomain, DomainViolationError

# Exact conversion used for flux→flow (1 m² = 10.76391041671 ft²)
M2_TO_FT2 = 10.76391041671

LATENT_COLUMNS = (
    "true_health",
    "clean_permeability_gfd_per_psi",
    "true_ndp_psi",
    "true_permeate_flow_clean_gpd",
    "scenario_family",
    "latent_decay_per_day",
    "latent_accel_per_day2",
    "split",
)


@dataclass
class SeriesSpec:
    series_id: str
    family: str
    split: str
    seed_offset: int
    health_decay_per_day: float = 0.0
    health_accel_per_day2: float = 0.0
    operation: str = "constant_pressure"
    target_permeate_flow_gpd: float | None = None
    max_feed_pressure_psi: float | None = None
    temp_amplitude_c: float = 0.0
    conc_amplitude_mg_l: float = 0.0
    clean_day: int | None = None
    clean_recovery_fraction: float = 0.0
    continue_past_threshold: bool = False
    censor_no_clean_crossing: bool = False


def load_benchmark_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Benchmark config must be a mapping: {path}"
        raise TypeError(msg)
    return loaded


def _series_from_config(item: dict[str, Any]) -> SeriesSpec:
    target = item.get("target_permeate_flow_gpd", item.get("target_permeate_flow"))
    return SeriesSpec(
        series_id=str(item["id"]),
        family=str(item["family"]),
        split=str(item["split"]),
        seed_offset=int(item["seed_offset"]),
        health_decay_per_day=float(item.get("health_decay_per_day", 0.0)),
        health_accel_per_day2=float(item.get("health_accel_per_day2", 0.0)),
        operation=str(item.get("operation", "constant_pressure")),
        target_permeate_flow_gpd=float(target) if target is not None else None,
        max_feed_pressure_psi=(
            float(item["max_feed_pressure_psi"])
            if item.get("max_feed_pressure_psi") is not None
            else None
        ),
        temp_amplitude_c=float(item.get("temp_amplitude_c", 0.0)),
        conc_amplitude_mg_l=float(item.get("conc_amplitude_mg_l", 0.0)),
        clean_day=int(item["clean_day"]) if item.get("clean_day") is not None else None,
        clean_recovery_fraction=float(item.get("clean_recovery_fraction", 0.0)),
        continue_past_threshold=bool(item.get("continue_past_threshold", False)),
        censor_no_clean_crossing=bool(item.get("censor_no_clean_crossing", False)),
    )


def flux_gfd_to_flow_gpd(flux_gfd: float, area_m2: float) -> float:
    """Dimensionally correct GFD × ft² → gpd."""
    return float(flux_gfd) * float(area_m2) * M2_TO_FT2


def _health_trajectory(
    n_days: int,
    spec: SeriesSpec,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    health = np.ones(n_days, dtype=float)
    meta: dict[str, Any] = {
        "censor_no_clean_crossing": spec.censor_no_clean_crossing,
        "would_cross_without_clean": None,
        "crossed_threshold": False,
        "intervention_before_crossing": False,
    }
    for t in range(1, n_days):
        decay = spec.health_decay_per_day + spec.health_accel_per_day2 * t
        health[t] = max(0.05, health[t - 1] - decay)
        if spec.clean_day is not None and t == spec.clean_day and spec.clean_recovery_fraction > 0:
            health[t] = health[t] + (1.0 - health[t]) * spec.clean_recovery_fraction

    # Unobserved no-cleaning trajectory (for censoring metadata)
    if spec.clean_day is not None or spec.censor_no_clean_crossing:
        h_nc = np.ones(n_days, dtype=float)
        for t in range(1, n_days):
            decay = spec.health_decay_per_day + spec.health_accel_per_day2 * t
            h_nc[t] = max(0.05, h_nc[t - 1] - decay)
        cross_idx = np.flatnonzero(h_nc <= threshold)
        meta["would_cross_without_clean"] = int(cross_idx[0]) if cross_idx.size else None
        if (
            spec.clean_day is not None
            and meta["would_cross_without_clean"] is not None
            and spec.clean_day < int(meta["would_cross_without_clean"])
        ):
            meta["intervention_before_crossing"] = True
            meta["censor_no_clean_crossing"] = True

    if np.any(health <= threshold):
        meta["crossed_threshold"] = True
        meta["first_cross_day"] = int(np.flatnonzero(health <= threshold)[0])
    elif spec.continue_past_threshold and meta.get("would_cross_without_clean") is None:
        # No-intervention series that never crosses: still OK (stable family).
        pass
    return health, meta


def simulate_series(
    spec: SeriesSpec,
    *,
    cfg: dict[str, Any],
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (raw_measurements, ground_truth, series_meta)."""
    domain_cfg = cfg["domain"]
    sim_cfg = cfg["simulation"]
    forecast_cfg = cfg["forecasting"]
    domain = BrackishDomain(max_c_fc_mg_l=float(domain_cfg["max_c_fc_mg_l"]))

    rng = np.random.default_rng(base_seed + spec.seed_offset)
    n_days = int(sim_cfg["n_days"])
    start = date.fromisoformat(str(sim_cfg["start_date"]))
    threshold = float(forecast_cfg["threshold_relative_health"])

    a25 = float(domain_cfg["clean_permeability_gfd_per_psi_at_25c"])
    area_m2 = float(domain_cfg["membrane_area_m2"])
    area_ft2 = area_m2 * M2_TO_FT2

    feed_c0 = float(np.mean(domain_cfg["feed_concentration_mg_l_range"]))
    temp0 = float(cfg.get("reference", {}).get("temperature_c", 25.0))
    recovery0 = float(np.mean(domain_cfg["recovery_range"]))
    p_feed0 = float(np.mean(domain_cfg["feed_pressure_psi_range"]))
    p_perm = float(domain_cfg["permeate_pressure_psi"])
    dp = float(domain_cfg["device_dp_psi"])

    # Validate declared baseline operating point once (no silent repair).
    c_fc0 = float(feed_concentrate_concentration_mg_l(feed_c0, recovery0)[0])
    domain.validate_operating_point(
        feed_concentration_mg_l=feed_c0,
        recovery=recovery0,
        temp_c=temp0,
        c_fc_mg_l=c_fc0,
    )

    health, health_meta = _health_trajectory(n_days, spec, threshold)

    # Coherent clean reference at matching T, NDP, flow (noise-free day-0 clean).
    pi_fc0 = float(osmotic_pressure_filmtec_psi(c_fc0, temp0)[0])
    ndp0 = float(net_driving_pressure(p_feed0, p_perm, dp, pi_fc0, 0.0)[0])
    if ndp0 <= 0:
        msg = (
            f"Baseline NDP={ndp0} psi is non-positive at declared reference conditions. "
            "Adjust feed_pressure_psi_range / osmotic inputs; invalid scenarios are not repaired."
        )
        raise DomainViolationError(msg)
    tcf0 = float(filmtec_temperature_correction_factor(temp0)[0])
    a0 = a25 * tcf0
    q_ref_gpd = flux_gfd_to_flow_gpd(a0 * 1.0 * ndp0, area_m2)

    rows_raw: list[dict[str, Any]] = []
    rows_gt: list[dict[str, Any]] = []

    for t in range(n_days):
        temp = temp0 + spec.temp_amplitude_c * np.sin(2 * np.pi * t / 365.0)
        feed_c = feed_c0 + spec.conc_amplitude_mg_l * np.sin(2 * np.pi * t / 180.0)
        recovery = recovery0
        p_feed = p_feed0

        c_fc = float(feed_concentrate_concentration_mg_l(feed_c, recovery)[0])
        domain.validate_operating_point(
            feed_concentration_mg_l=feed_c,
            recovery=recovery,
            temp_c=temp,
            c_fc_mg_l=c_fc,
        )

        pi_fc = float(osmotic_pressure_filmtec_psi(c_fc, temp)[0])
        pi_p = 0.0
        tcf = float(filmtec_temperature_correction_factor(temp)[0])
        a_t = a25 * tcf

        if spec.operation == "constant_flow" and spec.target_permeate_flow_gpd is not None:
            # Q_gpd = A*h*NDP * area_ft2  →  NDP = Q / (A*h*area_ft2)
            denom = a_t * health[t] * area_ft2
            if denom <= 0:
                msg = "Non-positive permeability×health×area in constant-flow controller."
                raise DomainViolationError(msg)
            ndp_needed = spec.target_permeate_flow_gpd / denom
            p_feed = ndp_needed + dp / 2.0 + p_perm + pi_fc - pi_p
            if spec.max_feed_pressure_psi is not None and p_feed > spec.max_feed_pressure_psi:
                p_feed = float(spec.max_feed_pressure_psi)

        ndp = float(net_driving_pressure(p_feed, p_perm, dp, pi_fc, pi_p)[0])
        if ndp <= 0:
            msg = (
                f"NDP={ndp} psi non-positive at t={t} for {spec.series_id}. "
                "Invalid operating point is not silently repaired."
            )
            raise DomainViolationError(msg)

        flux_clean = a_t * 1.0 * ndp
        flux = a_t * health[t] * ndp
        q_clean = flux_gfd_to_flow_gpd(flux_clean, area_m2)
        q_p = flux_gfd_to_flow_gpd(flux, area_m2)

        # After pressure cap, realized flow may be below target — keep mass balance consistent.
        q_f = q_p / recovery
        q_c = q_f - q_p
        salt_passage = 0.01
        c_perm = feed_c * salt_passage
        c_conc = (feed_c * q_f - c_perm * q_p) / max(q_c, 1e-9)

        noise_std = float(sim_cfg.get("noise_relative_std", 0.01))

        def _noise(x: float, *, std: float = noise_std) -> float:
            return float(x * (1.0 + rng.normal(0.0, std)))

        # Raw measurements only — no latents / split labels as predictor features.
        # split/scenario_family kept as metadata columns excluded from forecasting inputs.
        raw = {
            "date": (start + timedelta(days=t)).isoformat(),
            "source_series_id": spec.series_id,
            "feed_temperature_c": _noise(temp),
            "feed_pressure_psi": _noise(p_feed),
            "concentrate_pressure_psi": _noise(p_feed - dp),
            "permeate_pressure_psi": _noise(p_perm),
            "device_pressure_drop_psi": _noise(dp),
            "feed_flow_gpd": _noise(q_f),
            "permeate_flow_gpd": _noise(q_p),
            "concentrate_flow_gpd": _noise(q_c),
            "feed_concentration_mg_l": _noise(feed_c),
            "permeate_concentration_mg_l": _noise(c_perm),
            "concentrate_concentration_mg_l": _noise(c_conc),
            "cip_flag": 1 if (spec.clean_day is not None and t == spec.clean_day) else 0,
            "days_since_replacement": t,
        }

        if rng.random() < float(sim_cfg.get("missing_probability", 0.0)):
            raw["permeate_flow_gpd"] = np.nan
        if rng.random() < float(sim_cfg.get("outlier_probability", 0.0)):
            pf_val = raw["permeate_flow_gpd"]
            raw["permeate_flow_gpd"] = (
                float(pf_val) * 1.5
                if isinstance(pf_val, (int, float)) and np.isfinite(float(pf_val))
                else np.nan
            )

        gt = {
            "date": raw["date"],
            "source_series_id": spec.series_id,
            "true_health": float(health[t]),
            "clean_permeability_gfd_per_psi": a_t,
            "true_ndp_psi": ndp,
            "true_permeate_flow_clean_gpd": q_clean,
            "true_permeate_flow_gpd": q_p,
            "true_flux_gfd": flux,
            "scenario_family": spec.family,
            "latent_decay_per_day": spec.health_decay_per_day,
            "latent_accel_per_day2": spec.health_accel_per_day2,
            "split": spec.split,
        }
        rows_raw.append(raw)
        rows_gt.append(gt)

    raw_df = pd.DataFrame(rows_raw)
    gt_df = pd.DataFrame(rows_gt)

    ref = {
        "mode": "documented_clean_by_construction",
        "provenance": cfg["reference"]["provenance"],
        "available_from": (start).isoformat(),
        "temperature_c": temp0,
        "ndp_psi": ndp0,
        "permeate_flow": q_ref_gpd,
        "permeate_flow_unit": "gpd",
        "membrane_area_m2": area_m2,
        "membrane_area_ft2": area_ft2,
        "clean_permeability_gfd_per_psi_at_ref": a0,
        "series_id": spec.series_id,
        "note": (
            "Noise-free clean reference at matching T/NDP/flow; "
            "not taken from noisy day-0 measurements."
        ),
    }
    meta = {
        "series": spec.series_id,
        "family": spec.family,
        "split": spec.split,
        "reference": ref,
        "health_meta": health_meta,
        "domain": domain.to_dict(),
        "flow_units": "gpd",
        "flux_units": "GFD",
        "flux_to_flow": "Q_gpd = flux_GFD * area_m2 * M2_TO_FT2",
        "m2_to_ft2": M2_TO_FT2,
        "temperature_treatment": (
            "FilmTec TCF only (filmtec_exponential_tcf_v1); "
            "no additional viscosity TCF (no double-counting)."
        ),
        "latent_columns_excluded_from_forecast_inputs": list(LATENT_COLUMNS),
        "predictor_excluded_metadata_columns": ["split", "scenario_family"],
        "cleaning_threshold": {
            "relative_health": threshold,
            "justification": forecast_cfg.get("threshold_justification"),
        },
    }
    return raw_df, gt_df, meta


def generate_synthetic_benchmark(
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate all synthetic series; write raw/, ground_truth/, meta."""
    cfg = load_benchmark_config(config_path)
    out = Path(output_dir or cfg["output_dir"])
    raw_dir = out / "raw"
    gt_dir = out / "ground_truth"
    ref_dir = out / "references"
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg["seed"])
    manifests: list[dict[str, Any]] = []
    for item in cfg["simulation"]["series"]:
        spec = _series_from_config(item)
        raw_df, gt_df, meta = simulate_series(spec, cfg=cfg, base_seed=base_seed)
        raw_path = raw_dir / f"{spec.series_id}.csv"
        gt_path = gt_dir / f"{spec.series_id}.csv"
        raw_df.to_csv(raw_path, index=False)
        gt_df.to_csv(gt_path, index=False)
        # Documented reference YAML for shared normalise_csv path
        ref = meta["reference"]
        ref_yaml = {
            "mode": "documented",
            "provenance": ref["provenance"],
            "available_from": ref["available_from"],
            "temperature_c": ref["temperature_c"],
            "ndp_psi": ref["ndp_psi"],
            "permeate_flow": ref["permeate_flow"],
            "permeate_flow_unit": ref["permeate_flow_unit"],
            "notes": ref.get("note"),
        }
        ref_path = ref_dir / f"{spec.series_id}_reference.yaml"
        ref_path.write_text(yaml.safe_dump(ref_yaml, sort_keys=False), encoding="utf-8")
        manifests.append(
            {
                "series_id": spec.series_id,
                "split": spec.split,
                "family": spec.family,
                "raw_path": str(raw_path),
                "ground_truth_path": str(gt_path),
                "reference_path": str(ref_path),
                "meta": meta,
            }
        )

    manifest = {
        "config": str(config_path),
        "seed": base_seed,
        "domain": cfg["domain"],
        "mapping": "configs/canonical_synthetic_schema.yaml",
        "series": manifests,
        "note": (
            "Ground-truth latents must never be used as forecasting inputs. "
            "Synthetic benchmark performance does not establish plant accuracy."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "domain_assumptions.json").write_text(
        json.dumps(
            {
                "domain": cfg["domain"],
                "literature_vs_experimental": {
                    "literature_supported": cfg["domain"].get("literature_supported"),
                    "experimental_scenario_ranges": cfg["domain"].get(
                        "experimental_scenario_ranges"
                    ),
                },
                "not_seawater": True,
                "validity_at": "concentrate_side_C_fc",
                "flow_conversion": {
                    "formula": "Q_gpd = flux_GFD * area_m2 * M2_TO_FT2",
                    "m2_to_ft2": M2_TO_FT2,
                },
                "cleaning_threshold": cfg["forecasting"].get("threshold_justification"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
