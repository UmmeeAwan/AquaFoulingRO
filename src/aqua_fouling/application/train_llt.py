"""Train the local-linear-trend Kalman candidate on designated training series."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize  # type: ignore[import-untyped]

from aqua_fouling.application.json_safe import json_safe
from aqua_fouling.application.synthetic_benchmark import (
    SYNTHETIC_MAPPING,
    _health_via_canonical_normalise,
)
from aqua_fouling.forecasting.artifact import (
    MODEL_VERSION,
    LLTArtifact,
    save_artifact,
    sha256_file,
    sha256_text,
)
from aqua_fouling.forecasting.evaluation import build_segments_for_series
from aqua_fouling.forecasting.llt_kalman import (
    LLTParameters,
    SegmentSeries,
    total_training_loglik,
)
from aqua_fouling.simulation.simulator import generate_synthetic_benchmark, load_benchmark_config


class TrainingConvergenceError(RuntimeError):
    """All training candidates failed to converge; existing artifact must be preserved."""


def _day_ordinals_from_dates(dates: list[Any]) -> np.ndarray:
    out: list[int] = []
    for d in dates:
        try:
            out.append(date.fromisoformat(str(d)[:10]).toordinal())
        except ValueError:
            out.append(-1)
    return np.asarray(out, dtype=np.int64)


def load_training_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Training config must be a mapping: {path}"
        raise ValueError(msg)
    return raw


def _segments_for_series(
    *,
    series_id: str,
    dates: Any,
    health: Any,
    cip: Any,
    dsr: Any,
) -> list[SegmentSeries]:
    import pandas as pd

    date_s = pd.Series(dates)
    health_s = pd.to_numeric(pd.Series(health), errors="coerce")
    segments, _audit, seg_ids = build_segments_for_series(
        date_s,
        cip=pd.Series(cip) if cip is not None else None,
        days_since_replacement=pd.Series(dsr) if dsr is not None else None,
    )
    day_ords = _day_ordinals_from_dates([str(x) for x in date_s.tolist()])
    vals = health_s.to_numpy(dtype=float)
    out: list[SegmentSeries] = []
    for sid, _seg in enumerate(segments):
        mask = seg_ids == sid
        if not np.any(mask):
            continue
        d = day_ords[mask]
        v = vals[mask]
        ok = d >= 0
        if not np.any(ok):
            continue
        if not np.any(np.isfinite(v[ok])):
            continue
        out.append(
            SegmentSeries(
                series_id=series_id,
                segment_id=sid,
                day_ordinals=d[ok],
                values=v[ok],
            )
        )
    return out


def _require_file_hash(path: Path, *, label: str) -> str:
    if not path.is_file():
        msg = (
            f"Cannot compute provenance hash for {label}: file not found at {path}. "
            "Refusing to fabricate a hash for an unverifiable file identity."
        )
        raise FileNotFoundError(msg)
    return sha256_file(path)


def compute_training_provenance(
    manifest: dict[str, Any],
    *,
    splits: set[str],
    cfg: dict[str, Any],
    mapping_path: Path,
    training_config_path: Path,
    benchmark_config_path: Path,
) -> dict[str, Any]:
    """Deterministic content hashes of training CSVs, references, mapping, configs."""
    content_hashes: dict[str, str] = {}
    series_ids: list[str] = []
    for item in manifest["series"]:
        if item["split"] not in splits:
            continue
        series_id = str(item["series_id"])
        series_ids.append(series_id)
        raw_path = Path(item["raw_path"])
        content_hashes[f"csv:{series_id}"] = _require_file_hash(
            raw_path, label=f"training CSV {series_id}"
        )
        ref_path = Path(item.get("reference_path") or "")
        if not ref_path.is_file():
            ref_path = Path(cfg["output_dir"]) / "references" / f"{series_id}_reference.yaml"
        content_hashes[f"reference:{series_id}"] = _require_file_hash(
            ref_path, label=f"training reference {series_id}"
        )
    content_hashes["mapping"] = _require_file_hash(mapping_path, label="adapter mapping")
    training_config_hash = _require_file_hash(training_config_path, label="training config")
    benchmark_config_hash = _require_file_hash(benchmark_config_path, label="benchmark config")
    # Combined data hash depends only on content digests (order-stable).
    training_data_hash = sha256_text(
        json.dumps(
            {k: content_hashes[k] for k in sorted(content_hashes)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return {
        "series_ids": series_ids,
        "content_hashes": content_hashes,
        "training_data_hash": training_data_hash,
        "training_config_hash": training_config_hash,
        "benchmark_config_hash": benchmark_config_hash,
        "mapping_path": str(mapping_path),
    }


def collect_split_segments(
    manifest: dict[str, Any],
    *,
    splits: set[str],
    cfg: dict[str, Any],
) -> tuple[list[SegmentSeries], list[str]]:
    """Load normalised health segments for the requested splits only."""
    segments: list[SegmentSeries] = []
    series_ids: list[str] = []
    import pandas as pd

    for item in manifest["series"]:
        if item["split"] not in splits:
            continue
        series_id = str(item["series_id"])
        raw = pd.read_csv(item["raw_path"])
        ref_path = Path(item.get("reference_path") or "")
        if not ref_path.is_file():
            ref_path = Path(cfg["output_dir"]) / "references" / f"{series_id}_reference.yaml"
        health, _meta = _health_via_canonical_normalise(
            Path(item["raw_path"]),
            series_id=series_id,
            reference_path=ref_path,
        )
        segs = _segments_for_series(
            series_id=series_id,
            dates=raw["date"],
            health=health,
            cip=raw.get("cip_flag"),
            dsr=raw.get("days_since_replacement"),
        )
        segments.extend(segs)
        series_ids.append(series_id)
    return segments, series_ids


def _neg_loglik(theta: np.ndarray, segments: list[SegmentSeries]) -> float:
    try:
        params = LLTParameters.from_unconstrained(theta)
    except Exception:  # noqa: BLE001
        return 1e12
    ll, n = total_training_loglik(segments, params)
    if n <= 0 or not np.isfinite(ll):
        return 1e12
    return float(-ll)


def fit_llt_parameters(
    segments: list[SegmentSeries],
    *,
    seed: int,
    maxiter: int = 200,
    initial: LLTParameters | None = None,
) -> tuple[LLTParameters, dict[str, Any]]:
    """Minimize negative innovation log-likelihood (L-BFGS-B on log-variances)."""
    rng = np.random.default_rng(seed)
    if initial is None:
        initial = LLTParameters(
            r=1e-3,
            q_level=1e-5,
            q_slope=1e-7,
            p0_level=1e-2,
            p0_slope=1e-4,
        )
    theta0 = initial.to_unconstrained()
    theta0 = theta0 + rng.normal(0.0, 0.01, size=theta0.shape)

    history: list[float] = []

    def _cb(th: np.ndarray) -> None:
        history.append(float(_neg_loglik(th, segments)))

    result = minimize(
        _neg_loglik,
        theta0,
        args=(segments,),
        method="L-BFGS-B",
        callback=_cb,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    params = LLTParameters.from_unconstrained(np.asarray(result.x, dtype=float))
    final_nll = float(result.fun)
    ll, n_obs = total_training_loglik(segments, params)
    # converged_flag reflects optimiser success only — a finite objective after
    # one iteration does not establish convergence.
    convergence = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "n_iterations": int(result.nit),
        "n_function_evaluations": int(result.nfev),
        "final_negative_loglik": final_nll,
        "final_loglik": float(ll),
        "n_training_observations": int(n_obs),
        "n_training_segments": len(segments),
        "objective_trace_tail": history[-20:],
        "converged_flag": bool(result.success),
        "optimizer": "scipy.optimize.minimize/L-BFGS-B",
        "parameterization": "log-variance (exp) with clip to [1e-12, 1e2]",
    }
    if not np.isfinite(final_nll):
        msg = "Training failed: non-finite final negative log-likelihood."
        raise RuntimeError(msg)
    return params, convergence


def select_settings_on_development(
    train_params_by_setting: dict[str, LLTParameters],
    *,
    development_segments: list[SegmentSeries],
    setting_grid: list[dict[str, Any]],
    convergence_by_setting: dict[str, dict[str, Any]],
) -> tuple[str, LLTParameters, dict[str, Any]]:
    """Pick among *successfully converged* settings by development log-likelihood."""
    rows: list[dict[str, Any]] = []
    best_key = ""
    best_ll = -np.inf
    best_params: LLTParameters | None = None
    for spec in setting_grid:
        key = str(spec["name"])
        conv = convergence_by_setting.get(key) or {}
        converged = bool(conv.get("converged_flag"))
        params = train_params_by_setting[key]
        ll, n = total_training_loglik(development_segments, params)
        row = {
            "setting": key,
            "converged": converged,
            "excluded_from_selection": not converged,
            "development_loglik": float(ll) if converged else None,
            "n_development_observations": int(n),
            "parameters": params.to_dict(),
            "convergence_message": conv.get("message"),
            "spec": spec,
        }
        rows.append(row)
        if not converged:
            continue
        if n > 0 and ll > best_ll:
            best_ll = float(ll)
            best_key = key
            best_params = params
    if best_params is None:
        failed = [
            {
                "setting": r["setting"],
                "message": r.get("convergence_message"),
            }
            for r in rows
        ]
        msg = (
            "Development selection failed: no training candidates converged. "
            "Unsuccessful optimiser runs are excluded from selection. "
            "The existing serving artifact (if any) must be preserved — "
            "this training run must not overwrite it. "
            f"Failed candidates: {failed}."
        )
        raise TrainingConvergenceError(msg)
    return (
        best_key,
        best_params,
        {
            "candidates": rows,
            "selected": best_key,
            "selection_rule": (
                "Among candidates with converged_flag=True only; "
                "highest development innovation log-likelihood wins."
            ),
        },
    )


def run_training(config_path: Path) -> dict[str, Any]:
    """Full training pipeline → versioned JSON artifact.

    If all candidates fail to converge, raises ``TrainingConvergenceError`` and
    does **not** overwrite an existing artifact at the configured path.
    """
    train_cfg = load_training_config(config_path)
    bench_path = Path(train_cfg["synthetic_benchmark_config"])
    bench = load_benchmark_config(bench_path)
    regenerate = bool(train_cfg.get("regenerate_synthetic", True))
    if regenerate:
        manifest = generate_synthetic_benchmark(bench_path)
    else:
        out = Path(bench["output_dir"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    seed = int(train_cfg.get("random_seed", bench.get("seed", 0)))
    mapping_path = Path(train_cfg.get("mapping_path") or SYNTHETIC_MAPPING)
    provenance = compute_training_provenance(
        manifest,
        splits={"train"},
        cfg=bench,
        mapping_path=mapping_path,
        training_config_path=config_path,
        benchmark_config_path=bench_path,
    )
    train_segments, train_ids = collect_split_segments(manifest, splits={"train"}, cfg=bench)
    banned = {
        str(item["series_id"])
        for item in manifest["series"]
        if item["split"] in {"development", "test"}
    }
    if any(s.series_id in banned for s in train_segments):
        msg = "Training segments incorrectly include development/test series."
        raise RuntimeError(msg)
    if set(train_ids) & banned:
        msg = "Training series IDs overlap development/test."
        raise RuntimeError(msg)

    dev_segments, dev_ids = collect_split_segments(manifest, splits={"development"}, cfg=bench)
    test_ids = [str(item["series_id"]) for item in manifest["series"] if item["split"] == "test"]

    setting_grid = list(train_cfg.get("development_selection_grid") or [])
    if not setting_grid:
        setting_grid = [
            {
                "name": "default_init",
                "initial": {
                    "r": 1.0e-3,
                    "q_level": 1.0e-5,
                    "q_slope": 1.0e-7,
                    "p0_level": 1.0e-2,
                    "p0_slope": 1.0e-4,
                },
            }
        ]

    fitted: dict[str, LLTParameters] = {}
    conv_by_setting: dict[str, Any] = {}
    maxiter = int(train_cfg.get("optimizer", {}).get("maxiter", 200))
    out_path = Path(train_cfg["artifact_path"])
    existing_artifact_preserved = out_path.is_file()

    for spec in setting_grid:
        init = LLTParameters.from_dict(spec["initial"])
        params, conv = fit_llt_parameters(train_segments, seed=seed, maxiter=maxiter, initial=init)
        fitted[str(spec["name"])] = params
        conv_by_setting[str(spec["name"])] = conv

    try:
        selected_name, _selected_params, selection = select_settings_on_development(
            fitted,
            development_segments=dev_segments,
            setting_grid=setting_grid,
            convergence_by_setting=conv_by_setting,
        )
    except TrainingConvergenceError:
        # Do not overwrite the serving artifact.
        raise

    selected_init = LLTParameters.from_dict(
        next(s["initial"] for s in setting_grid if s["name"] == selected_name)
    )
    frozen_params, frozen_conv = fit_llt_parameters(
        train_segments, seed=seed, maxiter=maxiter, initial=selected_init
    )
    if not frozen_conv.get("converged_flag"):
        msg = (
            f"Final freeze fit for setting '{selected_name}' did not converge "
            f"({frozen_conv.get('message')}). Existing serving artifact at "
            f"{out_path} was not overwritten "
            f"(existed_before={existing_artifact_preserved})."
        )
        raise TrainingConvergenceError(msg)

    artifact = LLTArtifact(
        parameters=frozen_params.to_dict(),
        training_series_ids=sorted(train_ids),
        development_series_ids=sorted(dev_ids),
        held_out_from_training=sorted(banned),
        random_seed=seed,
        training_config_hash=provenance["training_config_hash"],
        training_data_hash=provenance["training_data_hash"],
        convergence=frozen_conv,
        selected_settings={
            "selected_grid_name": selected_name,
            "min_history_observations": int(train_cfg.get("min_history_observations", 5)),
            "development_selection": selection,
            "all_setting_convergence": conv_by_setting,
        },
        development_selection_note=(
            "A small grid of initialisation settings was fitted on training data "
            "only; only optimiser-successful (converged_flag=True) candidates were "
            "eligible for development log-likelihood selection, frozen before test "
            "evaluation. The inspected Phase 3 test set remains an inspected "
            "benchmark, not a pristine unseen holdout."
        ),
        provenance={
            "training_config": str(config_path),
            "synthetic_benchmark_config": str(bench_path),
            "benchmark_config_hash": provenance["benchmark_config_hash"],
            "content_hashes": provenance["content_hashes"],
            "mapping_path": provenance["mapping_path"],
            "test_series_ids_not_used": test_ids,
            "cpu_only": True,
            "no_simulator_latent_inputs": True,
            "hash_policy": (
                "training_data_hash is SHA-256 over the sorted JSON map of "
                "SHA-256 content digests of training CSVs, reference YAMLs, and "
                "the adapter mapping. Config hashes are file content digests. "
                "Hashes are never fabricated for missing files."
            ),
        },
        assumptions=[
            "Measured relative_health from shared normalisation pipeline only.",
            "No latent simulator health, decay, scenario labels, or future rows.",
            "Filter resets between series and intervention/calendar-gap segments.",
            "Forecasts assume no future cleaning intervention.",
            "Improving slopes are not clipped.",
            "Gaussian innovation intervals are model-based, not calibrated confidence.",
            "Prior mean health=1.0 (clean-reference scale) at segment start; "
            "mid-fouling histories are not re-centred.",
        ],
    )
    save_artifact(artifact, out_path)

    report = {
        "artifact_path": str(out_path),
        "model_version": MODEL_VERSION,
        "parameters": frozen_params.to_dict(),
        "convergence": frozen_conv,
        "selected_settings": artifact.selected_settings,
        "training_series_ids": artifact.training_series_ids,
        "development_series_ids": artifact.development_series_ids,
        "held_out_from_training": artifact.held_out_from_training,
        "training_config_hash": provenance["training_config_hash"],
        "benchmark_config_hash": provenance["benchmark_config_hash"],
        "training_data_hash": provenance["training_data_hash"],
        "content_hashes": provenance["content_hashes"],
    }
    report_path = Path(train_cfg.get("training_report_path", "reports/phase4/training_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    return report
