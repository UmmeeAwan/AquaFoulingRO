"""Typer CLI entry point: ``poetry run aqua ...``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml

from aqua_fouling.application.audit import run_data_audit
from aqua_fouling.application.evaluate_trained import run_evaluation
from aqua_fouling.application.forecast_errors import ForecastServiceError
from aqua_fouling.application.forecast_service import forecast_csv
from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.application.ocwd_benchmark import run_ocwd_baseline_benchmark
from aqua_fouling.application.phase3_report import write_benchmark_summary
from aqua_fouling.application.readiness import write_normalisation_readiness_report
from aqua_fouling.application.readiness_quant import run_quantitative_readiness
from aqua_fouling.application.synthetic_benchmark import run_synthetic_baseline_benchmark
from aqua_fouling.application.train_llt import run_training
from aqua_fouling.application.validate import validate_csv
from aqua_fouling.settings import get_settings
from aqua_fouling.simulation.simulator import generate_synthetic_benchmark

app = typer.Typer(
    name="aqua",
    help=(
        "Aqua Fouling RO — audit, validate, normalise, forecast, baselines, "
        "and trained LLT train/evaluate."
    ),
    add_completion=False,
    no_args_is_help=True,
)


@app.command("audit-data")
def audit_data(
    archive: Annotated[
        Path,
        typer.Option("--archive", help="Path to the source ZIP archive."),
    ] = Path("data/raw/dataverse_files.zip"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for audit artefacts."),
    ] = Path("reports/data-audit"),
    extract_dir: Annotated[
        Path | None,
        typer.Option(
            "--extract-dir",
            help="Directory for safe extraction (default: data/extracted).",
        ),
    ] = None,
) -> None:
    """Safely extract the archive and write a reproducible data audit."""
    settings = get_settings()
    dest = extract_dir or settings.default_extract_dir
    typer.echo(f"Auditing archive: {archive}")
    typer.echo(f"Extracting to: {dest}")
    typer.echo(f"Writing reports to: {output}")
    result = run_data_audit(archive=archive, output=output, extract_dir=dest)
    summary = result["summary"]
    typer.echo(
        f"Done. series={summary['n_csv_series']} "
        f"dates={summary['global_min_date']}→{summary['global_max_date']} "
        f"cip_rows={summary['total_cip_flagged_rows']}"
    )
    typer.echo(
        f"Reports: {output}/audit.json, audit.md, per_series_summary.csv, column_inventory.csv"
    )


@app.command("validate")
def validate_cmd(
    csv: Annotated[Path, typer.Option("--csv", help="Source CSV path.")],
    source_series_id: Annotated[str, typer.Option("--series-id")],
    mapping: Annotated[
        Path | None, typer.Option("--mapping", help="Adapter YAML (optional).")
    ] = None,
    reference: Annotated[
        Path | None, typer.Option("--reference", help="Reference YAML (optional).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional JSON output path.")
    ] = None,
) -> None:
    """Validate a CSV against adapter/reference configuration."""
    result = validate_csv(
        csv,
        source_series_id=source_series_id,
        mapping_path=mapping,
        reference_path=reference,
    )
    text = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)


@app.command("normalise")
def normalise_cmd(
    csv: Annotated[Path, typer.Option("--csv", help="Source CSV path.")],
    source_series_id: Annotated[str, typer.Option("--series-id")],
    mapping: Annotated[Path | None, typer.Option("--mapping")] = None,
    reference: Annotated[Path | None, typer.Option("--reference")] = None,
    concentration_mg_l: Annotated[
        float | None,
        typer.Option("--concentration-mg-l", help="Feed concentration mg/L if known."),
    ] = None,
    calibration: Annotated[
        Path | None,
        typer.Option("--calibration", help="Optional conductivity calibration YAML/JSON."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional JSON output path.")
    ] = None,
) -> None:
    """Run normalisation (independent path when concentration/reference allow)."""
    cal: dict[str, Any] | None = None
    if calibration is not None:
        raw = calibration.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw) if calibration.suffix == ".json" else yaml.safe_load(raw)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            typer.echo(
                json.dumps(
                    {
                        "error": {
                            "code": "malformed_calibration",
                            "message": f"Malformed calibration file: {exc}",
                            "actionable_correction": (
                                "Provide valid JSON or YAML for --calibration."
                            ),
                            "details": {"path": str(calibration)},
                        }
                    },
                    indent=2,
                ),
                err=True,
            )
            raise typer.Exit(code=1) from exc
        if not isinstance(parsed, dict):
            typer.echo(
                json.dumps(
                    {
                        "error": {
                            "code": "malformed_calibration",
                            "message": "Calibration file must be a JSON/YAML object.",
                            "actionable_correction": (
                                "Provide a mapping of calibration parameters."
                            ),
                            "details": {"path": str(calibration)},
                        }
                    },
                    indent=2,
                ),
                err=True,
            )
            raise typer.Exit(code=1)
        cal = parsed
    result = normalise_csv(
        csv,
        source_series_id=source_series_id,
        mapping_path=mapping,
        reference_path=reference,
        concentration_mg_l=concentration_mg_l,
        conductivity_calibration=cal,
    )
    text = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)


@app.command("readiness-report")
def readiness_report(
    audit_json: Annotated[Path, typer.Option("--audit-json")] = Path(
        "reports/data-audit/audit.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path(
        "reports/phase2/normalisation_readiness.md"
    ),
) -> None:
    """Write the Phase 2 normalisation readiness report."""
    path = write_normalisation_readiness_report(audit_json=audit_json, output=output)
    typer.echo(f"Wrote {path}")


@app.command("readiness-quant")
def readiness_quant(
    csv_dir: Annotated[Path, typer.Option("--csv-dir")] = Path("data/extracted/CSV files"),
    output: Annotated[Path, typer.Option("--output")] = Path("reports/phase3"),
) -> None:
    """Quantitative per-series readiness (segments, gaps, eligible origins)."""
    summary = run_quantitative_readiness(csv_dir, output)
    typer.echo(
        f"Wrote {output}/readiness_quantitative.md "
        f"(series={summary['n_series']} segments={summary['total_segments']})"
    )


@app.command("generate-synthetic")
def generate_synthetic(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/synthetic_benchmark.yaml"),
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Generate reproducible synthetic raw + ground-truth series."""
    manifest = generate_synthetic_benchmark(config, output_dir=output_dir)
    typer.echo(
        f"Generated {len(manifest['series'])} series under "
        f"{output_dir or 'artifacts/synthetic'} (seed={manifest['seed']})"
    )


@app.command("benchmark-synthetic")
def benchmark_synthetic(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/synthetic_benchmark.yaml"),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports/phase3"),
    regenerate: Annotated[bool, typer.Option("--regenerate/--no-regenerate")] = True,
) -> None:
    """Run primary synthetic baseline benchmark."""
    payload = run_synthetic_baseline_benchmark(config, report_dir, regenerate=regenerate)
    typer.echo(f"Wrote {report_dir}/synthetic_metrics.json")
    typer.echo(
        f"Test eligible records: {payload.get('n_test_eligible_records')} "
        "(synthetic ≠ plant accuracy)"
    )


@app.command("benchmark-ocwd")
def benchmark_ocwd(
    config: Annotated[Path, typer.Option("--config")] = Path(
        "configs/ocwd_forecast_benchmark.yaml"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports/phase3"),
) -> None:
    """Run separate OCWD s3sf baseline experiment (not pooled with synthetic)."""
    payload = run_ocwd_baseline_benchmark(config, report_dir)
    typer.echo(f"Wrote {report_dir}/ocwd_metrics.json")
    typer.echo(
        f"Test records: {payload.get('n_test_records')} "
        "(external s3sf provenance; normalisation blockers unresolved)"
    )


@app.command("benchmark-summary")
def benchmark_summary(
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports/phase3"),
) -> None:
    """Assemble benchmark_summary.md from Phase 3 JSON artefacts."""
    syn = json.loads((report_dir / "synthetic_metrics.json").read_text(encoding="utf-8"))
    ocwd = json.loads((report_dir / "ocwd_metrics.json").read_text(encoding="utf-8"))
    readiness = None
    rq = report_dir / "readiness_quantitative.json"
    if rq.is_file():
        readiness = json.loads(rq.read_text(encoding="utf-8"))
    path = write_benchmark_summary(report_dir, synthetic=syn, ocwd=ocwd, readiness=readiness)
    typer.echo(f"Wrote {path}")


@app.command("train")
def train_cmd(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/training.yaml"),
) -> None:
    """Train the local-linear-trend Kalman candidate (CPU) and write a versioned artifact."""
    report = run_training(config)
    conv = report.get("convergence") or {}
    typer.echo(f"Wrote artifact: {report.get('artifact_path')}")
    typer.echo(f"Model version: {report.get('model_version')}")
    typer.echo(
        f"Convergence: success={conv.get('success')} nit={conv.get('n_iterations')} "
        f"final_nll={conv.get('final_negative_loglik')} "
        f"n_obs={conv.get('n_training_observations')}"
    )
    typer.echo(f"Training report: {config and 'reports/phase4/training_report.json'}")


@app.command("evaluate")
def evaluate_cmd(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/evaluation.yaml"),
) -> None:
    """Evaluate frozen trained LLT vs persistence and recent linear trend."""
    payload = run_evaluation(config)
    typer.echo(f"Wrote {payload.get('markdown_summary')}")
    typer.echo(
        f"Artifact unchanged during inference: "
        f"{payload.get('artifact_unchanged_during_inference')}"
    )
    typer.echo(
        "Test set is an inspected benchmark (not a pristine holdout). "
        "Synthetic ≠ plant / OCWD transfer."
    )


@app.command("forecast")
def forecast_cmd(
    csv: Annotated[Path, typer.Option("--csv", help="Plant operating-history CSV.")],
    source_series_id: Annotated[
        str | None,
        typer.Option(
            "--series-id",
            help="Optional; inferred from CSV or defaults to uploaded_series.",
        ),
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(
            "--reference",
            help="Documented reference YAML. Omit to receive reference_required.",
        ),
    ] = None,
    mapping: Annotated[
        Path | None,
        typer.Option(
            "--mapping",
            help="Adapter mapping YAML (else aliases / default canonical schema).",
        ),
    ] = None,
    mapping_json: Annotated[
        Path | None,
        typer.Option("--mapping-json", help="Optional JSON mapping file (Swagger parity)."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Optional serving YAML (informational; artifact from AQUA_ARTIFACT_PATH).",
        ),
    ] = None,
    horizon_days: Annotated[
        int | None, typer.Option("--horizon-days", help="Default 14; allowed: 7, 14, 30.")
    ] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Relative-health threshold.")
    ] = None,
    concentration_mg_l: Annotated[float | None, typer.Option("--concentration-mg-l")] = None,
    artifact: Annotated[
        Path | None,
        typer.Option(
            "--artifact",
            help="Override server artifact path (CLI only; not available via HTTP).",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional JSON output path.")
    ] = None,
) -> None:
    """CSV-first forecast using the configured trained artifact.

    Does not train. Does not mutate the artifact. One series per request.
    """
    _ = config  # documented for parity with HTTP serving config; settings/env win
    settings = get_settings()
    mj: dict[str, Any] | str | None = None
    if mapping_json is not None:
        mj = mapping_json.read_text(encoding="utf-8")
    try:
        result = forecast_csv(
            csv,
            source_series_id=source_series_id,
            mapping_path=mapping,
            mapping_json=mj,
            reference_path=reference,
            horizon_days=horizon_days,
            threshold=threshold,
            artifact_path=artifact or Path(settings.artifact_path),
            concentration_mg_l=concentration_mg_l,
        )
    except ForecastServiceError as exc:
        typer.echo(json.dumps(exc.to_dict(), indent=2), err=True)
        raise typer.Exit(code=1) from exc
    text = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)
    if result.get("status") == "reference_required":
        typer.echo("status=reference_required (documented reference required)", err=True)
        raise typer.Exit(code=2)
    res = result.get("result") or {}
    model = (result.get("metadata") or {}).get("model") or {}
    typer.echo(
        f"crossing_status={res.get('crossing_status')} "
        f"days_to_threshold={res.get('days_to_threshold')} "
        f"model={model.get('model_version')}",
        err=True,
    )


@app.command("serve")
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Start the API (health/ready/capabilities/validate/normalise/forecast)."""
    settings = get_settings()
    uvicorn.run(
        "aqua_fouling.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    app()
