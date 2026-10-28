"""Unit tests for audit metrics on synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aqua_fouling.application.audit_metrics import (
    analyse_date_coverage,
    check_s3sf_relationship,
    compare_schemas,
    longest_consecutive_missing_runs,
    missing_measurement_mask,
)


def test_duplicate_and_missing_dates(fixtures_dir: Path) -> None:
    frame = pd.read_csv(fixtures_dir / "synthetic" / "series_a.csv")
    coverage = analyse_date_coverage(frame["date"])

    assert coverage.n_duplicate_date_rows == 1
    assert coverage.duplicate_dates == ["2020-01-02"]
    assert coverage.n_missing_dates == 1
    assert coverage.missing_dates == ["2020-01-03"]
    assert coverage.min_date == "2020-01-01"
    assert coverage.max_date == "2020-01-07"


def test_missing_measurement_vs_missing_date(fixtures_dir: Path) -> None:
    frame = pd.read_csv(fixtures_dir / "synthetic" / "series_a.csv")
    coverage = analyse_date_coverage(frame["date"])
    # Calendar gap is 2020-01-03 — no row exists for that day.
    assert "2020-01-03" in coverage.missing_dates

    # Measurement gaps exist on rows that do have dates.
    mask = missing_measurement_mask(frame["meas"])
    assert int(mask.sum()) == 3
    # Dates present where measurement missing include 2020-01-02, 01-05, 01-06
    missing_meas_dates = set(frame.loc[mask, "date"].astype(str))
    assert "2020-01-05" in missing_meas_dates
    assert "2020-01-03" not in missing_meas_dates


def test_consecutive_missing_run(fixtures_dir: Path) -> None:
    frame = pd.read_csv(fixtures_dir / "synthetic" / "series_a.csv")
    mask = missing_measurement_mask(frame["meas"])
    runs = longest_consecutive_missing_runs(mask, top_n=3)
    assert runs[0].length == 2
    assert runs[0].start_index == 4
    assert runs[0].end_index == 5


def test_schema_differences(fixtures_dir: Path) -> None:
    a = list(pd.read_csv(fixtures_dir / "synthetic" / "series_a.csv").columns)
    b = list(pd.read_csv(fixtures_dir / "synthetic" / "series_b.csv").columns)
    comparison = compare_schemas({"A": a, "B": b})
    assert comparison["n_distinct_schemas"] == 2
    assert "date" in comparison["shared_columns_all_series"]
    assert "cip" in comparison["shared_columns_all_series"]
    assert "s3sf" not in comparison["shared_columns_all_series"]


def test_s3sf_relationship_reports_exception(fixtures_dir: Path) -> None:
    frame = pd.read_csv(fixtures_dir / "synthetic" / "series_a.csv")
    result = check_s3sf_relationship(frame)
    assert result["checked"] is True
    assert result["n_exceptions"] >= 1
    assert result["rtol"] == 1e-4
    assert result["atol"] == 1e-6
    assert "absolute_error_max" in result
    assert "relative_error_max" in result
    exc = result["exceptions"][0]
    assert exc["row_index"] == 3
    assert exc["date"] == "2020-01-04"
    assert exc["observed_s3sf"] == 999.0
    assert "calculated_value" in exc
    assert "absolute_error" in exc
    assert "relative_error" in exc
