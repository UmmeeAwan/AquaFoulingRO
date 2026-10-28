"""Integration tests: quantity-aware unit validation closes adapter bypass."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aqua_fouling.application.normalise import normalise_csv
from aqua_fouling.physics.units import PSI_PER_BAR


def _write_mapping(
    path: Path,
    *,
    pressure_unit: str = "psi",
    temp_unit: str = "deg_C",
    conc_unit: str = "mg_L_NaCl_eq",
    flow_unit: str = "gpd",
) -> None:
    path.write_text(
        f"""
name: quantity_unit_test
schema_id: canonical_generic_v1
source_series_pattern: ".+"
timestamp: {{column: observation_date, format: ISO8601, timezone: UTC}}
fields:
  - {{external_name: observation_date, canonical_name: observation_date_calendar,
     unit: {{symbol: calendar_date}}, kind: measured, measurement_scope: whole_unit}}
  - {{external_name: temperature_c, canonical_name: temperature_c,
     unit: {{symbol: {temp_unit}}}, kind: measured, measurement_scope: whole_unit}}
  - {{external_name: feed_pressure, canonical_name: stage_3_feed_pressure_psi,
     unit: {{symbol: {pressure_unit}}}, kind: measured, measurement_scope: stage_3}}
  - {{external_name: permeate_pressure, canonical_name: permeate_pressure_psi,
     unit: {{symbol: {pressure_unit}}}, kind: measured, measurement_scope: whole_unit}}
  - {{external_name: device_dp, canonical_name: stage_3_device_pressure_drop_psi,
     unit: {{symbol: {pressure_unit}}}, kind: measured, measurement_scope: stage_3}}
  - {{external_name: permeate_flow, canonical_name: stage_3_permeate_flow_source_unit,
     unit: {{symbol: {flow_unit}}}, kind: measured, measurement_scope: stage_3}}
  - {{external_name: feed_flow, canonical_name: stage_3_feed_flow_source_unit,
     unit: {{symbol: {flow_unit}}}, kind: measured, measurement_scope: stage_3}}
  - {{external_name: feed_concentration, canonical_name: feed_concentration_mg_l,
     unit: {{symbol: {conc_unit}}}, kind: measured, measurement_scope: whole_unit}}
  - {{external_name: cip, canonical_name: cip_flag_0_1,
     unit: {{symbol: flag_0_1}}, kind: measured, measurement_scope: whole_unit}}
""",
        encoding="utf-8",
    )


def _write_csv(path: Path, *, p_feed: float, p_perm: float, dp: float) -> None:
    path.write_text(
        "observation_date,temperature_c,feed_pressure,permeate_pressure,device_dp,"
        "permeate_flow,feed_flow,feed_concentration,cip\n"
        f"2020-01-01,25.0,{p_feed},{p_perm},{dp},6500.0,10000.0,2000.0,0\n"
        f"2020-01-02,25.0,{p_feed},{p_perm},{dp},6400.0,9846.0,2000.0,0\n",
        encoding="utf-8",
    )


def _write_ref(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "documented",
                "provenance": "unit quantity test reference",
                "available_from": "2020-01-01",
                "temperature_c": 25.0,
                "ndp_psi": 150.0,
                "permeate_flow": 6500.0,
                "permeate_flow_unit": "gpd",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_atm_pressure_mapping_cannot_succeed(tmp_path: Path) -> None:
    """A pressure mapping declared as atm must not return successful normalisation."""
    mapping = tmp_path / "map_atm.yaml"
    csv = tmp_path / "data.csv"
    ref = tmp_path / "ref.yaml"
    _write_mapping(mapping, pressure_unit="atm")
    _write_csv(csv, p_feed=14.0, p_perm=1.0, dp=1.5)
    _write_ref(ref)
    result = normalise_csv(csv, source_series_id="ATM", mapping_path=mapping, reference_path=ref)
    assert result["status"] == "invalid"
    val_errors = (result.get("validation") or {}).get("errors") or []
    err_text = " ".join(e.get("problem", "") for e in val_errors)
    err_text += " ".join(str(e) for e in result.get("errors") or [])
    lower = err_text.lower()
    assert "atm" in lower or "incompatible" in lower or "unsupported" in lower
    derived = result.get("derived_normalised_values")
    assert derived is None or not (
        (derived.get("independent_physical_normalisation") or {}).get("available")
    )


def test_bar_pressure_equivalent_to_psi(tmp_path: Path) -> None:
    """Supported bar inputs produce equivalent results to equivalent psi inputs."""
    ref = tmp_path / "ref.yaml"
    _write_ref(ref)

    psi_map = tmp_path / "map_psi.yaml"
    psi_csv = tmp_path / "psi.csv"
    _write_mapping(psi_map, pressure_unit="psi")
    _write_csv(psi_csv, p_feed=200.0, p_perm=15.0, dp=20.0)

    bar_map = tmp_path / "map_bar.yaml"
    bar_csv = tmp_path / "bar.csv"
    _write_mapping(bar_map, pressure_unit="bar")
    _write_csv(
        bar_csv,
        p_feed=200.0 / PSI_PER_BAR,
        p_perm=15.0 / PSI_PER_BAR,
        dp=20.0 / PSI_PER_BAR,
    )

    r_psi = normalise_csv(psi_csv, source_series_id="PSI", mapping_path=psi_map, reference_path=ref)
    r_bar = normalise_csv(bar_csv, source_series_id="BAR", mapping_path=bar_map, reference_path=ref)
    assert r_psi["status"] in {"ok", "ok_with_limitations"}
    assert r_bar["status"] in {"ok", "ok_with_limitations"}
    h_psi = r_psi["normalised"]["derived"]["independent_physical_normalisation"]
    h_bar = r_bar["normalised"]["derived"]["independent_physical_normalisation"]
    assert h_psi["available"] is True
    assert h_bar["available"] is True
    for a, b in zip(h_psi["relative_health"], h_bar["relative_health"], strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert float(a) == pytest.approx(float(b), rel=1e-5, abs=1e-5)
    for a, b in zip(h_psi["ndp_operating_psi"], h_bar["ndp_operating_psi"], strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert float(a) == pytest.approx(float(b), rel=1e-5, abs=1e-4)


def test_incompatible_temperature_unit_actionable_error(tmp_path: Path) -> None:
    mapping = tmp_path / "map_f.yaml"
    csv = tmp_path / "data.csv"
    ref = tmp_path / "ref.yaml"
    _write_mapping(mapping, temp_unit="deg_F")
    _write_csv(csv, p_feed=200.0, p_perm=15.0, dp=20.0)
    _write_ref(ref)
    result = normalise_csv(csv, source_series_id="TF", mapping_path=mapping, reference_path=ref)
    assert result["status"] == "invalid"
    errors = (result.get("validation") or {}).get("errors") or []
    parts = [f"{e.get('problem', '')} {e.get('actionable_correction', '')}" for e in errors]
    blob = " ".join(parts).lower()
    assert "temperature" in blob or "deg_f" in blob or "fahrenheit" in blob
    assert "actionable" in blob or "convert" in blob or "alias" in blob


def test_incompatible_concentration_conductivity_rejected(tmp_path: Path) -> None:
    mapping = tmp_path / "map_ec.yaml"
    csv = tmp_path / "data.csv"
    ref = tmp_path / "ref.yaml"
    _write_mapping(mapping, conc_unit="uS/cm")
    _write_csv(csv, p_feed=200.0, p_perm=15.0, dp=20.0)
    _write_ref(ref)
    result = normalise_csv(csv, source_series_id="EC", mapping_path=mapping, reference_path=ref)
    assert result["status"] == "invalid"
    errors = (result.get("validation") or {}).get("errors") or []
    blob = " ".join(e.get("problem", "") for e in errors).lower()
    assert "conductivity" in blob or "us/cm" in blob or "concentration" in blob
    assert "mg/l" in blob or "never silently" in blob or "incompatible" in blob


def test_ocwd_unresolved_flow_still_loads_for_diagnostic() -> None:
    """Unresolved OCWD flows remain adaptable; independent path stays blocked."""
    from aqua_fouling.application.validate import validate_csv

    csv = Path("data/extracted/CSV files/orange_county_ro_A01.csv")
    if not csv.is_file():
        pytest.skip("OCWD extracted CSV not available")
    result = validate_csv(csv, source_series_id="A01")
    # Unresolved OCWD flows are warnings; independent path stays unavailable.
    assert result["status"] in {"valid", "valid_with_warnings"}
    from aqua_fouling.application.normalise import normalise_csv as norm

    n = norm(csv, source_series_id="A01")
    ip = (n.get("derived_normalised_values") or {}).get("independent_physical_normalisation") or {}
    assert ip.get("available") is not True
