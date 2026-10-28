"""Independent scientific tests for the physics module."""

from __future__ import annotations

import pytest

from aqua_fouling.physics.ndp import NDPError, net_driving_pressure, require_positive_ndp
from aqua_fouling.physics.normalisation import (
    NormalisationError,
    normalised_permeate_flow,
    normalised_permeate_flow_from_temperature,
    relative_health,
)
from aqua_fouling.physics.osmotic import (
    OsmoticPressureError,
    feed_concentrate_concentration_mg_l,
    osmotic_pressure_filmtec_bar,
)
from aqua_fouling.physics.recovery import RecoveryError, recovery_from_flows
from aqua_fouling.physics.temperature import (
    filmtec_temperature_correction_factor,
    tcf_at_reference_temperature,
)
from aqua_fouling.physics.units import BAR_PER_PSI, PSI_PER_BAR, bar_to_psi, psi_to_bar


def test_unit_conversion_constants() -> None:
    assert pytest.approx(14.503773773) == PSI_PER_BAR
    assert bar_to_psi(1.0) == pytest.approx(PSI_PER_BAR)
    assert psi_to_bar(PSI_PER_BAR) == pytest.approx(1.0)
    assert pytest.approx(1.0) == BAR_PER_PSI * PSI_PER_BAR


def test_tcf_at_reference_temperature_is_one() -> None:
    assert tcf_at_reference_temperature() == pytest.approx(1.0, abs=1e-12)
    assert float(filmtec_temperature_correction_factor(25.0)[0]) == pytest.approx(1.0)


def test_tcf_direction_colder_reduces_tcf() -> None:
    # Colder water → lower permeability → TCF < 1 for FilmTec convention.
    tcf_10 = float(filmtec_temperature_correction_factor(10.0)[0])
    tcf_25 = float(filmtec_temperature_correction_factor(25.0)[0])
    tcf_30 = float(filmtec_temperature_correction_factor(30.0)[0])
    assert tcf_10 < tcf_25 < tcf_30


def test_filmtec_worked_example_tcf_and_normalisation() -> None:
    """FILMTEC Form 609-02084-0905 worked example (provenance: fetched PDF).

    Pressures in bar as published; flows taken from the gpm column of the same
    example (operating 559 gpm → normalised ≈636 gpm). The PDF narrative also
    lists 127 m³/h for operating flow; using 127 with the published NDP/TCF
    ratios does not reproduce 636, so the gpm figures are used for the
    arithmetic check (consistent units within the worked example).
    """
    tcf_s = float(filmtec_temperature_correction_factor(15.0)[0])
    tcf_o = float(filmtec_temperature_correction_factor(10.0)[0])
    assert tcf_s == pytest.approx(0.70, rel=0.02)
    assert tcf_o == pytest.approx(0.58, rel=0.02)

    ndp_s = 25.0 - 1.5 - 1.0 - 2.5  # 20 bar
    ndp_o = 28.0 - 2.0 - 2.0 - 2.72  # 21.28 bar
    q_o_gpm = 559.0
    q_s = float(normalised_permeate_flow(q_o_gpm, ndp_o, ndp_s, tcf_o, tcf_s)[0])
    assert q_s == pytest.approx(636.0, rel=0.02)


def test_filmtec_osmotic_example_cfc() -> None:
    # Cfc=3671 mg/L, T=15°C → π≈2.5 bar in the worked example.
    c_fc = float(feed_concentrate_concentration_mg_l(1986.0, 0.75)[0])
    assert c_fc == pytest.approx(3671.0, rel=0.01)
    pi = float(osmotic_pressure_filmtec_bar(c_fc, 15.0)[0])
    assert pi == pytest.approx(2.5, rel=0.02)


def test_constant_permeability_invariance() -> None:
    """Same permeability under changed operating NDP/T should normalise to same Q_ref."""
    # Construct operating points from a fixed permeability model:
    # Q = K * NDP * TCF  with K constant; normalisation should recover K*NDP_ref*TCF_ref.
    k = 2.0
    ndp_ref, t_ref = 100.0, 25.0
    tcf_ref = float(filmtec_temperature_correction_factor(t_ref)[0])
    q_ref = k * ndp_ref * tcf_ref
    for ndp_op, t_op in [(80.0, 20.0), (120.0, 30.0), (90.0, 15.0)]:
        tcf_op = float(filmtec_temperature_correction_factor(t_op)[0])
        q_meas = k * ndp_op * tcf_op
        q_n = float(
            normalised_permeate_flow_from_temperature(q_meas, ndp_op, ndp_ref, t_op, t_ref)[0]
        )
        assert q_n == pytest.approx(q_ref, rel=1e-9)


def test_deterioration_preserved_after_correction() -> None:
    ndp_ref, t_ref = 100.0, 25.0
    # Healthy then fouled (lower Q at same NDP/T)
    q_healthy = 200.0
    q_fouled = 150.0
    qn_h = float(
        normalised_permeate_flow_from_temperature(q_healthy, 100.0, ndp_ref, 25.0, t_ref)[0]
    )
    qn_f = float(
        normalised_permeate_flow_from_temperature(q_fouled, 100.0, ndp_ref, 25.0, t_ref)[0]
    )
    assert qn_f < qn_h
    assert float(relative_health(qn_f, qn_h)[0]) == pytest.approx(0.75)


def test_permeate_backpressure_reduces_ndp() -> None:
    ndp_low_back = float(net_driving_pressure(200, 10, 20, 30, 0)[0])
    ndp_high_back = float(net_driving_pressure(200, 40, 20, 30, 0)[0])
    assert ndp_high_back < ndp_low_back


def test_invalid_nonpositive_ndp() -> None:
    with pytest.raises(NDPError):
        require_positive_ndp(0.0)
    with pytest.raises(NDPError):
        require_positive_ndp(-1.0)
    with pytest.raises((NormalisationError, NDPError)):
        normalised_permeate_flow(10.0, 0.0, 100.0, 1.0, 1.0)


def test_invalid_recovery_and_osmotic_range() -> None:
    with pytest.raises(RecoveryError):
        recovery_from_flows(10.0, 0.0)
    with pytest.raises(RecoveryError):
        recovery_from_flows(12.0, 10.0)  # Y > 1
    with pytest.raises(OsmoticPressureError):
        osmotic_pressure_filmtec_bar(25_000.0, 25.0)


def test_health_requires_nonzero_reference() -> None:
    with pytest.raises(NormalisationError):
        relative_health(1.0, 0.0)
