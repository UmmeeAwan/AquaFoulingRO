"""Reference configuration and segment behaviour tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from aqua_fouling.application.segments import contiguous_segments, flag_abrupt_changes
from aqua_fouling.data.reference import ReferenceConfig, ReferenceMode


def test_reference_rejects_window_before_available() -> None:
    with pytest.raises(ValidationError):
        ReferenceConfig(
            mode=ReferenceMode.EXPLICIT_WINDOW,
            provenance="test",
            available_from=date(2020, 1, 5),
            window_start=date(2020, 1, 1),
            window_end=date(2020, 1, 10),
        )


def test_reference_respects_prediction_cutoff() -> None:
    with pytest.raises(ValidationError):
        ReferenceConfig(
            mode=ReferenceMode.EXPLICIT_WINDOW,
            provenance="test",
            available_from=date(2020, 2, 2),
            prediction_cutoff=date(2020, 2, 1),
            window_start=date(2020, 1, 1),
            window_end=date(2020, 2, 1),
        )


def test_documented_reference_ok() -> None:
    ref = ReferenceConfig(
        mode=ReferenceMode.DOCUMENTED,
        provenance="start-up report",
        available_from=date(2019, 1, 15),
        temperature_c=25.0,
        ndp_psi=100.0,
        permeate_flow=300.0,
        permeate_flow_unit="gpm",
    )
    assert "first_row_as_healthy" in ref.to_provenance_dict()["forbidden_practices_not_used"]


def test_gap_and_cip_split_segments() -> None:
    dates = pd.Series(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-05",  # gap
            "2020-01-06",
            "2020-01-07",
        ]
    )
    cip = pd.Series([0, 0, 0, 1, 0])
    segments, audit = contiguous_segments(dates, cip=cip)
    assert len(segments) >= 2
    assert any(a["reason"] == "calendar_gap" for a in audit)
    assert any(a["reason"] == "cip_flag_present" for a in audit)


def test_abrupt_change_flagged_not_auto_cleaning() -> None:
    values = pd.Series([100.0, 100.0, 40.0, 40.0])
    flags = flag_abrupt_changes(values, relative_threshold=0.3)
    assert flags
    assert flags[0]["action"] == "flag_only_not_auto_cleaning"
