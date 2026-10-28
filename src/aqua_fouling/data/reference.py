"""Reference-condition configuration and validation.

Never:
- Use the first row as healthy automatically
- Use the maximum over the entire dataset as the reference
- Treat dss=0 as proof of cleaning
- Use observations after a historical prediction cutoff

An explicitly selected post-cleaning window is allowed, but analysis that
depends on it may begin only after the window has completed.
"""

from __future__ import annotations

import math
from datetime import date
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from aqua_fouling.physics.units import is_flow_unit, is_unresolved_unit, normalize_unit_symbol


class ReferenceMode(str, Enum):
    DOCUMENTED = "documented"
    EXPLICIT_WINDOW = "explicit_window"


class ReferenceConfig(BaseModel):
    """Documented reference or explicitly selected stable window."""

    mode: ReferenceMode
    provenance: Annotated[str, Field(min_length=1)]
    available_from: date = Field(
        description=(
            "Earliest date at which this reference may be used. "
            "For an explicit window this is the day after the window end."
        )
    )
    prediction_cutoff: date | None = Field(
        default=None,
        description=(
            "Optional historical cutoff: observations on/after this date must "
            "not be used when establishing the reference."
        ),
    )
    # Documented fixed reference values (independent path)
    temperature_c: float | None = None
    ndp_psi: float | None = None
    permeate_flow: float | None = None
    permeate_flow_unit: str | None = None
    # Explicit window (dates inclusive)
    window_start: date | None = None
    window_end: date | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> ReferenceConfig:
        if self.mode == ReferenceMode.DOCUMENTED:
            missing = [
                name
                for name, value in (
                    ("temperature_c", self.temperature_c),
                    ("ndp_psi", self.ndp_psi),
                    ("permeate_flow", self.permeate_flow),
                    ("permeate_flow_unit", self.permeate_flow_unit),
                )
                if value is None
            ]
            if missing:
                msg = (
                    "Documented reference requires temperature_c, ndp_psi, "
                    f"permeate_flow and permeate_flow_unit; missing: {missing}."
                )
                raise ValueError(msg)
            assert self.temperature_c is not None
            assert self.ndp_psi is not None
            assert self.permeate_flow is not None
            assert self.permeate_flow_unit is not None
            for name, value, lo, hi in (
                ("temperature_c", self.temperature_c, -5.0, 50.0),
                ("ndp_psi", self.ndp_psi, 1e-6, 2000.0),
                ("permeate_flow", self.permeate_flow, 1e-9, 1e9),
            ):
                if not math.isfinite(value):
                    msg = (
                        f"Documented reference {name}={value} is not finite. "
                        "Provide a finite, physically valid value."
                    )
                    raise ValueError(msg)
                if not (lo < value < hi):
                    msg = (
                        f"Documented reference {name}={value} outside physically "
                        f"plausible open interval ({lo}, {hi}). Correct the reference YAML."
                    )
                    raise ValueError(msg)
            unit = self.permeate_flow_unit
            if is_unresolved_unit(unit) or not is_flow_unit(unit):
                msg = (
                    f"Documented reference permeate_flow_unit='{unit}' is unsupported. "
                    "Use an explicit convertible flow unit (gpd, gpm, or m3/d). "
                    "unresolved_flow is not allowed for reference-dependent health."
                )
                raise ValueError(msg)
            self.permeate_flow_unit = normalize_unit_symbol(unit)
        elif self.mode == ReferenceMode.EXPLICIT_WINDOW:
            if self.window_start is None or self.window_end is None:
                msg = "explicit_window reference requires window_start and window_end."
                raise ValueError(msg)
            if self.window_end < self.window_start:
                msg = "window_end must be on or after window_start."
                raise ValueError(msg)
            # Analysis may begin only after the window has completed.
            expected = date.fromordinal(self.window_end.toordinal() + 1)
            if self.available_from < expected:
                msg = (
                    f"available_from ({self.available_from}) must be on or after "
                    f"the day following window_end ({expected}). "
                    "Analysis using a post-cleaning window cannot begin before "
                    "the window has completed."
                )
                raise ValueError(msg)
            if self.prediction_cutoff is not None and self.window_end >= self.prediction_cutoff:
                msg = (
                    "Reference window must end before prediction_cutoff; "
                    "observations on/after the cutoff must not establish the reference."
                )
                raise ValueError(msg)
        return self

    def to_provenance_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "provenance": self.provenance,
            "available_from": self.available_from.isoformat(),
            "prediction_cutoff": (
                self.prediction_cutoff.isoformat() if self.prediction_cutoff else None
            ),
            "temperature_c": self.temperature_c,
            "ndp_psi": self.ndp_psi,
            "permeate_flow": self.permeate_flow,
            "permeate_flow_unit": self.permeate_flow_unit,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "notes": self.notes,
            "forbidden_practices_not_used": [
                "first_row_as_healthy",
                "dataset_maximum_as_reference",
                "dss_zero_as_clean_proof",
                "post_cutoff_reference_establishment",
                "automatic_reset_after_every_cleaning",
            ],
        }
