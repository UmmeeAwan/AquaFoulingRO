"""Brackish / NaCl-equivalent demonstration domain for Phase 3 synthetic data.

Compatible with implemented physics:
- filmtec_exponential_tcf_v1
- filmtec_cfc_bar_lt_20000_v1 (validated only for C_fc < 20000 mg/L)

This is NOT a validated seawater model. Validity is checked at effective
concentrate-side conditions (C_fc), not only at feed concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DomainViolationError(ValueError):
    """Unsupported physical condition for the demonstration domain."""


@dataclass(frozen=True)
class BrackishDomain:
    """Documented domain limits for the synthetic benchmark."""

    id: str = "brackish_nacl_demo_v1"
    max_c_fc_mg_l: float = 20_000.0
    min_temp_c: float = 5.0
    max_temp_c: float = 40.0
    min_recovery: float = 0.05
    max_recovery: float = 0.90
    not_seawater: bool = True

    def validate_operating_point(
        self,
        *,
        feed_concentration_mg_l: float,
        recovery: float,
        temp_c: float,
        c_fc_mg_l: float,
    ) -> None:
        """Reject unsupported conditions with actionable messages."""
        if not (self.min_temp_c <= temp_c <= self.max_temp_c):
            msg = (
                f"Temperature {temp_c} °C outside demonstration domain "
                f"[{self.min_temp_c}, {self.max_temp_c}]. "
                "Adjust scenario temperature_c_range."
            )
            raise DomainViolationError(msg)
        if not (self.min_recovery < recovery < self.max_recovery):
            msg = (
                f"Recovery {recovery} outside open interval "
                f"({self.min_recovery}, {self.max_recovery}). "
                "Choose a physically plausible recovery."
            )
            raise DomainViolationError(msg)
        if feed_concentration_mg_l < 0:
            msg = "Feed concentration must be non-negative (mg/L as NaCl-equivalent)."
            raise DomainViolationError(msg)
        if c_fc_mg_l >= self.max_c_fc_mg_l:
            msg = (
                f"Effective concentrate-side C_fc={c_fc_mg_l:.1f} mg/L exceeds "
                f"domain limit {self.max_c_fc_mg_l:g} mg/L "
                "(FilmTec filmtec_cfc_bar_lt_20000_v1). "
                "This demonstration domain is not a validated seawater model; "
                "reduce feed concentration and/or recovery."
            )
            raise DomainViolationError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "max_c_fc_mg_l": self.max_c_fc_mg_l,
            "temperature_c_bounds": [self.min_temp_c, self.max_temp_c],
            "recovery_bounds_open": [self.min_recovery, self.max_recovery],
            "not_seawater": self.not_seawater,
            "validity_checked_at": "effective_concentrate_side_C_fc",
            "literature_supported_methods": [
                "filmtec_exponential_tcf_v1",
                "filmtec_cfc_bar_lt_20000_v1",
                "filmtec_astm_style_ndp_v1",
            ],
        }
