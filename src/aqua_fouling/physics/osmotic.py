"""Osmotic-pressure methods with explicit scope and validity checks.

Method implemented
------------------
``filmtec_cfc_bar_lt_20000_v1`` — FilmTec practical approximation for
feed-concentrate mixture concentration C_fc < 20 000 mg/L:

    π (bar) = C_fc * (T + 320) / 491_000

with T in °C and C_fc in mg/L (as used in the FilmTec worked example).

Source: FILMTEC Form No. 609-02084-0905 (Record Keeping / normalisation excerpt).
URL: https://www.bigbrandwater.com/assets/library/filmtec/filmtec-record-keeping.pdf

Scope / restrictions
--------------------
- Documented for C_fc < 20 000 mg/L in that excerpt.
- Requires a concentration input (mg/L). Conductivity alone is insufficient.
- Not labelled as universally valid for seawater or wastewater without
  appropriate concentration definition and validation.
- The high-salinity (>20 000 mg/L) FilmTec branch is intentionally not
  implemented here because OCR/parsing of that branch was ambiguous;
  concentrations above the limit raise a validity error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aqua_fouling.physics.recovery import average_feed_concentrate_factor
from aqua_fouling.physics.units import as_float_array, bar_to_psi

METHOD_ID = "filmtec_cfc_bar_lt_20000_v1"
MAX_CFC_MG_PER_L = 20_000.0
SOURCE_URL = "https://www.bigbrandwater.com/assets/library/filmtec/filmtec-record-keeping.pdf"
SOURCE_FORM = "FILMTEC Form No. 609-02084-0905"
SOURCE_EQUATION = "pi_bar = C_fc * (T_C + 320) / 491000  for C_fc < 20000 mg/L"


class OsmoticPressureError(ValueError):
    """Invalid osmotic-pressure inputs or out-of-scope concentrations."""


def feed_concentrate_concentration_mg_l(
    feed_concentration_mg_l: ArrayLike,
    recovery: ArrayLike,
) -> NDArray[np.floating]:
    """C_fc = C_f * ln(1/(1-Y)) / Y (FilmTec / ASTM-style average)."""
    cf = as_float_array(feed_concentration_mg_l)
    if np.any(~np.isfinite(cf)) or np.any(cf < 0):
        msg = "Feed concentration must be finite and non-negative (mg/L)."
        raise OsmoticPressureError(msg)
    factor = average_feed_concentrate_factor(recovery)
    return cf * factor


def osmotic_pressure_filmtec_bar(
    c_fc_mg_l: ArrayLike,
    temp_c: ArrayLike,
) -> NDArray[np.floating]:
    """FilmTec π in bar for C_fc < 20 000 mg/L."""
    c = as_float_array(c_fc_mg_l)
    t = as_float_array(temp_c)
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(t)):
        msg = "C_fc and temperature must be finite."
        raise OsmoticPressureError(msg)
    if np.any(c < 0):
        msg = "C_fc must be non-negative."
        raise OsmoticPressureError(msg)
    if np.any(c >= MAX_CFC_MG_PER_L):
        msg = (
            f"Method {METHOD_ID} is only validated for C_fc < {MAX_CFC_MG_PER_L:g} mg/L "
            f"(FilmTec Form 609-02084). Got max C_fc={float(np.max(c)):.1f}. "
            "Do not apply this dilute/practical branch to seawater or high-TDS wastewater."
        )
        raise OsmoticPressureError(msg)
    return c * (t + 320.0) / 491_000.0


def osmotic_pressure_filmtec_psi(
    c_fc_mg_l: ArrayLike,
    temp_c: ArrayLike,
) -> NDArray[np.floating]:
    """FilmTec π converted to psi."""
    return bar_to_psi(osmotic_pressure_filmtec_bar(c_fc_mg_l, temp_c))
