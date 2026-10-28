"""Membrane temperature correction — FilmTec/Dow exponential convention.

Convention name
---------------
``filmtec_exponential_tcf_v1``

Definition used here
--------------------
TCF(T) := permeability(T) / permeability(T_ref) with T_ref = 25 °C, so
TCF(25 °C) = 1.

Source
------
FILMTEC Membranes, System Operation: Record Keeping, Form No. 609-02084-0905
(Dow Liquid Separations / FilmTec technical manual excerpt):

  TCF = exp[2640 * (1/298 - 1/(273 + T))]   for T >= 25 °C
  TCF = exp[3020 * (1/298 - 1/(273 + T))]   for T <= 25 °C

where T is temperature in °C. At T = 25 °C both expressions yield 1.

Do not assume a dataset ``tcf_water`` column uses this convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aqua_fouling.physics.units import as_float_array

CONVENTION_ID = "filmtec_exponential_tcf_v1"
T_REF_C = 25.0
SOURCE_URL = "https://www.bigbrandwater.com/assets/library/filmtec/filmtec-record-keeping.pdf"
SOURCE_FORM = "FILMTEC Form No. 609-02084-0905"
SOURCE_EQUATION = (
    "TCF=EXP[2640*(1/298-1/(273+T))] (T>=25°C); " "TCF=EXP[3020*(1/298-1/(273+T))] (T<=25°C)"
)


def filmtec_temperature_correction_factor(
    temp_c: ArrayLike,
) -> NDArray[np.floating]:
    """Return FilmTec exponential TCF relative to 25 °C."""
    t = as_float_array(temp_c)
    # U = 2640 above/equal 25 °C, 3020 below/equal 25 °C; at 25 either is 1.
    u = np.where(t >= T_REF_C, 2640.0, 3020.0)
    with np.errstate(over="ignore", invalid="ignore"):
        tcf = np.exp(u * (1.0 / 298.0 - 1.0 / (273.0 + t)))
    return np.asarray(tcf, dtype=float)


def tcf_at_reference_temperature() -> float:
    """TCF at the convention reference temperature must be 1."""
    return float(filmtec_temperature_correction_factor(T_REF_C)[0])
