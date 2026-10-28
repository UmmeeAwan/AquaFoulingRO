"""Independent physical normalisation of permeate flow / permeability.

Convention
----------
``filmtec_exponential_tcf_v1`` temperature correction with

    Q_normalised = Q_measured * (NDP_reference / NDP_operating)
                             * (TCF_reference / TCF_operating)

where TCF(T) = permeability(T)/permeability(T_ref) and T_ref = 25 °C for the
FilmTec exponential convention implemented in ``physics.temperature``.

This matches the structure of FilmTec Form 609-02084 and ASTM D4516-style
standardisation (standard/reference over operating).

Important
---------
- Do **not** assume a dataset ``tcf_water`` column uses this TCF convention.
- Do **not** apply temperature correction twice (e.g. multiply by both our
  FilmTec TCF and a source-calculated TCF factor in the same path).
- Source-calculated specific flux / NDP comparisons belong in a separately
  labelled diagnostic path, not in this independent normalisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aqua_fouling.physics.ndp import NDPError, require_positive_ndp
from aqua_fouling.physics.temperature import (
    CONVENTION_ID,
    filmtec_temperature_correction_factor,
)
from aqua_fouling.physics.units import as_float_array

NORMALISATION_METHOD_ID = "independent_q_norm_filmtec_tcf_v1"
NORMALISATION_VERSION = "2.0.0"


class NormalisationError(ValueError):
    """Invalid normalisation inputs."""


def normalised_permeate_flow(
    q_measured: ArrayLike,
    ndp_operating: ArrayLike,
    ndp_reference: ArrayLike | float,
    tcf_operating: ArrayLike,
    tcf_reference: ArrayLike | float,
) -> NDArray[np.floating]:
    """Q_n = Q_m * NDP_ref/NDP_op * TCF_ref/TCF_op."""
    qm = as_float_array(q_measured)
    ndp_op = require_positive_ndp(ndp_operating)
    ndp_ref = require_positive_ndp(ndp_reference)
    tcf_op = as_float_array(tcf_operating)
    tcf_ref = as_float_array(tcf_reference)
    if np.any(~np.isfinite(qm)):
        msg = "Measured flow must be finite."
        raise NormalisationError(msg)
    if np.any(~np.isfinite(tcf_op)) or np.any(tcf_op <= 0):
        msg = "Operating TCF must be finite and strictly positive."
        raise NormalisationError(msg)
    if np.any(~np.isfinite(tcf_ref)) or np.any(tcf_ref <= 0):
        msg = "Reference TCF must be finite and strictly positive."
        raise NormalisationError(msg)
    return qm * (ndp_ref / ndp_op) * (tcf_ref / tcf_op)


def normalised_permeate_flow_from_temperature(
    q_measured: ArrayLike,
    ndp_operating: ArrayLike,
    ndp_reference: ArrayLike | float,
    temp_operating_c: ArrayLike,
    temp_reference_c: ArrayLike | float,
) -> NDArray[np.floating]:
    """Normalise using FilmTec TCF computed from temperatures (no source TCF)."""
    tcf_op = filmtec_temperature_correction_factor(temp_operating_c)
    tcf_ref = filmtec_temperature_correction_factor(temp_reference_c)
    return normalised_permeate_flow(
        q_measured,
        ndp_operating,
        ndp_reference,
        tcf_op,
        tcf_ref,
    )


def relative_health(
    normalised_value: ArrayLike,
    reference_normalised_value: ArrayLike | float,
) -> NDArray[np.floating]:
    """Health = Q_norm / Q_norm_ref against a fixed explicit reference."""
    qn = as_float_array(normalised_value)
    qref = as_float_array(reference_normalised_value)
    if np.any(~np.isfinite(qn)) or np.any(~np.isfinite(qref)):
        msg = "Health inputs must be finite."
        raise NormalisationError(msg)
    if np.any(qref == 0):
        msg = "Reference normalised value must be non-zero."
        raise NormalisationError(msg)
    return qn / qref


def method_metadata() -> dict[str, Any]:
    return {
        "normalisation_method_id": NORMALISATION_METHOD_ID,
        "normalisation_version": NORMALISATION_VERSION,
        "temperature_convention_id": CONVENTION_ID,
        "formula": ("Q_normalised = Q_measured * NDP_ref/NDP_op * TCF_ref/TCF_op"),
        "notes": [
            "Independent path does not use source tcf_water as TCF.",
            "Source-calculated NDP/s3sf comparisons are diagnostic only.",
        ],
    }


# Re-export for callers that need NDPError together with normalisation.
__all__ = [
    "NORMALISATION_METHOD_ID",
    "NORMALISATION_VERSION",
    "NDPError",
    "NormalisationError",
    "method_metadata",
    "normalised_permeate_flow",
    "normalised_permeate_flow_from_temperature",
    "relative_health",
]
