"""Net driving pressure (NDP).

Approximation (FilmTec / ASTM D4516-style, stage or device):

    NDP = P_feed - (ΔP_device / 2) - P_permeate - π_fc + π_permeate

where π_fc is the effective feed-concentrate osmotic pressure and
π_permeate is the permeate-side osmotic pressure (often ≈ 0 for high rejection).

Recovery enters via C_fc = C_f * ln(1/(1-Y))/Y when osmotic pressure is
computed from concentration (see ``physics.osmotic``).

This module computes NDP from explicit hydraulic and osmotic inputs.
It does not treat a source-calculated NDP column as validation of our
concentration correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aqua_fouling.physics.units import as_float_array

METHOD_ID = "filmtec_astm_style_ndp_v1"


class NDPError(ValueError):
    """Invalid NDP inputs."""


def net_driving_pressure(
    feed_pressure: ArrayLike,
    permeate_pressure: ArrayLike,
    device_pressure_drop: ArrayLike,
    osmotic_feed_concentrate: ArrayLike,
    osmotic_permeate: ArrayLike | float = 0.0,
) -> NDArray[np.floating]:
    """Compute NDP in the same pressure unit as the inputs (typically psi)."""
    pf = as_float_array(feed_pressure)
    pp = as_float_array(permeate_pressure)
    dp = as_float_array(device_pressure_drop)
    pi_fc = as_float_array(osmotic_feed_concentrate)
    pi_p = as_float_array(osmotic_permeate)
    arrays = (pf, pp, dp, pi_fc, pi_p)
    if any(np.any(~np.isfinite(a)) for a in arrays):
        msg = "All NDP inputs must be finite."
        raise NDPError(msg)
    if np.any(dp < 0):
        msg = "Device pressure drop must be non-negative."
        raise NDPError(msg)
    ndp = pf - (dp / 2.0) - pp - pi_fc + pi_p
    return np.asarray(ndp, dtype=float)


def require_positive_ndp(ndp: ArrayLike) -> NDArray[np.floating]:
    """Return NDP array or raise if any value is non-positive."""
    values = as_float_array(ndp)
    if np.any(~np.isfinite(values)):
        msg = "NDP contains non-finite values."
        raise NDPError(msg)
    if np.any(values <= 0):
        msg = (
            f"Net driving pressure must be strictly positive; "
            f"got min={float(np.nanmin(values))}."
        )
        raise NDPError(msg)
    return values
