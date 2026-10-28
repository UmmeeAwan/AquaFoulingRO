"""Recovery ratio calculations with validity checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aqua_fouling.physics.units import as_float_array


class RecoveryError(ValueError):
    """Invalid recovery inputs or out-of-range recovery."""


def recovery_from_flows(
    permeate_flow: ArrayLike,
    feed_flow: ArrayLike,
) -> NDArray[np.floating]:
    """Y = Q_permeate / Q_feed. Requires feed_flow > 0 and 0 <= Y < 1."""
    qp = as_float_array(permeate_flow)
    qf = as_float_array(feed_flow)
    if np.any(~np.isfinite(qp) | ~np.isfinite(qf)):
        msg = "Recovery inputs must be finite."
        raise RecoveryError(msg)
    if np.any(qf <= 0):
        msg = "Feed flow must be strictly positive for recovery calculation."
        raise RecoveryError(msg)
    y = qp / qf
    if np.any(y < 0) or np.any(y >= 1.0):
        msg = (
            f"Recovery must satisfy 0 <= Y < 1; got min={float(np.min(y))}, "
            f"max={float(np.max(y))}."
        )
        raise RecoveryError(msg)
    return y


def average_feed_concentrate_factor(recovery: ArrayLike) -> NDArray[np.floating]:
    """Concentration polarisation / average factor: ln(1/(1-Y)) / Y.

    Used by FilmTec / ASTM-style feed-concentrate average concentration:
    C_fc = C_f * ln(1/(1-Y)) / Y
    """
    y = as_float_array(recovery)
    if np.any(~np.isfinite(y)) or np.any(y <= 0) or np.any(y >= 1.0):
        msg = "Recovery for C_fc factor must satisfy 0 < Y < 1."
        raise RecoveryError(msg)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.log(1.0 / (1.0 - y)) / y, dtype=float)
