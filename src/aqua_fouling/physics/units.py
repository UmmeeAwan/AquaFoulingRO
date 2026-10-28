"""Supported unit conversions with documented constants."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Exact / conventional conversion factors (CODATA / standard engineering).
PSI_PER_BAR = 14.503773773
BAR_PER_PSI = 1.0 / PSI_PER_BAR
PA_PER_PSI = 6894.757293168
PSI_PER_PA = 1.0 / PA_PER_PSI
KPA_PER_PSI = PA_PER_PSI / 1000.0
GPD_PER_M3_PER_D = 264.172052358  # US gallon per day per m³/day
M3_PER_D_PER_GPD = 1.0 / GPD_PER_M3_PER_D
# 1 US gallon per minute → gallons per day
GPD_PER_GPM = 1440.0
GPM_PER_GPD = 1.0 / GPD_PER_GPM

# Canonical units used after adapter conversion.
CANONICAL_FLOW_UNIT = "gpd"
CANONICAL_PRESSURE_UNIT = "psi"
CANONICAL_TEMPERATURE_UNIT = "deg_c"
CANONICAL_CONCENTRATION_UNIT = "mg_l"

FLOW_ALIASES = {
    "gpd": "gpd",
    "gal/d": "gpd",
    "gal_per_day": "gpd",
    "gpm": "gpm",
    "gal/min": "gpm",
    "gal_per_min": "gpm",
    "m3/d": "m3_per_d",
    "m3_per_d": "m3_per_d",
    "m³/d": "m3_per_d",
}
PRESSURE_ALIASES = {
    "psi": "psi",
    "bar": "bar",
    "pa": "pa",
    "kpa": "kpa",
}
TEMPERATURE_ALIASES = {
    "deg_c": "deg_c",
    "degc": "deg_c",
    "c": "deg_c",
    "celsius": "deg_c",
    "°c": "deg_c",
}
CONCENTRATION_ALIASES = {
    "mg_l": "mg_l",
    "mg/l": "mg_l",
    "mg_per_l": "mg_l",
    "mgperl": "mg_l",
    "mg_l_nacl_eq": "mg_l",
    "mg_per_l_nacl_eq": "mg_l",
    "ppm": "mg_l",  # treated as mg/L NaCl-equivalent when explicitly mapped as concentration
}
CONDUCTIVITY_ALIASES = {
    "us_cm": "conductivity",
    "us/cm": "conductivity",
    "µs/cm": "conductivity",
    "microsiemens_per_cm": "conductivity",
    "ms_cm": "conductivity",
    "ms/cm": "conductivity",
    "unresolved_conductivity": "conductivity",
}


class UnitConversionError(ValueError):
    """Unsupported or incompatible unit conversion."""


def as_float_array(values: ArrayLike) -> NDArray[np.floating]:
    return np.atleast_1d(np.asarray(values, dtype=float))


def psi_to_bar(psi: ArrayLike) -> NDArray[np.floating]:
    return as_float_array(psi) * BAR_PER_PSI


def bar_to_psi(bar: ArrayLike) -> NDArray[np.floating]:
    return as_float_array(bar) * PSI_PER_BAR


def celsius_to_kelvin(temp_c: ArrayLike) -> NDArray[np.floating]:
    return as_float_array(temp_c) + 273.15


def normalize_unit_symbol(symbol: str) -> str:
    cleaned = symbol.strip().lower().replace(" ", "").replace("_", "")
    # Re-insert common separators after strip for alias tables that use _
    compact = symbol.strip().lower().replace(" ", "")
    for table in (
        FLOW_ALIASES,
        PRESSURE_ALIASES,
        TEMPERATURE_ALIASES,
        CONCENTRATION_ALIASES,
        CONDUCTIVITY_ALIASES,
    ):
        if compact in table:
            return table[compact]
        if cleaned in {k.replace("_", "").replace("/", "") for k in table}:
            for k, v in table.items():
                if k.replace("_", "").replace("/", "") == cleaned:
                    return v
    return compact


def is_temperature_unit(symbol: str) -> bool:
    return normalize_unit_symbol(symbol) == "deg_c"


def is_concentration_unit(symbol: str) -> bool:
    return normalize_unit_symbol(symbol) == "mg_l"


def is_conductivity_unit(symbol: str) -> bool:
    s = normalize_unit_symbol(symbol)
    return s == "conductivity" or "conduct" in s


def convert_flow(
    values: ArrayLike,
    *,
    from_unit: str,
    to_unit: str = CANONICAL_FLOW_UNIT,
) -> NDArray[np.floating]:
    """Convert flow between gpd, gpm, and m3/d."""
    src = normalize_unit_symbol(from_unit)
    dst = normalize_unit_symbol(to_unit)
    arr = as_float_array(values)
    if src == dst:
        return arr
    # Convert to gpd intermediate
    if src == "gpd":
        gpd = arr
    elif src == "gpm":
        gpd = arr * GPD_PER_GPM
    elif src == "m3_per_d":
        gpd = arr * GPD_PER_M3_PER_D
    else:
        msg = (
            f"Unsupported source flow unit '{from_unit}'. "
            f"Supported: gpd, gpm, m3/d. Convert externally or update the adapter."
        )
        raise UnitConversionError(msg)
    if dst == "gpd":
        return gpd
    if dst == "gpm":
        return gpd * GPM_PER_GPD
    if dst == "m3_per_d":
        return gpd * M3_PER_D_PER_GPD
    msg = f"Unsupported target flow unit '{to_unit}'. " f"Supported: gpd, gpm, m3/d."
    raise UnitConversionError(msg)


def convert_pressure(
    values: ArrayLike,
    *,
    from_unit: str,
    to_unit: str = CANONICAL_PRESSURE_UNIT,
) -> NDArray[np.floating]:
    """Convert pressure between psi, bar, Pa, kPa."""
    src = normalize_unit_symbol(from_unit)
    dst = normalize_unit_symbol(to_unit)
    arr = as_float_array(values)
    if src == dst:
        return arr
    if src == "psi":
        psi = arr
    elif src == "bar":
        psi = bar_to_psi(arr)
    elif src == "pa":
        psi = arr * PSI_PER_PA
    elif src == "kpa":
        psi = arr * PSI_PER_PA * 1000.0
    else:
        msg = (
            f"Unsupported source pressure unit '{from_unit}'. "
            f"Supported: psi, bar, Pa, kPa. Convert externally or update the adapter."
        )
        raise UnitConversionError(msg)
    if dst == "psi":
        return psi
    if dst == "bar":
        return psi_to_bar(psi)
    if dst == "pa":
        return psi * PA_PER_PSI
    if dst == "kpa":
        return psi * KPA_PER_PSI
    msg = f"Unsupported target pressure unit '{to_unit}'. Supported: psi, bar, Pa, kPa."
    raise UnitConversionError(msg)


def is_flow_unit(symbol: str) -> bool:
    return normalize_unit_symbol(symbol) in {"gpd", "gpm", "m3_per_d"}


def is_pressure_unit(symbol: str) -> bool:
    return normalize_unit_symbol(symbol) in {"psi", "bar", "pa", "kpa"}


def is_unresolved_unit(symbol: str) -> bool:
    s = normalize_unit_symbol(symbol)
    return s in {"unresolved_flow", "unresolved", "unknown", ""} or s.startswith("unresolved")


def convert_quantity(
    values: ArrayLike,
    *,
    from_unit: str,
    to_unit: str,
    quantity: str,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Convert a quantity or raise UnitConversionError with actionable context."""
    if is_unresolved_unit(from_unit) or is_unresolved_unit(to_unit):
        msg = (
            f"Cannot convert {quantity}: unit '{from_unit}' → '{to_unit}' is unresolved. "
            "Declare an explicit unit (e.g. gpd/gpm for flow, psi/bar for pressure) "
            "in the adapter or reference YAML."
        )
        raise UnitConversionError(msg)
    if quantity == "flow" or is_flow_unit(from_unit) or is_flow_unit(to_unit):
        out = convert_flow(values, from_unit=from_unit, to_unit=to_unit)
        return out, {
            "from_unit": normalize_unit_symbol(from_unit),
            "to_unit": normalize_unit_symbol(to_unit),
            "quantity": "flow",
        }
    if quantity == "pressure" or is_pressure_unit(from_unit) or is_pressure_unit(to_unit):
        out = convert_pressure(values, from_unit=from_unit, to_unit=to_unit)
        return out, {
            "from_unit": normalize_unit_symbol(from_unit),
            "to_unit": normalize_unit_symbol(to_unit),
            "quantity": "pressure",
        }
    if normalize_unit_symbol(from_unit) == normalize_unit_symbol(to_unit):
        return as_float_array(values), {
            "from_unit": normalize_unit_symbol(from_unit),
            "to_unit": normalize_unit_symbol(to_unit),
            "quantity": quantity,
        }
    msg = (
        f"No conversion rule for {quantity} units '{from_unit}' → '{to_unit}'. "
        "Update the adapter to a supported unit or convert the source data."
    )
    raise UnitConversionError(msg)
