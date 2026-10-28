"""Physics helpers for Phase 2 independent normalisation."""

from aqua_fouling.physics.ndp import net_driving_pressure, require_positive_ndp
from aqua_fouling.physics.normalisation import (
    NORMALISATION_METHOD_ID,
    NORMALISATION_VERSION,
    method_metadata,
    normalised_permeate_flow,
    normalised_permeate_flow_from_temperature,
    relative_health,
)
from aqua_fouling.physics.osmotic import (
    feed_concentrate_concentration_mg_l,
    osmotic_pressure_filmtec_bar,
    osmotic_pressure_filmtec_psi,
)
from aqua_fouling.physics.recovery import recovery_from_flows
from aqua_fouling.physics.temperature import (
    CONVENTION_ID as TCF_CONVENTION_ID,
)
from aqua_fouling.physics.temperature import (
    filmtec_temperature_correction_factor,
)
from aqua_fouling.physics.units import bar_to_psi, psi_to_bar

__all__ = [
    "NORMALISATION_METHOD_ID",
    "NORMALISATION_VERSION",
    "TCF_CONVENTION_ID",
    "bar_to_psi",
    "feed_concentrate_concentration_mg_l",
    "filmtec_temperature_correction_factor",
    "method_metadata",
    "net_driving_pressure",
    "normalised_permeate_flow",
    "normalised_permeate_flow_from_temperature",
    "osmotic_pressure_filmtec_bar",
    "osmotic_pressure_filmtec_psi",
    "psi_to_bar",
    "recovery_from_flows",
    "relative_health",
    "require_positive_ndp",
]
