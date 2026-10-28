"""Synthetic RO simulation for Phase 3 primary benchmark."""

from aqua_fouling.simulation.domain import BrackishDomain, DomainViolationError
from aqua_fouling.simulation.simulator import generate_synthetic_benchmark

__all__ = [
    "BrackishDomain",
    "DomainViolationError",
    "generate_synthetic_benchmark",
]
