"""Ensure API/CLI numeric payloads never contain NaN or Infinity."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe_number(value: Any) -> float | None:
    """Convert a numeric value to float, or null if non-finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return json_safe_number(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
