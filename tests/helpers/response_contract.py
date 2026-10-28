"""Helpers for the pre-release public response contract."""

from __future__ import annotations

from typing import Any


def forecast_result(body: dict[str, Any]) -> dict[str, Any]:
    return body.get("result") or {}


def forecast_points(body: dict[str, Any]) -> list[dict[str, Any]]:
    return list(forecast_result(body).get("forecast") or [])


def normalised_observations(body: dict[str, Any]) -> list[dict[str, Any]]:
    return list((body.get("normalised_data") or {}).get("observations") or [])


def n_normalised(body: dict[str, Any]) -> int:
    nd = body.get("normalised_data") or {}
    if "n_observations" in nd:
        return int(nd["n_observations"])
    return len(normalised_observations(body))


def model_version(body: dict[str, Any]) -> str | None:
    return ((body.get("metadata") or {}).get("model") or {}).get("model_version")


def resolved_mapping(body: dict[str, Any]) -> list[dict[str, Any]]:
    if "mapping" in body and isinstance(body["mapping"], dict):
        return list(body["mapping"].get("resolved") or [])
    return list(((body.get("metadata") or {}).get("mapping") or {}).get("resolved") or [])


def mapping_source(body: dict[str, Any]) -> str | None:
    if "mapping" in body and isinstance(body["mapping"], dict):
        return body["mapping"].get("source")
    return ((body.get("metadata") or {}).get("mapping") or {}).get("source")


def error_code(body: dict[str, Any]) -> str | None:
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("code")
    return body.get("code")


def batch_counts(body: dict[str, Any]) -> dict[str, Any]:
    return (body.get("summary") or {}).get("counts") or {}


def indep_norm(body: dict[str, Any]) -> dict[str, Any]:
    derived = (body.get("normalised") or {}).get("derived") or {}
    return dict(derived.get("independent_physical_normalisation") or {})


def forecast_core(body: dict[str, Any]) -> dict[str, Any]:
    """Comparable forecast fields for equivalence tests."""
    res = forecast_result(body)
    return {
        "status": body["status"],
        "source_series_id": res.get("source_series_id") or body.get("source_series_id"),
        "crossing_status": res.get("crossing_status"),
        "days_to_threshold": res.get("days_to_threshold"),
        "horizon_days": res.get("horizon_days"),
        "n_normalised_observations": n_normalised(body),
        "model_version": model_version(body),
        "threshold": (res.get("threshold") or {}).get("relative_health"),
        "forecast": [
            {
                "date": p["date"],
                "relative_health_point": p["relative_health_point"],
                "latent_health_interval": p["latent_health_interval"],
                "future_observation_interval": p["future_observation_interval"],
            }
            for p in forecast_points(body)
        ],
    }
