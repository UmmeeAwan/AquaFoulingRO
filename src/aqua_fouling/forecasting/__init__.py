"""Forecasting package — baselines and trained local-linear-trend Kalman."""

from aqua_fouling.forecasting.baselines import (
    LinearTrendForecaster,
    PersistenceForecaster,
    get_baseline,
)
from aqua_fouling.forecasting.interface import ForecastRequest, ForecastResult
from aqua_fouling.forecasting.trained_llt import TrainedLLTForecaster, load_trained_forecaster

__all__ = [
    "ForecastRequest",
    "ForecastResult",
    "LinearTrendForecaster",
    "PersistenceForecaster",
    "TrainedLLTForecaster",
    "get_baseline",
    "load_trained_forecaster",
]
