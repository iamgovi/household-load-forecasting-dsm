
"""Common interface for controlled household-load forecasting."""

import math
from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """Base interface for all forecasting implementations."""

    model_name: str

    @abstractmethod
    def fit(self, training_data: pd.DataFrame) -> "Forecaster":
        """Fit using training data only."""

    @abstractmethod
    def predict(self, history: pd.DataFrame) -> list[float]:
        """Return exactly 24 hourly forecasts from the supplied history."""

    def validate_history(self, history: pd.DataFrame) -> None:
        """Validate historical inputs without imputing missing observations."""
        if not isinstance(history, pd.DataFrame):
            raise TypeError("History must be a pandas DataFrame.")

        required = {"timestamp", "meter_id", "load"}
        missing = required - set(history.columns)

        if missing:
            raise ValueError(f"Missing history columns: {sorted(missing)}")

        if len(history) < 168:
            raise ValueError(
                "At least 168 historical observations are required"
            )

        if history[["timestamp", "meter_id"]].isnull().any().any():
            raise ValueError("History contains missing required values")

        meters = history["meter_id"].astype(str).str.strip()

        if meters.eq("").any():
            raise ValueError("History contains empty meter identifiers")

        if meters.nunique() != 1:
            raise ValueError("History must contain exactly one meter")

        timestamps = pd.to_datetime(
            history["timestamp"], errors="raise"
        )

        if timestamps.isna().any():
            raise ValueError("History contains invalid timestamps")

        if timestamps.duplicated().any():
            raise ValueError("History contains duplicate timestamps")

        if not timestamps.is_monotonic_increasing:
            raise ValueError("History must be chronologically sorted")

        intervals = timestamps.diff().iloc[1:]

        if not intervals.eq(pd.Timedelta(hours=1)).all():
            raise ValueError(
                "History must contain consecutive hourly observations"
            )

        loads = pd.to_numeric(history["load"], errors="coerce")

        if loads.isna().any() or not all(
            math.isfinite(float(value)) for value in loads
        ):
            raise ValueError(
                "History contains missing or non-finite loads"
            )

    def validate_predictions(self, predictions) -> list[float]:
        """Require exactly 24 finite numerical forecasts."""
        try:
            values = [float(value) for value in predictions]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Predictions must contain numeric values"
            ) from exc

        if len(values) != 24:
            raise ValueError(
                "Forecaster must return exactly 24 predictions"
            )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Predictions must all be finite")

        return values