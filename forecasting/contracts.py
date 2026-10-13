"""Shared prediction-output validation for forecasting experiments."""

import numpy as np
import pandas as pd

from forecasting.config import HORIZON, PREDICTION_COLUMNS


def validate_predictions(
    predictions: pd.DataFrame,
    require_complete_horizon: bool = True,
) -> None:
    """Validate predictions against the canonical forecasting contract.

    Raises ValueError when the output is invalid.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Predictions must be a pandas DataFrame.")

    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    if predictions.empty:
        raise ValueError("Predictions must not be empty.")

    data = predictions[PREDICTION_COLUMNS].copy()

    if data.isnull().any().any():
        raise ValueError("Prediction output contains missing values.")

    # Validate required identifiers.
    identifier_columns = ("model", "city", "meter", "split", "run_id")

    for column in identifier_columns:
        identifiers = data[column].astype(str).str.strip()

        if identifiers.eq("").any():
            raise ValueError(f"{column} must not contain empty values.")

    # Validate timestamps.
    data["forecast_origin"] = pd.to_datetime(
        data["forecast_origin"], errors="raise"
    )
    data["target_timestamp"] = pd.to_datetime(
        data["target_timestamp"], errors="raise"
    )

    # Forecast origins and target timestamps must use a consistent
    # timezone convention throughout the prediction output.
    for column in ("forecast_origin", "target_timestamp"):
        timezone_flags = data[column].map(
            lambda timestamp: timestamp.tzinfo is not None
        )

        if timezone_flags.nunique() > 1:
            raise ValueError(
                f"{column} must use a consistent timezone convention."
            )

    # Validate forecast horizons.
    data["horizon"] = pd.to_numeric(data["horizon"], errors="raise")

    if not np.isfinite(data["horizon"].to_numpy(dtype=float)).all():
        raise ValueError("Horizon values must be finite.")

    if not data["horizon"].between(1, HORIZON).all():
        raise ValueError(f"Horizon must be between 1 and {HORIZON}.")

    if not (data["horizon"] % 1 == 0).all():
        raise ValueError("Horizon values must be integers.")

    data["horizon"] = data["horizon"].astype(int)

    # Validate numeric prediction fields and reject NaN/inf values.
    for column in ("actual", "prediction", "inference_time_ms"):
        data[column] = pd.to_numeric(data[column], errors="raise")

        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"{column} must contain only finite values."
            )

    if not data["inference_time_ms"].ge(0).all():
        raise ValueError("Inference time cannot be negative.")

    # Each target timestamp must correspond to its forecast origin
    # and one-based hourly forecast horizon.
    expected_timestamps = (
        data["forecast_origin"]
        + pd.to_timedelta(data["horizon"], unit="h")
    )

    if not data["target_timestamp"].eq(expected_timestamps).all():
        raise ValueError(
            "Target timestamps must equal forecast origin plus horizon hours."
        )

    # Prevent duplicate predictions for the same forecast and horizon.
    key_columns = [
        "model",
        "city",
        "meter",
        "split",
        "forecast_origin",
        "horizon",
        "run_id",
    ]

    if data.duplicated(key_columns).any():
        raise ValueError("Duplicate prediction rows found.")

    # Every forecast must contain exactly one prediction for each
    # horizon from 1 through HORIZON.
    if require_complete_horizon:
        group_columns = [
            "model",
            "city",
            "meter",
            "split",
            "forecast_origin",
            "run_id",
        ]

        for _, group in data.groupby(group_columns, dropna=False):
            horizons = sorted(group["horizon"].tolist())

            if horizons != list(range(1, HORIZON + 1)):
                raise ValueError(
                    f"Each forecast must contain horizons 1 through {HORIZON}."
                )