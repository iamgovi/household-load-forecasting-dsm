"""Tests for the shared prediction-output contract."""

import pandas as pd
import pytest

from forecasting.config import PREDICTION_COLUMNS
from forecasting.contracts import validate_predictions


def make_predictions():
    origin = pd.Timestamp("2025-01-10 00:00:00")

    rows = []
    for horizon in range(1, 25):
        row = {
            "model": "test_model",
            "city": "test_city",
            "meter": "test_meter",
            "split": "validation",
            "forecast_origin": origin,
            "horizon": horizon,
            "target_timestamp": origin + pd.Timedelta(hours=horizon),
            "actual": 1.0,
            "prediction": 1.1,
            "inference_time_ms": 2.0,
            "run_id": "test_run",
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def test_valid_complete_forecast_passes():
    validate_predictions(make_predictions())


def test_incomplete_forecast_fails():
    predictions = make_predictions().iloc[:-1]

    with pytest.raises(ValueError, match="horizons 1 through 24"):
        validate_predictions(predictions)


def test_incorrect_target_timestamp_fails():
    predictions = make_predictions()
    predictions.loc[0, "target_timestamp"] += pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="Target timestamps"):
        validate_predictions(predictions)


def test_missing_column_fails():
    predictions = make_predictions().drop(columns=["prediction"])

    with pytest.raises(ValueError, match="Missing prediction columns"):
        validate_predictions(predictions)


def test_partial_horizon_can_be_allowed():
    predictions = make_predictions().iloc[:5]

    validate_predictions(predictions, require_complete_horizon=False)
