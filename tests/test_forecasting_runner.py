import numpy as np
import pandas as pd
import pytest

from forecasting.baselines import SeasonalNaiveForecaster
from forecasting.runner import forecast_one_origin


def make_frame(start, count, meter="m1"):
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=count, freq="h"),
        "meter_id": [meter] * count,
        "load": np.arange(count, dtype=float) + 1,
        "location": ["test-city"] * count,
    })


def test_runner_returns_valid_complete_forecast():
    training = make_frame("2025-01-01", 240)
    history = make_frame("2025-01-11", 168)
    origin = history.timestamp.iloc[-1]
    future = make_frame(origin + pd.Timedelta(hours=1), 24)

    result = forecast_one_origin(
        SeasonalNaiveForecaster(), training, history, future,
        city="unspecified", split="validation",
        forecast_origin=origin, run_id="runner-test",
    )

    assert len(result) == 24
    assert result["horizon"].tolist() == list(range(1, 25))
    assert result["model"].eq("seasonal_naive").all()
    assert result["city"].eq("test-city").all()


def test_runner_rejects_wrong_history_length():
    training = make_frame("2025-01-01", 240)
    history = make_frame("2025-01-11", 167)
    future = make_frame(history.timestamp.iloc[-1] + pd.Timedelta(hours=1), 24)

    with pytest.raises(ValueError, match="168"):
        forecast_one_origin(
            SeasonalNaiveForecaster(), training, history, future,
            city="x", split="validation",
            forecast_origin=history.timestamp.iloc[-1], run_id="bad-history",
        )


def test_runner_rejects_future_timestamp_gap():
    training = make_frame("2025-01-01", 240)
    history = make_frame("2025-01-11", 168)
    origin = history.timestamp.iloc[-1]
    future = make_frame(origin + pd.Timedelta(hours=2), 24)

    with pytest.raises(ValueError, match="future timestamps"):
        forecast_one_origin(
            SeasonalNaiveForecaster(), training, history, future,
            city="x", split="validation",
            forecast_origin=origin, run_id="bad-future",
        )


def test_runner_rejects_test_split_typo():
    training = make_frame("2025-01-01", 240)
    history = make_frame("2025-01-11", 168)
    origin = history.timestamp.iloc[-1]
    future = make_frame(origin + pd.Timedelta(hours=1), 24)

    with pytest.raises(ValueError, match="split"):
        forecast_one_origin(
            SeasonalNaiveForecaster(), training, history, future,
            city="x", split="train",
            forecast_origin=origin, run_id="bad-split",
        )

def test_runner_rejects_training_data_after_origin():
    training = make_frame("2025-01-01", 240)
    history = make_frame("2025-01-11", 168)
    origin = history.timestamp.iloc[-1]
    training.loc[0, "timestamp"] = origin + pd.Timedelta(hours=1)
    future = make_frame(origin + pd.Timedelta(hours=1), 24)

    with pytest.raises(ValueError, match="after forecast origin"):
        forecast_one_origin(
            SeasonalNaiveForecaster(), training, history, future,
            city="x", split="validation",
            forecast_origin=origin, run_id="future-training",
        )


def test_runner_rejects_training_from_another_meter():
    training = make_frame("2025-01-01", 240, meter="other-meter")
    history = make_frame("2025-01-11", 168, meter="m1")
    origin = history.timestamp.iloc[-1]
    future = make_frame(origin + pd.Timedelta(hours=1), 24)

    with pytest.raises(ValueError, match="forecast meter"):
        forecast_one_origin(
            SeasonalNaiveForecaster(), training, history, future,
            city="x", split="validation",
            forecast_origin=origin, run_id="wrong-meter",
        )
