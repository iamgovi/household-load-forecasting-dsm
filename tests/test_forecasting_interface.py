import numpy as np
import pandas as pd
import pytest

from forecasting.interface import Forecaster


class DummyForecaster(Forecaster):
    model_name = "dummy"

    def fit(self, training_data):
        return self

    def predict(self, history):
        self.validate_history(history)
        return self.validate_predictions([1.0] * 24)


def valid_history():
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=168, freq="h"),
        "meter_id": ["meter-1"] * 168,
        "load": np.ones(168),
    })


def test_valid_history_and_forecast():
    model = DummyForecaster()
    history = valid_history()
    assert model.fit(history) is model
    assert len(model.predict(history)) == 24


def test_insufficient_history_fails():
    with pytest.raises(ValueError, match="168"):
        DummyForecaster().validate_history(valid_history().iloc[:100])


def test_missing_observation_fails():
    history = valid_history()
    history.loc[50, "load"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        DummyForecaster().validate_history(history)


def test_non_hourly_history_fails():
    history = valid_history()
    history.loc[50, "timestamp"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="consecutive hourly"):
        DummyForecaster().validate_history(history)


def test_invalid_prediction_length_fails():
    with pytest.raises(ValueError, match="exactly 24"):
        DummyForecaster().validate_predictions([1.0] * 23)


def test_non_finite_prediction_fails():
    with pytest.raises(ValueError, match="finite"):
        DummyForecaster().validate_predictions([float("nan")] + [1.0] * 23)
