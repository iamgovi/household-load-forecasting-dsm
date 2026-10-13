import numpy as np
import pandas as pd
import pytest

from forecasting.arima_lstm import ARIMALSTMForecaster
from forecasting.registry import get_baseline_forecaster


def make_data(n=210):
    values = 5 + np.sin(np.arange(n) * 2 * np.pi / 24)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "meter_id": ["m1"] * n,
        "load": values,
    })


def test_arima_lstm_rejects_prediction_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        ARIMALSTMForecaster().predict(make_data().iloc[-168:])


def test_arima_lstm_is_registered():
    assert isinstance(
        get_baseline_forecaster("arima_lstm"), ARIMALSTMForecaster
    )


def test_arima_lstm_rejects_insufficient_training_data():
    with pytest.raises(ValueError, match="Insufficient data"):
        ARIMALSTMForecaster(epochs=1).fit(make_data(180))


def test_arima_lstm_forecasts_24_hours():
    data = make_data()
    model = ARIMALSTMForecaster(hidden_size=4, epochs=1, batch_size=32)
    model.fit(data)
    prediction = model.predict(data.iloc[-168:])
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()
