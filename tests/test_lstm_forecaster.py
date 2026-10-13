import numpy as np
import pandas as pd
import pytest

from forecasting.lstm import LSTMForecaster
from forecasting.registry import get_baseline_forecaster


def make_data(n=240):
    values = 3 + np.sin(np.arange(n) * 2 * np.pi / 24)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "meter_id": ["m1"] * n,
        "load": values,
    })


def test_lstm_can_fit_and_predict_24_hours():
    model = LSTMForecaster(hidden_size=4, epochs=1, batch_size=32)
    data = make_data()
    model.fit(data)
    predictions = model.predict(data.iloc[-168:])
    assert len(predictions) == 24
    assert np.isfinite(predictions).all()


def test_lstm_requires_fit_before_prediction():
    with pytest.raises(RuntimeError, match="fit"):
        LSTMForecaster().predict(make_data().iloc[-168:])


def test_lstm_is_available_through_registry():
    model = get_baseline_forecaster("lstm")
    assert isinstance(model, LSTMForecaster)


def test_lstm_rejects_multiple_training_meters():
    data = make_data()
    other = data.copy()
    other["timestamp"] += pd.Timedelta(hours=len(data))
    other["meter_id"] = "m2"
    combined = pd.concat([data, other], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one meter"):
        LSTMForecaster(epochs=1).fit(combined)
