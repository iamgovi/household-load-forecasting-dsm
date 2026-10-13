import numpy as np
import pandas as pd
import pytest

from forecasting.baselines import (
    LinearRegressionForecaster,
    SeasonalNaiveForecaster,
)


def make_data(n=400):
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    load = 5 + np.sin(np.arange(n) * 2 * np.pi / 24)
    return pd.DataFrame({
        "timestamp": timestamps,
        "meter_id": ["meter-1"] * n,
        "load": load,
    })


def test_seasonal_naive_repeats_last_day():
    data = make_data()
    model = SeasonalNaiveForecaster().fit(data)
    result = model.predict(data.iloc[-168:])
    assert len(result) == 24
    np.testing.assert_allclose(result, data["load"].to_numpy()[-24:])


def test_linear_regression_forecasts_24_values():
    data = make_data()
    model = LinearRegressionForecaster().fit(data.iloc[:300])
    result = model.predict(data.iloc[300:468] if len(data) >= 468 else data.iloc[-168:])
    assert len(result) == 24
    assert np.isfinite(result).all()


def test_linear_regression_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        LinearRegressionForecaster().predict(make_data().iloc[-168:])


def test_linear_regression_rejects_too_little_training_data():
    with pytest.raises(ValueError, match="lookback"):
        LinearRegressionForecaster().fit(make_data(190))