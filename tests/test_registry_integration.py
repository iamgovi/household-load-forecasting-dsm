
import pytest

from forecasting.registry import (
    get_baseline_forecaster,
    get_model_spec,
)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("seasonal_naive", "SeasonalNaiveForecaster"),
        ("linear_regression", "LinearRegressionForecaster"),
        ("arima_lstm", "ARIMALSTMForecaster"),
    ],
)
def test_registry_returns_implemented_baselines(name, expected_class):
    model = get_baseline_forecaster(name)
    assert type(model).__name__ == expected_class
    assert get_model_spec(name).name == name


def test_unimplemented_model_is_not_returned_as_ready():
    with pytest.raises(ValueError, match="No implemented baseline"):
        get_baseline_forecaster("arima")
