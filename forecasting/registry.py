"""Central registry of supported forecasting model identifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    category: str
    description: str
    requires_training: bool = True


MODEL_REGISTRY = {
    "seasonal_naive": ModelSpec(
        "seasonal_naive", "baseline", "Seasonal naive baseline", False
    ),
    "holt_winters": ModelSpec(
        "holt_winters", "statistical", "Holt-Winters exponential smoothing"
    ),
    "arima": ModelSpec(
        "arima", "statistical", "ARIMA forecasting"
    ),
    "sarima": ModelSpec(
        "sarima", "statistical", "Seasonal ARIMA forecasting"
    ),
    "linear_regression": ModelSpec(
        "linear_regression", "machine_learning", "Linear regression"
    ),
    "xgboost": ModelSpec(
        "xgboost", "machine_learning", "XGBoost regression"
    ),
    "gru": ModelSpec(
        "gru", "deep_learning", "GRU neural network"
    ),
    "lstm": ModelSpec(
        "lstm", "deep_learning", "LSTM neural network"
    ),
    "arima_lstm": ModelSpec(
        "arima_lstm", "hybrid", "ARIMA with LSTM residual correction"
    ),
    "arima_gru": ModelSpec(
        "arima_gru", "hybrid", "ARIMA with GRU residual correction"
    ),
    "sarima_gru": ModelSpec(
        "sarima_gru", "hybrid", "SARIMA with GRU residual correction"
    ),
    "pso_arima_gru": ModelSpec(
        "pso_arima_gru", "hybrid", "PSO-tuned ARIMA-GRU"
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return a registered model specification."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown forecasting model: {name!r}") from None


def list_models(category: str | None = None) -> list[str]:
    """List registered model identifiers, optionally filtered by category."""
    if category is None:
        return list(MODEL_REGISTRY)

    return [
        name
        for name, spec in MODEL_REGISTRY.items()
        if spec.category == category
    ]

def get_baseline_forecaster(model_name: str):
    """Return an instantiated, implemented forecaster."""
    from forecasting.baselines import (
        LinearRegressionForecaster,
        SeasonalNaiveForecaster,
    )
    from forecasting.lstm import LSTMForecaster
    from forecasting.arima_lstm import ARIMALSTMForecaster
    from forecasting.sarima_gru import SARIMAGRUForecaster

    implementations = {
        "seasonal_naive": SeasonalNaiveForecaster,
        "linear_regression": LinearRegressionForecaster,
        "lstm": LSTMForecaster,
        "arima_lstm": ARIMALSTMForecaster,
        "sarima_gru": SARIMAGRUForecaster,
    }

    try:
        return implementations[model_name]()
    except KeyError:
        raise ValueError(
            f"No implemented baseline forecaster for {model_name!r}"
        ) from None
