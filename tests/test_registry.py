"""Tests for the forecasting model registry."""

import pytest

from forecasting.registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_models,
)


EXPECTED_MODELS = {
    "seasonal_naive",
    "holt_winters",
    "arima",
    "sarima",
    "linear_regression",
    "xgboost",
    "gru",
    "lstm",
    "arima_lstm",
    "arima_gru",
    "sarima_gru",
    "pso_arima_gru",
}


def test_all_required_models_are_registered():
    assert set(MODEL_REGISTRY) == EXPECTED_MODELS


def test_model_categories_are_available():
    assert "lstm" in list_models("deep_learning")
    assert "arima_gru" in list_models("hybrid")
    assert "xgboost" in list_models("machine_learning")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown forecasting model"):
        get_model_spec("unknown_model")


def test_registry_spec_is_immutable():
    spec = get_model_spec("lstm")

    with pytest.raises(AttributeError):
        spec.name = "changed"
