import numpy as np
import pytest

from forecasting.hybrid import (
    autoregressive_design,
    fit_ar_coefficients,
    recursive_ar_forecast,
    residual_targets,
)


def test_ar_design_orders_lags_most_recent_first():
    np.testing.assert_array_equal(
        autoregressive_design([1, 2, 3], order=3),
        [1, 3, 2, 1],
    )


def test_ar_fit_and_recursive_forecast_are_finite():
    values = np.sin(np.arange(100) / 5)
    coefficients = fit_ar_coefficients(values, order=3)
    prediction = recursive_ar_forecast(coefficients, values, horizon=24)
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()


def test_ar_rejects_non_finite_training_data():
    values = np.arange(20, dtype=float)
    values[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_ar_coefficients(values)


def test_residual_targets():
    result = residual_targets([10, 12, 14], [8, 11, 13])
    np.testing.assert_array_equal(result, [2, 1, 1])


def test_residual_targets_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        residual_targets([1, 2], [1])


def test_recursive_forecast_rejects_bad_coefficients():
    with pytest.raises(ValueError, match="Coefficient count"):
        recursive_ar_forecast([1, 2], [1, 2, 3])
