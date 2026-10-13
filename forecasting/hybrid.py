"""Reusable ARIMA residual forecasting helpers."""

import numpy as np


def autoregressive_design(history, order=3):
    """Construct an intercept-plus-lag design vector."""
    if order < 1 or len(history) < order:
        raise ValueError("History must contain at least `order` values")

    values = np.asarray(history, dtype=float)
    if not np.isfinite(values[-order:]).all():
        raise ValueError("AR history contains non-finite values")

    return np.array([1.0, *values[-order:][::-1]], dtype=float)


def fit_ar_coefficients(history, order=3):
    """Fit AR coefficients from a single chronological training series."""
    values = np.asarray(history, dtype=float)
    if order < 1 or len(values) <= order:
        raise ValueError("Insufficient training observations for AR model")
    if not np.isfinite(values).all():
        raise ValueError("Training history contains missing or non-finite values")

    X = np.stack([
        autoregressive_design(values[:i], order)
        for i in range(order, len(values))
    ])
    y = values[order:]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def recursive_ar_forecast(coefficients, history, horizon=24, order=3):
    """Forecast recursively without using future observations."""
    beta = np.asarray(coefficients, dtype=float)
    values = list(np.asarray(history, dtype=float))

    if len(beta) != order + 1:
        raise ValueError("Coefficient count does not match AR order")
    if horizon < 1 or len(values) < order:
        raise ValueError("Invalid horizon or insufficient history")
    if not np.isfinite(beta).all() or not np.isfinite(values[-order:]).all():
        raise ValueError("AR inputs must be finite")

    result = []
    for _ in range(horizon):
        estimate = float(autoregressive_design(values, order) @ beta)
        result.append(estimate)
        values.append(estimate)

    return np.asarray(result)


def residual_targets(actual, baseline):
    """Calculate residual targets after shape and finite-value validation."""
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if actual.shape != baseline.shape:
        raise ValueError("Actual and baseline arrays must have identical shapes")
    if not np.isfinite(actual).all() or not np.isfinite(baseline).all():
        raise ValueError("Actual and baseline arrays must be finite")

    return actual - baseline
