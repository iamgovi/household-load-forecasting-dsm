"""Lightweight forecasting baselines using the common interface."""

import numpy as np
from sklearn.linear_model import LinearRegression

from forecasting.interface import Forecaster
from forecasting.config import HORIZON, LOOKBACK


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the most recent 24-hour profile."""

    model_name = "seasonal_naive"

    def fit(self, training_data):
        self.validate_history(training_data)
        self.fitted_ = True
        return self

    def predict(self, history):
        self.validate_history(history)
        values = history["load"].to_numpy(dtype=float)
        return self.validate_predictions(values[-HORIZON:])


class LinearRegressionForecaster(Forecaster):
    """Direct multi-output regression from 168 lags to 24 future values."""

    model_name = "linear_regression"

    def fit(self, training_data):
        self.validate_history(training_data)

        values = training_data["load"].to_numpy(dtype=float)
        if len(values) <= LOOKBACK + HORIZON:
            raise ValueError(
                "Training data must contain more than lookback + horizon observations"
            )

        X, y = [], []
        for end in range(LOOKBACK, len(values) - HORIZON + 1):
            X.append(values[end - LOOKBACK:end])
            y.append(values[end:end + HORIZON])

        if not X:
            raise ValueError("No training windows could be constructed")

        self.model_ = LinearRegression()
        self.model_.fit(np.asarray(X), np.asarray(y))
        self.fitted_ = True
        return self

    def predict(self, history):
        self.validate_history(history)
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() before predict()")

        values = history["load"].to_numpy(dtype=float)
        prediction = self.model_.predict(values[-LOOKBACK:].reshape(1, -1))[0]
        return self.validate_predictions(prediction)