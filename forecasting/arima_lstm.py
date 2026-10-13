
"""Autoregressive residual baseline plus LSTM correction model."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.config import HORIZON, LOOKBACK, SEED
from forecasting.hybrid import fit_ar_coefficients, recursive_ar_forecast
from forecasting.interface import Forecaster


class _ResidualLSTM(nn.Module):
    """LSTM that predicts a 24-hour residual correction."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1])


class ARIMALSTMForecaster(Forecaster):
    """AR baseline with an LSTM trained to predict residual corrections."""

    model_name = "arima_lstm"

    def __init__(self, hidden_size=16, epochs=5, batch_size=32, seed=SEED):
        if hidden_size < 1 or epochs < 1 or batch_size < 1:
            raise ValueError(
                "hidden_size, epochs, and batch_size must be positive"
            )

        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.fitted_ = False

    @staticmethod
    def _one_step_residuals(values, start_index=4):
        """Generate expanding-window AR residuals with zero warm-up.

        The first four residual positions are initialized to zero because
        the AR(3) model needs at least four observations before its first fit.
        These are model-initialization values, not imputed load measurements.
        """
        values = np.asarray(values, dtype=np.float64)
        residuals = np.zeros(len(values), dtype=np.float64)

        for i in range(start_index, len(values)):
            prior = values[:i]

            beta = fit_ar_coefficients(prior, order=3)
            forecast = recursive_ar_forecast(
                beta,
                prior,
                horizon=1,
                order=3,
            )[0]

            residuals[i] = values[i] - forecast

        return residuals

    def fit(self, training_data):
        self.fitted_ = False
        self.validate_history(training_data)

        data = training_data.sort_values("timestamp").copy()

        if data["meter_id"].astype(str).nunique() != 1:
            raise ValueError(
                "ARIMA-LSTM training data must contain exactly one meter"
            )

        values = data["load"].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("Training data contains non-finite loads")

        if len(values) < LOOKBACK + HORIZON + 4:
            raise ValueError(
                "Insufficient data for ARIMA-LSTM residual windows"
            )

        # Generate residuals after the AR(3) warm-up.
        residuals = self._one_step_residuals(
            values,
            start_index=4,
        )

        X, y = [], []

        # Each example uses a 168-hour residual context to predict
        # the next 24 hours of AR forecast errors.
        for end in range(
            LOOKBACK,
            len(values) - HORIZON + 1,
        ):
            context = residuals[end - LOOKBACK:end]
            future = values[end:end + HORIZON]

            if not np.isfinite(context).all():
                continue

            if not np.isfinite(future).all():
                continue

            # The baseline uses only observations before the target window.
            beta = fit_ar_coefficients(values[:end], order=3)
            baseline = recursive_ar_forecast(
                beta,
                values[:end],
                horizon=HORIZON,
                order=3,
            )

            X.append(context)
            y.append(future - baseline)

        if not X:
            raise ValueError("No valid ARIMA-LSTM training windows")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Learn normalization parameters from training windows only.
        self.x_mean_ = float(X.mean())
        self.x_std_ = float(X.std()) or 1.0
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0

        X = (X - self.x_mean_) / self.x_std_
        y = (y - self.y_mean_) / self.y_std_

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.network_ = _ResidualLSTM(self.hidden_size)
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=1e-3,
        )
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            self.network_.train()

            for xb, yb in loader:
                optimizer.zero_grad()
                predictions = self.network_(xb)
                loss = loss_fn(predictions, yb)
                loss.backward()
                optimizer.step()

        self.fitted_ = True
        return self

    def predict(self, history):
        self.validate_history(history)

        if not self.fitted_:
            raise RuntimeError("Call fit() before predict()")

        data = history.sort_values("timestamp")
        values = data["load"].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("History contains non-finite loads")

        # Produce the AR baseline from the supplied history only.
        beta = fit_ar_coefficients(values, order=3)
        baseline = recursive_ar_forecast(
            beta,
            values,
            horizon=HORIZON,
            order=3,
        )

        # The warm-up ensures exactly LOOKBACK residuals are available
        # when the input history contains LOOKBACK observations.
        residuals = self._one_step_residuals(
            values,
            start_index=4,
        )

        context = residuals[-LOOKBACK:]

        if len(context) != LOOKBACK or not np.isfinite(context).all():
            raise ValueError("Insufficient valid residual history")

        context = (context - self.x_mean_) / self.x_std_

        x = torch.tensor(
            context,
            dtype=torch.float32,
        ).reshape(1, LOOKBACK, 1)

        self.network_.eval()

        with torch.no_grad():
            normalized_correction = (
                self.network_(x).cpu().numpy().ravel()
            )

        correction = (
            normalized_correction * self.y_std_ + self.y_mean_
        )

        prediction = baseline + correction

        return np.asarray(
            self.validate_predictions(prediction),
            dtype=np.float64,
        )