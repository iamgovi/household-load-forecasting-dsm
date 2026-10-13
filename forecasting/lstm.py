"""Standalone LSTM for direct 24-hour household-load forecasting."""

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.config import HORIZON, LOOKBACK, SEED
from forecasting.interface import Forecaster


class _LSTMNetwork(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1])


class LSTMForecaster(Forecaster):
    """Train on historical windows and predict the next 24 hours."""

    model_name = "lstm"

    def __init__(
        self, hidden_size=32, epochs=10, batch_size=64,
        learning_rate=0.001, seed=SEED
    ):
        if hidden_size < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("Model dimensions and epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.fitted_ = False

    def fit(self, training_data):
        self.validate_history(training_data)

        data = training_data.sort_values("timestamp").copy()
        if data["meter_id"].astype(str).nunique() != 1:
            raise ValueError("LSTM training data must contain exactly one meter")

        values = data["load"].to_numpy(dtype=np.float32)
        if len(values) <= LOOKBACK + HORIZON:
            raise ValueError(
                "Training data must exceed lookback plus forecast horizon"
            )

        # Fit normalization using training observations only.
        self.mean_ = float(values.mean())
        self.std_ = float(values.std()) or 1.0
        scaled = (values - self.mean_) / self.std_

        X, y = [], []
        for end in range(LOOKBACK, len(scaled) - HORIZON + 1):
            X.append(scaled[end - LOOKBACK:end])
            y.append(scaled[end:end + HORIZON])

        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.network_ = _LSTMNetwork(self.hidden_size)
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate
        )
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            self.network_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network_(xb), yb)
                loss.backward()
                optimizer.step()

        self.network_.eval()
        self.fitted_ = True
        return self

    def predict(self, history):
        self.validate_history(history)
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict()")

        values = history.sort_values("timestamp")["load"].to_numpy(
            dtype=np.float32
        )
        scaled = (values[-LOOKBACK:] - self.mean_) / self.std_
        x = torch.tensor(scaled, dtype=torch.float32).reshape(1, LOOKBACK, 1)

        self.network_.eval()
        with torch.no_grad():
            prediction = self.network_(x).cpu().numpy().ravel()

        prediction = prediction * self.std_ + self.mean_
        return self.validate_predictions(prediction)
