
"""Experimental SARIMA-GRU residual hybrid forecaster.

Uses chronological rolling-origin SARIMA forecasts to construct training
targets and a GRU to predict a direct 24-hour residual correction.

This module does not load datasets, access the final test split, or write
research reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

from forecasting.config import HORIZON, LOOKBACK, SEED
from forecasting.interface import Forecaster


class _ResidualGRU(nn.Module):
    """Unidirectional GRU producing a direct 24-step correction."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(hidden_size, HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output(hidden[-1])


class SARIMAGRUForecaster(Forecaster):
    """SARIMA baseline with a GRU residual-correction model.

    Supports one meter per fit/predict call. Inputs require timestamp,
    meter_id, and load columns with consecutive hourly observations.

    Experimental note:
    The current GRU feature design uses residuals from a SARIMA fit to
    the historical prefix. This is causal, but the residual-generation
    procedure still requires rigorous alignment validation before the
    hybrid should be used for final model comparison.
    """

    model_name = "sarima_gru"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 24),
        hidden_size: int = 32,
        max_training_windows: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        maxiter: int = 20,
        seed: int = SEED,
    ):
        if len(order) != 3 or len(seasonal_order) != 4:
            raise ValueError("Invalid SARIMA order configuration")
        if seasonal_order[3] < 2:
            raise ValueError("Seasonal period must be at least 2")
        if hidden_size < 1 or max_training_windows < 4 or epochs < 1:
            raise ValueError("Invalid GRU training configuration")
        if learning_rate <= 0 or maxiter < 1:
            raise ValueError("learning_rate and maxiter must be positive")

        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.hidden_size = hidden_size
        self.max_training_windows = max_training_windows
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.seed = seed

        self._sarima_result = None
        self._gru: _ResidualGRU | None = None
        self._meter_id = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        self._fitted = False

    @staticmethod
    def _prepare_frame(
        data: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """Validate a single meter's complete hourly series."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{label} must be a pandas DataFrame")

        required = {"timestamp", "meter_id", "load"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"{label} missing columns: {sorted(missing)}"
            )
        if data.empty:
            raise ValueError(f"{label} cannot be empty")

        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="raise"
        )

        if frame["timestamp"].isna().any():
            raise ValueError(f"{label} contains invalid timestamps")

        if frame["meter_id"].isna().any():
            raise ValueError(f"{label} contains missing meter identifiers")

        meter_ids = frame["meter_id"].astype(str).str.strip()
        if meter_ids.eq("").any() or meter_ids.nunique() != 1:
            raise ValueError(
                f"{label} must contain exactly one valid meter"
            )

        frame["meter_id"] = meter_ids
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        if frame["timestamp"].duplicated().any():
            raise ValueError(f"{label} contains duplicate timestamps")

        if not frame["timestamp"].diff().iloc[1:].eq(
            pd.Timedelta(hours=1)
        ).all():
            raise ValueError(
                f"{label} must contain consecutive hourly observations"
            )

        frame["load"] = pd.to_numeric(frame["load"], errors="coerce")
        loads = frame["load"].to_numpy(dtype=float)

        if not np.isfinite(loads).all():
            raise ValueError(
                f"{label} contains missing or non-finite loads"
            )

        frame["load"] = loads
        return frame

    def _fit_sarima(self, values: np.ndarray):
        """Fit SARIMA using only the supplied historical observations."""
        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=self.maxiter)

    @staticmethod
    def _forecast_24(result) -> np.ndarray:
        """Return exactly 24 finite baseline forecasts."""
        forecast = np.asarray(
            result.get_forecast(steps=HORIZON).predicted_mean,
            dtype=float,
        ).reshape(-1)

        if forecast.shape != (HORIZON,):
            raise ValueError("SARIMA must return exactly 24 forecasts")
        if not np.isfinite(forecast).all():
            raise ValueError("SARIMA returned non-finite forecasts")

        return forecast

    def fit(
        self,
        training_data: pd.DataFrame,
    ) -> "SARIMAGRUForecaster":
        """Fit the hybrid from the supplied training partition only."""
        frame = self._prepare_frame(training_data, "training_data")
        values = frame["load"].to_numpy(dtype=float)
        self._meter_id = frame["meter_id"].iloc[0]

        first_origin = LOOKBACK - 1
        if len(values) < first_origin + 1 + HORIZON:
            raise ValueError(
                "Insufficient data for rolling SARIMA forecasts"
            )

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        # Origins advance by 24 hours, so forecast targets do not overlap.
        for origin_idx in range(
            first_origin,
            len(values) - HORIZON,
            HORIZON,
        ):
            historical_values = values[: origin_idx + 1]

            # The model sees only observations up to this origin.
            origin_result = self._fit_sarima(historical_values)
            baseline = self._forecast_24(origin_result)

            # Use residuals from the historical prefix only.
            historical_residuals = np.asarray(
                origin_result.resid, dtype=float
            ).reshape(-1)
            residual_window = historical_residuals[-LOOKBACK:]

            if (
                len(residual_window) != LOOKBACK
                or not np.isfinite(residual_window).all()
            ):
                continue

            actual_future = values[
                origin_idx + 1 : origin_idx + 1 + HORIZON
            ]
            if len(actual_future) != HORIZON:
                continue

            features.append(residual_window[:, None])
            targets.append(actual_future - baseline)

            if len(features) >= self.max_training_windows:
                break

        if len(features) < 4:
            raise ValueError(
                "Insufficient valid rolling-origin windows; "
                "at least four are required"
            )

        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)

        # Scaling is fitted exclusively on generated training windows.
        self._x_mean = float(X.mean())
        self._x_std = max(float(X.std()), 1e-8)
        self._y_mean = float(y.mean())
        self._y_std = max(float(y.std()), 1e-8)

        X_scaled = (X - self._x_mean) / self._x_std
        y_scaled = (y - self._y_mean) / self._y_std

        torch.manual_seed(self.seed)
        self._gru = _ResidualGRU(self.hidden_size)
        optimizer = torch.optim.Adam(
            self._gru.parameters(),
            lr=self.learning_rate,
        )
        loss_fn = nn.MSELoss()

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        self._gru.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            predictions = self._gru(x_tensor)
            loss = loss_fn(predictions, y_tensor)
            loss.backward()
            optimizer.step()

        self._gru.eval()

        # Final baseline parameters are learned from training data only.
        self._sarima_result = self._fit_sarima(values)
        self._fitted = True
        return self

    def predict(self, history: pd.DataFrame) -> list[float]:
        """Forecast the next 24 hours using the supplied history."""
        if not self._fitted or self._gru is None:
            raise RuntimeError(
                "Call fit(training_data) before predict(history)"
            )

        frame = self._prepare_frame(history, "history")
        self.validate_history(frame)

        if frame["meter_id"].iloc[0] != self._meter_id:
            raise ValueError("History meter differs from fitted meter")

        history_values = frame["load"].to_numpy(dtype=float)

        # Apply training-fitted SARIMA parameters to the supplied context.
        # This updates the state using history without re-estimating params.
        history_model = SARIMAX(
            history_values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        filtered_result = history_model.filter(
            self._sarima_result.params
        )

        baseline = self._forecast_24(filtered_result)

        residual_history = np.asarray(
            filtered_result.resid, dtype=float
        ).reshape(-1)

        if len(residual_history) < LOOKBACK:
            raise ValueError("Insufficient residual history")

        residual_window = residual_history[-LOOKBACK:]
        if not np.isfinite(residual_window).all():
            raise ValueError(
                "SARIMA residual history is non-finite; "
                "cannot produce a controlled GRU correction"
            )

        X = (
            (residual_window[:, None] - self._x_mean) / self._x_std
        ).astype(np.float32)

        self._gru.eval()
        with torch.no_grad():
            correction_scaled = self._gru(
                torch.tensor(X[None, :, :], dtype=torch.float32)
            ).cpu().numpy()[0]

        correction = correction_scaled * self._y_std + self._y_mean
        prediction = baseline + correction

        return self.validate_predictions(prediction)
