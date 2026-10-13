"""
THE MODEL FILE: give it any household's usage history, get the next 7 days back.

Works on data it has NEVER seen before -- not just your 62 known households.
Only requirement: at least 168 hours (1 week) of hourly kWh readings, ideally
2-4+ weeks for a better ARIMA fit.

Requires: general_gru_model.pt, general_model_meta.joblib (from train_general_model.py)

USAGE (from Python):
    import pandas as pd
    from forecast_model import forecast_next_week

    history = pd.read_csv("some_new_household.csv", parse_dates=["timestamp"])
    history = history.set_index("timestamp")["kwh"]   # a plain hourly Series

    forecast = forecast_next_week(history)
    print(forecast)   # 168 rows: timestamp, predicted_kwh

USAGE (from the command line, on a CSV with "timestamp" and "kwh" columns):
    python forecast_model.py my_new_household.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import joblib

warnings.filterwarnings("ignore")

DEVICE = "cpu"  # inference on one household is fast enough without a GPU


class GeneralGRUForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, horizon):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


def _load_model():
    meta = joblib.load("general_model_meta.joblib")
    model = GeneralGRUForecaster(
        n_features=len(meta["feature_cols"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        horizon=meta["horizon"],
    )
    model.load_state_dict(torch.load("general_gru_model.pt", map_location=DEVICE))
    model.eval()
    return model, meta


def _add_cyclical_calendar(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"x_Timestamp": timestamps})
    df["hour_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["is_weekend"] = df["x_Timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
    return df


def _to_regular_hourly(series: pd.Series) -> pd.Series:
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    reindexed = series.reindex(full_index)
    if reindexed.isna().any():
        reindexed = reindexed.interpolate(limit_direction="both")
    return reindexed


def forecast_n_days(history: pd.Series, n_days: int = 30) -> pd.DataFrame:
    """
    INPUT:  history -- a pandas Series of hourly kWh readings, indexed by
            datetime, for ANY household. Needs at least 168 hours (1 week);
            2-4+ weeks gives a noticeably better ARIMA fit.
            n_days -- how many days ahead to forecast (e.g. 7, 30).

    OUTPUT: a DataFrame with n_days*24 rows -- timestamp, predicted_kwh --
            covering the period immediately after the input history ends.

    HOW IT WORKS: fits ARIMA fresh on the given history, predicts 24 hours
    at a time, and for each 24-hour block also runs the pretrained GRU to
    correct ARIMA's residual error. Since there's no real future data to
    check against as we go further out, each block's OWN forecast is used
    as if it were real to build the next block's input -- standard practice
    for multi-step-ahead forecasting, but it means later days are less
    certain than the first day. Say this explicitly if presenting.
    """
    model, meta = _load_model()
    seq_len, horizon = meta["seq_len"], meta["horizon"]
    x_mean, x_std = meta["x_mean"], meta["x_std"]
    y_mean, y_std = meta["y_mean"], meta["y_std"]
    feature_cols = meta["feature_cols"]
    t_kwh_idx = feature_cols.index("t_kWh")

    if len(history) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} hours of history, got {len(history)}. "
            f"Provide more data (ideally several weeks) for a usable forecast."
        )

    history_reg = _to_regular_hourly(history)
    arima_fit = ARIMA(history_reg, order=meta["arima_order"]).fit()

    buffer = history_reg.copy()
    all_predictions = []
    n_blocks = int(np.ceil(n_days * 24 / horizon))

    for block_num in range(n_blocks):
        arima_block_forecast = arima_fit.forecast(steps=horizon)
        block_timestamps = arima_block_forecast.index

        input_window_index = pd.date_range(end=block_timestamps[0] - pd.Timedelta(hours=1),
                                            periods=seq_len, freq="h")
        seg = _add_cyclical_calendar(input_window_index)
        seg["t_kWh"] = buffer.reindex(input_window_index).values

        if seg[feature_cols].isna().any(axis=None):
            raise RuntimeError(
                f"Not enough continuous history to build block {block_num + 1}'s input window. "
                f"This shouldn't happen with clean input data -- check for gaps in your history."
            )

        X_input = seg[feature_cols].values.astype(np.float32)
        X_input[:, t_kwh_idx] = (X_input[:, t_kwh_idx] - x_mean) / x_std
        X_tensor = torch.tensor(X_input).unsqueeze(0)

        with torch.no_grad():
            pred_residual_scaled = model(X_tensor).numpy()[0]
        pred_residual = pred_residual_scaled * y_std + y_mean

        block_forecast = np.clip(arima_block_forecast.values + pred_residual, a_min=0, a_max=None)
        block_series = pd.Series(block_forecast, index=block_timestamps)
        all_predictions.append(block_series)

        # Use this block's own forecast as if it were observed, to extend
        # both the ARIMA model and the buffer for the next block. If the
        # update fails, we surface it (instead of silently ignoring it, which
        # previously caused the model to forecast the same period twice and
        # crash later with a duplicate-timestamp error) and drop back to a
        # fresh ARIMA fit on the extended buffer so the loop can keep going.
        buffer = pd.concat([buffer, block_series])
        buffer = buffer[~buffer.index.duplicated(keep="last")]

        try:
            arima_fit = arima_fit.append(block_series, refit=False)
        except Exception as e:
            print(f"Warning: ARIMA state update failed at block {block_num + 1} ({e}); "
                  f"refitting fresh on extended history instead.")
            arima_fit = ARIMA(buffer, order=meta["arima_order"]).fit()

    full_forecast = pd.concat(all_predictions).iloc[: n_days * 24]
    return pd.DataFrame({
        "timestamp": full_forecast.index,
        "predicted_kwh": full_forecast.values,
    }).reset_index(drop=True)


def forecast_next_week(history: pd.Series) -> pd.DataFrame:
    """Convenience wrapper -- 7 days ahead. See forecast_n_days for details."""
    return forecast_n_days(history, n_days=7)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python forecast_model.py <path_to_csv_with_timestamp_and_kwh_columns>")
        sys.exit(1)

    input_path = sys.argv[1]
    raw = pd.read_csv(input_path, parse_dates=["timestamp"])
    history = raw.set_index("timestamp")["kwh"]

    print(f"Loaded {len(history)} hours of history for this household.")
    print("Fitting ARIMA and generating the 7-day forecast...")

    result = forecast_next_week(history)

    print(f"\n7-day forecast ({len(result)} hours):\n")
    print(result.to_string(index=False))

    daily_totals = result.set_index("timestamp")["predicted_kwh"].resample("D").sum()
    print("\nDaily totals:")
    print(daily_totals)

    output_path = "week_forecast_output.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved full forecast to {output_path}")