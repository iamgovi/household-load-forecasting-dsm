"""
Trains a GENERAL-PURPOSE residual GRU -- one that does NOT need to know which
specific household it's looking at. The earlier GRU had a per-household
"identity slot" (embedding) that only worked for the 62 households it was
trained on. This version drops that, so it works on ANY household's data,
including ones it has never seen.

Run ONCE: python train_general_model.py
Reads: train.csv, val.csv, test.csv
Writes: general_gru_model.pt, general_model_meta.joblib
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
import joblib

warnings.filterwarnings("ignore")

ARIMA_ORDER = (2, 1, 2)
SEQ_LEN = 168      # one week of history needed as input
HORIZON = 24       # predicts 24 hours at a time (used 7x to build a full week)
BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only raw usage + calendar info -- NOTHING specific to a particular
# household. This is what makes the model reusable on new data.
FEATURE_COLS = ["t_kWh", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def add_cyclical_calendar(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["is_weekend"] = df["x_Timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
    return df


def to_regular_hourly(series):
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    reindexed = series.reindex(full_index)
    if reindexed.isna().any():
        reindexed = reindexed.interpolate(limit_direction="both")
    return reindexed


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GeneralGRUForecaster(nn.Module):
    """No embedding, no household identity -- works on any input sequence
    that has the right shape and features."""
    def __init__(self, n_features, hidden_size, num_layers, horizon):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


def build_residual_windows(df, residual_by_meter, seq_len, horizon):
    X_list, y_list = [], []
    for meter_id, g in df.groupby("meter"):
        if meter_id not in residual_by_meter:
            continue
        residual = residual_by_meter[meter_id]
        g = g.sort_values("x_Timestamp").reset_index(drop=True)
        g = g[g["x_Timestamp"].isin(residual.index)].reset_index(drop=True)
        time_diff = g["x_Timestamp"].diff().dt.total_seconds().div(3600).fillna(1.0)
        block_id = (time_diff != 1.0).cumsum()

        for _, block in g.groupby(block_id):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < seq_len + horizon:
                continue
            feats = block[FEATURE_COLS].values
            target_residuals = residual.reindex(block["x_Timestamp"]).values
            for start in range(0, n - seq_len - horizon + 1):
                target_slice = target_residuals[start + seq_len: start + seq_len + horizon]
                if np.isnan(target_slice).any():
                    continue
                X_list.append(feats[start: start + seq_len])
                y_list.append(target_slice)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, epochs_without_improvement = val_loss, model.state_dict(), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def main():
    print(f"Using device: {DEVICE}")
    train = pd.read_csv("train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("test.csv", parse_dates=["x_Timestamp"])
    train = add_cyclical_calendar(train)
    val = add_cyclical_calendar(val)
    test = add_cyclical_calendar(test)
    all_df = pd.concat([train, val, test], ignore_index=True)

    print("Fitting per-household ARIMA models to build residual training targets...")
    residual_by_meter = {}
    for i, meter_id in enumerate(sorted(all_df["meter"].unique()), 1):
        m_all = all_df[all_df["meter"] == meter_id]
        if len(m_all) < SEQ_LEN * 3:
            continue
        history = m_all.set_index("x_Timestamp")["t_kWh"].sort_index()
        history_reg = to_regular_hourly(history)
        try:
            model_fit = ARIMA(history_reg, order=ARIMA_ORDER).fit()
        except Exception:
            continue
        residual_by_meter[meter_id] = (history_reg - model_fit.fittedvalues).dropna()
        if i % 10 == 0:
            print(f"  {i} households done")

    print(f"ARIMA succeeded for {len(residual_by_meter)} households")

    print("\nBuilding pooled residual training windows (no household identity used)...")
    X_all, y_all = build_residual_windows(all_df, residual_by_meter, SEQ_LEN, HORIZON)
    print(f"Total windows: {len(X_all):,}")

    n_val = int(len(X_all) * 0.1)
    idx = np.random.RandomState(42).permutation(len(X_all))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    t_kwh_idx = FEATURE_COLS.index("t_kWh")
    x_mean, x_std = X_all[train_idx][:, :, t_kwh_idx].mean(), X_all[train_idx][:, :, t_kwh_idx].std()
    x_std = x_std if x_std > 1e-6 else 1.0
    y_mean, y_std = y_all[train_idx].mean(), y_all[train_idx].std()
    y_std = y_std if y_std > 1e-6 else 1.0

    X_all[:, :, t_kwh_idx] = (X_all[:, :, t_kwh_idx] - x_mean) / x_std
    y_scaled = (y_all - y_mean) / y_std

    train_loader = DataLoader(WindowDataset(X_all[train_idx], y_scaled[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_all[val_idx], y_scaled[val_idx]), batch_size=BATCH_SIZE)

    model = GeneralGRUForecaster(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, HORIZON).to(DEVICE)
    print(f"\nTraining general GRU ({sum(p.numel() for p in model.parameters()):,} params)...")
    model = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), "general_gru_model.pt")
    joblib.dump({
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "arima_order": ARIMA_ORDER,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "x_mean": float(x_mean), "x_std": float(x_std),
        "y_mean": float(y_mean), "y_std": float(y_std),
    }, "general_model_meta.joblib")

    print("\nSaved general_gru_model.pt, general_model_meta.joblib")
    print("This model works on ANY household's data now -- use forecast_model.py to run it.")


if __name__ == "__main__":
    main()