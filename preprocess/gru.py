"""
Step 7: Plain GRU, DIRECT 24-hour multi-output forecasting.

Deliberately NOT recursive/autoregressive -- step 5 vs 5b already showed
recursive forecasting hurts accuracy on this data. The GRU here reads a
window of past hours and outputs all 24 future hours in a single forward
pass (sequence-to-vector), matching the design that won for Random Forest.

This is your first real deep-learning data point. Compare its MAE/RMSE/sMAPE
directly against RandomForest's numbers from step5_ml_direct.py.

Requires: pip install torch

Run: python step7_gru_direct.py
Reads: train.csv, val.csv, test.csv
Writes: gru_results_per_meter.csv, gru_results_summary.csv, gru_model.pt
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 168          # one week of hourly history as input
HORIZON = 24           # predict the next 24 hours in one shot
BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5           # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = ["t_kWh", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
EMBEDDING_DIM = 8      # learned per-household identity vector, concatenated into the GRU input


def add_cyclical_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos encoding instead of raw hour/day-of-week integers -- lets the
    GRU see that hour 23 and hour 0 are adjacent, which a raw integer can't
    represent (23 and 0 look maximally far apart to a plain numeric feature)."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["is_weekend"] = df["x_Timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
    return df


def build_windows(df: pd.DataFrame, seq_len: int, horizon: int, meter_code_map: dict):
    """Builds (input_window, target_window) pairs PER METER, only within
    contiguous stretches of exactly-hourly timestamps -- never sliding a
    window across a gap, which would silently feed the model a false sense
    of continuity it didn't actually have. Also returns each window's meter
    code, so the model can learn a per-household identity embedding."""
    X_list, y_list, meta_list, code_list = [], [], [], []

    for meter_id, g in df.groupby("meter"):
        g = g.sort_values("x_Timestamp").reset_index(drop=True)
        time_diff = g["x_Timestamp"].diff().dt.total_seconds().div(3600).fillna(1.0)
        # a new "block" starts whenever the gap to the previous row isn't exactly 1 hour
        block_id = (time_diff != 1.0).cumsum()
        meter_code = meter_code_map[meter_id]

        for _, block in g.groupby(block_id):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < seq_len + horizon:
                continue
            feats = block[FEATURE_COLS].values
            for start in range(0, n - seq_len - horizon + 1):
                X_list.append(feats[start: start + seq_len])
                y_list.append(block["t_kWh"].values[start + seq_len: start + seq_len + horizon])
                meta_list.append(meter_id)
                code_list.append(meter_code)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meters = np.array(meta_list)
    codes = np.array(code_list, dtype=np.int64)
    return X, y, meters, codes


class WindowDataset(Dataset):
    def __init__(self, X, y, codes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.codes = torch.tensor(codes, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.codes[idx], self.y[idx]


class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, horizon, n_meters, embedding_dim):
        super().__init__()
        self.meter_embedding = nn.Embedding(n_meters, embedding_dim)
        self.gru = nn.GRU(n_features + embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x, meter_code):
        # meter embedding is per-window (one household), broadcast to every timestep
        emb = self.meter_embedding(meter_code)               # (batch, embedding_dim)
        emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq_len, embedding_dim)
        x_with_id = torch.cat([x, emb_seq], dim=-1)
        _, h_n = self.gru(x_with_id)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-6
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def compute_metrics(y_true, y_pred):
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "sMAPE": float(smape(y_true, y_pred)),
    }


def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, code_batch, y_batch in train_loader:
            X_batch, code_batch, y_batch = X_batch.to(DEVICE), code_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch, code_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, code_batch, y_batch in val_loader:
                X_batch, code_batch, y_batch = X_batch.to(DEVICE), code_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch, code_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}: train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    return model


def main():
    print(f"Using device: {DEVICE}")

    train = pd.read_csv("../train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("../val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])

    train = add_cyclical_calendar(train)
    val = add_cyclical_calendar(val)
    test = add_cyclical_calendar(test)

    print("Building sliding windows (only within contiguous hourly stretches)...")
    all_meters = sorted(set(train["meter"]) | set(val["meter"]) | set(test["meter"]))
    meter_code_map = {m: i for i, m in enumerate(all_meters)}

    X_train, y_train, _, codes_train = build_windows(train, SEQ_LEN, HORIZON, meter_code_map)
    X_val, y_val, _, codes_val = build_windows(val, SEQ_LEN, HORIZON, meter_code_map)
    X_test, y_test, meters_test, codes_test = build_windows(test, SEQ_LEN, HORIZON, meter_code_map)
    print(f"train windows: {len(X_train):,}  val windows: {len(X_val):,}  test windows: {len(X_test):,}")

    if len(X_train) == 0:
        raise RuntimeError(
            "No training windows built -- your train split per meter is probably "
            "shorter than SEQ_LEN + HORIZON. Reduce SEQ_LEN or check for excessive "
            "gap-splitting in build_windows."
        )

    # Scale t_kWh (the first feature column) using train statistics only
    t_kwh_idx = FEATURE_COLS.index("t_kWh")
    mean, std = X_train[:, :, t_kwh_idx].mean(), X_train[:, :, t_kwh_idx].std()
    std = std if std > 1e-6 else 1.0
    for arr in (X_train, X_val, X_test):
        arr[:, :, t_kwh_idx] = (arr[:, :, t_kwh_idx] - mean) / std
    y_train_scaled = (y_train - mean) / std
    y_val_scaled = (y_val - mean) / std
    # y_test stays in original units -- we'll rescale predictions back for honest metrics

    train_loader = DataLoader(WindowDataset(X_train, y_train_scaled, codes_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val_scaled, codes_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(WindowDataset(X_test, y_test, codes_test), batch_size=BATCH_SIZE)  # y_test unscaled

    model = GRUForecaster(n_features=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LAYERS, horizon=HORIZON,
                           n_meters=len(all_meters), embedding_dim=EMBEDDING_DIM).to(DEVICE)

    print(f"\nTraining GRU ({sum(p.numel() for p in model.parameters()):,} parameters)...")
    model = train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), "../gru_model.pt")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, code_batch, _ in test_loader:
            X_batch, code_batch = X_batch.to(DEVICE), code_batch.to(DEVICE)
            preds_scaled = model(X_batch, code_batch).cpu().numpy()
            preds = preds_scaled * std + mean  # back to real kWh units
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)

    overall_metrics = compute_metrics(y_test, all_preds)
    print("\n" + "=" * 60)
    print("GRU (direct 24h multi-output) -- OVERALL METRICS")
    print("=" * 60)
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.5f}")

    per_meter_rows = []
    for meter_id in np.unique(meters_test):
        mask = meters_test == meter_id
        m = compute_metrics(y_test[mask], all_preds[mask])
        m["meter"] = meter_id
        m["n_windows"] = int(mask.sum())
        per_meter_rows.append(m)
    per_meter_df = pd.DataFrame(per_meter_rows)
    per_meter_df.to_csv("gru_results_per_meter.csv", index=False)

    pd.Series(overall_metrics).to_csv("../gru_results_summary.csv")
    print("\nSaved gru_results_per_meter.csv, gru_results_summary.csv, gru_model.pt")
    print(
        "\nNEXT: compare these MAE/RMSE/sMAPE numbers directly against "
        "RandomForest's from step5_ml_direct.py. Whichever wins becomes the "
        "benchmark for the ARIMA-GRU hybrid to beat."
    )


if __name__ == "__main__":
    main()