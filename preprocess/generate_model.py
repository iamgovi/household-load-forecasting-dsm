"""
Precomputes the ARIMA-GRU hybrid forecast for EVERY household's next 24 hours
(from the most recent available data point), and saves it to forecasts.csv.

WHY PRECOMPUTE instead of running ARIMA-GRU live in the app:
ARIMA-GRU is slow (per-meter ARIMA fitting) and has more moving parts (torch +
statsmodels both loaded, buffer-building logic) than Random Forest. Running it
live in a Streamlit app risks slow load times or a crash during your demo.
Precomputing once and having the app just read a CSV removes that risk
entirely -- the app becomes simple and fast regardless of which model produced
the numbers.

Run ONCE (before your demo): python generate_forecasts_arima_gru.py
Reads: train.csv, val.csv, test.csv
Writes: forecasts_arima_gru.csv (one row per household: meter, city,
        last_timestamp, forecast_h1 ... forecast_h24)
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ARIMA_ORDER = (2, 1, 2)
SEQ_LEN = 168
HORIZON = 24
BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EMBEDDING_DIM = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = ["t_kWh", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                "lag_24", "lag_168", "roll_mean_24"]
MIN_HISTORY_HOURS = SEQ_LEN * 3


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
    def __init__(self, X, y, codes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.codes = torch.tensor(codes, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.codes[idx], self.y[idx]


class GRUResidualForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, horizon, n_meters, embedding_dim):
        super().__init__()
        self.meter_embedding = nn.Embedding(n_meters, embedding_dim)
        self.gru = nn.GRU(n_features + embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x, meter_code):
        emb = self.meter_embedding(meter_code)
        emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x_with_id = torch.cat([x, emb_seq], dim=-1)
        _, h_n = self.gru(x_with_id)
        return self.fc(h_n[-1])


def build_residual_windows(df, residual_by_meter, seq_len, horizon, meter_code_map):
    X_list, y_list, code_list = [], [], []
    for meter_id, g in df.groupby("meter"):
        if meter_id not in residual_by_meter:
            continue
        residual = residual_by_meter[meter_id]
        g = g.sort_values("x_Timestamp").reset_index(drop=True)
        g = g[g["x_Timestamp"].isin(residual.index)].reset_index(drop=True)
        time_diff = g["x_Timestamp"].diff().dt.total_seconds().div(3600).fillna(1.0)
        block_id = (time_diff != 1.0).cumsum()
        meter_code = meter_code_map[meter_id]

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
                code_list.append(meter_code)

    return (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32),
            np.array(code_list, dtype=np.int64))


def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_batch, code_batch, y_batch in train_loader:
            X_batch, code_batch, y_batch = X_batch.to(DEVICE), code_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch, code_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, code_batch, y_batch in val_loader:
                X_batch, code_batch, y_batch = X_batch.to(DEVICE), code_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch, code_batch), y_batch).item() * len(X_batch)
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
    train = pd.read_csv("../train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("../val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])
    train = add_cyclical_calendar(train)
    val = add_cyclical_calendar(val)
    test = add_cyclical_calendar(test)

    all_meters = sorted(set(train["meter"]) | set(val["meter"]) | set(test["meter"]))
    meter_code_map = {m: i for i, m in enumerate(all_meters)}

    # Fit ARIMA on the FULL available history (train+val+test) per meter --
    # for the deployed app we want the best possible forecast starting from
    # "now" (the most recent real data point), so we use everything we have.
    print("\nFitting per-meter ARIMA models on full history...")
    residual_by_meter, arima_fits, full_histories = {}, {}, {}
    for i, meter_id in enumerate(all_meters, 1):
        m_all = pd.concat([
            train[train["meter"] == meter_id],
            val[val["meter"] == meter_id],
            test[test["meter"] == meter_id],
        ])
        if len(m_all) < MIN_HISTORY_HOURS:
            continue
        history = m_all.set_index("x_Timestamp")["t_kWh"].sort_index()
        history_reg = to_regular_hourly(history)
        try:
            model_fit = ARIMA(history_reg, order=ARIMA_ORDER).fit()
        except Exception as e:
            print(f"  [{i}/{len(all_meters)}] {meter_id}: ARIMA failed ({e}) -- skipping")
            continue
        fitted = model_fit.fittedvalues
        residual_by_meter[meter_id] = (history_reg - fitted).dropna()
        arima_fits[meter_id] = model_fit
        full_histories[meter_id] = history_reg
        if i % 10 == 0:
            print(f"  [{i}/{len(all_meters)}] done")

    print(f"ARIMA succeeded for {len(residual_by_meter)}/{len(all_meters)} meters")

    # Train the residual GRU on ALL data (train+val+test) pooled -- for the
    # final deployed forecast we're not holding out a test set anymore, we
    # want the best model using everything available.
    print("\nBuilding residual windows across full history...")
    all_df = pd.concat([train, val, test], ignore_index=True)
    X_all, y_all, codes_all = build_residual_windows(all_df, residual_by_meter, SEQ_LEN, HORIZON, meter_code_map)
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

    train_loader = DataLoader(WindowDataset(X_all[train_idx], y_scaled[train_idx], codes_all[train_idx]),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_all[val_idx], y_scaled[val_idx], codes_all[val_idx]),
                             batch_size=BATCH_SIZE)

    gru = GRUResidualForecaster(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, HORIZON,
                                 len(all_meters), EMBEDDING_DIM).to(DEVICE)
    print(f"\nTraining final residual GRU ({sum(p.numel() for p in gru.parameters()):,} params)...")
    gru = train_model(gru, train_loader, val_loader)

    # Save the trained hybrid pieces in case you want to regenerate forecasts later
    torch.save(gru.state_dict(), "gru_residual_model.pt")

    # ---- Generate the actual "tomorrow" forecast per household ----
    print("\nGenerating next-24h forecasts per household...")
    rows = []
    for meter_id in sorted(residual_by_meter.keys()):
        history_reg = full_histories[meter_id]
        model_fit = arima_fits[meter_id]
        last_time = history_reg.index.max()

        arima_forecast = model_fit.forecast(steps=HORIZON).values

        input_window_index = pd.date_range(end=last_time, periods=SEQ_LEN, freq="h")
        if not all(t in history_reg.index for t in input_window_index):
            continue

        seg = pd.DataFrame({"x_Timestamp": input_window_index})
        seg["t_kWh"] = history_reg.reindex(input_window_index).values
        seg = add_cyclical_calendar(seg)
        seg["lag_24"] = history_reg.reindex(input_window_index - pd.Timedelta(hours=24)).values
        seg["lag_168"] = history_reg.reindex(input_window_index - pd.Timedelta(hours=168)).values
        seg["roll_mean_24"] = [
            history_reg.reindex(pd.date_range(t - pd.Timedelta(hours=24), t, freq="h", inclusive="left")).mean()
            for t in input_window_index
        ]
        if seg[FEATURE_COLS].isna().any(axis=None):
            continue

        X_input = seg[FEATURE_COLS].values.astype(np.float32)
        X_input[:, t_kwh_idx] = (X_input[:, t_kwh_idx] - x_mean) / x_std
        X_tensor = torch.tensor(X_input).unsqueeze(0).to(DEVICE)
        code_tensor = torch.tensor([meter_code_map[meter_id]], dtype=torch.long).to(DEVICE)

        gru.eval()
        with torch.no_grad():
            pred_residual_scaled = gru(X_tensor, code_tensor).cpu().numpy()[0]
        pred_residual = pred_residual_scaled * y_std + y_mean

        final_forecast = np.clip(arima_forecast + pred_residual, a_min=0, a_max=None)

        city = all_df[all_df["meter"] == meter_id]["city"].iloc[0]
        row = {"meter": meter_id, "city": city, "last_timestamp": last_time}
        for h in range(HORIZON):
            row[f"forecast_h{h+1}"] = final_forecast[h]
        rows.append(row)

    forecasts_df = pd.DataFrame(rows)
    forecasts_df.to_csv("forecasts_arima_gru.csv", index=False)
    print(f"\nSaved forecasts_arima_gru.csv with {len(forecasts_df)} households")


if __name__ == "__main__":
    main()