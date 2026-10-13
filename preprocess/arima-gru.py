"""
Step 8: ARIMA-GRU residual hybrid.

Design:
  1. ARIMA models the "linear" part of each household's series.
  2. The GRU is trained on the RESIDUAL (actual - ARIMA's one-step fitted
     values) instead of the raw series -- it learns whatever nonlinear
     structure ARIMA left on the table.
  3. Final forecast = ARIMA's 24h rolling forecast + GRU's 24h residual forecast.

This is the actual test of your original hypothesis: does adding GRU on top
of ARIMA's residual beat plain GRU (step7) or Random Forest (step5)? Given
ARIMA alone lost badly to Naive in step 3, go in expecting the residual may
carry almost the whole signal -- meaning this could end up performing close
to plain GRU. That's a legitimate, reportable outcome for your ablation study,
not a failure of the hybrid.

Requires: pip install torch statsmodels

Run: python step8_arima_gru.py
Reads: train.csv, val.csv, test.csv
Writes: arima_gru_results_per_meter.csv, arima_gru_results_summary.csv
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ARIMA_ORDER = (2, 1, 2)          # same fixed order as step3, for a fair comparison
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

MIN_HISTORY_HOURS = SEQ_LEN * 3   # skip meters without enough history for ARIMA + GRU windows


def add_cyclical_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["x_Timestamp"].dt.dayofweek / 7)
    df["is_weekend"] = df["x_Timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
    return df


def to_regular_hourly(series: pd.Series):
    """Same reasoning as step3/step4 -- ARIMA needs a strictly regular index."""
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    reindexed = series.reindex(full_index)
    if reindexed.isna().any():
        reindexed = reindexed.interpolate(limit_direction="both")
    return reindexed


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-6
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "sMAPE": float(smape(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# ARIMA component
# ---------------------------------------------------------------------------

def fit_arima_and_get_residual(history: pd.Series, order):
    """Fits ARIMA, returns (fitted_model, residual_series). Residual = actual
    minus ARIMA's in-sample one-step fitted values -- this is the signal the
    GRU will be trained to predict instead of the raw series."""
    model_fit = ARIMA(history, order=order).fit()
    fitted = model_fit.fittedvalues
    residual = (history - fitted).dropna()
    return model_fit, residual


def rolling_arima_forecast(model_fit, test_series: pd.Series, horizon: int):
    """Same rolling-origin logic as step3: forecast a block, record it, then
    update the model with the TRUE observed block before forecasting the next
    one. Returns a Series of predictions aligned to test_series' index."""
    preds, idx = [], []
    pos, n = 0, len(test_series)
    while pos < n:
        window = test_series.iloc[pos: pos + horizon]
        forecast = model_fit.forecast(steps=len(window))
        preds.extend(forecast.values if hasattr(forecast, "values") else forecast)
        idx.extend(window.index)
        try:
            model_fit = model_fit.append(window, refit=False)
        except Exception:
            pass
        pos += horizon
    return pd.Series(preds, index=idx)


# ---------------------------------------------------------------------------
# GRU component (predicts RESIDUAL, not raw t_kWh)
# ---------------------------------------------------------------------------

def build_residual_windows(df: pd.DataFrame, residual_by_meter: dict, seq_len: int,
                            horizon: int, meter_code_map: dict):
    """Same windowing logic as step7, but the TARGET is the residual series
    (aligned by timestamp), not the raw t_kWh. Inputs stay raw+engineered
    features -- only the prediction target changes."""
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

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    codes = np.array(code_list, dtype=np.int64)
    return X, y, codes


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

    # ---- Stage 1: per-meter ARIMA fit -> residual series (train+val) ----
    print("\nFitting per-meter ARIMA models and computing residuals (this is the slow part)...")
    residual_by_meter = {}
    arima_fits = {}
    histories = {}

    for i, meter_id in enumerate(all_meters, 1):
        m_train = train[train["meter"] == meter_id]
        m_val = val[val["meter"] == meter_id]
        if len(m_train) + len(m_val) < MIN_HISTORY_HOURS:
            continue
        history = pd.concat([m_train, m_val]).set_index("x_Timestamp")["t_kWh"].sort_index()
        history_reg = to_regular_hourly(history)
        try:
            model_fit, residual = fit_arima_and_get_residual(history_reg, ARIMA_ORDER)
        except Exception as e:
            print(f"  [{i}/{len(all_meters)}] {meter_id}: ARIMA fit failed ({e}) -- skipping")
            continue
        residual_by_meter[meter_id] = residual
        arima_fits[meter_id] = model_fit
        histories[meter_id] = history_reg
        if i % 10 == 0:
            print(f"  [{i}/{len(all_meters)}] done")

    print(f"ARIMA fit succeeded for {len(residual_by_meter)}/{len(all_meters)} meters")

    # ---- Stage 2: train GRU on residuals ----
    print("\nBuilding residual training windows...")
    X_train, y_train, codes_train = build_residual_windows(train, residual_by_meter, SEQ_LEN, HORIZON, meter_code_map)
    X_val, y_val, codes_val = build_residual_windows(val, residual_by_meter, SEQ_LEN, HORIZON, meter_code_map)
    print(f"train windows: {len(X_train):,}  val windows: {len(X_val):,}")

    if len(X_train) == 0:
        raise RuntimeError("No residual training windows built -- check MIN_HISTORY_HOURS / SEQ_LEN.")

    t_kwh_idx = FEATURE_COLS.index("t_kWh")
    x_mean, x_std = X_train[:, :, t_kwh_idx].mean(), X_train[:, :, t_kwh_idx].std()
    x_std = x_std if x_std > 1e-6 else 1.0
    y_mean, y_std = y_train.mean(), y_train.std()
    y_std = y_std if y_std > 1e-6 else 1.0

    for arr in (X_train, X_val):
        arr[:, :, t_kwh_idx] = (arr[:, :, t_kwh_idx] - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    train_loader = DataLoader(WindowDataset(X_train, y_train_scaled, codes_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val_scaled, codes_val), batch_size=BATCH_SIZE)

    gru = GRUResidualForecaster(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, HORIZON,
                                 len(all_meters), EMBEDDING_DIM).to(DEVICE)
    print(f"\nTraining residual GRU ({sum(p.numel() for p in gru.parameters()):,} parameters)...")
    gru = train_model(gru, train_loader, val_loader)

    # ---- Stage 3: evaluate the hybrid on test -- ARIMA forecast + GRU residual forecast ----
    print("\nEvaluating ARIMA + GRU hybrid on test set...")
    results = []
    for meter_id in sorted(set(test["meter"])):
        if meter_id not in arima_fits:
            continue
        m_test = test[test["meter"] == meter_id].sort_values("x_Timestamp")
        if len(m_test) < HORIZON:
            continue

        test_series = m_test.set_index("x_Timestamp")["t_kWh"]
        test_series_reg = to_regular_hourly(test_series)

        arima_forecast = rolling_arima_forecast(arima_fits[meter_id], test_series_reg, HORIZON)

        # Build GRU inputs for the same test windows, using a buffer that grows
        # with TRUE observed values between blocks -- same rolling-origin logic.
        buffer = histories[meter_id].copy()
        m_test_cal = m_test.copy()
        pos, n = 0, len(test_series_reg)
        gru_residual_preds = []
        gru_pred_index = []

        while pos < n:
            window_index = test_series_reg.index[pos: pos + HORIZON]
            input_window_index = pd.date_range(end=window_index[0] - pd.Timedelta(hours=1), periods=SEQ_LEN, freq="h")

            if not all(t in buffer.index for t in input_window_index):
                pos += HORIZON
                continue  # not enough clean history yet for this block, skip it

            seg = pd.DataFrame({"x_Timestamp": input_window_index})
            seg["t_kWh"] = buffer.reindex(input_window_index).values
            seg = add_cyclical_calendar(seg)
            for col in ["lag_24", "lag_168", "roll_mean_24"]:
                if col == "lag_24":
                    seg[col] = buffer.reindex(input_window_index - pd.Timedelta(hours=24)).values
                elif col == "lag_168":
                    seg[col] = buffer.reindex(input_window_index - pd.Timedelta(hours=168)).values
                elif col == "roll_mean_24":
                    seg[col] = [buffer.reindex(pd.date_range(t - pd.Timedelta(hours=24), t, freq="h", inclusive="left")).mean()
                                 for t in input_window_index]
            if seg[FEATURE_COLS].isna().any(axis=None):
                pos += HORIZON
                continue

            X_input = seg[FEATURE_COLS].values.astype(np.float32)
            X_input[:, t_kwh_idx] = (X_input[:, t_kwh_idx] - x_mean) / x_std
            X_tensor = torch.tensor(X_input).unsqueeze(0).to(DEVICE)
            code_tensor = torch.tensor([meter_code_map[meter_id]], dtype=torch.long).to(DEVICE)

            gru.eval()
            with torch.no_grad():
                pred_scaled = gru(X_tensor, code_tensor).cpu().numpy()[0]
            pred_residual = pred_scaled * y_std + y_mean

            gru_residual_preds.extend(pred_residual[:len(window_index)])
            gru_pred_index.extend(window_index)

            true_window = test_series_reg.loc[window_index]
            buffer = pd.concat([buffer, true_window])
            pos += HORIZON

        if not gru_pred_index:
            continue

        gru_residual_series = pd.Series(gru_residual_preds, index=gru_pred_index)
        common_idx = arima_forecast.index.intersection(gru_residual_series.index)
        if len(common_idx) == 0:
            continue

        final_forecast = arima_forecast.loc[common_idx] + gru_residual_series.loc[common_idx]
        actual = test_series_reg.loc[common_idx]

        metrics = compute_metrics(actual.values, final_forecast.values)
        arima_only_metrics = compute_metrics(actual.values, arima_forecast.loc[common_idx].values)

        row = {"meter": meter_id, "city": m_test["city"].iloc[0], "n_points": len(common_idx)}
        for k, v in metrics.items():
            row[f"Hybrid_{k}"] = v
        for k, v in arima_only_metrics.items():
            row[f"ARIMA_only_{k}"] = v
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("arima_gru_results_per_meter.csv", index=False)

    metric_cols = [c for c in results_df.columns if any(c.endswith(m) for m in ["_MAE", "_RMSE", "_sMAPE"])]
    summary = results_df[metric_cols].mean().sort_index()
    print("\n" + "=" * 60)
    print("ARIMA-GRU HYBRID -- OVERALL MEAN METRICS")
    print("(includes ARIMA_only columns so you can see what GRU's residual actually added)")
    print("=" * 60)
    print(summary)
    summary.to_csv("arima_gru_results_summary.csv")

    print(
        "\nCompare Hybrid_* against ARIMA_only_* (does the GRU residual help at all?), "
        "and against your step5/step7 numbers (does the hybrid beat plain RF or plain GRU?)."
    )


if __name__ == "__main__":
    main()