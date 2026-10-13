"""
Step 5: Classical ML models, evaluated the SAME way as your statistical baselines
(24-hour-ahead rolling forecasts) so the comparison is actually fair.
IMPORTANT DIFFERENCE FROM A NAIVE ML EVALUATION:
Your train/val/test CSVs already contain lag_1, lag_24, roll_mean_24 etc. computed
from TRUE past values. If we just fed those columns straight into a trained model
and scored it against test.csv, we'd be doing a "one-step-ahead using ground-truth
history" evaluation -- NOT comparable to the 24-step recursive forecasts your
ARIMA/Holt-Winters baselines were scored on. That would make ML look artificially
better than it really is for actual 24h-ahead deployment.
So this script instead does RECURSIVE forecasting: within each 24-hour window, it
feeds each prediction back in to compute the next hour's lag features, exactly
like a real deployed forecaster would have to. Between 24h windows, it resets
using the true observed values (same "rolling origin" logic as step4).
Models: Linear Regression, Random Forest (always available via sklearn).
XGBoost / LightGBM included if installed (pip install xgboost lightgbm) --
skipped with a note if not, rather than crashing.
Run: python step5_ml_models.py
Reads: train.csv, val.csv, test.csv
Writes: ml_results_per_meter.csv, ml_results_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- skipping. `pip install xgboost` to include it.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed -- skipping. `pip install lightgbm` to include it.")

FORECAST_HORIZON = 24
LAGS = [1, 2, 3, 24, 48, 168]              # must match step2_pipeline.py
ROLLING_WINDOWS = [3, 24, 168]              # must match step2_pipeline.py
MAX_METERS = None                           # set small while testing


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-6
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "sMAPE": smape(y_true, y_pred),
    }


def build_feature_row(buffer: pd.Series, timestamp: pd.Timestamp, meter_code: int, city_code: int) -> dict:
    """Recreate exactly the feature set step2_pipeline.py builds, but from a
    live buffer that may contain predicted (not just true) recent values."""
    row = {}
    for lag in LAGS:
        idx = timestamp - pd.Timedelta(hours=lag)
        row[f"lag_{lag}"] = buffer.get(idx, np.nan)
    for window in ROLLING_WINDOWS:
        window_start = timestamp - pd.Timedelta(hours=window)
        window_vals = buffer[(buffer.index >= window_start) & (buffer.index < timestamp)]
        row[f"roll_mean_{window}"] = window_vals.mean() if len(window_vals) else np.nan
        row[f"roll_std_{window}"] = window_vals.std() if len(window_vals) else np.nan
    row["hour"] = timestamp.hour
    row["day_of_week"] = timestamp.dayofweek
    row["is_weekend"] = int(timestamp.dayofweek in (5, 6))
    row["month"] = timestamp.month
    row["meter_code"] = meter_code
    row["city_code"] = city_code
    return row


def recursive_forecast(model, feature_cols, buffer, forecast_timestamps, meter_code, city_code):
    """Predict one step, append it to the buffer, use it to build the next
    step's features -- repeated across the full window. This is what makes it
    a genuine multi-step forecast rather than a lookup of true future lags."""
    buffer = buffer.copy()
    preds = []
    for ts in forecast_timestamps:
        feat_row = build_feature_row(buffer, ts, meter_code, city_code)
        X = pd.DataFrame([feat_row])[feature_cols]
        if X.isna().any(axis=None):
            # not enough history yet (e.g. lag_168 unavailable this early) -- fall back
            # to the most recent known value rather than crashing or feeding NaN to the model
            pred = buffer.iloc[-1]
        else:
            pred = model.predict(X)[0]
        preds.append(pred)
        buffer.loc[ts] = pred
    return preds


def get_feature_cols():
    cols = [f"lag_{l}" for l in LAGS]
    cols += [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    cols += [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    cols += ["hour", "day_of_week", "is_weekend", "month", "meter_code", "city_code"]
    return cols


def run_for_meter(meter_id, meter_code, city_code, models, feature_cols, train, val, test):
    m_test = test[test["meter"] == meter_id].sort_values("x_Timestamp")
    if len(m_test) < FORECAST_HORIZON:
        return None

    # buffer starts as this meter's true history from train+val (needed to seed lags)
    m_hist = pd.concat([
        train[train["meter"] == meter_id], val[val["meter"] == meter_id]
    ]).sort_values("x_Timestamp")
    buffer_full = pd.date_range(m_hist["x_Timestamp"].min(), m_hist["x_Timestamp"].max(), freq="h")
    buffer = m_hist.set_index("x_Timestamp")["t_kWh"].reindex(buffer_full).interpolate(limit_direction="both")

    test_series = m_test.set_index("x_Timestamp")["t_kWh"]
    row = {"meter": meter_id, "city": m_hist["city"].iloc[0] if len(m_hist) else None}

    pos = 0
    n = len(test_series)
    per_model_preds = {name: [] for name in models}
    actuals = []

    while pos < n:
        window = test_series.iloc[pos: pos + FORECAST_HORIZON]
        actuals.extend(window.values)
        for name, model in models.items():
            preds = recursive_forecast(model, feature_cols, buffer, window.index, meter_code, city_code)
            per_model_preds[name].extend(preds)
        # advance the shared buffer with TRUE observed values before the next window
        buffer = pd.concat([buffer, window])
        pos += FORECAST_HORIZON

    for name, preds in per_model_preds.items():
        metrics = compute_metrics(actuals, preds)
        for k, v in metrics.items():
            row[f"{name}_{k}"] = v

    return row


def main():
    train = pd.read_csv("../train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("../val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])

    all_meters = sorted(set(train["meter"]) | set(val["meter"]))
    meter_code_map = {m: i for i, m in enumerate(all_meters)}
    city_code_map = {c: i for i, c in enumerate(train["city"].unique())}

    for df in (train, val, test):
        df["meter_code"] = df["meter"].map(meter_code_map)
        df["city_code"] = df["city"].map(city_code_map)

    feature_cols = get_feature_cols()
    train_fit = train.dropna(subset=feature_cols + ["t_kWh"])

    print(f"Training on {len(train_fit):,} rows, {train_fit['meter'].nunique()} meters (pooled model)")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                          n_jobs=-1, random_state=42)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                            n_jobs=-1, random_state=42, verbosity=-1)

    print(f"Fitting {list(models.keys())} on the pooled training set...")
    X_train = train_fit[feature_cols]
    y_train = train_fit["t_kWh"]
    for name, model in models.items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)

    meters = sorted(test["meter"].unique())
    if MAX_METERS:
        meters = meters[:MAX_METERS]
        print(f"MAX_METERS set -- running on {len(meters)} meters only")

    results = []
    for i, meter_id in enumerate(meters, 1):
        print(f"[{i}/{len(meters)}] {meter_id}...")
        row = run_for_meter(
            meter_id, meter_code_map[meter_id], city_code_map[test[test['meter'] == meter_id]['city'].iloc[0]],
            models, feature_cols, train, val, test,
        )
        if row:
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("ml_results_per_meter.csv", index=False)
    print(f"\nSaved per-meter results -> ml_results_per_meter.csv")

    metric_cols = [c for c in results_df.columns if any(c.endswith(m) for m in ["_MAE", "_RMSE", "_sMAPE"])]
    summary = results_df[metric_cols].mean().sort_index()
    print("\n" + "=" * 60)
    print("OVERALL MEAN METRICS (compare directly against your step4 baseline numbers)")
    print("=" * 60)
    print(summary)
    summary.to_csv("ml_results_summary.csv")

    print(
        "\nNEXT: compare this to Naive's RMSE/sMAPE from step 4. If nothing here beats "
        "Naive either, that's a genuinely important finding about this dataset -- "
        "report it honestly rather than picking the least-bad ML model and calling it a win."
    )


if __name__ == "__main__":
    main()