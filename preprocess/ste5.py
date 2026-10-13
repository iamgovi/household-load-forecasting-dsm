"""
Step 5b: DIRECT 24-hour-ahead forecasting (no recursion).

This removes the compounding-error effect from step 5. Instead of predicting
hour-by-hour and feeding predictions back in, each model predicts the value
24 hours from now DIRECTLY, using only information available right now (lags,
rolling stats, and calendar features of the TARGET time, which are known in
advance since calendar structure is deterministic).

Why this is a fair test: it isolates the question "does ML find real structure
better than persistence?" from the separate question "does recursive forecasting
hurt ML models more than it hurts Naive?" -- which step 5 could not distinguish.

Baseline for comparison: at exactly a 24-hour horizon, plain persistence
("assume t+24 looks like right now") and seasonal naive ("assume t+24 looks
like the same hour 24h ago, i.e. now") are mathematically the same baseline --
so there's only one relevant naive baseline here, not two.

Run: python step5b_direct_forecast.py
Reads: train.csv, val.csv, test.csv
Writes: direct_results_per_meter.csv, direct_results_summary.csv
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
    print("xgboost not installed -- skipping.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed -- skipping.")

HORIZON_HOURS = 24
LAG_COLS = ["lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168"]
ROLL_COLS = ["roll_mean_3", "roll_std_3", "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168"]


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


def build_direct_dataset(df: pd.DataFrame, meter_code_map: dict, city_code_map: dict) -> pd.DataFrame:
    """Adds: target_24 (the value we're actually trying to predict), and
    calendar features describing the TARGET time (known in advance -- these
    are just calendar arithmetic, not real future information)."""
    df = df.sort_values(["meter", "x_Timestamp"]).copy()
    df["target_24"] = df.groupby("meter")["t_kWh"].shift(-HORIZON_HOURS)

    target_time = df["x_Timestamp"] + pd.Timedelta(hours=HORIZON_HOURS)
    df["target_hour"] = target_time.dt.hour
    df["target_day_of_week"] = target_time.dt.dayofweek
    df["target_is_weekend"] = df["target_day_of_week"].isin([5, 6]).astype(int)
    df["target_month"] = target_time.dt.month

    df["meter_code"] = df["meter"].map(meter_code_map)
    df["city_code"] = df["city"].map(city_code_map)

    return df.dropna(subset=["target_24"] + LAG_COLS + ROLL_COLS)


def get_feature_cols():
    return LAG_COLS + ROLL_COLS + [
        "target_hour", "target_day_of_week", "target_is_weekend", "target_month",
        "meter_code", "city_code",
    ]


def main():
    train = pd.read_csv("../train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("../val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])

    all_meters = sorted(set(train["meter"]) | set(val["meter"]) | set(test["meter"]))
    meter_code_map = {m: i for i, m in enumerate(all_meters)}
    city_code_map = {c: i for i, c in enumerate(train["city"].unique())}

    train_d = build_direct_dataset(train, meter_code_map, city_code_map)
    test_d = build_direct_dataset(test, meter_code_map, city_code_map)

    feature_cols = get_feature_cols()
    print(f"Direct-forecast training set: {len(train_d):,} rows "
          f"(after dropping rows near each meter's split boundary with no 24h-ahead target)")
    print(f"Direct-forecast test set: {len(test_d):,} rows")

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

    X_train = train_d[feature_cols]
    y_train = train_d["target_24"]

    print(f"Fitting {list(models.keys())}...")
    for name, model in models.items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)

    # per-meter evaluation, single-shot direct predictions -- no loop/recursion needed
    results = []
    for meter_id, g in test_d.groupby("meter"):
        row = {"meter": meter_id, "city": g["city"].iloc[0], "n_test_rows": len(g)}
        X_test = g[feature_cols]
        y_test = g["target_24"].values

        # the naive baseline for a 24h horizon: assume t+24 looks like right now (t_kWh at time t)
        naive_pred = g["t_kWh"].values
        for k, v in compute_metrics(y_test, naive_pred).items():
            row[f"Naive24_{k}"] = v

        for name, model in models.items():
            preds = model.predict(X_test)
            for k, v in compute_metrics(y_test, preds).items():
                row[f"{name}_{k}"] = v
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("direct_results_per_meter.csv", index=False)

    metric_cols = [c for c in results_df.columns if any(c.endswith(m) for m in ["_MAE", "_RMSE", "_sMAPE"])]
    summary = results_df[metric_cols].mean().sort_index()
    print("\n" + "=" * 60)
    print("DIRECT 24H-AHEAD FORECAST -- OVERALL MEAN METRICS")
    print("=" * 60)
    print(summary)
    summary.to_csv("direct_results_summary.csv")

    print(
        "\nCompare Naive24 against the ML models above. If ML now beats Naive24 "
        "(it didn't in the recursive version), that confirms the compounding-error "
        "explanation. If Naive24 still wins, the noisy-data explanation is the real story."
    )


if __name__ == "__main__":
    main()