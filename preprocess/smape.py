"""
Step 1 (diagnostic): does Random Forest's worse sMAPE (vs Naive24) come from
low-consumption hours specifically, where sMAPE is known to be unstable?

Splits the direct-forecast test set into "low usage" (bottom 10% of actual
t_kWh at the target time) vs "normal usage" (the rest), and computes sMAPE
for both Naive24 and Random Forest separately on each group.

Run: python step1_smape_diagnostic.py
Reads: train.csv, test.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HORIZON_HOURS = 24
LAG_COLS = ["lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168"]
ROLL_COLS = ["roll_mean_3", "roll_std_3", "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168"]
LOW_USAGE_PERCENTILE = 10


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-6
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def build_direct_dataset(df, meter_code_map, city_code_map):
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
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])

    all_meters = sorted(set(train["meter"]) | set(test["meter"]))
    meter_code_map = {m: i for i, m in enumerate(all_meters)}
    city_code_map = {c: i for i, c in enumerate(train["city"].unique())}

    train_d = build_direct_dataset(train, meter_code_map, city_code_map)
    test_d = build_direct_dataset(test, meter_code_map, city_code_map)
    feature_cols = get_feature_cols()

    print("Fitting Random Forest (same config as step5b)...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(train_d[feature_cols], train_d["target_24"])

    test_d = test_d.copy()
    test_d["naive24_pred"] = test_d["t_kWh"]
    test_d["rf_pred"] = rf.predict(test_d[feature_cols])

    threshold = np.percentile(test_d["target_24"], LOW_USAGE_PERCENTILE)
    print(f"\nLow-usage threshold (bottom {LOW_USAGE_PERCENTILE}%): target_24 <= {threshold:.4f} kWh")

    low = test_d[test_d["target_24"] <= threshold]
    normal = test_d[test_d["target_24"] > threshold]

    print(f"\nLow-usage rows: {len(low):,} ({len(low)/len(test_d)*100:.1f}% of test set)")
    print(f"Normal-usage rows: {len(normal):,}")

    print("\n" + "=" * 60)
    print("sMAPE BY USAGE BUCKET")
    print("=" * 60)
    rows = []
    for label, subset in [("Low usage (bottom 10%)", low), ("Normal usage (rest)", normal), ("All", test_d)]:
        rows.append({
            "bucket": label,
            "n_rows": len(subset),
            "Naive24_sMAPE": smape(subset["target_24"], subset["naive24_pred"]),
            "RandomForest_sMAPE": smape(subset["target_24"], subset["rf_pred"]),
            "Naive24_MAE": np.mean(np.abs(subset["target_24"] - subset["naive24_pred"])),
            "RandomForest_MAE": np.mean(np.abs(subset["target_24"] - subset["rf_pred"])),
        })
    report = pd.DataFrame(rows).set_index("bucket")
    print(report)

    print(
        "\nIf RandomForest_sMAPE is much worse than Naive24_sMAPE specifically in the "
        "'Low usage' row, but the two are close (or RF wins) in 'Normal usage', that "
        "confirms sMAPE instability at low values is driving the earlier headline "
        "sMAPE gap -- not a genuine RF weakness. Check MAE in the same rows too: if "
        "RF's MAE is competitive or better even in the low-usage bucket, that reinforces "
        "the same conclusion."
    )


if __name__ == "__main__":
    main()