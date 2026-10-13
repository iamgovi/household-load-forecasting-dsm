"""
Step 1: Verify and clean the Mathura & Bareilly smart meter dataset.

Run this FIRST, before any feature engineering or modeling. It answers:
  - how many households (meters) do we actually have usable data for?
  - is t_kWh cumulative (a running counter) or interval consumption?
  - how consistent is the sampling frequency per meter?
  - where are the gaps / missing periods?
  - are there garbage readings (negative, zero-stuck, voltage/current outliers)?

Update CSV_PATH below, then run: python step1_verify_clean.py
"""

import pandas as pd
import numpy as np

CSV_PATH = "../archive (1)/CEEW - Smart meter data Mathura 2020.csv"  # <-- update to your actual filename
OUTPUT_PATH = "../cleaned_meter_data.parquet"

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df['meter'].nunique()} unique meters")
    print("\nColumn dtypes before parsing:")
    print(df.dtypes)

    df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"], errors="coerce")
    n_bad_ts = df["x_Timestamp"].isna().sum()
    if n_bad_ts:
        print(f"\nWARNING: {n_bad_ts} rows had unparseable timestamps — dropping them")
        df = df.dropna(subset=["x_Timestamp"])

    return df


def inventory(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PER-METER INVENTORY")
    print("=" * 60)
    summary = df.groupby("meter")["x_Timestamp"].agg(
        first="min", last="max", n_rows="count"
    )
    summary["span_days"] = (summary["last"] - summary["first"]).dt.total_seconds() / 86400
    print(summary.sort_values("n_rows"))

    print("\nMeters with suspiciously few rows (< 10th percentile):")
    low_cutoff = summary["n_rows"].quantile(0.10)
    print(summary[summary["n_rows"] < low_cutoff])


def check_sampling_interval(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SAMPLING INTERVAL CHECK (per meter, then overall)")
    print("=" * 60)
    df_sorted = df.sort_values(["meter", "x_Timestamp"])
    diffs = df_sorted.groupby("meter")["x_Timestamp"].diff()
    print("\nMost common time gaps between consecutive readings:")
    print(diffs.value_counts().head(10))
    print(
        "\nIf this is NOT a single dominant value (e.g. mostly 3 min with a long tail),"
        "\nyour data has irregular sampling and you'll need to resample/interpolate"
        "\nbefore feature engineering."
    )


def check_cumulative_vs_interval(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("IS t_kWh CUMULATIVE OR INTERVAL CONSUMPTION?")
    print("=" * 60)
    df_sorted = df.sort_values(["meter", "x_Timestamp"])
    is_monotonic = df_sorted.groupby("meter")["t_kWh"].apply(
        lambda s: s.is_monotonic_increasing
    )
    pct_monotonic = is_monotonic.mean() * 100
    print(f"{pct_monotonic:.1f}% of meters have monotonically increasing t_kWh")
    if pct_monotonic > 80:
        print(
            "-> t_kWh is almost certainly a CUMULATIVE counter. "
            "You MUST use .diff() per meter to get actual consumption per interval "
            "before treating it as 'load'. Watch for counter resets (diff goes negative) "
            "and handle those explicitly (drop or treat as a meter reset event)."
        )
    else:
        print(
            "-> t_kWh looks like it's already interval consumption (not cumulative). "
            "Still worth spot-checking a few meters manually to confirm."
        )


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("OUTLIER / GARBAGE READING CHECK")
    print("=" * 60)
    n_neg_kwh = (df["t_kWh"] < 0).sum()
    n_zero_voltage = (df["z_Avg Voltage (Volt)"] == 0).sum()
    n_zero_current = (df["z_Avg Current (Amp)"] == 0).sum()
    print(f"Negative t_kWh readings: {n_neg_kwh}")
    print(f"Zero-voltage readings (likely outage or sensor fault): {n_zero_voltage}")
    print(f"Zero-current readings (no load drawn, could be legit or fault): {n_zero_current}")

    # Flag rather than silently drop -- decide handling after seeing the scale of the issue
    df["flag_zero_voltage"] = df["z_Avg Voltage (Volt)"] == 0
    df["flag_negative_kwh"] = df["t_kWh"] < 0
    return df


def missingness_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("MISSINGNESS BY COLUMN")
    print("=" * 60)
    print((df.isna().mean() * 100).round(2).astype(str) + " %")


def main():
    df = load_data(CSV_PATH)
    inventory(df)
    check_sampling_interval(df)
    check_cumulative_vs_interval(df)
    df = flag_outliers(df)
    missingness_report(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved flagged (not yet cleaned) data to {OUTPUT_PATH}")
    print(
        "\nNEXT: read the printed output above and decide, in writing (for your methods "
        "section), how you're handling: counter resets, zero-voltage periods, sampling "
        "gaps, and any meters with too little data to be usable. Don't silently drop rows "
        "without logging what and why."
    )


if __name__ == "__main__":
    main()