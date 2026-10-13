"""
Legacy compatibility helpers for preprocessing.

The authoritative research entry point is ``research_pipeline.py``. Running
this module delegates there and therefore does not overwrite root-level legacy
splits.

Pipeline order matters — this follows it strictly:
  1. Load + tag city
  2. Drop meters with too little history to be usable
  3. Resample from 3-min to hourly (configurable) per meter
  4. Chronological train/val/test split PER METER (before any feature fitting)
  5. Build lag/rolling/calendar features
  6. Fit any scaler ONLY on the training split, apply to val/test

Update the two CSV paths below, then run: python step2_pipeline.py
Outputs: train.csv, val.csv, test.csv in the same folder.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 30)

BAREILLY_CSV = "archive (1)/CEEW - Smart meter data Bareilly 2020.csv"
MATHURA_CSV = "archive (1)/CEEW - Smart meter data Mathura 2020.csv"

RESAMPLE_FREQ = "1h"          # try "15min" as an alternative and compare
MIN_COMPLETENESS = 0.60       # keep a meter only if it has >= 60% of the readings
                               # expected over ITS OWN observed date span (at 3-min
                               # intervals = 20 readings/hour). This is scaled per
                               # meter rather than a fixed row count, so it doesn't
                               # penalize cities/meters with a shorter observed span.
READINGS_PER_HOUR = 20         # 3-minute sampling -> 20 readings/hour

# Chronological split boundaries (per meter, applied to its own available range)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 goes to test

LAGS = [1, 2, 3, 24, 48, 168]          # in units of RESAMPLE_FREQ (hours, if "1h")
ROLLING_WINDOWS = [3, 24, 168]


def load_and_tag(path: str, city: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"], errors="coerce")
    df = df.dropna(subset=["x_Timestamp"])
    df["city"] = city
    return df


def combine(bareilly_path: str, mathura_path: str) -> pd.DataFrame:
    df_br = load_and_tag(bareilly_path, "Bareilly")
    df_mh = load_and_tag(mathura_path, "Mathura")
    df = pd.concat([df_br, df_mh], ignore_index=True)
    print(f"Combined: {len(df):,} rows, {df['meter'].nunique()} meters, "
          f"{df['city'].nunique()} cities")
    return df


def analyze_missingness_pattern(df: pd.DataFrame, expected_interval_minutes: int = 3) -> None:
    """Distinguishes two very different explanations for low completeness:
      - OUTAGE-LIKE: missingness concentrated in a few long, contiguous gaps
        (consistent with real power outages -- meter has nothing to transmit)
      - FAULT-LIKE: missingness scattered across many small/random gaps
        (more consistent with a flaky sensor/transmission link, not real outages)
    This matters because it changes how you're allowed to interpret and write
    up the Mathura vs. Bareilly completeness difference."""
    expected_gap = pd.Timedelta(minutes=expected_interval_minutes)
    rows = []

    for meter_id, g in df.groupby("meter"):
        g = g.sort_values("x_Timestamp")
        city = g["city"].iloc[0]
        diffs = g["x_Timestamp"].diff().dropna()
        gaps = diffs[diffs > expected_gap]
        if len(gaps) == 0:
            continue
        missing_time = gaps.sum()
        long_gaps = gaps[gaps > pd.Timedelta(hours=1)]
        very_long_gaps = gaps[gaps > pd.Timedelta(hours=6)]
        rows.append({
            "meter": meter_id,
            "city": city,
            "n_gap_events": len(gaps),
            "total_missing_hours": missing_time.total_seconds() / 3600,
            "n_gaps_over_1h": len(long_gaps),
            "n_gaps_over_6h": len(very_long_gaps),
            "pct_missing_time_in_gaps_over_1h": (
                long_gaps.sum().total_seconds() / missing_time.total_seconds() * 100
                if missing_time.total_seconds() > 0 else 0
            ),
        })

    report = pd.DataFrame(rows).set_index("meter")
    print("\n" + "=" * 60)
    print("MISSINGNESS PATTERN: OUTAGE-LIKE vs FAULT-LIKE")
    print("=" * 60)

    print("\nBy city -- average number of distinct gap events per meter, and")
    print("average total missing hours per meter (this is what actually")
    print("distinguishes 'more outages' from 'longer outages'):")
    print(report.groupby("city")[["n_gap_events", "total_missing_hours",
                                   "n_gaps_over_6h"]].mean().round(1))

    print("\nBy city -- share of missing time sitting in gaps LONGER than 1 hour")
    print("(high % => missingness is dominated by a few long outages;")
    print(" low % => missingness is scattered across many short/random gaps):")
    print(report.groupby("city")["pct_missing_time_in_gaps_over_1h"].describe())

    print("\nTop 10 meters by total missing time, with their gap-length profile:")
    print(
        report.sort_values("total_missing_hours", ascending=False)
        .head(10)[["city", "n_gap_events", "total_missing_hours", "n_gaps_over_1h", "n_gaps_over_6h"]]
    )


def drop_low_completeness_meters(df: pd.DataFrame, min_completeness: float,
                                  readings_per_hour: int) -> pd.DataFrame:
    """Drop meters whose actual row count is too low RELATIVE TO their own
    observed date span -- not relative to a fixed number. This avoids penalizing
    a whole city just because its meters were installed later / have a shorter
    span in this export, which is what happened with a flat row-count cutoff."""
    stats = df.groupby(["meter", "city"])["x_Timestamp"].agg(first="min", last="max", n_rows="count")
    span_hours = (stats["last"] - stats["first"]).dt.total_seconds() / 3600
    expected_rows = span_hours * readings_per_hour
    stats["completeness"] = stats["n_rows"] / expected_rows

    print("\nCompleteness by city (meter-level):")
    print(stats.groupby("city")["completeness"].describe())

    keep = stats[stats["completeness"] >= min_completeness].index.get_level_values("meter")
    dropped = stats[stats["completeness"] < min_completeness]
    if len(dropped):
        print(f"\nDropping {len(dropped)} meters with < {min_completeness:.0%} completeness:")
        print(dropped[["n_rows", "completeness"]])

    df = df[df["meter"].isin(keep)].copy()
    kept_by_city = df.groupby("city")["meter"].nunique()
    print(f"\nRemaining: {df['meter'].nunique()} meters, {len(df):,} rows")
    print(f"By city:\n{kept_by_city}")
    return df


def resample_per_meter(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample without turning empty hours into zero or imputing energy."""
    out = []
    for meter_id, g in df.groupby("meter"):
        # City is invariant meter metadata. Taking "first" inside each resample
        # bin loses it for bins with no readings, so attach it after resampling.
        meter_cities = g["city"].dropna().unique()
        if len(meter_cities) != 1:
            raise ValueError(
                f"Meter {meter_id!r} must map to exactly one city; found {meter_cities.tolist()}"
            )
        meter_city = meter_cities[0]
        g = g.set_index("x_Timestamp").sort_index()
        agg = g.resample(freq).agg({
            "t_kWh": lambda values: values.sum(min_count=1),
            "z_Avg Voltage (Volt)": "mean",
            "z_Avg Current (Amp)": "mean",
            "y_Freq (Hz)": "mean",
        })
        agg["city"] = meter_city
        agg["meter"] = meter_id
        out.append(agg)
    result = pd.concat(out).reset_index()
    n_missing = result["t_kWh"].isna().sum()
    print(f"\nResampled to {freq}: {len(result):,} rows. "
          f"{n_missing} rows with no readings in that window (gaps).")
    return result


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each meter's own timeline into train/val/test by position, not by a
    global date cutoff — this handles meters with different start/end dates
    fairly instead of assuming a single calendar boundary."""
    train_parts, val_parts, test_parts = [], [], []
    for meter_id, g in df.groupby("meter"):
        g = g.sort_values("x_Timestamp").reset_index(drop=True)
        n = len(g)
        train_end = int(n * TRAIN_FRAC)
        val_end = train_end + int(n * VAL_FRAC)
        train_parts.append(g.iloc[:train_end])
        val_parts.append(g.iloc[train_end:val_end])
        test_parts.append(g.iloc[val_end:])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    print(f"\nSplit sizes -> train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train, val, test


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["x_Timestamp"].dt.hour
    df["day_of_week"] = df["x_Timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["x_Timestamp"].dt.month
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """IMPORTANT: apply this separately to train/val/test, each sorted by time
    per meter, so no lag ever pulls a value from across the split boundary."""
    df = df.sort_values(["meter", "x_Timestamp"]).copy()
    grouped = df.groupby("meter")["t_kWh"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
        df[f"roll_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window).std()
        )

    return df


def main():
    df = combine(BAREILLY_CSV, MATHURA_CSV)
    analyze_missingness_pattern(df)
    df = drop_low_completeness_meters(df, MIN_COMPLETENESS, READINGS_PER_HOUR)
    df = resample_per_meter(df, RESAMPLE_FREQ)
    df = add_calendar_features(df)

    # IMPORTANT FIX: build lag/rolling features on each meter's FULL continuous
    # series BEFORE splitting, not separately within train/val/test. Building
    # them per-split was needlessly dropping the first ~168 hours of EVERY
    # split (not just the very start of the whole series) -- including a real,
    # avoidable gap at the start of val and test: a test-period row CAN
    # legitimately use val's true past values as lookback (that's not leakage
    # -- it's still only using data from before that row's own timestamp, the
    # same rule as always). This was quietly costing about a week of usable
    # data at every split boundary, per meter, and it's what caused a genuine
    # hole between where val.csv ended and where test.csv started -- no saved
    # file covered that week at all.
    df = add_lag_and_rolling_features(df)

    train, val, test = chronological_split(df)

    # Drop rows where lag/rolling features couldn't be computed -- now this
    # only removes the very start of each meter's OVERALL history (inside
    # train), not the start of val or test.
    feature_cols = [c for c in train.columns if c.startswith("lag_") or c.startswith("roll_")]
    train = train.dropna(subset=feature_cols)
    val = val.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)

    print(f"\nAfter dropping rows with incomplete lag/rolling features:")
    print(f"train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("\nSaved train.csv, val.csv, test.csv")
    print(
        "\nNOTE: no scaler/normalizer is applied here. If your next model needs "
        "scaled inputs (e.g. GRU/LSTM), fit the scaler on train.csv ONLY, then "
        "transform val/test with that same fitted scaler -- never refit on val/test."
    )


if __name__ == "__main__":
    # Keep a single authoritative implementation and preserve legacy artifacts.
    from research_pipeline import main as research_main
    research_main()
