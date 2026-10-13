"""Leakage-safe preprocessing that preserves legacy root-level artifacts.

Outputs are written to ``data/research`` and ``reports/preprocessing``. No
energy value is imputed; incomplete hours remain unusable (NaN).
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from research_config import (
    CITY_COLUMN, EXPECTED_READINGS_PER_HOUR, FORECAST_HORIZON, INPUT_WINDOW,
    LAGS, METER_COLUMN, MIN_HOURLY_COVERAGE, MIN_METER_COMPLETENESS,
    OUTPUT_DIR, RANDOM_SEED, RAW_FILES, REPORT_DIR, REPORT_OUTPUT,
    RESAMPLE_FREQUENCY, ROLLING_WINDOWS, SPLIT_INTEGRITY_OUTPUT,
    SUMMARY_OUTPUT, TARGET_COLUMN, TEST_OUTPUT, TIMESTAMP_COLUMN, TRAIN_OUTPUT,
    TRAIN_RATIO, VALIDATION_OUTPUT, VALIDATION_RATIO,
)

ELECTRICAL_COLUMNS = (
    "z_Avg Voltage (Volt)", "z_Avg Current (Amp)", "y_Freq (Hz)",
)


def set_deterministic_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_and_tag(path, city: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {TIMESTAMP_COLUMN, TARGET_COLUMN, METER_COLUMN, *ELECTRICAL_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    bad = int(df[TIMESTAMP_COLUMN].isna().sum())
    if bad:
        print(f"Dropping {bad:,} invalid timestamps from {path}")
        df = df.dropna(subset=[TIMESTAMP_COLUMN])
    df[CITY_COLUMN] = city
    return df


def combine() -> pd.DataFrame:
    frames = [load_and_tag(path, city) for city, path in RAW_FILES.items()]
    df = pd.concat(frames, ignore_index=True)
    duplicates = int(df.duplicated([METER_COLUMN, TIMESTAMP_COLUMN]).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate meter-timestamps")
    return df


def meter_completeness(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby([METER_COLUMN, CITY_COLUMN])[TIMESTAMP_COLUMN].agg(
        first="min", last="max", raw_rows="count"
    )
    seconds = (stats["last"] - stats["first"]).dt.total_seconds()
    stats["expected_raw_rows"] = (seconds // 180 + 1).astype("int64")
    stats["completeness"] = stats["raw_rows"] / stats["expected_raw_rows"]
    return stats.reset_index()


def filter_meters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = meter_completeness(df)
    keep = set(stats.loc[stats.completeness >= MIN_METER_COMPLETENESS, METER_COLUMN])
    print(f"Retained {len(keep)}/{len(stats)} meters at >=60% raw completeness")
    return df[df[METER_COLUMN].isin(keep)].copy(), stats


def _energy_sum_with_missing(values: pd.Series) -> float:
    """Unlike default resample sum, an entirely empty bin remains NaN."""
    return values.sum(min_count=1)


def resample_per_meter(
    df: pd.DataFrame,
    freq: str = RESAMPLE_FREQUENCY,
    expected_readings: int = EXPECTED_READINGS_PER_HOUR,
    min_coverage: float = MIN_HOURLY_COVERAGE,
) -> pd.DataFrame:
    """Aggregate and classify hours without forward/back filling.

    complete: >=20 valid readings and usable target;
    partial: 1--19 readings, observed sum retained only for auditing;
    missing: 0 readings, observed sum and target both NaN.
    """
    if freq != "1h":
        raise ValueError("Coverage policy is defined for hourly resampling")
    parts = []
    for meter_id, group in df.groupby(METER_COLUMN, sort=True):
        cities = group[CITY_COLUMN].dropna().unique()
        if len(cities) != 1:
            raise ValueError(
                f"Meter {meter_id!r} must map to exactly one city; found {cities.tolist()}"
            )
        indexed = group.set_index(TIMESTAMP_COLUMN).sort_index()
        hourly = indexed.resample(freq).agg({
            TARGET_COLUMN: [_energy_sum_with_missing, "count"],
            ELECTRICAL_COLUMNS[0]: "mean",
            ELECTRICAL_COLUMNS[1]: "mean",
            ELECTRICAL_COLUMNS[2]: "mean",
        })
        hourly.columns = ["observed_t_kWh", "readings_in_hour", *ELECTRICAL_COLUMNS]
        hourly["readings_in_hour"] = hourly["readings_in_hour"].astype("int16")
        hourly["hourly_coverage"] = (
            hourly["readings_in_hour"] / expected_readings
        ).clip(upper=1.0)
        hourly["hourly_quality"] = np.select(
            [hourly.readings_in_hour.eq(0), hourly.hourly_coverage.ge(min_coverage)],
            ["missing", "complete"], default="partial",
        )
        hourly[TARGET_COLUMN] = hourly.observed_t_kWh.where(
            hourly.hourly_quality.eq("complete")
        )
        hourly[CITY_COLUMN] = cities[0]
        hourly[METER_COLUMN] = meter_id
        parts.append(hourly.reset_index())
    result = pd.concat(parts, ignore_index=True)
    return result[[
        TIMESTAMP_COLUMN, TARGET_COLUMN, "observed_t_kWh", "readings_in_hour",
        "hourly_coverage", "hourly_quality", *ELECTRICAL_COLUMNS,
        CITY_COLUMN, METER_COLUMN,
    ]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COLUMN]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df.day_of_week.isin([5, 6]).astype("int8")
    df["month"] = ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.day_of_week / 7)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """All load-derived features use only timestamps earlier than the row."""
    df = df.sort_values([METER_COLUMN, TIMESTAMP_COLUMN]).copy()
    grouped = df.groupby(METER_COLUMN, sort=False)[TARGET_COLUMN]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        for meter_id, index in df.groupby(METER_COLUMN, sort=False).groups.items():
            past = df.loc[index, TARGET_COLUMN].shift(1)
            roll = past.rolling(window)
            df.loc[index, f"roll_mean_{window}"] = roll.mean().to_numpy()
            df.loc[index, f"roll_std_{window}"] = roll.std().to_numpy()
            df.loc[index, f"roll_min_{window}"] = roll.min().to_numpy()
            df.loc[index, f"roll_max_{window}"] = roll.max().to_numpy()
    return df


def chronological_split(df: pd.DataFrame):
    train_parts, validation_parts, test_parts = [], [], []
    for _, group in df.groupby(METER_COLUMN, sort=True):
        group = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        n = len(group)
        train_end = int(n * TRAIN_RATIO)
        validation_end = train_end + int(n * VALIDATION_RATIO)
        train_parts.append(group.iloc[:train_end])
        validation_parts.append(group.iloc[train_end:validation_end])
        test_parts.append(group.iloc[validation_end:])
    return tuple(pd.concat(parts, ignore_index=True) for parts in (
        train_parts, validation_parts, test_parts,
    ))


def count_usable_windows(group: pd.DataFrame) -> tuple[int, int]:
    """Count 168-input/24-target windows contained wholly in one split."""
    valid = group.sort_values(TIMESTAMP_COLUMN)[TARGET_COLUMN].notna().to_numpy(np.int8)
    width = INPUT_WINDOW + FORECAST_HORIZON
    candidates = max(0, len(valid) - width + 1)
    if not candidates:
        return 0, 0
    sums = np.convolve(valid, np.ones(width, dtype=np.int16), mode="valid")
    usable = int((sums == width).sum())
    return usable, candidates - usable


def _range_fields(group: pd.DataFrame, prefix: str) -> dict:
    usable, dropped = count_usable_windows(group)
    return {
        f"{prefix}_start": group[TIMESTAMP_COLUMN].min(),
        f"{prefix}_end": group[TIMESTAMP_COLUMN].max(),
        f"{prefix}_row_count": len(group),
        f"usable_{prefix}_windows": usable,
        f"dropped_{prefix}_windows_missing": dropped,
    }


def build_split_integrity(hourly, train, validation, test) -> pd.DataFrame:
    rows = []
    for meter_id, all_group in hourly.groupby(METER_COLUMN, sort=True):
        row = {
            CITY_COLUMN: all_group[CITY_COLUMN].iloc[0], METER_COLUMN: meter_id,
            "first_timestamp": all_group[TIMESTAMP_COLUMN].min(),
            "last_timestamp": all_group[TIMESTAMP_COLUMN].max(),
            "total_hourly_rows": len(all_group),
            "valid_hourly_rows": int(all_group[TARGET_COLUMN].notna().sum()),
            "incomplete_hourly_rows": int(all_group.hourly_quality.eq("partial").sum()),
            "fully_missing_hourly_rows": int(all_group.hourly_quality.eq("missing").sum()),
        }
        for name, split in (("train", train), ("validation", validation), ("test", test)):
            row.update(_range_fields(split[split[METER_COLUMN].eq(meter_id)], name))
        rows.append(row)
    return pd.DataFrame(rows)


def validate_split_boundaries(train, validation, test) -> None:
    for meter_id in train[METER_COLUMN].unique():
        tr = train[train[METER_COLUMN].eq(meter_id)][TIMESTAMP_COLUMN]
        va = validation[validation[METER_COLUMN].eq(meter_id)][TIMESTAMP_COLUMN]
        te = test[test[METER_COLUMN].eq(meter_id)][TIMESTAMP_COLUMN]
        if not (tr.max() < va.min() and va.max() < te.min()):
            raise AssertionError(f"Non-chronological boundary for {meter_id}")
        if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
            raise AssertionError(f"Overlapping timestamps for {meter_id}")


def write_outputs(stats, hourly, integrity, train, validation, test) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUTPUT, index=False)
    validation.to_csv(VALIDATION_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)
    integrity.to_csv(SPLIT_INTEGRITY_OUTPUT, index=False)
    retained = stats[stats.completeness >= MIN_METER_COMPLETENESS]
    summary = {
        "random_seed": RANDOM_SEED,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "input_window_hours": INPUT_WINDOW,
        "resample_frequency": RESAMPLE_FREQUENCY,
        "expected_readings_per_hour": EXPECTED_READINGS_PER_HOUR,
        "minimum_hourly_coverage": MIN_HOURLY_COVERAGE,
        "gap_policy": "No imputation; only hours with >=20 valid readings are usable",
        "meter_retention_threshold": MIN_METER_COMPLETENESS,
        "raw_meters": int(len(stats)), "retained_meters": int(len(retained)),
        "retained_by_city": retained.groupby(CITY_COLUMN)[METER_COLUMN].nunique().to_dict(),
        "hourly_rows": int(len(hourly)),
        "valid_hourly_rows": int(hourly[TARGET_COLUMN].notna().sum()),
        "partial_hourly_rows": int(hourly.hourly_quality.eq("partial").sum()),
        "fully_missing_hourly_rows": int(hourly.hourly_quality.eq("missing").sum()),
        "true_zero_valid_hours": int(hourly[TARGET_COLUMN].eq(0).sum()),
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "usable_windows": {
            "train": int(integrity.usable_train_windows.sum()),
            "validation": int(integrity.usable_validation_windows.sum()),
            "test": int(integrity.usable_test_windows.sum()),
        },
        "dropped_windows_missing": {
            "train": int(integrity.dropped_train_windows_missing.sum()),
            "validation": int(integrity.dropped_validation_windows_missing.sum()),
            "test": int(integrity.dropped_test_windows_missing.sum()),
        },
        "scaling": "None; model scalers must fit train only",
        "legacy_root_splits_modified": False,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    REPORT_OUTPUT.write_text(_report_text(summary))
    return summary


def _report_text(summary: dict) -> str:
    return f"""# Leakage-safe preprocessing report

## Methodological decisions

- Retain meters with at least {MIN_METER_COMPLETENESS:.0%} raw completeness over their observed span.
- Define hourly coverage as valid readings / {EXPECTED_READINGS_PER_HOUR} expected readings.
- A complete hour has >= {EXPECTED_READINGS_PER_HOUR} readings; its interval energies are summed.
- A partial hour has 1--19 readings. Its observed sum is audit-only and `t_kWh` is NaN.
- A missing hour has zero readings. Its observed sum and `t_kWh` remain NaN.
- No forward fill, backward fill, interpolation or other energy imputation is used.
- Split every meter chronologically 70%/15%/15% on the full hourly grid.
- Compute only backward lags and `shift(1)` rolling mean/std/min/max.
- Count 168-input/24-target windows only inside one split and only when every load is usable.
- Apply no shared scaling; future model scalers must fit training data only.

## Results

- Retained meters: {summary['retained_meters']} ({summary['retained_by_city']}).
- Hourly rows: {summary['hourly_rows']:,}.
- Complete usable hours: {summary['valid_hourly_rows']:,}.
- Partial unusable hours: {summary['partial_hourly_rows']:,}.
- Fully missing hours: {summary['fully_missing_hourly_rows']:,}.
- True observed zero-energy complete hours: {summary['true_zero_valid_hours']:,}.
- Split rows: {summary['split_rows']}.
- Usable 168-to-24 windows: {summary['usable_windows']}.
- Windows dropped for missing inputs/targets: {summary['dropped_windows_missing']}.

## Legacy isolation

New splits are under `data/research`; root-level splits and all model artifacts/results remain unchanged.
Legacy statistical/hybrid evaluators still use bidirectional interpolation and are unsafe for new scientific
results until a later modelling milestone. `train_general_model.py` combines all splits and randomly partitions
overlapping windows; it is a deployment path, not valid research evaluation evidence.

## Unresolved assumptions

The authoritative `t_kWh` definition, source URL, licence and timezone remain UNVERIFIED. Hourly SUM remains
conditional on confirming interval kWh. The 100% hourly threshold is deliberately conservative.
"""


def main() -> None:
    set_deterministic_seed()
    print("Loading and validating raw files...")
    raw = combine()
    filtered, stats = filter_meters(raw)
    print("Resampling without imputation...")
    hourly = add_calendar_features(resample_per_meter(filtered))
    hourly = add_lag_and_rolling_features(hourly)
    train, validation, test = chronological_split(hourly)
    validate_split_boundaries(train, validation, test)
    integrity = build_split_integrity(hourly, train, validation, test)
    summary = write_outputs(stats, hourly, integrity, train, validation, test)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
