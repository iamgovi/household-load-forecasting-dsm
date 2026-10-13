"""Reproducible, read-only audit of the two 2020 smart-meter source files.

The script streams the large CSVs, so it does not require loading the complete
dataset into memory. It writes machine-readable tables, a metadata report and
five presentation-ready figures. It does not clean or overwrite source data.

Run from the repository root:
    python preprocess/dataset_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "Bareilly": ROOT / "archive (1)" / "CEEW - Smart meter data Bareilly 2020.csv",
    "Mathura": ROOT / "archive (1)" / "CEEW - Smart meter data Mathura 2020.csv",
}
EXPECTED_COLUMNS = [
    "x_Timestamp", "t_kWh", "z_Avg Voltage (Volt)",
    "z_Avg Current (Amp)", "y_Freq (Hz)", "meter",
]


@dataclass
class MeterStats:
    city: str
    meter: str
    rows: int = 0
    first: datetime | None = None
    last: datetime | None = None
    previous_timestamp: datetime | None = None
    previous_load: float | None = None
    duplicate_timestamps: int = 0
    out_of_order_timestamps: int = 0
    irregular_intervals: int = 0
    gap_events: int = 0
    missing_intervals_estimated: int = 0
    negative_loads: int = 0
    zero_loads: int = 0
    valid_loads: int = 0
    load_decreases: int = 0
    interval_counts: Counter = field(default_factory=Counter)


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: str) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def retained_meter_ids(split_paths: list[Path]) -> set[str]:
    retained: set[str] = set()
    for path in split_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("meter"):
                    retained.add(row["meter"])
    return retained


def audit_file(city: str, path: Path, retained: set[str]):
    meters: dict[str, MeterStats] = {}
    missing = Counter()
    invalid_timestamps = 0
    total_rows = 0
    hourly_energy = defaultdict(float)
    interval_counts = Counter()
    representative_meter = None

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")

        for row in reader:
            total_rows += 1
            for column in EXPECTED_COLUMNS:
                if row[column].strip() == "":
                    missing[column] += 1

            meter = row["meter"].strip()
            if not meter:
                continue
            representative_meter = representative_meter or meter
            stats = meters.setdefault(meter, MeterStats(city=city, meter=meter))
            stats.rows += 1

            timestamp = parse_timestamp(row["x_Timestamp"])
            if timestamp is None:
                invalid_timestamps += 1
                continue
            stats.first = timestamp if stats.first is None else min(stats.first, timestamp)
            stats.last = timestamp if stats.last is None else max(stats.last, timestamp)

            if stats.previous_timestamp is not None:
                delta = int((timestamp - stats.previous_timestamp).total_seconds())
                stats.interval_counts[delta] += 1
                interval_counts[delta] += 1
                if delta == 0:
                    stats.duplicate_timestamps += 1
                elif delta < 0:
                    stats.out_of_order_timestamps += 1
                elif delta != 180:
                    stats.irregular_intervals += 1
                    if delta > 180:
                        stats.gap_events += 1
                        stats.missing_intervals_estimated += max(0, delta // 180 - 1)
            stats.previous_timestamp = timestamp

            load = parse_float(row["t_kWh"])
            if load is not None:
                stats.valid_loads += 1
                stats.negative_loads += int(load < 0)
                stats.zero_loads += int(load == 0)
                if stats.previous_load is not None:
                    stats.load_decreases += int(load < stats.previous_load)
                stats.previous_load = load
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_energy[(meter, hour)] += load

    rows = []
    for meter, stats in meters.items():
        comparisons = max(stats.valid_loads - 1, 0)
        expected = (
            int((stats.last - stats.first).total_seconds() // 180) + 1
            if stats.first and stats.last else 0
        )
        rows.append({
            "city": city,
            "meter": meter,
            "retained_in_current_splits": meter in retained,
            "raw_rows": stats.rows,
            "first_timestamp": stats.first.isoformat(sep=" ") if stats.first else "",
            "last_timestamp": stats.last.isoformat(sep=" ") if stats.last else "",
            "expected_3min_rows_over_span": expected,
            "completeness_pct": round(100 * stats.rows / expected, 4) if expected else None,
            "duplicate_timestamps": stats.duplicate_timestamps,
            "out_of_order_timestamps": stats.out_of_order_timestamps,
            "irregular_intervals": stats.irregular_intervals,
            "gap_events": stats.gap_events,
            "missing_intervals_estimated": stats.missing_intervals_estimated,
            "negative_load_values": stats.negative_loads,
            "zero_load_values": stats.zero_loads,
            "valid_load_values": stats.valid_loads,
            "load_decrease_pct": round(100 * stats.load_decreases / comparisons, 4) if comparisons else None,
            "dominant_interval_seconds": stats.interval_counts.most_common(1)[0][0] if stats.interval_counts else None,
        })

    missing_rows = [{
        "city": city,
        "column": column,
        "missing_count": missing[column],
        "missing_pct": round(100 * missing[column] / total_rows, 6) if total_rows else 0,
    } for column in EXPECTED_COLUMNS]
    return (
        rows, missing_rows, hourly_energy, interval_counts,
        invalid_timestamps, total_rows, representative_meter,
    )


def save_plots(hourly: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    representatives = hourly.sort_values("timestamp").groupby("city")["meter"].first()
    fig, axes = plt.subplots(len(representatives), 1, figsize=(12, 6), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (city, meter) in zip(axes, representatives.items()):
        sample = hourly[(hourly.city == city) & (hourly.meter == meter)].head(24 * 14)
        axis.plot(sample.timestamp, sample.t_kWh, linewidth=0.8)
        axis.set_title(f"{city}: representative meter {meter} (first 14 days)")
        axis.set_ylabel("Hourly energy (kWh*)")
    axes[-1].set_xlabel("Timestamp (timezone UNVERIFIED)")
    fig.tight_layout(); fig.savefig(figures / "representative_household_timeseries.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for city, group in coverage.groupby("city"):
        ax.hist(group.completeness_pct, bins=20, alpha=0.6, label=city)
    ax.set(xlabel="Estimated 3-minute completeness (%)", ylabel="Meters",
           title="Raw meter completeness distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "missing_data_distribution.png", dpi=180); plt.close(fig)

    hourly["hour"] = hourly.timestamp.dt.hour
    profile = hourly.groupby(["city", "hour"], as_index=False).t_kWh.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for city, group in profile.groupby("city"):
        ax.plot(group.hour, group.t_kWh, marker="o", label=city)
    ax.set(xlabel="Hour", ylabel="Mean hourly energy (kWh*)", title="Mean daily load profile")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "hourly_mean_load_profile.png", dpi=180); plt.close(fig)

    hourly["day_type"] = np.where(hourly.timestamp.dt.dayofweek >= 5, "Weekend", "Weekday")
    day_profile = hourly.groupby(["city", "day_type", "hour"], as_index=False).t_kWh.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (city, group) in zip(axes, day_profile.groupby("city")):
        for day_type, values in group.groupby("day_type"):
            axis.plot(values.hour, values.t_kWh, label=day_type)
        axis.set_title(city); axis.set_xlabel("Hour"); axis.legend()
    axes[0].set_ylabel("Mean hourly energy (kWh*)")
    fig.suptitle("Weekday and weekend profiles"); fig.tight_layout(); fig.savefig(figures / "weekday_weekend_profile.png", dpi=180); plt.close(fig)

    hourly["day_of_week"] = hourly.timestamp.dt.day_name().str[:3]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = hourly.groupby(["city", "day_of_week"], as_index=False).t_kWh.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for city, group in weekly.groupby("city"):
        values = group.set_index("day_of_week").reindex(order)
        ax.plot(order, values.t_kWh, marker="o", label=city)
    ax.set(ylabel="Mean hourly energy (kWh*)", title="Weekly seasonality")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "weekly_seasonality.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "dataset_audit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    retained = retained_meter_ids([ROOT / "train.csv", ROOT / "val.csv", ROOT / "test.csv"])
    coverage_rows, missing_rows, audit_rows, hourly_parts = [], [], [], []
    interval_totals = Counter()

    for city, path in DEFAULT_INPUTS.items():
        (coverage, missing, hourly_energy, interval_counts, bad_ts,
         total_rows, representative) = audit_file(city, path, retained)
        coverage_rows.extend(coverage); missing_rows.extend(missing)
        interval_totals.update(interval_counts)
        audit_rows.append({
            "city": city, "source_file": str(path.relative_to(ROOT)), "raw_rows": total_rows,
            "raw_meters": len(coverage), "retained_meters": sum(r["retained_in_current_splits"] for r in coverage),
            "invalid_timestamps": bad_ts, "representative_meter": representative,
            "timestamp_column": "x_Timestamp", "load_column": "t_kWh",
            "nominal_sampling_seconds": 180,
        })
        hourly_parts.append(pd.DataFrame([
            {"city": city, "meter": meter, "timestamp": timestamp, "t_kWh": value}
            for (meter, timestamp), value in hourly_energy.items()
        ]))

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["city", "meter"])
    missing_df = pd.DataFrame(missing_rows)
    audit_df = pd.DataFrame(audit_rows)
    hourly_df = pd.concat(hourly_parts, ignore_index=True)
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])

    audit_df.to_csv(output / "dataset_audit.csv", index=False)
    coverage_df.to_csv(output / "meter_coverage.csv", index=False)
    missing_df.to_csv(output / "missingness.csv", index=False)
    pd.DataFrame([
        {"interval_seconds": interval, "consecutive_timestamp_pairs": count}
        for interval, count in interval_totals.most_common()
    ]).to_csv(output / "sampling_intervals.csv", index=False)

    total_comparisons = (coverage_df.valid_load_values - 1).clip(lower=0).sum()
    decrease_weighted = (
        coverage_df.load_decrease_pct.mul((coverage_df.valid_load_values - 1).clip(lower=0)).sum()
        / total_comparisons if total_comparisons else None
    )
    summary = {
        "generated_from": [str(path.relative_to(ROOT)) for path in DEFAULT_INPUTS.values()],
        "dataset_source": "UNVERIFIED",
        "source_url": "UNVERIFIED",
        "licence": "UNVERIFIED",
        "timezone": "UNVERIFIED",
        "load_semantics_authoritative": "UNVERIFIED",
        "observed_cities": sorted(DEFAULT_INPUTS),
        "raw_meter_count": int(len(coverage_df)),
        "retained_meter_count_current_splits": int(coverage_df.retained_in_current_splits.sum()),
        "retention_rule_in_pipeline": ">=60% rows expected over each meter's observed span at 3-minute frequency",
        "timestamp_column": "x_Timestamp",
        "load_column": "t_kWh",
        "unit_from_column_label": "kWh",
        "dominant_sampling_interval_seconds": 180,
        "raw_date_min": coverage_df.first_timestamp.min(),
        "raw_date_max": coverage_df.last_timestamp.max(),
        "duplicate_timestamps": int(coverage_df.duplicate_timestamps.sum()),
        "out_of_order_timestamps": int(coverage_df.out_of_order_timestamps.sum()),
        "gap_events": int(coverage_df.gap_events.sum()),
        "estimated_missing_3min_intervals": int(coverage_df.missing_intervals_estimated.sum()),
        "negative_load_values": int(coverage_df.negative_load_values.sum()),
        "zero_load_values": int(coverage_df.zero_load_values.sum()),
        "weighted_pct_consecutive_load_decreases": round(float(decrease_weighted), 4),
        "t_kWh_interpretation": "Empirically consistent with per-interval energy, not a cumulative counter; authoritative definition UNVERIFIED",
        "hourly_aggregation_assessment": "SUM is scientifically appropriate if t_kWh is interval energy; conditional on authoritative unit verification",
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(hourly_df, coverage_df, output)

    report = f"""# Dataset audit and provenance report

Generated reproducibly by `preprocess/dataset_audit.py` from the two 2020 raw CSV files.

## Verified from files

- Cities encoded by filenames: Bareilly and Mathura.
- Raw meters: {summary['raw_meter_count']}.
- Meters represented in current train/validation/test files: {summary['retained_meter_count_current_splits']}.
- Timestamp column: `x_Timestamp`; observed dominant interval: 180 seconds.
- Load/energy column: `t_kWh`; its values fluctuate rather than behave like a cumulative counter.
- Duplicate meter-timestamps: {summary['duplicate_timestamps']}.
- Out-of-order timestamps: {summary['out_of_order_timestamps']}.
- Negative `t_kWh` values: {summary['negative_load_values']}.
- Estimated missing 3-minute intervals inside meter coverage spans: {summary['estimated_missing_3min_intervals']}.
- Raw date extent across meters: {summary['raw_date_min']} to {summary['raw_date_max']}.

## UNVERIFIED metadata

- Official dataset title and publisher/source: **UNVERIFIED**.
- Source URL/reference: **UNVERIFIED**.
- Licence and redistribution terms: **UNVERIFIED**.
- Timestamp timezone: **UNVERIFIED**.
- Authoritative definition and measurement process for `t_kWh`: **UNVERIFIED**.
- Whether each meter ID maps one-to-one to a household: **UNVERIFIED**.

## Interpretation of `t_kWh` and hourly aggregation

The column name identifies kWh and consecutive values frequently decrease
({summary['weighted_pct_consecutive_load_decreases']:.2f}% of valid consecutive comparisons), so it is not a
monotonically increasing cumulative register. It is empirically consistent with energy recorded per
three-minute interval. On that interpretation, hourly **SUM** is appropriate because energy quantities
are additive. This conclusion remains conditional until the source data dictionary confirms the field.

## Current preprocessing issues

- Hourly resampling calls pandas `sum` without `min_count=1`. A completely empty hour can therefore
  become zero instead of missing, contrary to the pipeline's diagnostic message. Any missing values
  that remain are forward-filled for up to three hours, which repeats energy. Both behaviours require
  replacement by an explicit, documented gap policy before regenerating modelling datasets.
- Existing statistical/hybrid scripts use bidirectional interpolation, including on test data. That can
  use future observations and must not be used for research evaluation.
- Empty resampling bins previously lost `city`; the pipeline now assigns city as invariant meter metadata.
- The 60% retention rule is calculated only over each meter's observed first-to-last span and therefore
  does not penalize missing periods before installation or after final observation.

An asterisk on plot axes (`kWh*`) indicates that the unit comes from the column label but is not yet
confirmed by authoritative metadata.
"""
    (output / "DATASET_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(f"Wrote audit outputs to {output}")


if __name__ == "__main__":
    main()
