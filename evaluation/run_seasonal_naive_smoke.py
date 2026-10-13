"""Validation-only Seasonal Naive smoke test for the common contract."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.contract import (
    DEPLOYMENT_NON_RESEARCH, HORIZON, LOOKBACK, METRIC_DEFINITIONS,
    ORIGIN_STRIDE_HOURS, PREDICTION_COLUMNS, REPORT_DIR, SEED, TEST_PATH,
    TRAIN_PATH, VALIDATION_PATH,
)
from evaluation.metrics import evaluate_predictions, validate_prediction_schema
from evaluation.origins import build_canonical_origins, load_research_splits, validate_origins

MODEL = "SeasonalNaive_Daily"
SPLIT_PATHS = {"train": TRAIN_PATH, "validation": VALIDATION_PATH, "test": TEST_PATH}


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TRAIN_PATH.parents[2],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def seasonal_naive_predictions(origins: pd.DataFrame, splits, run_id: str) -> pd.DataFrame:
    full = pd.concat(list(splits.values()), ignore_index=True)
    series_by_meter = {
        meter: group.set_index("x_Timestamp")["t_kWh"].sort_index()
        for meter, group in full.groupby("meter")
    }
    records = []
    for origin in origins.itertuples(index=False):
        target_times = pd.date_range(origin.first_target_timestamp, periods=HORIZON, freq="h")
        series = series_by_meter[origin.meter]
        started = time.perf_counter()
        prediction = series.reindex(target_times - pd.Timedelta(hours=24)).to_numpy()
        elapsed_ms = (time.perf_counter() - started) * 1000
        actual = series.reindex(target_times).to_numpy()
        for horizon, (timestamp, actual_value, predicted_value) in enumerate(
            zip(target_times, actual, prediction), start=1
        ):
            records.append({
                "model": MODEL, "city": origin.city, "meter": origin.meter,
                "split": "validation", "forecast_origin": origin.forecast_origin,
                "horizon": horizon, "target_timestamp": timestamp,
                "actual": actual_value, "prediction": predicted_value,
                "inference_time_ms": elapsed_ms, "run_id": run_id,
            })
    predictions = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    validate_prediction_schema(predictions)
    return predictions


def save_representative_figure(predictions: pd.DataFrame, output: Path) -> None:
    first = predictions.sort_values(["meter", "forecast_origin", "horizon"])
    first = first[(first.meter == first.meter.iloc[0]) &
                  (first.forecast_origin == first.forecast_origin.iloc[0])]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(first.horizon, first.actual, marker="o", label="Actual")
    axis.plot(first.horizon, first.prediction, marker="o", label="Seasonal Naive")
    axis.set(xlabel="Forecast horizon (hours)", ylabel="Hourly energy (kWh*)",
             title=f"Validation profile: {first.meter.iloc[0]}, origin {first.forecast_origin.iloc[0]}")
    axis.legend(); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_research_splits(SPLIT_PATHS)
    origin_tables = {}
    origin_counts = {}
    for split in ("validation", "test"):
        accepted, rejected, counts = build_canonical_origins(splits, split)
        validate_origins(accepted, splits)
        accepted.to_csv(REPORT_DIR / f"{split}_origins.csv", index=False)
        rejected.to_csv(REPORT_DIR / f"{split}_origin_rejections.csv", index=False)
        counts.to_csv(REPORT_DIR / f"{split}_origin_counts_per_meter.csv", index=False)
        origin_tables[split] = accepted
        origin_counts[split] = {
            "candidates": int(len(accepted) + len(rejected)),
            "accepted": int(len(accepted)), "rejected": int(len(rejected)),
            "rejections_by_reason": rejected.rejection_reason.value_counts().to_dict(),
        }

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"seasonal_naive_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    predictions = seasonal_naive_predictions(origin_tables["validation"], splits, run_id)
    prediction_path = REPORT_DIR / "seasonal_naive_validation_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    results = evaluate_predictions(predictions)
    for level, frame in results.items():
        frame.to_csv(REPORT_DIR / f"seasonal_naive_validation_{level}_metrics.csv", index=False)
    save_representative_figure(predictions, REPORT_DIR / "seasonal_naive_validation_profile.png")

    manifest = {
        "run_id": run_id, "created_at_utc": created_at, "model": MODEL,
        "split": "validation", "research_eligible": True,
        "prediction_path": str(prediction_path.resolve()),
        "data_paths": {key: str(value.resolve()) for key, value in SPLIT_PATHS.items()},
        "frequency": "1h", "forecast_horizon": HORIZON, "lookback": LOOKBACK,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "meter_count": int(predictions.meter.nunique()),
        "forecast_origin_count": int(predictions.forecast_origin.groupby(predictions.meter).count().sum() / HORIZON),
        "canonical_origin_counts": origin_counts, "metric_definitions": METRIC_DEFINITIONS,
        "seasonal_naive_definition": "prediction(t+h) = actual(t+h-24), h=1..24",
        "seed": SEED, "git_commit": git_hash(),
        "deployment_non_research_artifacts": DEPLOYMENT_NON_RESEARCH,
        "test_predictions_generated": False,
    }
    (REPORT_DIR / "seasonal_naive_validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "canonical_origins": origin_counts,
        "validation_smoke_overall": results["overall"].to_dict(orient="records"),
        "note": "Validation smoke test only; no model ranking or test predictions.",
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
