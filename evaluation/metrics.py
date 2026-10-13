"""Uniform metrics and aggregation for standard long-format predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.contract import HORIZON, PREDICTION_COLUMNS

ORIGIN_KEY = ["model", "city", "meter", "split", "forecast_origin", "run_id"]


def validate_prediction_schema(predictions: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction records missing columns: {sorted(missing)}")
    if predictions[PREDICTION_COLUMNS].isna().drop(columns=["inference_time_ms"]).any().any():
        raise ValueError("Prediction identity/actual/prediction fields cannot be missing")
    for _, group in predictions.groupby(ORIGIN_KEY, dropna=False):
        if len(group) != HORIZON or sorted(group.horizon.tolist()) != list(range(1, HORIZON + 1)):
            raise ValueError("Every forecast origin must contain horizons 1..24 exactly once")
        expected = group.forecast_origin.iloc[0] + pd.to_timedelta(group.horizon, unit="h")
        if not (pd.to_datetime(group.target_timestamp).to_numpy() == expected.to_numpy()).all():
            raise ValueError("Target timestamps do not align with origin+horizon")


def smape_percent(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = np.abs(actual) + np.abs(prediction)
    terms = np.divide(
        200.0 * np.abs(prediction - actual), denominator,
        out=np.zeros_like(denominator, dtype=float), where=denominator != 0,
    )
    return float(np.mean(terms))


def point_metrics(actual, prediction) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_actual = float(np.mean(actual))
    sst = float(np.sum((actual - mean_actual) ** 2))
    r2 = float(1 - np.sum(error ** 2) / sst) if len(actual) >= 2 and sst > 0 else np.nan
    return {
        "MAE": mae, "RMSE": rmse, "sMAPE_pct": smape_percent(actual, prediction),
        "R2": r2, "NRMSE_mean": rmse / mean_actual if mean_actual > 0 else np.nan,
        "n_predictions": int(len(actual)),
    }


def per_origin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_prediction_schema(predictions)
    rows = []
    for key, group in predictions.sort_values("horizon").groupby(ORIGIN_KEY, dropna=False):
        actual, prediction = group.actual.to_numpy(), group.prediction.to_numpy()
        metrics = point_metrics(actual, prediction)
        actual_peak, predicted_peak = float(actual.max()), float(prediction.max())
        peak_error = abs(actual_peak - predicted_peak)
        metrics.update({
            **dict(zip(ORIGIN_KEY, key)),
            "actual_peak": actual_peak, "predicted_peak": predicted_peak,
            "peak_load_error": peak_error,
            "normalized_peak_error": peak_error / actual_peak if actual_peak > 0 else np.nan,
            "peak_timing_error_hours": int(abs(np.argmax(actual) - np.argmax(prediction))),
            "inference_time_ms": float(group.inference_time_ms.iloc[0])
                if group.inference_time_ms.notna().any() else np.nan,
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def grouped_point_metrics(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        row.update(point_metrics(group.actual, group.prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    origins = per_origin_metrics(predictions)
    per_meter = grouped_point_metrics(predictions, ["model", "split", "city", "meter", "run_id"])
    peak_means = origins.groupby(["model", "split", "city", "meter", "run_id"], as_index=False)[
        ["peak_load_error", "normalized_peak_error", "peak_timing_error_hours", "inference_time_ms"]
    ].mean()
    per_meter = per_meter.merge(peak_means, on=["model", "split", "city", "meter", "run_id"])
    per_city = grouped_point_metrics(predictions, ["model", "split", "city", "run_id"])
    city_peaks = origins.groupby(["model", "split", "city", "run_id"], as_index=False)[
        ["peak_load_error", "normalized_peak_error", "peak_timing_error_hours", "inference_time_ms"]
    ].mean()
    per_city = per_city.merge(city_peaks, on=["model", "split", "city", "run_id"])
    per_horizon = grouped_point_metrics(predictions, ["model", "split", "horizon", "run_id"])
    pooled = grouped_point_metrics(predictions, ["model", "split", "run_id"])
    pooled_peaks = origins.groupby(["model", "split", "run_id"], as_index=False)[
        ["peak_load_error", "normalized_peak_error", "peak_timing_error_hours", "inference_time_ms"]
    ].mean()
    pooled = pooled.merge(pooled_peaks, on=["model", "split", "run_id"])
    macro = per_meter.groupby(["model", "split", "run_id"], as_index=False)[
        ["MAE", "RMSE", "sMAPE_pct", "R2", "NRMSE_mean",
         "peak_load_error", "normalized_peak_error", "peak_timing_error_hours", "inference_time_ms"]
    ].mean()
    macro["n_predictions"] = len(predictions)
    macro["aggregation"] = "macro_meter_mean"
    pooled["aggregation"] = "pooled_predictions"
    overall = pd.concat([pooled, macro], ignore_index=True, sort=False)
    return {"per_origin": origins, "per_meter": per_meter, "per_city": per_city,
            "per_horizon": per_horizon, "overall": overall}
