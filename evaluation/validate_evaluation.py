"""Synthetic contract/evaluator tests plus optional generated-artifact checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.comparison import common_origin_predictions, discover_research_prediction_files
from evaluation.contract import HORIZON, PREDICTION_COLUMNS, REPORT_DIR
from evaluation.metrics import (
    per_origin_metrics, point_metrics, smape_percent, validate_prediction_schema,
)
from evaluation.origins import build_canonical_origins, validate_origins


def prediction_frame(model="A", origin="2020-02-01 00:00", actual=None, prediction=None):
    origin = pd.Timestamp(origin)
    actual = np.arange(1, 25, dtype=float) if actual is None else np.asarray(actual, dtype=float)
    prediction = actual.copy() if prediction is None else np.asarray(prediction, dtype=float)
    return pd.DataFrame([{
        "model": model, "city": "C", "meter": "M", "split": "validation",
        "forecast_origin": origin, "horizon": h,
        "target_timestamp": origin + pd.Timedelta(hours=h),
        "actual": actual[h - 1], "prediction": prediction[h - 1],
        "inference_time_ms": 1.0, "run_id": "run",
    } for h in range(1, 25)], columns=PREDICTION_COLUMNS)


def synthetic_splits(invalid_history=False, invalid_target=False):
    ts = pd.date_range("2020-01-01", periods=500, freq="h")
    full = pd.DataFrame({"x_Timestamp": ts, "t_kWh": 1.0, "city": "C", "meter": "M"})
    train, validation, test = full.iloc[:300].copy(), full.iloc[300:400].copy(), full.iloc[400:].copy()
    if invalid_history:
        train.loc[299, "t_kWh"] = np.nan
    if invalid_target:
        validation.loc[305, "t_kWh"] = np.nan
    return {"train": train, "validation": validation, "test": test}


class EvaluationContractTests(unittest.TestCase):
    def test_prediction_schema_requires_24_aligned_horizons(self):
        frame = prediction_frame()
        validate_prediction_schema(frame)
        self.assertEqual(frame.horizon.tolist(), list(range(1, 25)))
        with self.assertRaises(ValueError):
            validate_prediction_schema(frame.iloc[:-1])
        broken = frame.copy(); broken.loc[0, "target_timestamp"] += pd.Timedelta(hours=1)
        with self.assertRaises(ValueError):
            validate_prediction_schema(broken)

    def test_validation_and_test_targets_stay_in_split(self):
        splits = synthetic_splits()
        for split in ("validation", "test"):
            accepted, _, _ = build_canonical_origins(splits, split)
            validate_origins(accepted, splits)
            boundary = splits[split].x_Timestamp
            self.assertTrue((accepted.first_target_timestamp >= boundary.min()).all())
            self.assertTrue((accepted.last_target_timestamp <= boundary.max()).all())

    def test_input_history_is_strictly_before_targets(self):
        accepted, _, _ = build_canonical_origins(synthetic_splits(), "validation")
        self.assertTrue((accepted.forecast_origin < accepted.first_target_timestamp).all())

    def test_origins_reject_invalid_history(self):
        accepted, rejected, _ = build_canonical_origins(synthetic_splits(invalid_history=True), "validation")
        self.assertIn("invalid_history_load", set(rejected.rejection_reason))
        self.assertNotIn(pd.Timestamp("2020-01-13 11:00"), set(accepted.forecast_origin))

    def test_origins_reject_missing_target_window(self):
        _, rejected, _ = build_canonical_origins(synthetic_splits(invalid_target=True), "validation")
        self.assertIn("invalid_target_load", set(rejected.rejection_reason))

    def test_known_metric_answers_and_degenerate_r2(self):
        perfect = point_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(perfect["MAE"], 0)
        self.assertEqual(perfect["RMSE"], 0)
        self.assertEqual(perfect["R2"], 1)
        shifted = point_metrics([1, 2], [2, 3])
        self.assertEqual(shifted["MAE"], 1)
        self.assertEqual(shifted["RMSE"], 1)
        self.assertTrue(np.isnan(point_metrics([1, 1], [1, 2])["R2"]))

    def test_smape_zero_pairs_are_safe(self):
        self.assertEqual(smape_percent([0, 0], [0, 0]), 0)
        self.assertTrue(np.isfinite(smape_percent([0, 1], [1, 1])))

    def test_peak_metrics_use_complete_profile(self):
        actual = np.arange(1, 25, dtype=float)
        predicted = actual.copy(); predicted[-1] = 20; predicted[-2] = 30
        row = per_origin_metrics(prediction_frame(actual=actual, prediction=predicted)).iloc[0]
        self.assertEqual(row.peak_load_error, 6)
        self.assertEqual(row.peak_timing_error_hours, 1)

    def test_comparison_uses_common_complete_origins(self):
        a1 = prediction_frame("A")
        b1 = prediction_frame("B")
        a2 = prediction_frame("A", "2020-02-02 00:00")
        b2 = prediction_frame("B", "2020-02-02 00:00").iloc[:-1]
        common, report = common_origin_predictions(pd.concat([a1, b1, a2, b2]), ["A", "B"])
        self.assertEqual(common.forecast_origin.nunique(), 1)
        self.assertEqual(len(common), 48)
        self.assertFalse(report.complete_forecast.all())

    def test_deployment_artifacts_are_not_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "deployment_manifest.json").write_text(json.dumps({
                "research_eligible": False, "prediction_path": "general_gru_model.pt"}))
            (path / "research_manifest.json").write_text(json.dumps({
                "research_eligible": True, "prediction_path": str(path / "predictions.csv")}))
            found = discover_research_prediction_files(path)
            self.assertEqual(found, [path / "predictions.csv"])

    def test_generated_artifacts_when_present(self):
        validation_path = REPORT_DIR / "validation_origins.csv"
        test_path = REPORT_DIR / "test_origins.csv"
        prediction_path = REPORT_DIR / "seasonal_naive_validation_predictions.csv"
        if not all(path.exists() for path in (validation_path, test_path, prediction_path)):
            self.skipTest("Smoke-test artifacts not generated yet")
        validation = pd.read_csv(validation_path, parse_dates=[
            "forecast_origin", "first_target_timestamp", "last_target_timestamp"])
        test = pd.read_csv(test_path, parse_dates=[
            "forecast_origin", "first_target_timestamp", "last_target_timestamp"])
        self.assertFalse(validation.empty); self.assertFalse(test.empty)
        predictions = pd.read_csv(
            prediction_path,
            parse_dates=["forecast_origin", "target_timestamp"])
        validate_prediction_schema(predictions)
        self.assertEqual(set(predictions.split), {"validation"})
        prediction_keys = set(zip(predictions.meter, predictions.forecast_origin))
        canonical_keys = set(zip(validation.meter, validation.forecast_origin))
        self.assertEqual(prediction_keys, canonical_keys)
        manifest = json.loads((REPORT_DIR / "seasonal_naive_validation_manifest.json").read_text())
        self.assertFalse(manifest["test_predictions_generated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
