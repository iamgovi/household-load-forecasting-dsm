"""Focused, reproducible validation checks for Milestone 2.

Run from the repository root:
    python preprocess/validate_preprocessing.py
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_config import FORECAST_HORIZON, INPUT_WINDOW, METER_COLUMN, TIMESTAMP_COLUMN
from research_pipeline import (
    add_lag_and_rolling_features, chronological_split, resample_per_meter,
    validate_split_boundaries,
)


def raw_rows(city="Bareilly", meter="M1") -> pd.DataFrame:
    timestamps = list(pd.date_range("2020-01-01 00:00", periods=20, freq="3min"))
    timestamps += list(pd.date_range("2020-01-01 02:00", periods=20, freq="3min"))
    return pd.DataFrame({
        "x_Timestamp": timestamps,
        "t_kWh": [0.0] * 20 + [0.1] * 20,
        "z_Avg Voltage (Volt)": [230.0] * 40,
        "z_Avg Current (Amp)": [1.0] * 40,
        "y_Freq (Hz)": [50.0] * 40,
        "city": [city] * 40,
        "meter": [meter] * 40,
    })


class PreprocessingValidation(unittest.TestCase):
    def test_missing_zero_and_city_are_distinct(self):
        hourly = resample_per_meter(raw_rows())
        self.assertEqual(hourly.loc[0, "t_kWh"], 0.0)
        self.assertEqual(hourly.loc[0, "hourly_quality"], "complete")
        self.assertTrue(pd.isna(hourly.loc[1, "t_kWh"]))
        self.assertTrue(pd.isna(hourly.loc[1, "observed_t_kWh"]))
        self.assertEqual(hourly.loc[1, "hourly_quality"], "missing")
        self.assertEqual(hourly["city"].tolist(), ["Bareilly"] * 3)

    def test_meter_cannot_map_to_multiple_cities(self):
        raw = raw_rows()
        raw.loc[0, "city"] = "Mathura"
        with self.assertRaisesRegex(ValueError, "exactly one city"):
            resample_per_meter(raw)

    def test_partial_hour_is_auditable_but_not_a_target(self):
        raw = raw_rows().iloc[:10].copy()
        hourly = resample_per_meter(raw)
        self.assertEqual(hourly.loc[0, "readings_in_hour"], 10)
        self.assertEqual(hourly.loc[0, "hourly_coverage"], 0.5)
        self.assertEqual(hourly.loc[0, "hourly_quality"], "partial")
        self.assertEqual(hourly.loc[0, "observed_t_kWh"], 0.0)
        self.assertTrue(pd.isna(hourly.loc[0, "t_kWh"]))

    def test_split_overlap_and_order(self):
        n = 1000
        frame = pd.DataFrame({
            TIMESTAMP_COLUMN: pd.date_range("2020-01-01", periods=n, freq="h"),
            "t_kWh": np.arange(n, dtype=float), METER_COLUMN: "M1",
        })
        train, validation, test = chronological_split(frame)
        validate_split_boundaries(train, validation, test)
        self.assertFalse(set(train.x_Timestamp) & set(validation.x_Timestamp))
        self.assertFalse(set(train.x_Timestamp) & set(test.x_Timestamp))
        self.assertFalse(set(validation.x_Timestamp) & set(test.x_Timestamp))

    def test_forecast_windows_cannot_cross_boundaries(self):
        n = 2000
        frame = pd.DataFrame({
            TIMESTAMP_COLUMN: pd.date_range("2020-01-01", periods=n, freq="h"),
            "t_kWh": 1.0, METER_COLUMN: "M1",
        })
        train, validation, test = chronological_split(frame)
        width = INPUT_WINDOW + FORECAST_HORIZON
        for current, later in ((train, validation), (validation, test)):
            for start in range(max(0, len(current) - width + 1)):
                target = current.iloc[start + INPUT_WINDOW:start + width]
                self.assertLess(target.x_Timestamp.max(), later.x_Timestamp.min())

    def test_rolling_features_are_strictly_past_only(self):
        frame = pd.DataFrame({
            TIMESTAMP_COLUMN: pd.date_range("2020-01-01", periods=5, freq="h"),
            "t_kWh": [1.0, 1.0, 1.0, 999.0, 2.0], METER_COLUMN: "M1",
        })
        featured = add_lag_and_rolling_features(frame)
        self.assertEqual(featured.loc[3, "lag_1"], 1.0)
        self.assertEqual(featured.loc[3, "roll_mean_3"], 1.0)
        self.assertEqual(featured.loc[3, "roll_max_3"], 1.0)

    def test_gap_handling_never_uses_future_observation(self):
        hourly = resample_per_meter(raw_rows())
        self.assertTrue(pd.isna(hourly.loc[1, "t_kWh"]))
        self.assertAlmostEqual(hourly.loc[2, "t_kWh"], 2.0)
        source = (Path(__file__).parent / "research_pipeline.py").read_text().lower()
        for forbidden in (".ffill(", ".bfill(", ".interpolate("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
