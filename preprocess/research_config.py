"""Central configuration for leakage-safe research preprocessing only."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {
    "Bareilly": ROOT / "archive (1)" / "CEEW - Smart meter data Bareilly 2020.csv",
    "Mathura": ROOT / "archive (1)" / "CEEW - Smart meter data Mathura 2020.csv",
}

RANDOM_SEED = 42
FORECAST_HORIZON = 24
INPUT_WINDOW = 168
RESAMPLE_FREQUENCY = "1h"
EXPECTED_READINGS_PER_HOUR = 20
MIN_HOURLY_COVERAGE = 1.0
MIN_METER_COMPLETENESS = 0.60
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

TIMESTAMP_COLUMN = "x_Timestamp"
TARGET_COLUMN = "t_kWh"
METER_COLUMN = "meter"
CITY_COLUMN = "city"
LAGS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (3, 24, 168)

OUTPUT_DIR = ROOT / "data" / "research"
REPORT_DIR = ROOT / "reports" / "preprocessing"
TRAIN_OUTPUT = OUTPUT_DIR / "train.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.csv"
TEST_OUTPUT = OUTPUT_DIR / "test.csv"
SPLIT_INTEGRITY_OUTPUT = REPORT_DIR / "split_integrity.csv"
SUMMARY_OUTPUT = REPORT_DIR / "preprocessing_summary.json"
REPORT_OUTPUT = REPORT_DIR / "PREPROCESSING_REPORT.md"
