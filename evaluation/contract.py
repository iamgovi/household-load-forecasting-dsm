"""Central specification for every scientific forecasting experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research"
REPORT_DIR = ROOT / "reports" / "evaluation"
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

TIMESTAMP_COLUMN = "x_Timestamp"
TARGET_COLUMN = "t_kWh"
METER_COLUMN = "meter"
CITY_COLUMN = "city"
FREQUENCY = "1h"
LOOKBACK = 168
HORIZON = 24
ORIGIN_STRIDE_HOURS = 24
SEED = 42

PREDICTION_COLUMNS = [
    "model", "city", "meter", "split", "forecast_origin", "horizon",
    "target_timestamp", "actual", "prediction", "inference_time_ms", "run_id",
]
ORIGIN_COLUMNS = [
    "city", "meter", "forecast_origin", "first_target_timestamp",
    "last_target_timestamp", "horizon_length", "split",
]

METRIC_DEFINITIONS = {
    "MAE": "mean(abs(prediction - actual))",
    "RMSE": "sqrt(mean((prediction - actual)^2))",
    "sMAPE_pct": "mean(200*abs(prediction-actual)/(abs(actual)+abs(prediction))); zero/zero contributes 0",
    "R2": "1-SSE/SST on the stated group; NaN when actual variance is zero or group has fewer than 2 rows",
    "NRMSE_mean": "RMSE/mean(actual); NaN when mean(actual) <= 0",
    "peak_load_error": "per-origin abs(max(actual_24h)-max(prediction_24h))",
    "normalized_peak_error": "peak_load_error/max(actual_24h); NaN when actual peak is zero",
    "peak_timing_error_hours": "absolute difference between first actual and predicted peak horizons",
    "inference_time_ms": "measured model prediction time, aggregated as mean per origin where available",
}

DEPLOYMENT_NON_RESEARCH = [
    "general_gru_model.pt", "general_model_meta.joblib",
    "preprocess/generate_model.py", "train_general_model.py", "forecast_model.py",
]
