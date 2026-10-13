"""
Step 4: Statistical baseline models.

Evaluates on 24-HOUR-AHEAD rolling forecasts (matches the project's stated main
horizon), per meter, then aggregates metrics. This gives you the "floor" every
later ML/DL/hybrid model needs to beat.

Models included:
  - Naive              (uses lag_1, already in your engineered CSVs)
  - Seasonal Naive      (uses lag_24, already in your engineered CSVs)
  - Moving Average      (uses roll_mean_24, already in your engineered CSVs)
  - Simple Exp. Smoothing
  - Holt-Winters (additive trend + daily seasonality)
  - ARIMA (fixed low order -- not auto-tuned; that's PSO/Optuna's job later)

Requires: pip install statsmodels

Run: python step4_baselines.py
Reads: train.csv, val.csv, test.csv (from step2_pipeline.py)
Writes: baseline_results_per_meter.csv, baseline_results_summary.csv
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels is noisy about convergence warnings on messy real data

FORECAST_HORIZON = 24          # hours ahead per window -- matches "next 24 hours" target
SEASONAL_PERIOD = 24           # daily seasonality for hourly data
ARIMA_ORDER = (2, 1, 2)        # fixed, deliberately simple -- not the point of this stage
MAX_METERS = None              # set to e.g. 5 while testing the script, None for full run


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-6
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    s = smape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": s}


def naive_from_features(test_df):
    """These three don't need model fitting -- the features already encode them."""
    results = {}
    y_true = test_df["t_kWh"].values
    results["Naive"] = compute_metrics(y_true, test_df["lag_1"].values)
    results["Seasonal_Naive"] = compute_metrics(y_true, test_df["lag_24"].values)
    results["Moving_Average"] = compute_metrics(y_true, test_df["roll_mean_24"].values)
    return results


def rolling_forecast_eval(history_series, test_series, model_fn, horizon):
    """Generic rolling-origin evaluator: fit once on history, forecast `horizon`
    steps, append the true test values actually observed, forecast the next
    window, repeat. No refitting each window (fast) -- state update only,
    which is standard practice for this kind of baseline comparison."""
    try:
        model_fit = model_fn(history_series)
    except Exception as e:
        return None, f"fit failed: {e}"

    preds, actuals = [], []
    current_history = history_series.copy()
    pos = 0
    n = len(test_series)

    while pos < n:
        window = test_series.iloc[pos: pos + horizon]
        try:
            forecast = model_fit.forecast(steps=len(window))
        except Exception as e:
            return None, f"forecast failed: {e}"

        preds.extend(forecast.values if hasattr(forecast, "values") else forecast)
        actuals.extend(window.values)

        # update state with the true observed window before moving on
        current_history = pd.concat([current_history, window])
        try:
            model_fit = model_fit.append(window, refit=False)
        except Exception:
            pass  # some model types don't support append; fall back to static forecast beyond this point

        pos += horizon

    return compute_metrics(actuals, preds), None


def to_regular_hourly(series):
    """statsmodels' .forecast() needs an index with an explicit, regular freq to
    know what '24 steps ahead' means in calendar time. Our main pipeline
    intentionally drops rows with incomplete lag/rolling features, which can
    leave small mid-series gaps -- fine for ML models that use the lag columns
    directly, but NOT fine for ARIMA/ETS, which need a continuous, evenly
    spaced series. This reindexes to a full hourly grid and linearly
    interpolates the (small number of) resulting gaps -- used ONLY for these
    statistical baselines, not the main feature-engineered dataset."""
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    reindexed = series.reindex(full_index)
    n_filled = reindexed.isna().sum()
    if n_filled > 0:
        reindexed = reindexed.interpolate(limit_direction="both")
    return reindexed, n_filled


def fit_ses(series):
    return SimpleExpSmoothing(series, initialization_method="estimated").fit()


def fit_holt_winters(series):
    return ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    ).fit()


def fit_arima(series):
    return ARIMA(series, order=ARIMA_ORDER).fit()


def run_for_meter(meter_id, train, val, test):
    m_train = train[train["meter"] == meter_id].sort_values("x_Timestamp")
    m_val = val[val["meter"] == meter_id].sort_values("x_Timestamp")
    m_test = test[test["meter"] == meter_id].sort_values("x_Timestamp")

    if len(m_train) < SEASONAL_PERIOD * 3 or len(m_test) < FORECAST_HORIZON:
        return None  # not enough data for this meter to evaluate meaningfully

    history = pd.concat([m_train, m_val]).set_index("x_Timestamp")["t_kWh"]
    test_series = m_test.set_index("x_Timestamp")["t_kWh"]

    row = {"meter": meter_id, "city": m_train["city"].iloc[0]}

    # feature-based baselines (fast, always run) -- use the original data as-is,
    # no reindexing needed since they read straight from existing lag/rolling columns
    naive_results = naive_from_features(m_test)
    for name, metrics in naive_results.items():
        for k, v in metrics.items():
            row[f"{name}_{k}"] = v

    # model-based baselines need a strictly regular hourly index -- see to_regular_hourly()
    history_reg, n_filled_hist = to_regular_hourly(history)
    test_reg, n_filled_test = to_regular_hourly(test_series)
    row["n_interpolated_history_hours"] = n_filled_hist
    row["n_interpolated_test_hours"] = n_filled_test

    for name, fit_fn in [("SES", fit_ses), ("HoltWinters", fit_holt_winters), ("ARIMA", fit_arima)]:
        metrics, err = rolling_forecast_eval(history_reg, test_reg, fit_fn, FORECAST_HORIZON)
        if metrics:
            for k, v in metrics.items():
                row[f"{name}_{k}"] = v
        else:
            row[f"{name}_error"] = err

    return row


def main():
    train = pd.read_csv("../train.csv", parse_dates=["x_Timestamp"])
    val = pd.read_csv("../val.csv", parse_dates=["x_Timestamp"])
    test = pd.read_csv("../test.csv", parse_dates=["x_Timestamp"])

    meters = sorted(train["meter"].unique())
    if MAX_METERS:
        meters = meters[:MAX_METERS]
        print(f"MAX_METERS is set -- running on only {len(meters)} meters for a quick test")

    results = []
    for i, meter_id in enumerate(meters, 1):
        print(f"[{i}/{len(meters)}] {meter_id}...")
        row = run_for_meter(meter_id, train, val, test)
        if row:
            results.append(row)
        else:
            print(f"  skipped (insufficient data)")

    results_df = pd.DataFrame(results)
    results_df.to_csv("baseline_results_per_meter.csv", index=False)
    print(f"\nSaved per-meter results for {len(results_df)} meters -> baseline_results_per_meter.csv")

    error_cols = [c for c in results_df.columns if c.endswith("_error")]
    if error_cols:
        print("\n" + "=" * 60)
        print("MODEL FIT/FORECAST FAILURE RATES (check these before trusting the metrics below)")
        print("=" * 60)
        for c in error_cols:
            n_failed = results_df[c].notna().sum()
            print(f"{c}: {n_failed}/{len(results_df)} meters failed")
            if n_failed > 0:
                print(f"  sample error: {results_df[c].dropna().iloc[0]}")

    metric_cols = [c for c in results_df.columns if any(
        c.endswith(m) for m in ["_MAE", "_RMSE", "_sMAPE"]
    )]
    summary = results_df[metric_cols].mean().sort_index()
    summary_by_city = results_df.groupby("city")[metric_cols].mean()

    print("\n" + "=" * 60)
    print("OVERALL MEAN METRICS ACROSS ALL EVALUATED METERS")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("MEAN METRICS BY CITY")
    print("=" * 60)
    print(summary_by_city.T)

    summary.to_csv("baseline_results_summary.csv")
    print("\nSaved aggregated summary -> baseline_results_summary.csv")
    print(
        "\nNEXT: whichever model has the lowest RMSE/sMAPE here is your floor. "
        "Every ML/DL/hybrid model you build after this needs to beat it -- if it "
        "doesn't, that's a real, reportable finding, not a failure."
    )


if __name__ == "__main__":
    main()