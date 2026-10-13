
"""Controlled single-origin forecasting runner."""

import math
import time

import pandas as pd

from forecasting.contracts import validate_predictions
from forecasting.interface import Forecaster


def _validate_loads(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate and convert load values to finite floats."""
    loads = pd.to_numeric(frame["load"], errors="coerce")

    if loads.isna().any() or not loads.map(math.isfinite).all():
        raise ValueError(f"{label} contains missing or non-finite loads")

    frame["load"] = loads.astype(float)
    return frame


def _validate_hourly_sequence(
    frame: pd.DataFrame,
    label: str,
    *,
    sort: bool,
) -> pd.DataFrame:
    """Validate unique, consecutive hourly timestamps."""
    if sort:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{label} contains duplicate timestamps")

    differences = frame["timestamp"].diff().iloc[1:]

    if not differences.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            f"{label} must contain consecutive hourly observations "
            "in chronological order"
        )

    return frame


def forecast_one_origin(
    forecaster: Forecaster,
    training_data: pd.DataFrame,
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    city: str,
    split: str,
    forecast_origin,
    run_id: str,
) -> pd.DataFrame:
    """Generate one forecast without loading data or initiating experiments.

    The model is fitted only on training_data. History is used as prediction
    context, and future actual values are used only for output evaluation.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")

    required = {"timestamp", "meter_id", "load"}

    frames = (
        ("training_data", training_data),
        ("history", history),
        ("future", future),
    )

    for label, frame in frames:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{label} must be a pandas DataFrame")

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

        if frame.empty:
            raise ValueError(f"{label} cannot be empty")

    if len(history) != 168:
        raise ValueError("history must contain exactly 168 observations")

    if len(future) != 24:
        raise ValueError("future must contain exactly 24 observations")

    origin = pd.Timestamp(forecast_origin)

    if pd.isna(origin):
        raise ValueError("forecast_origin must be a valid timestamp")

    train = training_data.copy()
    history = history.copy()
    future = future.copy()

    # Parse timestamps and reject invalid or missing values.
    frames = (
        ("training_data", train),
        ("history", history),
        ("future", future),
    )

    for label, frame in frames:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="raise"
        )

        if frame["timestamp"].isna().any():
            raise ValueError(f"{label} contains invalid timestamps")

    # All timestamps and the forecast origin must use the same
    # timezone-awareness convention.
    origin_is_aware = origin.tzinfo is not None

    for label, frame in frames:
        timestamp_timezone = frame["timestamp"].dt.tz
        timestamps_are_aware = timestamp_timezone is not None

        if timestamps_are_aware != origin_is_aware:
            raise ValueError(
                f"{label} timezone convention differs from forecast_origin"
            )

    if train["timestamp"].dt.tz != history["timestamp"].dt.tz:
        raise ValueError("Training and history timezone conventions differ")

    if history["timestamp"].dt.tz != future["timestamp"].dt.tz:
        raise ValueError("History and future timezone conventions differ")

    # Validate meter identifiers.
    for label, frame in frames:
        if frame["meter_id"].isna().any():
            raise ValueError(f"{label} contains missing meter identifiers")

        identifiers = frame["meter_id"].astype(str).str.strip()

        if identifiers.eq("").any():
            raise ValueError(f"{label} contains empty meter identifiers")

    meters = history["meter_id"].astype(str).unique()

    if len(meters) != 1:
        raise ValueError("history must contain exactly one meter")

    meter = meters[0]

    if not train["meter_id"].astype(str).eq(meter).all():
        raise ValueError("training data must belong to the forecast meter")

    if not future["meter_id"].astype(str).eq(meter).all():
        raise ValueError("future observations must belong to the forecast meter")

    # Reject training observations after the origin before checking
    # chronological continuity, so the relevant error is reported.
    if train["timestamp"].max() > origin:
        raise ValueError(
            "Training data contains observations after forecast origin"
        )

    # Sort historical inputs and validate their hourly continuity.
    train = _validate_hourly_sequence(
        train, "training_data", sort=True
    )
    history = _validate_hourly_sequence(
        history, "history", sort=True
    )

    train = _validate_loads(train, "training_data")
    history = _validate_loads(history, "history")

    # The supplied context must end exactly at the origin.
    if history["timestamp"].iloc[-1] != origin:
        raise ValueError(
            "forecast_origin must equal the final history timestamp"
        )

    # Future rows are evaluation targets, never model inputs.
    # Sort them before validating the expected target timestamps.
    future = future.sort_values("timestamp").reset_index(drop=True)

    if future["timestamp"].duplicated().any():
        raise ValueError("future contains duplicate timestamps")

    expected = pd.date_range(
        start=origin + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
    )

    if not future["timestamp"].reset_index(drop=True).equals(
        pd.Series(expected, name="timestamp")
    ):
        raise ValueError(
            "future timestamps must be the 24 hours after the origin"
        )

    actual = pd.to_numeric(future["load"], errors="coerce")

    if actual.isna().any() or not actual.map(math.isfinite).all():
        raise ValueError(
            "future actuals contain missing or non-finite loads"
        )

    # Overlapping training and history observations must agree.
    # Overlap itself is permitted.
    train_history = train[["timestamp", "load"]].merge(
        history[["timestamp", "load"]],
        on="timestamp",
        how="inner",
        suffixes=("_train", "_history"),
    )

    if not train_history.empty:
        if not train_history["load_train"].eq(
            train_history["load_history"]
        ).all():
            raise ValueError(
                "Training data and history disagree on overlapping loads"
            )

    location = (
        str(history["location"].iloc[-1])
        if "location" in history.columns
        and pd.notna(history["location"].iloc[-1])
        else "unspecified"
    )

    output_city = location if city == "unspecified" else city

    # Fit only on training_data; future actuals are excluded from inputs.
    forecaster.fit(train)

    started = time.perf_counter()
    predictions = forecaster.predict(history)
    inference_ms = (time.perf_counter() - started) * 1000

    predictions = forecaster.validate_predictions(predictions)

    result = pd.DataFrame(
        [
            {
                "model": forecaster.model_name,
                "city": output_city,
                "meter": meter,
                "split": split,
                "forecast_origin": origin,
                "horizon": horizon,
                "target_timestamp": timestamp,
                "actual": float(truth),
                "prediction": float(estimate),
                "inference_time_ms": inference_ms,
                "run_id": run_id.strip(),
            }
            for horizon, (timestamp, truth, estimate) in enumerate(
                zip(
                    expected,
                    actual.to_numpy(dtype=float),
                    predictions,
                ),
                start=1,
            )
        ]
    )

    validate_predictions(result)
    return result
