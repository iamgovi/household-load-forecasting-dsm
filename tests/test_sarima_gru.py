"""Tests for the SARIMA-GRU preparation implementation."""

import numpy as np
import pandas as pd
import pytest

from forecasting.registry import get_baseline_forecaster, get_model_spec
from forecasting.sarima_gru import SARIMAGRUForecaster


def make_frame(count=500, meter="meter-test"):
    timestamps = pd.date_range(
        "2025-01-01", periods=count, freq="h"
    )
    t = np.arange(count, dtype=float)
    load = 10 + 2 * np.sin(2 * np.pi * t / 24) + 0.01 * t

    return pd.DataFrame({
        "timestamp": timestamps,
        "meter_id": meter,
        "load": load,
    })


def test_sarima_gru_is_registered():
    model = get_baseline_forecaster("sarima_gru")

    assert isinstance(model, SARIMAGRUForecaster)
    assert get_model_spec("sarima_gru").name == "sarima_gru"


def test_prepare_frame_rejects_missing_load():
    frame = make_frame()
    frame.loc[10, "load"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        SARIMAGRUForecaster._prepare_frame(frame, "test_data")


def test_prepare_frame_rejects_duplicate_timestamps():
    frame = make_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        SARIMAGRUForecaster._prepare_frame(frame, "test_data")


def test_prepare_frame_rejects_multiple_meters():
    frame = make_frame()
    frame.loc[0, "meter_id"] = "different-meter"

    with pytest.raises(ValueError, match="exactly one"):
        SARIMAGRUForecaster._prepare_frame(frame, "test_data")


def test_short_training_series_is_rejected():
    model = SARIMAGRUForecaster()
    frame = make_frame(count=100)

    with pytest.raises(ValueError, match="Insufficient data"):
        model.fit(frame)


def test_predict_requires_fit():
    model = SARIMAGRUForecaster()

    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict(make_frame(count=168))
