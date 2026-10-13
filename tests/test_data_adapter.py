"""Tests for the dataset-agnostic input adapter."""

import pandas as pd
import pytest

from forecasting.data_adapter import normalize_load_data


def test_standard_columns_are_accepted_and_sorted():
    source = pd.DataFrame({
        "timestamp": ["2025-01-01 02:00", "2025-01-01 01:00"],
        "meter_id": ["meter_b", "meter_b"],
        "load": [2.5, 1.5],
    })

    result = normalize_load_data(source)

    assert list(result.columns) == ["timestamp", "meter_id", "load"]
    assert result["timestamp"].is_monotonic_increasing
    assert result["load"].tolist() == [1.5, 2.5]


def test_custom_dataset_column_mapping():
    source = pd.DataFrame({
        "datetime": ["2025-01-01 01:00"],
        "house_id": ["house_001"],
        "t_kWh": [3.2],
        "district": ["example_location"],
        "coverage": [1.0],
    })

    result = normalize_load_data(
        source,
        column_map={
            "timestamp": "datetime",
            "meter_id": "house_id",
            "load": "t_kWh",
            "location": "district",
            "quality": "coverage",
        },
    )

    assert result.loc[0, "meter_id"] == "house_001"
    assert result.loc[0, "load"] == 3.2
    assert result.loc[0, "location"] == "example_location"
    assert result.loc[0, "quality"] == 1.0


def test_missing_required_column_fails():
    source = pd.DataFrame({
        "timestamp": ["2025-01-01 01:00"],
        "meter_id": ["meter_1"],
    })

    with pytest.raises(ValueError, match="Missing required source columns"):
        normalize_load_data(source)


def test_duplicate_meter_timestamp_fails():
    source = pd.DataFrame({
        "timestamp": ["2025-01-01 01:00", "2025-01-01 01:00"],
        "meter_id": ["meter_1", "meter_1"],
        "load": [1.0, 2.0],
    })

    with pytest.raises(ValueError, match="Duplicate timestamp/meter_id"):
        normalize_load_data(source)


def test_missing_load_fails():
    source = pd.DataFrame({
        "timestamp": ["2025-01-01 01:00"],
        "meter_id": ["meter_1"],
        "load": [None],
    })

    with pytest.raises(ValueError):
        normalize_load_data(source)


def test_input_is_not_modified():
    source = pd.DataFrame({
        "timestamp": ["2025-01-01 02:00", "2025-01-01 01:00"],
        "meter_id": ["meter_1", "meter_1"],
        "load": [2.0, 1.0],
    })
    original = source.copy(deep=True)

    normalize_load_data(source)

    pd.testing.assert_frame_equal(source, original)
