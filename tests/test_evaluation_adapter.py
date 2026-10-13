import pandas as pd
import pytest

from forecasting.evaluation_adapter import to_research_schema


def test_standardized_columns_map_to_research_contract():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 01:00"]),
        "meter_id": ["house_1"],
        "load": [2.5],
        "location": ["sample_location"],
    })

    result = to_research_schema(source)

    assert "x_Timestamp" in result.columns
    assert "meter" in result.columns
    assert "t_kWh" in result.columns
    assert "city" in result.columns
    assert result.loc[0, "meter"] == "house_1"
    assert result.loc[0, "t_kWh"] == 2.5
    assert result.loc[0, "city"] == "sample_location"


def test_missing_location_uses_neutral_placeholder():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 01:00"]),
        "meter_id": ["house_1"],
        "load": [2.5],
    })

    result = to_research_schema(source)

    assert result.loc[0, "city"] == "unspecified"


def test_missing_required_field_fails():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 01:00"]),
        "meter_id": ["house_1"],
    })

    with pytest.raises(ValueError, match="missing columns"):
        to_research_schema(source)
