"""Map standardized dataset fields to the existing research contract."""

import pandas as pd

from evaluation.contract import (
    CITY_COLUMN,
    METER_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)


def to_research_schema(
    data: pd.DataFrame,
    location_column: str | None = "location",
) -> pd.DataFrame:
    """Map standardized timestamp/meter_id/load data to research column names."""

    required = {"timestamp", "meter_id", "load"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Standardized data is missing columns: {sorted(missing)}"
        )

    result = data.rename(
        columns={
            "timestamp": TIMESTAMP_COLUMN,
            "meter_id": METER_COLUMN,
            "load": TARGET_COLUMN,
        }
    ).copy()

    if location_column and location_column in result.columns:
        result = result.rename(columns={location_column: CITY_COLUMN})
    elif CITY_COLUMN not in result.columns:
        result[CITY_COLUMN] = "unspecified"

    return result
