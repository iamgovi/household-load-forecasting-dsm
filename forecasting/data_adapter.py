"""Dataset adapter for standardized household load forecasting data."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "meter_id", "load"]
OPTIONAL_COLUMNS = ["location", "quality"]


def normalize_load_data(
    data: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert a dataset to the common forecasting input schema.

    Canonical fields:
        timestamp: observation timestamp
        meter_id: unique meter identifier
        load: numeric load/energy target
        location: optional location metadata
        quality: optional data-quality/coverage indicator

    column_map maps canonical names to source column names.
    Example: {"timestamp": "datetime", "meter_id": "house_id", "load": "t_kWh"}

    This function does not interpolate, impute, resample, or modify the input.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input dataset must not be empty.")

    column_map = column_map or {}
    allowed_names = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    unknown_names = set(column_map) - set(allowed_names)
    if unknown_names:
        raise ValueError(
            f"Unsupported canonical column mappings: {sorted(unknown_names)}"
        )

    source_names = {
        canonical: column_map.get(canonical, canonical)
        for canonical in allowed_names
    }

    required_sources = {
        canonical: source_names[canonical] for canonical in REQUIRED_COLUMNS
    }

    missing = [
        source for source in required_sources.values()
        if source not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    selected = {}
    for canonical in allowed_names:
        source = source_names[canonical]
        if source in data.columns:
            selected[canonical] = data[source].copy()

    result = pd.DataFrame(selected, index=data.index).reset_index(drop=True)

    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise"
    )

    if result["timestamp"].isna().any():
        raise ValueError("Timestamp values must not be missing.")

    if result["meter_id"].isna().any():
        raise ValueError("Meter IDs must not be missing.")

    result["meter_id"] = result["meter_id"].astype(str).str.strip()

    if result["meter_id"].eq("").any():
        raise ValueError("Meter IDs must not be empty.")

    result["load"] = pd.to_numeric(result["load"], errors="raise")

    if not np.isfinite(result["load"].to_numpy(dtype=float)).all():
        raise ValueError("Load values must be finite and non-missing.")

    duplicate_rows = result.duplicated(["meter_id", "timestamp"])

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate timestamp/meter_id observations found. "
            "Resolve duplicates explicitly before forecasting."
        )

    if "location" in result.columns:
        result["location"] = result["location"].astype("string")

    if "quality" in result.columns:
        if result["quality"].isna().any():
            raise ValueError("Quality indicators must not be missing.")
        # Preserve the source quality values; do not assume a particular scale.

    result = result.sort_values(
        ["meter_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)

    return result
