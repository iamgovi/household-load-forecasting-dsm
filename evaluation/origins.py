"""Canonical validation/test forecast-origin construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.contract import (
    CITY_COLUMN, FREQUENCY, HORIZON, LOOKBACK, METER_COLUMN, ORIGIN_COLUMNS,
    ORIGIN_STRIDE_HOURS, TARGET_COLUMN, TIMESTAMP_COLUMN,
)


def load_research_splits(paths: dict[str, object]) -> dict[str, pd.DataFrame]:
    result = {}
    for split, path in paths.items():
        frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
        frame["split_source"] = split
        result[split] = frame.sort_values([METER_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    return result


def build_canonical_origins(
    splits: dict[str, pd.DataFrame], evaluation_split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return accepted origins, rejected candidates, and per-meter counts.

    The 168 input timestamps may be in an earlier split, but the entire target
    interval must be in ``evaluation_split``. No values are filled.
    """
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("Canonical research origins are generated only for validation or test")
    full = pd.concat(list(splits.values()), ignore_index=True)
    full = full.sort_values([METER_COLUMN, TIMESTAMP_COLUMN])
    accepted, rejected, counts = [], [], []
    one_hour = pd.Timedelta(FREQUENCY)

    for meter_id, target_split in splits[evaluation_split].groupby(METER_COLUMN, sort=True):
        city = target_split[CITY_COLUMN].iloc[0]
        meter_all = full[full[METER_COLUMN].eq(meter_id)].set_index(TIMESTAMP_COLUMN).sort_index()
        target_split = target_split.set_index(TIMESTAMP_COLUMN).sort_index()
        split_start, split_end = target_split.index.min(), target_split.index.max()
        all_index = meter_all.index
        valid_load = meter_all[TARGET_COLUMN].notna().to_numpy()
        invalid_prefix = np.concatenate(([0], np.cumsum(~valid_load)))
        candidate_offsets = np.arange(0, len(target_split), ORIGIN_STRIDE_HOURS)
        candidate_times = target_split.index[candidate_offsets]
        positions = all_index.get_indexer(candidate_times)
        meter_reasons = {}
        accepted_count = 0

        for split_position, first_target, first_position in zip(
            candidate_offsets, candidate_times, positions
        ):
            origin = first_target - one_hour
            last_target = first_target + pd.Timedelta(hours=HORIZON - 1)
            reason = None
            if last_target > split_end or first_target < split_start:
                reason = "target_outside_split"
            elif first_position < LOOKBACK:
                reason = "missing_history_timestamps"
            elif all_index[first_position - LOOKBACK] != first_target - pd.Timedelta(hours=LOOKBACK):
                reason = "missing_history_timestamps"
            elif invalid_prefix[first_position] - invalid_prefix[first_position - LOOKBACK] > 0:
                reason = "invalid_history_load"
            elif split_position + HORIZON > len(target_split) or (
                target_split.index[split_position + HORIZON - 1] != last_target
            ):
                reason = "missing_target_timestamps"
            elif invalid_prefix[first_position + HORIZON] - invalid_prefix[first_position] > 0:
                reason = "invalid_target_load"

            common = {
                CITY_COLUMN: city, METER_COLUMN: meter_id,
                "forecast_origin": origin,
                "first_target_timestamp": first_target,
                "last_target_timestamp": last_target,
                "horizon_length": HORIZON, "split": evaluation_split,
            }
            if reason is None:
                accepted.append(common)
                accepted_count += 1
            else:
                rejected.append({**common, "rejection_reason": reason})
                meter_reasons[reason] = meter_reasons.get(reason, 0) + 1
        counts.append({
            CITY_COLUMN: city, METER_COLUMN: meter_id, "split": evaluation_split,
            "candidate_origins": len(candidate_times), "accepted_origins": accepted_count,
            "rejected_origins": len(candidate_times) - accepted_count,
            **{f"rejected_{key}": value for key, value in meter_reasons.items()},
        })

    accepted_df = pd.DataFrame(accepted, columns=ORIGIN_COLUMNS)
    rejected_df = pd.DataFrame(rejected)
    counts_df = pd.DataFrame(counts).fillna(0)
    return accepted_df, rejected_df, counts_df


def validate_origins(origins: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    one_hour = pd.Timedelta(FREQUENCY)
    for row in origins.itertuples(index=False):
        target = pd.date_range(row.first_target_timestamp, periods=HORIZON, freq=FREQUENCY)
        if row.horizon_length != HORIZON or target[-1] != row.last_target_timestamp:
            raise AssertionError("Invalid canonical horizon")
        if row.forecast_origin + one_hour != row.first_target_timestamp:
            raise AssertionError("Origin/target alignment is not hourly")
        split_meter = splits[row.split][splits[row.split][METER_COLUMN].eq(row.meter)]
        if not target.isin(split_meter[TIMESTAMP_COLUMN]).all():
            raise AssertionError("Canonical target leaves its evaluation split")
