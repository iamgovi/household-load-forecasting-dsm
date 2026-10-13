"""Input and output contracts for the optimisation engine.

Inputs
------
* Forecast   : DataFrame with ``timestamp`` and ``predicted_kwh`` (24 consecutive hourly rows).
               ``prediction`` / ``target_timestamp`` (the research long-schema names) are accepted
               as aliases so a filtered slice of the common prediction table can be passed directly.
* Tariff     : any :class:`optimisation.tariff.Tariff` (supplied externally; see TARIFF_INTERFACE.md).
* Appliances : list of :class:`Appliance` (JSON loader below).
* Household  : :class:`Household` (maximum load, objective weights).

Units: one schedule slot = one hour, so an hourly energy value in kWh is numerically the
average power in kW over that hour. All "kW" peaks are hourly-average kW.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HORIZON = 24

SIMULATION_LABEL = "SIMULATION / LITERATURE-BASED ASSUMPTION"


# --------------------------------------------------------------------------- forecast
def validate_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    """Return a clean ``timestamp, predicted_kwh`` frame or raise ``ValueError``."""
    frame = forecast.rename(columns={"target_timestamp": "timestamp", "prediction": "predicted_kwh"})
    missing = {"timestamp", "predicted_kwh"} - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast is missing required columns: {sorted(missing)}")
    frame = frame[["timestamp", "predicted_kwh"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame) != HORIZON:
        raise ValueError(f"Forecast must have exactly {HORIZON} hourly rows, got {len(frame)}")
    steps = frame["timestamp"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast timestamps must be 24 consecutive hours")
    values = frame["predicted_kwh"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast contains NaN or infinite values")
    if (values < 0).any():
        raise ValueError("Forecast contains negative load values")
    return frame


# --------------------------------------------------------------------------- appliances
@dataclass(frozen=True)
class Appliance:
    """One flexible appliance.

    ``earliest_start`` and ``latest_finish`` are clock hours (0–24). ``latest_finish`` is
    EXCLUSIVE: the appliance must be OFF by that hour (it is the deadline). If
    ``latest_finish <= earliest_start`` the window wraps past midnight (e.g. 22 → 6).
    ``preferred_start`` defines the un-optimised "before" behaviour.
    """

    appliance_id: str
    name: str
    power_kw: float
    duration_h: int
    earliest_start: int
    latest_finish: int
    interruptible: bool = False
    preferred_start: int | None = None
    priority: int | None = None
    source_class: str = SIMULATION_LABEL
    notes: str = ""

    def __post_init__(self):
        if self.power_kw <= 0:
            raise ValueError(f"{self.appliance_id}: power_kw must be > 0")
        if not isinstance(self.duration_h, int) or self.duration_h <= 0:
            raise ValueError(f"{self.appliance_id}: duration_h must be a positive integer")
        for key in ("earliest_start", "latest_finish"):
            if not 0 <= getattr(self, key) <= 24:
                raise ValueError(f"{self.appliance_id}: {key} must be in 0..24")
        if self.duration_h > len(self.allowed_clock_hours()):
            raise ValueError(f"{self.appliance_id}: duration longer than its allowed window")

    def allowed_clock_hours(self) -> set[int]:
        start, end = self.earliest_start % 24, self.latest_finish % 24
        if self.latest_finish == 24 and self.earliest_start < 24:
            end = 24
        if start < end:
            return set(range(start, end))
        if start == end:  # full day
            return set(range(24))
        return set(range(start, 24)) | set(range(0, end))

    @property
    def energy_kwh(self) -> float:
        return self.power_kw * self.duration_h


def load_appliances(path: str | Path) -> list[Appliance]:
    data = json.loads(Path(path).read_text())
    items = data["appliances"] if isinstance(data, dict) else data
    appliances = [Appliance(**item) for item in items]
    ids = [a.appliance_id for a in appliances]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate appliance_id values")
    return appliances


# --------------------------------------------------------------------------- household
@dataclass(frozen=True)
class Household:
    """Household limits and objective weights.

    Objective (stage 1):  w_cost * schedulable_cost / baseline_cost + w_peak * peak / max_power_kw
    Stage 2 (tie-break):  among stage-1-optimal schedules, minimise shifted energy.
    """

    max_power_kw: float = 3.5
    w_cost: float = 0.3
    w_peak: float = 0.7
    lexicographic_tie_break: bool = True
    preferences: dict[str, Any] = field(default_factory=dict)
    source_class: str = SIMULATION_LABEL

    def __post_init__(self):
        if self.max_power_kw <= 0:
            raise ValueError("max_power_kw must be > 0")
        if self.w_cost < 0 or self.w_peak < 0 or self.w_cost + self.w_peak == 0:
            raise ValueError("Weights must be non-negative and not both zero")


def load_household(path: str | Path) -> Household:
    return Household(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- output
HOURLY_COLUMNS = ["timestamp", "clock_hour", "tariff_rate", "base_load", "appliance_load_before",
                  "appliance_load_after", "total_before", "total_after"]

SUMMARY_KEYS = ["cost_before", "cost_after", "cost_reduction_pct", "peak_before", "peak_after",
                "peak_reduction_pct", "par_before", "par_after", "shifted_energy", "solver_status",
                "runtime"]


@dataclass
class OptimisationResult:
    """Everything the dashboard (later) and the reports need. See OPTIMISATION_OUTPUT_CONTRACT.md."""

    hourly: pd.DataFrame
    summary: dict[str, Any]
    appliance_schedule: pd.DataFrame
    before_matrix: np.ndarray
    after_matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        hourly = self.hourly.copy()
        hourly["timestamp"] = hourly["timestamp"].astype(str)
        return {"summary": self.summary, "metadata": self.metadata,
                "hourly": hourly.to_dict(orient="records"),
                "appliance_schedule": self.appliance_schedule.to_dict(orient="records")}

    def save(self, directory: str | Path, prefix: str = "") -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.hourly.to_csv(out / f"{prefix}hourly.csv", index=False)
        self.appliance_schedule.to_csv(out / f"{prefix}appliance_schedule.csv", index=False)
        (out / f"{prefix}summary.json").write_text(
            json.dumps({"summary": self.summary, "metadata": self.metadata}, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def appliance_to_dict(appliance: Appliance) -> dict[str, Any]:
    return asdict(appliance)
