"""Solver-independent scheduling problem.

Holds the inputs, the feasibility rules and the objective in one place so that the MILP,
the validation tests and any future GA/PSO benchmark all use *exactly* the same definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimisation.contracts import HORIZON, Appliance, Household, validate_forecast
from optimisation.tariff import Tariff


@dataclass
class SchedulingProblem:
    forecast: pd.DataFrame
    tariff: Tariff
    appliances: list[Appliance]
    household: Household
    timestamps: pd.DatetimeIndex = field(init=False)
    base: np.ndarray = field(init=False)
    rates: np.ndarray = field(init=False)
    allowed: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.appliances:
            raise ValueError("At least one appliance is required")
        self.forecast = validate_forecast(self.forecast)
        self.timestamps = pd.DatetimeIndex(self.forecast["timestamp"])
        self.base = self.forecast["predicted_kwh"].to_numpy(float)
        self.rates = np.asarray(self.tariff.scheduling_rates(self.timestamps), float)
        self.allowed = np.array([self._window_slots(a) for a in self.appliances], dtype=bool)

    def _window_slots(self, app: Appliance) -> np.ndarray:
        """Slots of the ONE window occurrence the appliance is scheduled in.

        A clock-hour window can appear twice in a horizon that does not start at midnight
        (e.g. 22:00–06:00 in a 04:00→03:00 horizon shows up as 04–05 *and* 22–03). Those are
        different nights, so only one occurrence is allowed: the run of consecutive in-window
        slots that contains the preferred start, otherwise the first run long enough for the duration.
        """
        clock = self.timestamps.hour.to_numpy()
        in_window = np.array([h in app.allowed_clock_hours() for h in clock])
        runs, start = [], None
        for t in range(HORIZON + 1):
            if t < HORIZON and in_window[t]:
                start = t if start is None else start
            elif start is not None:
                runs.append((start, t)); start = None
        chosen = None
        if app.preferred_start is not None:
            pref = np.flatnonzero(clock == app.preferred_start)
            chosen = next((r for r in runs if len(pref) and r[0] <= pref[0] < r[1]), None)
        if chosen is None:
            chosen = next((r for r in runs if r[1] - r[0] >= app.duration_h), None)
        mask = np.zeros(HORIZON, dtype=bool)
        if chosen is not None:
            mask[chosen[0]:chosen[1]] = True
        return mask

    # ------------------------------------------------------------------ basics
    @property
    def n_appliances(self) -> int:
        return len(self.appliances)

    @property
    def powers(self) -> np.ndarray:
        return np.array([a.power_kw for a in self.appliances], float)

    @property
    def cost_responsive(self) -> bool:
        return self.tariff.is_time_varying(self.timestamps)

    def feasible_starts(self, a: int) -> list[int]:
        """Start slots k such that slots k..k+d-1 are inside the horizon and inside the allowed window."""
        d = self.appliances[a].duration_h
        return [k for k in range(HORIZON - d + 1) if self.allowed[a, k:k + d].all()]

    def slot_of_clock_hour(self, hour: int) -> int:
        matches = np.flatnonzero(self.timestamps.hour.to_numpy() == hour)
        if len(matches) != 1:
            raise ValueError(f"Clock hour {hour} does not occur exactly once in the horizon")
        return int(matches[0])

    def appliance_load(self, schedule: np.ndarray) -> np.ndarray:
        return self.powers @ np.asarray(schedule, float)

    def total_load(self, schedule: np.ndarray) -> np.ndarray:
        return self.base + self.appliance_load(schedule)

    # ------------------------------------------------------------------ baseline ("before")
    def baseline_schedule(self) -> np.ndarray:
        """Un-optimised behaviour: each appliance runs as one block from its preferred start.

        If ``preferred_start`` is None, the earliest feasible start is used.
        """
        schedule = np.zeros((self.n_appliances, HORIZON), dtype=int)
        for a, app in enumerate(self.appliances):
            starts = self.feasible_starts(a)
            if not starts:
                raise ValueError(f"{app.appliance_id}: no feasible contiguous run in this horizon")
            k = starts[0] if app.preferred_start is None else self.slot_of_clock_hour(app.preferred_start)
            if k not in starts:
                raise ValueError(f"{app.appliance_id}: preferred start {app.preferred_start}:00 is not feasible")
            schedule[a, k:k + app.duration_h] = 1
        return schedule

    # ------------------------------------------------------------------ objective
    def cost_normaliser(self, baseline: np.ndarray) -> float:
        value = float(np.sum(self.total_load(baseline) * self.rates))
        return value if value > 1e-12 else 1.0

    def objective(self, schedule: np.ndarray, baseline: np.ndarray) -> float:
        """Stage-1 objective: w_cost * cost/cost_baseline + w_peak * peak/max_power."""
        total = self.total_load(schedule)
        cost = float(np.sum(total * self.rates))
        h = self.household
        return h.w_cost * cost / self.cost_normaliser(baseline) + h.w_peak * float(total.max()) / h.max_power_kw

    def shifted_energy_by_appliance(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """kWh of each appliance that runs in an hour where it did not run before (no netting across appliances)."""
        moved = np.clip(np.asarray(after, int) - np.asarray(before, int), 0, None)
        return self.powers * moved.sum(axis=1)

    # ------------------------------------------------------------------ feasibility
    def check_schedule(self, schedule: np.ndarray, tol: float = 1e-9) -> list[str]:
        """Return a list of rule violations (empty list = feasible)."""
        s = np.asarray(schedule)
        problems: list[str] = []
        if s.shape != (self.n_appliances, HORIZON):
            return [f"schedule shape {s.shape} != {(self.n_appliances, HORIZON)}"]
        if not np.isin(s, (0, 1)).all():
            problems.append("schedule is not binary")
        for a, app in enumerate(self.appliances):
            on = np.flatnonzero(s[a])
            if len(on) != app.duration_h:
                problems.append(f"{app.appliance_id}: runs {len(on)} h, requires {app.duration_h} h")
            if not self.allowed[a, on].all():
                problems.append(f"{app.appliance_id}: runs outside its allowed window / after its deadline")
            if not app.interruptible and len(on) > 1 and not np.all(np.diff(on) == 1):
                problems.append(f"{app.appliance_id}: non-interruptible appliance was split")
        if (self.total_load(s) > self.household.max_power_kw + tol).any():
            problems.append("household maximum load exceeded")
        return problems
