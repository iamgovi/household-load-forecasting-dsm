"""Before/after metrics. Formulas are documented in MILP_FORMULATION.md and OPTIMISATION_OUTPUT_CONTRACT.md.

cost_reduction_pct  = 100 * (cost_before - cost_after) / cost_before
peak_reduction_pct  = 100 * (peak_before - peak_after) / peak_before
PAR                 = peak / mean(total hourly load)
shifted_energy      = Σ_a P_a · Σ_t max(0, x_after[a,t] − x_before[a,t])
                      (energy, per appliance, that now runs in an hour it did not run in before;
                       counted once, on the arrival side, and never netted across appliances)
Positive *_reduction_pct values mean improvement.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from optimisation.contracts import HOURLY_COLUMNS
from optimisation.problem import SchedulingProblem


def pct_reduction(before: float, after: float) -> float:
    return float("nan") if abs(before) < 1e-12 else 100.0 * (before - after) / before


def summary_metrics(problem: SchedulingProblem, before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    ts = problem.timestamps
    tb, ta = problem.total_load(before), problem.total_load(after)
    fb, fa = problem.appliance_load(before), problem.appliance_load(after)
    cost_b, cost_a = problem.tariff.bill(tb, ts), problem.tariff.bill(ta, ts)
    peak_b, peak_a = float(tb.max()), float(ta.max())
    par_b, par_a = peak_b / float(tb.mean()), peak_a / float(ta.mean())
    shifted = problem.shifted_energy_by_appliance(before, after)
    return {
        "cost_before": cost_b, "cost_after": cost_a, "cost_saving": cost_b - cost_a,
        "cost_reduction_pct": pct_reduction(cost_b, cost_a),
        "peak_before": peak_b, "peak_after": peak_a, "peak_reduction_pct": pct_reduction(peak_b, peak_a),
        "par_before": par_b, "par_after": par_a, "par_reduction_pct": pct_reduction(par_b, par_a),
        "shifted_energy": float(shifted.sum()),
        "net_profile_shift_kwh": 0.5 * float(np.abs(fa - fb).sum()),  # legacy Review-3 definition, for comparison only
        "energy_before_kwh": float(tb.sum()), "energy_after_kwh": float(ta.sum()),
        "appliance_energy_kwh": float(fb.sum()),
        "energy_conserved": bool(abs(tb.sum() - ta.sum()) < 1e-9),
        "cost_responsive": problem.cost_responsive,
        "cost_note": ("Tariff varies by hour: cost changes reflect load shifting."
                      if problem.cost_responsive else
                      "Tariff is not time-varying: shifting cannot change the bill; only peak/PAR effects are claimed."),
        "baseline_exceeds_max_load": bool((tb > problem.household.max_power_kw + 1e-9).any()),
    }


def hourly_table(problem: SchedulingProblem, before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "timestamp": problem.timestamps, "clock_hour": problem.timestamps.hour,
        "tariff_rate": problem.rates, "base_load": problem.base,
        "appliance_load_before": problem.appliance_load(before), "appliance_load_after": problem.appliance_load(after),
    })
    frame["total_before"] = frame.base_load + frame.appliance_load_before
    frame["total_after"] = frame.base_load + frame.appliance_load_after
    frame = frame[HOURLY_COLUMNS]
    for a, app in enumerate(problem.appliances):
        frame[f"on_before__{app.appliance_id}"] = before[a]
        frame[f"on_after__{app.appliance_id}"] = after[a]
    return frame


def appliance_table(problem: SchedulingProblem, before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    clock = problem.timestamps.hour.to_numpy()
    shifted = problem.shifted_energy_by_appliance(before, after)
    rows = []
    for a, app in enumerate(problem.appliances):
        bh, ah = np.flatnonzero(before[a]), np.flatnonzero(after[a])
        rows.append({
            "appliance_id": app.appliance_id, "name": app.name, "power_kw": app.power_kw,
            "duration_h": app.duration_h, "energy_kwh": app.energy_kwh,
            "allowed_window": f"{app.earliest_start:02d}:00–{app.latest_finish:02d}:00",
            "interruptible": app.interruptible,
            "start_before": int(clock[bh[0]]), "start_after": int(clock[ah[0]]),
            "hours_before": ",".join(str(int(clock[h])) for h in bh),
            "hours_after": ",".join(str(int(clock[h])) for h in ah),
            "moved": bool(not np.array_equal(bh, ah)), "shifted_energy_kwh": float(shifted[a]),
            "scheduling_cost_before": float(app.power_kw * problem.rates[bh].sum()),
            "scheduling_cost_after": float(app.power_kw * problem.rates[ah].sum()),
            "source_class": app.source_class,
        })
    return pd.DataFrame(rows)
