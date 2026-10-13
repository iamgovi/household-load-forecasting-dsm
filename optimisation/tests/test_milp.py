"""Validation tests for the optimisation engine. Self-contained: synthetic inputs, no gitignored files.

Run:  python -m unittest discover -s optimisation/tests -t . -v
"""
from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from optimisation import (Appliance, FlatTariff, Household, SchedulingProblem, SlabTariff, TouTariff,
                          illustrative_tou_tariff, load_appliances, optimise, solve_milp, tariff_from_dict)
from optimisation.heuristic_interface import BaselineScheduler, MILPScheduler, benchmark

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXAMPLE = Path(__file__).resolve().parents[2] / "reports/review3/demand_side_optimisation/schedules/source_forecast_24h.csv"


def synthetic_forecast(start="2024-01-01 04:00", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=24, freq="h")
    hours = ts.hour.to_numpy()
    load = 0.35 + 0.25 * np.exp(-((hours - 20) % 24) ** 2 / 8) + 0.05 * rng.random(24)
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": load})


def illustrative_appliances():
    return load_appliances(CONFIGS / "appliances_illustrative.json")


class ScheduleRules(unittest.TestCase):
    """Every rule in the brief, checked on several forecasts/tariffs."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        tariffs = [illustrative_tou_tariff(), FlatTariff(rate=6.0),
                   SlabTariff(slabs=[{"upto_kwh": 100, "rate": 0}, {"upto_kwh": None, "rate": 5}])]
        for seed, tariff in itertools.product(range(3), tariffs):
            problem = SchedulingProblem(synthetic_forecast(seed=seed), tariff, illustrative_appliances(), Household())
            before = problem.baseline_schedule()
            after, info = solve_milp(problem, before)
            cls.cases.append((problem, before, after, info))

    def test_solver_returns_optimal(self):
        for _, _, after, info in self.cases:
            self.assertEqual(info["solver_status"], "optimal")
            self.assertIsNotNone(after)

    def test_binary(self):
        for _, _, after, _ in self.cases:
            self.assertTrue(np.isin(after, (0, 1)).all())

    def test_required_duration(self):
        for p, _, after, _ in self.cases:
            for a, app in enumerate(p.appliances):
                self.assertEqual(int(after[a].sum()), app.duration_h, app.appliance_id)

    def test_window_and_deadline(self):
        for p, _, after, _ in self.cases:
            clock = p.timestamps.hour.to_numpy()
            for a, app in enumerate(p.appliances):
                for h in clock[after[a] == 1]:
                    self.assertIn(int(h), app.allowed_clock_hours(), f"{app.appliance_id} ON at {h}:00")

    def test_non_interruptible_consecutive(self):
        for p, _, after, _ in self.cases:
            for a, app in enumerate(p.appliances):
                if not app.interruptible:
                    on = np.flatnonzero(after[a])
                    self.assertTrue(np.all(np.diff(on) == 1), app.appliance_id)

    def test_max_load_never_violated(self):
        for p, _, after, _ in self.cases:
            self.assertLessEqual(p.total_load(after).max(), p.household.max_power_kw + 1e-9)

    def test_energy_accounting(self):
        for p, before, after, _ in self.cases:
            self.assertAlmostEqual(p.appliance_load(before).sum(), p.appliance_load(after).sum(), places=9)
            self.assertAlmostEqual(p.appliance_load(after).sum(), sum(a.energy_kwh for a in p.appliances), places=9)
            self.assertAlmostEqual(p.total_load(after).sum(), p.base.sum() + p.appliance_load(after).sum(), places=9)

    def test_objective_matches_schedule(self):
        for p, before, after, info in self.cases:
            self.assertAlmostEqual(info["objective_value"], p.objective(after, before), places=6)

    def test_feasibility_checker_agrees(self):
        for p, before, after, _ in self.cases:
            self.assertEqual(p.check_schedule(after), [])
            self.assertEqual(p.check_schedule(before), [])

    def test_never_worse_than_baseline(self):
        for p, before, after, _ in self.cases:
            self.assertLessEqual(p.objective(after, before), p.objective(before, before) + 1e-9)


class Optimality(unittest.TestCase):
    def test_matches_brute_force(self):
        """Enumerate every combination of feasible starts and confirm the MILP finds the true optimum."""
        apps = [a for a in illustrative_appliances() if not a.interruptible]
        for seed in range(3):
            p = SchedulingProblem(synthetic_forecast(seed=seed), illustrative_tou_tariff(), apps, Household())
            base = p.baseline_schedule()
            best = np.inf
            for combo in itertools.product(*[p.feasible_starts(a) for a in range(len(apps))]):
                s = np.zeros((len(apps), 24), int)
                for a, k in enumerate(combo):
                    s[a, k:k + apps[a].duration_h] = 1
                if not p.check_schedule(s):
                    best = min(best, p.objective(s, base))
            after, info = solve_milp(p, base)
            self.assertAlmostEqual(info["objective_value"], best, places=7)


class InterruptibleAndWindows(unittest.TestCase):
    def _two_cheap_islands(self):
        rates = [9.0] * 24
        rates[1] = rates[5] = 1.0  # cheap hours that are NOT adjacent
        return TouTariff(hourly_rates=rates)

    def test_interruptible_may_split(self):
        app = Appliance("x", "X", 1.0, 2, 0, 8, interruptible=True, preferred_start=2)
        p = SchedulingProblem(synthetic_forecast("2024-01-01 00:00"), self._two_cheap_islands(), [app],
                              Household(w_cost=1.0, w_peak=0.0))
        after, _ = solve_milp(p)
        self.assertEqual(p.timestamps.hour[after[0] == 1].tolist(), [1, 5])

    def test_non_interruptible_never_splits(self):
        app = Appliance("x", "X", 1.0, 2, 0, 8, interruptible=False, preferred_start=2)
        p = SchedulingProblem(synthetic_forecast("2024-01-01 00:00"), self._two_cheap_islands(), [app],
                              Household(w_cost=1.0, w_peak=0.0))
        after, _ = solve_milp(p)
        on = np.flatnonzero(after[0])
        self.assertEqual(len(on), 2)
        self.assertEqual(on[1] - on[0], 1)

    def test_overnight_window(self):
        app = Appliance("ev", "EV", 0.7, 3, 22, 6, interruptible=True, preferred_start=22)
        self.assertEqual(app.allowed_clock_hours(), {22, 23, 0, 1, 2, 3, 4, 5})
        p = SchedulingProblem(synthetic_forecast(), illustrative_tou_tariff(), [app], Household())  # horizon 04:00→03:00
        after, _ = solve_milp(p)
        on = np.flatnonzero(after[0])
        self.assertTrue(set(p.timestamps.hour[on]) <= app.allowed_clock_hours())
        # Must charge on the night it is plugged in (slots 18..23), never 04:00–05:00 of the previous night.
        self.assertTrue((on >= 18).all(), on)

    def test_window_occurrence_respected_by_checker(self):
        app = Appliance("ev", "EV", 0.7, 2, 22, 6, interruptible=True, preferred_start=22)
        p = SchedulingProblem(synthetic_forecast(), FlatTariff(rate=1), [app], Household())
        wrong_night = np.zeros((1, 24), int); wrong_night[0, [0, 1]] = 1  # 04:00 and 05:00 slots
        self.assertTrue(p.check_schedule(wrong_night))

    def test_infeasible_preferred_start_rejected(self):
        app = Appliance("w", "W", 1.0, 2, 8, 10, preferred_start=9)  # 9→11 overruns the 10:00 deadline
        p = SchedulingProblem(synthetic_forecast(), FlatTariff(rate=5), [app], Household())
        with self.assertRaises(ValueError):
            p.baseline_schedule()


class TariffBehaviour(unittest.TestCase):
    def test_flat_tariff_claims_no_cost_saving(self):
        r = optimise(synthetic_forecast(), FlatTariff(rate=6.0), illustrative_appliances())
        self.assertFalse(r.summary["cost_responsive"])
        self.assertAlmostEqual(r.summary["cost_reduction_pct"], 0.0, places=9)

    def test_slab_bill_independent_of_schedule(self):
        slab = SlabTariff(slabs=[{"upto_kwh": 100, "rate": 0.0}, {"upto_kwh": 200, "rate": 2.5},
                                 {"upto_kwh": None, "rate": 6.0}], prior_cycle_kwh=95)
        r = optimise(synthetic_forecast(), slab, illustrative_appliances())
        self.assertFalse(r.summary["cost_responsive"])
        self.assertAlmostEqual(r.summary["cost_before"], r.summary["cost_after"], places=9)

    def test_slab_arithmetic(self):
        tele = SlabTariff(slabs=[{"upto_kwh": 100, "rate": 0.0}, {"upto_kwh": 200, "rate": 2.0},
                                 {"upto_kwh": None, "rate": 5.0}])
        self.assertAlmostEqual(tele.cycle_charge(250), 100 * 2.0 + 50 * 5.0)
        non = SlabTariff(slabs=tele.slabs, telescopic=False)
        self.assertAlmostEqual(non.cycle_charge(250), 250 * 5.0)

    def test_tou_reduces_cost_on_review3_style_case(self):
        r = optimise(synthetic_forecast(), illustrative_tou_tariff(), illustrative_appliances())
        self.assertTrue(r.summary["cost_responsive"])
        self.assertGreater(r.summary["cost_reduction_pct"], 0)

    def test_factory_and_official_guard(self):
        self.assertEqual(tariff_from_dict({"tariff_type": "flat", "rate": 5}).tariff_type, "flat")
        self.assertEqual(tariff_from_dict({"tariff_type": "tou", "hourly_rates": [1.0] * 24}).tariff_type, "tou")
        self.assertEqual(tariff_from_dict({"tariff_type": "slab", "slabs": [{"upto_kwh": None, "rate": 3}]}).tariff_type, "slab")
        with self.assertRaises(ValueError):
            tariff_from_dict({"tariff_type": "flat", "rate": 5, "is_official": True})
        with self.assertRaises(ValueError):
            TouTariff.from_periods([{"start_hour": 0, "end_hour": 12, "rate": 1}])  # gaps


class TieBreakAndMetrics(unittest.TestCase):
    def test_tie_break_removes_pointless_moves(self):
        fc = synthetic_forecast()
        apps = illustrative_appliances()
        on = optimise(fc, illustrative_tou_tariff(), apps, Household(lexicographic_tie_break=True))
        off = optimise(fc, illustrative_tou_tariff(), apps, Household(lexicographic_tie_break=False))
        self.assertAlmostEqual(on.summary["objective_value"], off.summary["objective_value"], places=6)
        self.assertLessEqual(on.summary["shifted_energy"], off.summary["shifted_energy"] + 1e-9)

    def test_metric_formulas(self):
        r = optimise(synthetic_forecast(), illustrative_tou_tariff(), illustrative_appliances())
        s, h = r.summary, r.hourly
        self.assertAlmostEqual(s["peak_before"], h.total_before.max())
        self.assertAlmostEqual(s["par_after"], h.total_after.max() / h.total_after.mean())
        self.assertAlmostEqual(s["cost_reduction_pct"], 100 * (s["cost_before"] - s["cost_after"]) / s["cost_before"])
        self.assertAlmostEqual(s["peak_reduction_pct"], 100 * (s["peak_before"] - s["peak_after"]) / s["peak_before"])
        self.assertAlmostEqual(s["shifted_energy"], r.appliance_schedule.shifted_energy_kwh.sum())

    def test_shifted_energy_not_netted(self):
        """Two appliances swapping hours: aggregate formula sees 0, per-appliance formula sees both moves."""
        apps = [Appliance("a", "A", 1.0, 1, 0, 24, preferred_start=10), Appliance("b", "B", 1.0, 1, 0, 24, preferred_start=11)]
        p = SchedulingProblem(synthetic_forecast("2024-01-01 00:00"), FlatTariff(rate=1), apps, Household())
        before = p.baseline_schedule()
        swapped = before[::-1].copy()
        self.assertAlmostEqual(p.shifted_energy_by_appliance(before, swapped).sum(), 2.0)
        self.assertAlmostEqual(0.5 * np.abs(p.appliance_load(swapped) - p.appliance_load(before)).sum(), 0.0)

    def test_output_contract_columns(self):
        r = optimise(synthetic_forecast(), illustrative_tou_tariff(), illustrative_appliances())
        for col in ["timestamp", "base_load", "appliance_load_before", "appliance_load_after", "total_before", "total_after"]:
            self.assertIn(col, r.hourly.columns)
        for key in ["cost_before", "cost_after", "cost_reduction_pct", "peak_before", "peak_after", "peak_reduction_pct",
                    "par_before", "par_after", "shifted_energy", "solver_status", "runtime"]:
            self.assertIn(key, r.summary)


class InputValidation(unittest.TestCase):
    def test_base_load_above_cap_reports_infeasible(self):
        fc = synthetic_forecast()
        fc.loc[5, "predicted_kwh"] = 10.0
        r = optimise(fc, FlatTariff(rate=5), illustrative_appliances())
        self.assertEqual(r.summary["solver_status"], "infeasible")

    def test_bad_forecasts_rejected(self):
        fc = synthetic_forecast()
        for broken in (fc.iloc[:23], fc.assign(predicted_kwh=-fc.predicted_kwh),
                       fc.assign(timestamp=fc.timestamp + pd.to_timedelta(np.arange(24) * 2, "h"))):
            with self.assertRaises(ValueError):
                SchedulingProblem(broken, FlatTariff(rate=1), illustrative_appliances(), Household())

    def test_research_long_schema_aliases_accepted(self):
        fc = synthetic_forecast().rename(columns={"timestamp": "target_timestamp", "predicted_kwh": "prediction"})
        self.assertEqual(optimise(fc, FlatTariff(rate=1), illustrative_appliances()).summary["solver_status"], "optimal")


class Review3Reproduction(unittest.TestCase):
    @unittest.skipUnless(EXAMPLE.exists(), "committed Review-3 example forecast not found")
    def test_same_cost_result_fewer_pointless_moves(self):
        legacy = load_appliances(CONFIGS / "appliances_review3_legacy.json")
        r = optimise(pd.read_csv(EXAMPLE), illustrative_tou_tariff(), legacy, Household())
        self.assertAlmostEqual(r.summary["cost_before"], 97.64381411, places=6)
        self.assertAlmostEqual(r.summary["cost_after"], 88.64381411, places=6)
        self.assertAlmostEqual(r.summary["shifted_energy"], 2.0, places=9)  # Review-3 reported 4.6 (tie moves)


class HeuristicInterface(unittest.TestCase):
    def test_benchmark_uses_same_scoring(self):
        p = SchedulingProblem(synthetic_forecast(), illustrative_tou_tariff(), illustrative_appliances(), Household())
        milp_eval, base_eval = benchmark(p, [MILPScheduler(), BaselineScheduler()])
        self.assertTrue(milp_eval.feasible and base_eval.feasible)
        self.assertLessEqual(milp_eval.objective, base_eval.objective + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
