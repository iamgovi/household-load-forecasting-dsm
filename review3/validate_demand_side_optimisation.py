"""Validation checks for the simulation-based demand-side MILP."""
from __future__ import annotations
import json,unittest
import numpy as np
import pandas as pd
from review3.demand_side_optimisation import (APPLIANCES,CONFIGS,FORECAST_MODEL,HOUSEHOLD_MAX_POWER_KW,
 OUTPUT,RUNS,SCHEDULES,TABLES,baseline_schedule,calculate_metrics,load_forecast,optimize_schedule,profile,tariff_table)

class DSOValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arr=np.load(RUNS/"schedule_arrays.npz"); cls.before=cls.arr["before"]; cls.after=cls.arr["after"]
        cls.base=cls.arr["base"]; cls.tariff=cls.arr["tariff"]
        cls.summary=pd.read_csv(TABLES/"optimisation_summary.csv").set_index("Metric")
        cls.manifest=json.loads((RUNS/"optimisation_manifest.json").read_text())

    def test_solver_is_optimal(self):
        timestamps=pd.DatetimeIndex(load_forecast().target_timestamp)
        schedule,result,_=optimize_schedule(self.base,self.tariff,timestamps); self.assertTrue(result.success); np.testing.assert_array_equal(schedule,self.after)

    def test_duration_windows_deadlines_and_horizon(self):
        self.assertEqual(self.after.shape,(len(APPLIANCES),24))
        timestamps=pd.DatetimeIndex(load_forecast().target_timestamp)
        for i,a in enumerate(APPLIANCES):
            hours=np.flatnonzero(self.after[i]); self.assertEqual(len(hours),a["duration_h"])
            clock=timestamps[hours].hour
            self.assertGreaterEqual(clock.min(),a["earliest_start"]); self.assertLess(clock.max(),a["latest_finish"])
            self.assertTrue(np.all((hours>=0)&(hours<24)))

    def test_non_interruptible_contiguous(self):
        for i,a in enumerate(APPLIANCES):
            if not a["interruptible"]:
                hours=np.flatnonzero(self.after[i]); np.testing.assert_array_equal(np.diff(hours),np.ones(max(len(hours)-1,0),dtype=int))

    def test_peak_constraint(self):
        total=self.base+profile(self.after); self.assertLessEqual(total.max(),HOUSEHOLD_MAX_POWER_KW+1e-9)

    def test_flexible_energy_conserved(self):
        self.assertAlmostEqual(profile(self.before).sum(),profile(self.after).sum(),places=10)
        self.assertAlmostEqual(self.summary.loc["Flexible Energy","Before"],self.summary.loc["Flexible Energy","After"],places=10)

    def test_cost_peak_par_reproduce(self):
        bm,_,bt=calculate_metrics(self.base,self.before,self.tariff); am,_,at=calculate_metrics(self.base,self.after,self.tariff)
        self.assertAlmostEqual(bm["electricity_cost_inr"],np.sum(bt*self.tariff),places=10)
        self.assertAlmostEqual(am["electricity_cost_inr"],np.sum(at*self.tariff),places=10)
        self.assertAlmostEqual(bm["peak_load_kw_equivalent"],bt.max(),places=10); self.assertAlmostEqual(am["peak_load_kw_equivalent"],at.max(),places=10)
        self.assertAlmostEqual(bm["PAR"],bt.max()/bt.mean(),places=10); self.assertAlmostEqual(am["PAR"],at.max()/at.mean(),places=10)
        for label,key in (("Electricity Cost","electricity_cost_inr"),("Peak Load","peak_load_kw_equivalent"),("Peak-to-Average Ratio","PAR")):
            self.assertAlmostEqual(self.summary.loc[label,"Before"],bm[key],places=9); self.assertAlmostEqual(self.summary.loc[label,"After"],am[key],places=9)

    def test_shifted_energy_reproduces(self):
        expected=.5*np.abs(profile(self.after)-profile(self.before)).sum(); self.assertAlmostEqual(self.summary.loc["Shifted Energy","After"],expected,places=10)

    def test_tariff_and_forecast_complete(self):
        forecast=load_forecast(); self.assertEqual(len(tariff_table(forecast.target_timestamp)),24); self.assertEqual(len(self.tariff),24); self.assertTrue((self.tariff>=0).all())
        self.assertEqual(len(forecast),24); self.assertEqual(forecast.horizon.tolist(),list(range(1,25)))
        generated=tariff_table(forecast.target_timestamp); self.assertEqual(generated.clock_hour.tolist(),pd.DatetimeIndex(forecast.target_timestamp).hour.tolist())

    def test_appliance_inputs_nonnegative(self):
        for a in APPLIANCES:
            self.assertGreater(a["power_kw"],0); self.assertGreater(a["duration_h"],0); self.assertLessEqual(a["latest_finish"],24)
            self.assertEqual(a["source_class"],"ILLUSTRATIVE / SIMULATED")

    def test_no_unsupported_sustainability_or_test_selection_claim(self):
        self.assertFalse(self.manifest["renewable_metrics_included"]); self.assertFalse(self.manifest["carbon_metrics_included"])
        self.assertFalse(self.manifest["test_data_used_for_model_selection"])
        report=(OUTPUT/"DEMAND_SIDE_OPTIMISATION_REVIEW3.md").read_text().lower()
        self.assertIn("simulation-based demand-side optimisation",report); self.assertIn("does not demonstrate reduced energy consumption",report)

if __name__=="__main__": unittest.main(verbosity=2)
