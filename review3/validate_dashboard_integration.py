"""Programmatic checks for the Review-3 dashboard integration layer."""
from __future__ import annotations
import io,unittest
import numpy as np
import pandas as pd
from review3.dashboard_integration import (MODEL_MANIFEST,MODEL_PATH,build_dashboard_forecaster,
 forecast_uploaded_history,load_validated_dashboard_example,run_validated_optimisation,validate_uploaded_history)
from review3.demand_side_optimisation import OUTPUT,TABLES

class DashboardIntegrationValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecast,cls.meta=load_validated_dashboard_example(); cls.result=run_validated_optimisation(cls.forecast)
        cls.summary=pd.read_csv(TABLES/"optimisation_summary.csv").set_index("Metric")
        cls.source=(OUTPUT.parents[2]/"dashboard.py").read_text()

    def test_validated_example_and_24_hours(self):
        self.assertEqual(len(self.forecast),24); self.assertEqual(self.meta["horizon_hours"],24); self.assertEqual(self.meta["model"],"LinearRegression")

    def test_milp_optimal_and_metrics_match_frozen_dso(self):
        m=self.result["metrics"]; self.assertEqual(m["solver_status"],"Optimal")
        mapping={"cost_before":("Electricity Cost","Before"),"cost_after":("Electricity Cost","After"),"peak_before":("Peak Load","Before"),
                 "peak_after":("Peak Load","After"),"par_before":("Peak-to-Average Ratio","Before"),"par_after":("Peak-to-Average Ratio","After"),
                 "shifted_energy_kwh":("Shifted Energy","After"),"flexible_before_kwh":("Flexible Energy","Before"),"flexible_after_kwh":("Flexible Energy","After")}
        for key,(row,col) in mapping.items(): self.assertAlmostEqual(m[key],self.summary.loc[row,col],places=8)

    def test_energy_conservation(self):
        m=self.result["metrics"]; self.assertAlmostEqual(m["flexible_before_kwh"],m["flexible_after_kwh"],places=10); self.assertAlmostEqual(m["flexible_difference_kwh"],0,places=10)

    def test_only_validated_appliances(self):
        self.assertEqual(set(self.result["appliance_schedule"].Appliance),{"Washing Machine","Dishwasher","Water Heater","Water Pump"})

    def test_legacy_default_logic_removed(self):
        forbidden=["forecast_n_days","FORECAST_DAYS","general_gru_model","solar self-consumption","EV Charger","comfort-bounded","30-Day Forecast","schedule_appliance"]
        for text in forbidden: self.assertNotIn(text,self.source)
        self.assertIn('index=0',self.source); self.assertIn("Review-3 validated example",self.source)

    def test_no_unsupported_claims(self):
        lower=self.source.lower()
        for text in ("carbon reduction","renewable utilisation","energy saved"): self.assertNotIn(text,lower)

    def test_valid_upload_and_latest_segment_selection(self):
        timestamps=pd.date_range("2020-01-01",periods=169,freq="h"); frame=pd.DataFrame({"timestamp":timestamps,"kwh":np.ones(169)})
        valid=validate_uploaded_history(io.StringIO(frame.to_csv(index=False))); self.assertEqual(len(valid),169)
        earlier=pd.DataFrame({"timestamp":pd.date_range("2019-12-01",periods=50,freq="h"),"kwh":1.})
        gap=pd.concat([earlier,frame],ignore_index=True)
        selected=validate_uploaded_history(io.StringIO(gap.to_csv(index=False)))
        self.assertEqual(len(selected),169); self.assertEqual(selected.attrs["gap_count"],1); self.assertEqual(selected.attrs["discarded_rows"],50)

    def test_upload_duplicate_and_short_rejection(self):
        timestamps=pd.date_range("2020-01-01",periods=168,freq="h"); frame=pd.DataFrame({"timestamp":timestamps,"kwh":1.})
        duplicate=frame.copy(); duplicate.loc[10,"timestamp"]=duplicate.loc[9,"timestamp"]
        with self.assertRaisesRegex(ValueError,"Duplicate"): validate_uploaded_history(io.StringIO(duplicate.to_csv(index=False)))
        with self.assertRaisesRegex(ValueError,"168"): validate_uploaded_history(io.StringIO(frame.head(100).to_csv(index=False)))

    def test_gapped_upload_without_168_hour_segment_rejected(self):
        a=pd.date_range("2020-01-01",periods=100,freq="h"); b=pd.date_range("2020-02-01",periods=100,freq="h")
        frame=pd.DataFrame({"timestamp":a.append(b),"kwh":1.})
        with self.assertRaisesRegex(ValueError,"longest valid hourly block has 100"):
            validate_uploaded_history(io.StringIO(frame.to_csv(index=False)))

    def test_uploaded_history_forecast_and_milp(self):
        build_dashboard_forecaster()
        self.assertTrue(MODEL_PATH.exists()); self.assertTrue(MODEL_MANIFEST.exists())
        history=validate_uploaded_history(OUTPUT.parents[2]/"dashboard_BR02.csv")
        self.assertEqual(len(history),949); self.assertEqual(history.attrs["gap_count"],2)
        forecast,meta=forecast_uploaded_history(history)
        self.assertEqual(len(forecast),24); self.assertTrue(np.isfinite(forecast.predicted_kwh).all())
        self.assertTrue((forecast.predicted_kwh>=0).all()); self.assertEqual(meta["horizon_hours"],24)
        result=run_validated_optimisation(forecast)
        self.assertEqual(result["metrics"]["solver_status"],"Optimal")
        self.assertAlmostEqual(result["metrics"]["flexible_difference_kwh"],0,places=10)

if __name__=="__main__": unittest.main(verbosity=2)
