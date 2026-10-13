"""Automated scientific-integrity checks for the Review-3 residual hybrid."""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from evaluation.metrics import point_metrics, validate_prediction_schema
from review3.arima_gru_hybrid import CHECKPOINTS, METRICS, OUTPUT, PREDICTIONS, RUNS, TABLES


class HybridValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred=pd.read_csv(PREDICTIONS/"arima_gru_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
        cls.residual=pd.read_csv(OUTPUT/"residuals"/"train_oos_arima_residuals.csv",parse_dates=["timestamp","fit_end_timestamp"])
        cls.manifest=json.loads((RUNS/"arima_gru_manifest.json").read_text())

    def test_training_residuals_are_strictly_oos(self):
        self.assertTrue(self.residual.out_of_sample.all())
        self.assertTrue((self.residual.fit_end_timestamp<self.residual.timestamp).all())
        self.assertTrue((self.residual.training_rows_before_prediction>=168).all())
        np.testing.assert_allclose(self.residual.residual,self.residual.actual-self.residual.arima_prediction,rtol=1e-10,atol=1e-10)

    def test_training_windows_are_temporally_sealed(self):
        w=pd.read_csv(TABLES/"residual_training_window_index.csv",parse_dates=["forecast_origin","residual_input_start","residual_input_end","first_target","last_target","arima_fit_end"])
        self.assertTrue((w.residual_input_end==w.forecast_origin).all())
        self.assertTrue((w.arima_fit_end<w.first_target).all())
        self.assertTrue((w.last_target-w.first_target==pd.Timedelta(hours=23)).all())
        self.assertTrue(w.targets_excluded_from_arima_fit.all())

    def test_exact_24h_and_canonical_origins(self):
        validate_prediction_schema(self.pred)
        canonical=pd.read_csv("reports/evaluation/validation_origins.csv",parse_dates=["forecast_origin"])
        have=set(map(tuple,self.pred[["meter","forecast_origin"]].drop_duplicates().to_records(index=False)))
        allowed=set(map(tuple,canonical[["meter","forecast_origin"]].to_records(index=False)))
        self.assertTrue(have.issubset(allowed)); self.assertEqual(len(have),1794)

    def test_decomposition_identity(self):
        np.testing.assert_allclose(self.pred.prediction,self.pred.arima_prediction+self.pred.residual_correction,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(self.pred.prediction,self.pred.hybrid_prediction,rtol=1e-8,atol=1e-8)

    def test_common_comparison_is_equal_sample(self):
        c=pd.read_csv(TABLES/"hybrid_model_comparison.csv")
        self.assertEqual(c.Forecast_Origins.nunique(),1); self.assertEqual(c.Forecast_Origins.iloc[0],1794)
        self.assertEqual(c.Meters.nunique(),1); self.assertEqual(c.Meters.iloc[0],58)

    def test_metrics_reproduce(self):
        saved=pd.read_csv(METRICS/"arima_gru_overall.csv").query("aggregation=='pooled_predictions'").iloc[0]
        fresh=point_metrics(self.pred.actual,self.pred.prediction)
        for metric in ("MAE","RMSE","sMAPE_pct","R2","NRMSE_mean"):
            # CSV round-trip introduces sub-nanounit floating-point changes.
            self.assertAlmostEqual(float(saved[metric]),float(fresh[metric]),delta=1e-6)

    def test_train_only_scaling_validation_only_and_no_attention_pso(self):
        self.assertEqual(self.manifest["scaler"]["fit_split"],"train")
        self.assertFalse(self.manifest["test_data_loaded"]); self.assertFalse(self.manifest["test_predictions_generated"])
        self.assertFalse(self.manifest["attention"]); self.assertFalse(self.manifest["pso"])
        self.assertIn("unidirectional",self.manifest["architecture"])
        checkpoint=__import__("torch").load(CHECKPOINTS/"arima_gru_validation_checkpoint.pt",weights_only=False)
        self.assertEqual(checkpoint["scaler"]["fit_split"],"train")

    def test_no_interpolation_or_future_fill_in_research_source(self):
        source=(OUTPUT.parents[2]/"review3"/"arima_gru_hybrid.py").read_text().lower()
        for forbidden in (".interpolate(",".ffill(",".bfill(","attention(","pso("):
            self.assertNotIn(forbidden,source)
        self.assertNotIn("test_path",source)


if __name__=="__main__": unittest.main(verbosity=2)
