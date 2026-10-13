"""Automated integrity checks for the temporal-attention ablation."""
from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import point_metrics, validate_prediction_schema
from review3.arima_gru_attention import BASELINE, CHECKPOINTS, METRICS, OUTPUT, PREDICTIONS, RUNS, TABLES, WEIGHTS, TemporalAttentionResidualGRU


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


class AttentionValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred=pd.read_csv(PREDICTIONS/"arima_gru_attention_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
        cls.manifest=json.loads((RUNS/"arima_gru_attention_manifest.json").read_text())

    def test_temporal_attention_architecture_only(self):
        model=TemporalAttentionResidualGRU(); x=torch.zeros(3,168,1)
        prediction,weights=model(x,return_attention=True)
        self.assertEqual(tuple(prediction.shape),(3,24)); self.assertEqual(tuple(weights.shape),(3,168))
        torch.testing.assert_close(weights.sum(1),torch.ones(3))
        self.assertFalse(model.gru.bidirectional); self.assertEqual(self.manifest["attention_type"],"temporal")
        self.assertFalse(self.manifest["pso"])

    def test_residual_pipeline_unchanged(self):
        self.assertTrue(self.manifest["residuals_match_frozen_baseline"])
        self.assertEqual(self.manifest["training_residual_rows"],286079)
        self.assertEqual(self.manifest["training_windows"],8289)

    def test_no_future_training_inputs(self):
        w=pd.read_csv(BASELINE/"tables"/"residual_training_window_index.csv",parse_dates=["residual_input_end","first_target","arima_fit_end"])
        self.assertTrue((w.residual_input_end<w.first_target).all()); self.assertTrue((w.arima_fit_end<w.first_target).all())

    def test_train_scaler_validation_selection_and_no_test(self):
        self.assertEqual(self.manifest["scaler"]["fit_split"],"train")
        self.assertFalse(self.manifest["test_data_loaded"]); self.assertFalse(self.manifest["test_predictions_generated"])
        self.assertGreaterEqual(self.manifest["selected_epoch"],1)

    def test_prediction_contract_and_same_origins(self):
        validate_prediction_schema(self.pred)
        baseline=pd.read_csv(BASELINE/"predictions"/"arima_gru_predictions.csv",parse_dates=["forecast_origin"])
        key=["city","meter","forecast_origin"]
        a=set(map(tuple,self.pred[key].drop_duplicates().to_records(index=False)))
        b=set(map(tuple,baseline[key].drop_duplicates().to_records(index=False)))
        self.assertEqual(a,b); self.assertEqual(len(a),1794)

    def test_attention_weights_align_and_normalize(self):
        weights=np.load(WEIGHTS/"validation_attention_weights.npz")["weights"]
        self.assertEqual(weights.shape,(1794,168)); np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
        representative=pd.read_csv(WEIGHTS/"representative_attention_weights.csv")
        self.assertEqual(len(representative),168); self.assertAlmostEqual(representative.attention_weight.sum(),1,places=6)

    def test_decomposition_and_metrics_reproduce(self):
        np.testing.assert_allclose(self.pred.prediction,self.pred.arima_prediction+self.pred.residual_correction,rtol=1e-6,atol=1e-7)
        saved=pd.read_csv(METRICS/"arima_gru_attention_overall.csv").query("aggregation=='pooled_predictions'").iloc[0]
        fresh=point_metrics(self.pred.actual,self.pred.prediction)
        for metric in ("MAE","RMSE","sMAPE_pct","R2","NRMSE_mean"): self.assertAlmostEqual(saved[metric],fresh[metric],delta=1e-6)

    def test_frozen_baseline_hashes_unchanged(self):
        self.assertTrue(self.manifest["baseline_outputs_untouched"])
        for path,expected in self.manifest["baseline_hashes_before"].items(): self.assertEqual(sha256(path),expected)

    def test_research_source_has_no_fill_test_or_pso(self):
        source=(OUTPUT.parents[2]/"review3"/"arima_gru_attention.py").read_text().lower()
        for forbidden in (".interpolate(",".ffill(",".bfill(","test_path","pso(","bidirectional=true"):
            self.assertNotIn(forbidden,source)


if __name__=="__main__": unittest.main(verbosity=2)
