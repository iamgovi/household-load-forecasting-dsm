"""Scientific-integrity checks for the PSO residual-GRU experiment."""
from __future__ import annotations
import hashlib,json,unittest
import numpy as np
import pandas as pd
import torch
from evaluation.metrics import point_metrics,validate_prediction_schema
from review3.arima_gru_pso import CHECKPOINTS,HYBRID,METRICS,OUTPUT,PREDICTIONS,RUNS,SEARCH,TABLES,TunedResidualGRU,decode

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

class PSOValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((RUNS/"pso_manifest.json").read_text())
        cls.pred=pd.read_csv(PREDICTIONS/"pso_arima_gru_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])

    def test_no_test_and_inner_validation_fitness(self):
        self.assertFalse(self.manifest["test_data_loaded"]); self.assertFalse(self.manifest["test_predictions_generated"])
        self.assertEqual(self.manifest["pso_fitness"],"inner-validation RMSE")
        split=pd.read_csv(SEARCH/"inner_chronological_split.csv",parse_dates=["forecast_origin"])
        for _,g in split.groupby("meter"):
            a=g[g.inner_partition.eq("inner_train")]; b=g[g.inner_partition.eq("inner_validation")]
            if len(a) and len(b): self.assertLess(a.forecast_origin.max(),b.forecast_origin.min())

    def test_residual_pipeline_and_no_future_inputs(self):
        self.assertTrue(self.manifest["residual_pipeline_matches_milestone4"])
        w=pd.read_csv(HYBRID/"tables"/"residual_training_window_index.csv",parse_dates=["residual_input_end","first_target","arima_fit_end"])
        self.assertTrue((w.residual_input_end<w.first_target).all()); self.assertTrue((w.arima_fit_end<w.first_target).all())

    def test_plain_no_attention_model(self):
        model=TunedResidualGRU(64,.3); self.assertFalse(model.gru.bidirectional)
        self.assertFalse(any("attention" in name.lower() for name,_ in model.named_modules()))
        self.assertIn("no attention",self.manifest["architecture"])

    def test_train_only_scalers(self):
        self.assertIn("inner-train",self.manifest["scaler_fit"])
        for seed in self.manifest["final_seeds"]:
            ck=torch.load(CHECKPOINTS/f"pso_arima_gru_seed{seed}_checkpoint.pt",map_location="cpu",weights_only=False)
            self.assertEqual(ck["scaler"]["fit_split"],"train")

    def test_particle_log_complete_and_selected_is_best(self):
        h=pd.read_csv(SEARCH/"pso_search_history.csv"); self.assertEqual(len(h),25)
        self.assertTrue(h.status.isin(["success","cached"]).all()); self.assertTrue(h.fitness_RMSE.notna().all())
        required=set(self.manifest["search_space"])|{"iteration","particle_id","seed","epochs_trained","fitness_RMSE","runtime_seconds"}
        self.assertTrue(required.issubset(h.columns))
        best=h.loc[h.fitness_RMSE.idxmin()]
        for name,value in self.manifest["selected_configuration"].items(): self.assertEqual(best[name],value)

    def test_convergence_monotonic(self):
        c=pd.read_csv(SEARCH/"pso_convergence.csv"); self.assertTrue((c.global_best_fitness.diff().dropna()<=1e-12).all())
        self.assertAlmostEqual(c.global_best_fitness.iloc[-1],self.manifest["best_inner_validation_RMSE"],places=12)

    def test_prediction_contract_common_origins_and_decomposition(self):
        validate_prediction_schema(self.pred)
        baseline=pd.read_csv(HYBRID/"predictions"/"arima_gru_predictions.csv",parse_dates=["forecast_origin"])
        key=["city","meter","forecast_origin"]
        self.assertEqual(set(map(tuple,self.pred[key].drop_duplicates().to_records(index=False))),set(map(tuple,baseline[key].drop_duplicates().to_records(index=False))))
        np.testing.assert_allclose(self.pred.prediction,self.pred.arima_prediction+self.pred.residual_correction,rtol=1e-6,atol=1e-7)

    def test_metrics_reproduce(self):
        saved=pd.read_csv(METRICS/"pso_arima_gru_overall.csv").query("aggregation=='pooled_predictions'").iloc[0]; fresh=point_metrics(self.pred.actual,self.pred.prediction)
        for metric in ("MAE","RMSE","sMAPE_pct","R2","NRMSE_mean"): self.assertAlmostEqual(saved[metric],fresh[metric],delta=1e-6)

    def test_seeds_recorded_and_baselines_untouched(self):
        self.assertEqual(self.manifest["final_seeds"],[42,123,2026]); self.assertEqual(self.manifest["pso_seed"],42)
        self.assertTrue(self.manifest["baseline_outputs_untouched"])
        for path,expected in self.manifest["baseline_hashes_before"].items(): self.assertEqual(sha256(path),expected)

if __name__=="__main__": unittest.main(verbosity=2)
