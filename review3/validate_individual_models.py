"""Integrity checks for Review-3 individual-model evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from evaluation.comparison import common_origin_predictions
from evaluation.metrics import evaluate_predictions, validate_prediction_schema
from review3.individual_models import FIGURES, METRICS, MODEL_ORDER, PREDICTIONS, RUNS, TABLES


class Review3Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames=[]; cls.manifests={}
        for model in MODEL_ORDER:
            frame=pd.read_csv(PREDICTIONS/f"{model.lower()}_predictions.csv",
                              parse_dates=["forecast_origin","target_timestamp"])
            validate_prediction_schema(frame); cls.frames.append(frame)
            cls.manifests[model]=json.loads((RUNS/f"{model.lower()}_manifest.json").read_text())
        cls.all=pd.concat(cls.frames,ignore_index=True)

    def test_only_corrected_research_data_and_validation(self):
        for manifest in self.manifests.values():
            self.assertIn("/data/research/train.csv",manifest["train_data"])
            self.assertFalse(manifest["test_predictions_generated"])
            self.assertEqual(manifest["split"],"validation")
        self.assertEqual(set(self.all.split),{"validation"})

    def test_every_model_has_same_complete_24h_origins(self):
        common,availability=common_origin_predictions(self.all,MODEL_ORDER)
        self.assertEqual(len(common),len(self.all))
        self.assertTrue(availability.complete_forecast.all())
        populations=[]
        for model,frame in self.all.groupby("model"):
            populations.append(set(zip(frame.meter,frame.forecast_origin)))
            self.assertEqual(set(frame.horizon),set(range(1,25)))
        self.assertTrue(all(population==populations[0] for population in populations[1:]))

    def test_no_hybrid_or_deployment_model_in_table(self):
        table=pd.read_csv(TABLES/"individual_model_comparison.csv")
        self.assertEqual(set(table.Model),set(MODEL_ORDER))
        forbidden=("ARIMA-GRU","Attention","PSO","general_gru_model")
        self.assertFalse(any(any(term in model for term in forbidden) for model in table.Model))

    def test_comparison_values_regenerate_from_predictions(self):
        regenerated=evaluate_predictions(self.all)["overall"].query("aggregation == 'pooled_predictions'")
        saved=pd.read_csv(TABLES/"individual_model_comparison.csv")
        for row in saved.itertuples(index=False):
            source=regenerated[regenerated.model.eq(row.Model)].iloc[0]
            for metric in ("MAE","RMSE","R2","NRMSE_mean"):
                self.assertAlmostEqual(getattr(row,metric),source[metric],places=10)

    def test_gru_contract(self):
        config=self.manifests["GRU"]["configuration"]
        self.assertEqual(config["scaler_fit"],"train only")
        self.assertIn("unidirectional",config["architecture"])
        history=pd.read_csv(RUNS/"gru_training_history.csv")
        self.assertGreater(len(history),0); self.assertLessEqual(len(history),config["max_epochs"])
        self.assertTrue((RUNS.parent/"checkpoints"/"gru_validation_checkpoint.pt").exists())

    def test_figures_are_nonempty_and_model_profiles_exist(self):
        for model in MODEL_ORDER:
            path=FIGURES/f"{model.lower()}_actual_vs_predicted.png"
            self.assertTrue(path.exists() and path.stat().st_size>1000)
        self.assertGreaterEqual(len(list(FIGURES.glob("*.png"))),17)

    def test_research_runner_contains_no_fill_or_interpolation(self):
        source=Path(__file__).with_name("individual_models.py").read_text().lower()
        for forbidden in (".interpolate(",".ffill(",".bfill("):
            self.assertNotIn(forbidden,source)


if __name__=="__main__": unittest.main(verbosity=2)
