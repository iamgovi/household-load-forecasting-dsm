"""Add a second untuned-hybrid seed for a modest seed-noise comparison."""
import json
import pandas as pd

from evaluation.contract import REPORT_DIR, TRAIN_PATH, VALIDATION_PATH
from evaluation.metrics import evaluate_predictions
from evaluation.origins import load_research_splits
from review3.arima_gru_hybrid import build_training_residuals_and_windows, build_validation_arrays
from review3.arima_gru_pso import HYBRID, PREDICTIONS, TABLES, train_final


def main():
    splits=load_research_splits({"train":TRAIN_PATH,"validation":VALIDATION_PATH})
    origins=pd.read_csv(REPORT_DIR/"validation_origins.csv",parse_dates=["forecast_origin","first_target_timestamp","last_target_timestamp"])
    residuals,_,X,y,_,models=build_training_residuals_and_windows(splits["train"],sorted(origins.meter.unique()))
    arrays,metadata,_=build_validation_arrays(splits["train"],splits["validation"],origins,residuals,models)
    Xv,yv,base,actual=arrays; config={"hidden_units":32,"learning_rate":.001,"dropout":0.,"lookback":168,"batch_size":128}
    model,pred,history,seconds,inference,scaler=train_final(config,X,y,Xv,yv,base,actual,metadata,123)
    pred["model"]="ARIMA-GRU-Untuned-Seed123"; pred.to_csv(PREDICTIONS/"untuned_arima_gru_seed123_predictions.csv",index=False)
    metrics=evaluate_predictions(pred)["overall"].query("aggregation=='pooled_predictions'").iloc[0]
    frozen=pd.read_csv(HYBRID/"tables"/"hybrid_model_comparison.csv").query("Model=='ARIMA-GRU'").iloc[0]
    rows=[{"seed":42,"MAE":frozen.MAE,"RMSE":frozen.RMSE,"source":"frozen Milestone 4","training_time_seconds":frozen.Train_Time_s},
          {"seed":123,"MAE":metrics.MAE,"RMSE":metrics.RMSE,"source":"independent reproduction","training_time_seconds":seconds}]
    pd.DataFrame(rows).to_csv(TABLES/"untuned_seed_variability.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
