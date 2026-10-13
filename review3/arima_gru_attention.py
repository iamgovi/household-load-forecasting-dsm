"""Temporal-attention ablation of the frozen Review-3 ARIMA-GRU hybrid."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.contract import HORIZON, LOOKBACK, PREDICTION_COLUMNS, REPORT_DIR, SEED, TRAIN_PATH, VALIDATION_PATH
from evaluation.metrics import evaluate_predictions, validate_prediction_schema
from evaluation.origins import load_research_splits
from review3.arima_gru_hybrid import build_training_residuals_and_windows, build_validation_arrays

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"reports"/"review3"/"arima_gru_attention"
PREDICTIONS,TABLES,FIGURES=OUTPUT/"predictions",OUTPUT/"tables",OUTPUT/"figures"
METRICS,RUNS,CHECKPOINTS,WEIGHTS=OUTPUT/"metrics",OUTPUT/"runs",OUTPUT/"checkpoints",OUTPUT/"attention_weights"
BASELINE=ROOT/"reports"/"review3"/"arima_gru_hybrid"


def prepare_dirs():
    for p in (PREDICTIONS,TABLES,FIGURES,METRICS,RUNS,CHECKPOINTS,WEIGHTS): p.mkdir(parents=True,exist_ok=True)


def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


class TemporalAttentionResidualGRU(nn.Module):
    """Additive temporal attention over the 168 GRU output states."""
    def __init__(self,hidden_size=32):
        super().__init__()
        self.gru=nn.GRU(1,hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.attention_projection=nn.Linear(hidden_size,hidden_size)
        self.attention_score=nn.Linear(hidden_size,1,bias=False)
        self.output=nn.Linear(hidden_size,HORIZON)

    def forward(self,x,return_attention=False):
        states,_=self.gru(x)
        scores=self.attention_score(torch.tanh(self.attention_projection(states))).squeeze(-1)
        weights=torch.softmax(scores,dim=1)
        context=torch.sum(states*weights.unsqueeze(-1),dim=1)
        prediction=self.output(context)
        return (prediction,weights) if return_attention else prediction


def train_model(X_train,y_train,X_val,y_val):
    x_mean,x_std=float(X_train.mean()),float(X_train.std()) or 1.0
    y_mean,y_std=float(y_train.mean()),float(y_train.std()) or 1.0
    Xs=(X_train-x_mean)/x_std; ys=(y_train-y_mean)/y_std
    Xv=(X_val-x_mean)/x_std; yv=(y_val-y_mean)/y_std
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(TensorDataset(torch.tensor(Xs),torch.tensor(ys)),batch_size=128,
                      shuffle=True,generator=generator)
    val_x,val_y=torch.tensor(Xv),torch.tensor(yv)
    model=TemporalAttentionResidualGRU(32); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.MSELoss()
    history=[]; best_loss=np.inf; best_state=None; stale=0; started=time.perf_counter()
    for epoch in range(1,21):
        model.train(); losses=[]
        for xb,yb in loader:
            optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); losses.append(loss.detach().item())
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(val_x),val_y).item()
        history.append({"epoch":epoch,"train_residual_MSE_scaled":float(np.mean(losses)),"validation_residual_MSE_scaled":val_loss})
        if val_loss<best_loss-1e-6:
            best_loss=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=4: break
    train_seconds=time.perf_counter()-started; model.load_state_dict(best_state); model.eval()
    started=time.perf_counter()
    with torch.no_grad(): scaled,attention=model(val_x,return_attention=True)
    inference_ms=(time.perf_counter()-started)*1000/max(len(X_val),1)
    correction=scaled.numpy()*y_std+y_mean
    scaler={"input_residual_mean":x_mean,"input_residual_std":x_std,
            "target_correction_mean":y_mean,"target_correction_std":y_std,"fit_split":"train"}
    return model,correction,attention.numpy(),train_seconds,inference_ms,history,scaler


def records(metadata,actual,base,correction,inference_ms,run_id):
    rows=[]
    for origin,truth,arima,residual in zip(metadata,actual,base,correction):
        targets=pd.date_range(origin.first_target_timestamp,periods=HORIZON,freq="h")
        for h,(ts,a,b,r) in enumerate(zip(targets,truth,arima,residual),1):
            prediction=b+r
            rows.append({"model":"ARIMA-GRU-Attention","city":origin.city,"meter":origin.meter,
              "split":"validation","forecast_origin":origin.forecast_origin,"horizon":h,"target_timestamp":ts,
              "actual":a,"prediction":prediction,"inference_time_ms":inference_ms,"run_id":run_id,
              "arima_prediction":b,"residual_correction":r,"hybrid_prediction":prediction})
    return pd.DataFrame(rows)


def main():
    prepare_dirs(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True,warn_only=True)
    protected=[BASELINE/"predictions"/"arima_gru_predictions.csv",BASELINE/"residuals"/"train_oos_arima_residuals.csv",
               BASELINE/"checkpoints"/"arima_gru_validation_checkpoint.pt",BASELINE/"runs"/"arima_gru_manifest.json"]
    hashes_before={str(p):sha256(p) for p in protected}
    splits=load_research_splits({"train":TRAIN_PATH,"validation":VALIDATION_PATH})
    origins=pd.read_csv(REPORT_DIR/"validation_origins.csv",parse_dates=["forecast_origin","first_target_timestamp","last_target_timestamp"])
    meters=sorted(origins.meter.unique()); generation_started=time.perf_counter()
    residuals,coverage,X_train,y_train,_,models=build_training_residuals_and_windows(splits["train"],meters)
    generation_seconds=time.perf_counter()-generation_started
    frozen=pd.read_csv(BASELINE/"residuals"/"train_oos_arima_residuals.csv",parse_dates=["timestamp","fit_end_timestamp"])
    check_cols=["city","meter","timestamp","actual","arima_prediction","residual","fit_end_timestamp","training_rows_before_prediction","out_of_sample"]
    pd.testing.assert_frame_equal(residuals[check_cols].reset_index(drop=True),frozen[check_cols].reset_index(drop=True),check_exact=False,rtol=1e-10,atol=1e-10)
    arrays,metadata,failures=build_validation_arrays(splits["train"],splits["validation"],origins,residuals,models)
    X_val,y_val,base,actual=arrays
    model,correction,attention,train_seconds,inference_ms,history,scaler=train_model(X_train,y_train,X_val,y_val)
    run_id=f"ARIMA-GRU-Attention_validation_{int(time.time())}"
    pred=records(metadata,actual,base,correction,inference_ms,run_id)
    validate_prediction_schema(pred[PREDICTION_COLUMNS]); pred.to_csv(PREDICTIONS/"arima_gru_attention_predictions.csv",index=False)
    failures.to_csv(RUNS/"arima_gru_attention_failures.csv",index=False)
    pd.DataFrame(history).to_csv(RUNS/"arima_gru_attention_training_history.csv",index=False)
    for level,table in evaluate_predictions(pred[PREDICTION_COLUMNS]).items(): table.to_csv(METRICS/f"arima_gru_attention_{level}.csv",index=False)
    np.savez_compressed(WEIGHTS/"validation_attention_weights.npz",weights=attention)
    first=metadata[0]; times=pd.date_range(end=first.forecast_origin,periods=LOOKBACK,freq="h")
    pd.DataFrame({"city":first.city,"meter":first.meter,"forecast_origin":first.forecast_origin,
                  "history_timestamp":times,"lag_hours":np.arange(LOOKBACK-1,-1,-1),"attention_weight":attention[0]}).to_csv(
                      WEIGHTS/"representative_attention_weights.csv",index=False)
    pd.DataFrame({"lag_hours":np.arange(LOOKBACK-1,-1,-1),"mean_attention_weight":attention.mean(axis=0),
                  "median_attention_weight":np.median(attention,axis=0)}).to_csv(WEIGHTS/"aggregate_attention_by_lag.csv",index=False)
    checkpoint={"model_state_dict":model.state_dict(),"scaler":scaler,"hidden_size":32,"lookback":LOOKBACK,
                "horizon":HORIZON,"attention_type":"additive_temporal_softmax","seed":SEED}
    torch.save(checkpoint,CHECKPOINTS/"arima_gru_attention_validation_checkpoint.pt")
    hashes_after={str(p):sha256(p) for p in protected}
    manifest={"model":"ARIMA-GRU-Attention","split":"validation","research_eligible":True,"run_id":run_id,
      "created_at_utc":datetime.now(timezone.utc).isoformat(),"train_data":str(TRAIN_PATH),"validation_data":str(VALIDATION_PATH),
      "test_data_loaded":False,"test_predictions_generated":False,"residual_pipeline":"identical Milestone-4 functions",
      "residuals_match_frozen_baseline":True,"training_residual_rows":len(residuals),"training_windows":len(X_train),
      "successful_validation_origins":len(metadata),"failed_validation_origins":len(failures),"meters":pred.meter.nunique(),
      "architecture":"unidirectional 1-layer GRU, 32 hidden, additive temporal softmax attention over 168 states, direct 24 output",
      "parameter_count":sum(p.numel() for p in model.parameters()),"attention_type":"temporal","attention_axis":"history_time",
      "attention_weight_shape":list(attention.shape),"scaler":scaler,"optimizer":"Adam","learning_rate":1e-3,"batch_size":128,
      "max_epochs":20,"patience":4,"selected_epoch":int(pd.DataFrame(history).validation_residual_MSE_scaled.idxmin()+1),
      "residual_generation_seconds":generation_seconds,"training_time_seconds":train_seconds,"inference_time_ms_per_origin":inference_ms,
      "pso":False,"baseline_hashes_before":hashes_before,"baseline_hashes_after":hashes_after,
      "baseline_outputs_untouched":hashes_before==hashes_after}
    (RUNS/"arima_gru_attention_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("parameter_count","selected_epoch","training_time_seconds","successful_validation_origins","baseline_outputs_untouched")},indent=2))


if __name__=="__main__": main()
