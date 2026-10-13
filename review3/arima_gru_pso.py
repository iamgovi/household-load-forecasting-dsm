"""Small PSO search for the plain leakage-safe ARIMA-GRU residual model."""
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

from evaluation.contract import HORIZON, PREDICTION_COLUMNS, REPORT_DIR, TRAIN_PATH, VALIDATION_PATH
from evaluation.metrics import evaluate_predictions, point_metrics, validate_prediction_schema
from evaluation.origins import load_research_splits
from review3.arima_gru_hybrid import build_training_residuals_and_windows, build_validation_arrays

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"reports"/"review3"/"arima_gru_pso"
SEARCH,PREDICTIONS,METRICS,TABLES,FIGURES,CHECKPOINTS,RUNS=[OUTPUT/x for x in
    ("search","predictions","metrics","tables","figures","checkpoints","runs")]
HYBRID=ROOT/"reports"/"review3"/"arima_gru_hybrid"
ATTENTION=ROOT/"reports"/"review3"/"arima_gru_attention"
INDIVIDUAL=ROOT/"reports"/"review3"/"individual_models"
SPACE={"hidden_units":[16,32,64],"learning_rate":[0.0005,0.001,0.002],"dropout":[0.0,0.1,0.2,0.3],
       "lookback":[24,48,72,168],"batch_size":[64,128]}
SWARM_SIZE=5; ITERATIONS=5; SCREEN_EPOCHS=4; PSO_SEED=42; FINAL_SEEDS=[42,123,2026]


def prepare_dirs():
    for p in (SEARCH,PREDICTIONS,METRICS,TABLES,FIGURES,CHECKPOINTS,RUNS): p.mkdir(parents=True,exist_ok=True)


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True,warn_only=True)


class TunedResidualGRU(nn.Module):
    def __init__(self,hidden_units,dropout):
        super().__init__(); self.gru=nn.GRU(1,hidden_units,batch_first=True,bidirectional=False)
        self.dropout=nn.Dropout(dropout); self.output=nn.Linear(hidden_units,HORIZON)
    def forward(self,x):
        _,hidden=self.gru(x); return self.output(self.dropout(hidden[-1]))


def decode(position):
    result={}
    for value,(name,choices) in zip(position,SPACE.items()):
        result[name]=choices[int(np.clip(np.rint(value),0,len(choices)-1))]
    return result


def inner_indices(metadata):
    frame=pd.DataFrame(metadata).reset_index(names="array_index").sort_values(["meter","forecast_origin"])
    train_idx=[]; val_idx=[]
    for _,g in frame.groupby("meter"):
        cut=max(1,int(len(g)*.8)); train_idx.extend(g.array_index.iloc[:cut]); val_idx.extend(g.array_index.iloc[cut:])
    return np.asarray(train_idx,int),np.asarray(val_idx,int)


def fit_fixed(config,X_train,y_train,X_val,y_val,seed,epochs):
    set_seed(seed); lookback=config["lookback"]
    Xt=X_train[:,-lookback:].copy(); Xv=X_val[:,-lookback:].copy()
    x_mean,x_std=float(Xt.mean()),float(Xt.std()) or 1.0; y_mean,y_std=float(y_train.mean()),float(y_train.std()) or 1.0
    Xt=(Xt-x_mean)/x_std; Xv=(Xv-x_mean)/x_std; yt=(y_train-y_mean)/y_std
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(yt)),batch_size=config["batch_size"],shuffle=True,generator=generator)
    model=TunedResidualGRU(config["hidden_units"],config["dropout"]); optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"]); loss_fn=nn.MSELoss()
    started=time.perf_counter(); history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); losses.append(loss.detach().item())
        model.eval()
        with torch.no_grad(): pred=model(torch.tensor(Xv)).numpy()*y_std+y_mean
        metrics=point_metrics(y_val.ravel(),pred.ravel()); history.append({"epoch":epoch,"train_MSE_scaled":float(np.mean(losses)),"validation_MAE":metrics["MAE"],"validation_RMSE":metrics["RMSE"]})
    runtime=time.perf_counter()-started
    scaler={"input_residual_mean":x_mean,"input_residual_std":x_std,"target_correction_mean":y_mean,"target_correction_std":y_std,"fit_split":"train"}
    return model,pred,history,runtime,scaler


def screen_particle(config,X,y,train_idx,val_idx,iteration,particle_id):
    started=time.perf_counter()
    try:
        _,pred,history,_,_=fit_fixed(config,X[train_idx],y[train_idx],X[val_idx],y[val_idx],PSO_SEED,SCREEN_EPOCHS)
        metrics=point_metrics(y[val_idx].ravel(),pred.ravel()); status="success"; reason=""
    except Exception as exc:
        metrics={"RMSE":1e9,"MAE":1e9}; history=[]; status="failed"; reason=repr(exc)
    return {"iteration":iteration,"particle_id":particle_id,**config,"seed":PSO_SEED,"epochs_trained":len(history),
            "fitness_RMSE":metrics["RMSE"],"MAE":metrics["MAE"],"runtime_seconds":time.perf_counter()-started,
            "status":status,"failure_reason":reason}


def run_pso(X,y,metadata):
    train_idx,val_idx=inner_indices(metadata); rng=np.random.default_rng(PSO_SEED); dims=len(SPACE)
    upper=np.array([len(v)-1 for v in SPACE.values()],float); positions=rng.uniform(0,upper,(SWARM_SIZE,dims)); velocities=np.zeros_like(positions)
    positions[0]=np.array([1,1,0,3,1],float)  # untuned reference is explicitly screened
    personal=positions.copy(); personal_fit=np.full(SWARM_SIZE,np.inf); global_pos=positions[0].copy(); global_fit=np.inf
    history=[]; convergence=[]; cache={}; started=time.perf_counter()
    for iteration in range(1,ITERATIONS+1):
        iteration_rows=[]
        for particle in range(SWARM_SIZE):
            config=decode(positions[particle]); key=tuple(config.items())
            if key in cache:
                row={**cache[key],"iteration":iteration,"particle_id":particle,"status":"cached","runtime_seconds":0.0}
            else:
                row=screen_particle(config,X,y,train_idx,val_idx,iteration,particle); cache[key]=row.copy()
            history.append(row); iteration_rows.append(row); fitness=row["fitness_RMSE"]
            if fitness<personal_fit[particle]: personal_fit[particle]=fitness; personal[particle]=positions[particle].copy()
            if fitness<global_fit: global_fit=fitness; global_pos=positions[particle].copy()
        best=min(iteration_rows,key=lambda r:r["fitness_RMSE"])
        convergence.append({"iteration":iteration,"iteration_best_fitness":best["fitness_RMSE"],"global_best_fitness":global_fit,**{f"best_{k}":v for k,v in decode(global_pos).items()}})
        r1=rng.random(positions.shape); r2=rng.random(positions.shape)
        velocities=.7*velocities+1.4*r1*(personal-positions)+1.4*r2*(global_pos-positions)
        positions=np.clip(positions+velocities,0,upper)
    search_seconds=time.perf_counter()-started
    pd.DataFrame(history).to_csv(SEARCH/"pso_search_history.csv",index=False); pd.DataFrame(convergence).to_csv(SEARCH/"pso_convergence.csv",index=False)
    split=pd.DataFrame(metadata); split["inner_partition"]="inner_train"; split.iloc[val_idx,split.columns.get_loc("inner_partition")]="inner_validation"
    split.to_csv(SEARCH/"inner_chronological_split.csv",index=False)
    return decode(global_pos),global_fit,search_seconds,len(cache)


def train_final(config,X_train,y_train,X_val,y_val,base,actual,metadata,seed):
    set_seed(seed); lookback=config["lookback"]; Xt=X_train[:,-lookback:].copy(); Xv=X_val[:,-lookback:].copy()
    x_mean,x_std=float(Xt.mean()),float(Xt.std()) or 1.; y_mean,y_std=float(y_train.mean()),float(y_train.std()) or 1.
    Xt=(Xt-x_mean)/x_std; Xv=(Xv-x_mean)/x_std; yt=(y_train-y_mean)/y_std; yv=(y_val-y_mean)/y_std
    gen=torch.Generator().manual_seed(seed); loader=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(yt)),batch_size=config["batch_size"],shuffle=True,generator=gen)
    val_x,val_y=torch.tensor(Xv),torch.tensor(yv); model=TunedResidualGRU(config["hidden_units"],config["dropout"])
    opt=torch.optim.Adam(model.parameters(),lr=config["learning_rate"]); loss_fn=nn.MSELoss(); best=np.inf; state=None; stale=0; hist=[]; started=time.perf_counter()
    for epoch in range(1,21):
        model.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.detach().item())
        model.eval()
        with torch.no_grad(): vl=loss_fn(model(val_x),val_y).item()
        hist.append({"seed":seed,"epoch":epoch,"train_MSE_scaled":float(np.mean(losses)),"validation_MSE_scaled":vl})
        if vl<best-1e-6: best=vl; state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=4: break
    train_seconds=time.perf_counter()-started; model.load_state_dict(state); model.eval(); started=time.perf_counter()
    with torch.no_grad(): correction=model(val_x).numpy()*y_std+y_mean
    inference_ms=(time.perf_counter()-started)*1000/max(len(X_val),1); rows=[]; run_id=f"PSO-ARIMA-GRU_seed{seed}"
    for origin,truth,arima,residual in zip(metadata,actual,base,correction):
        targets=pd.date_range(origin.first_target_timestamp,periods=HORIZON,freq="h")
        for h,(ts,a,b,r) in enumerate(zip(targets,truth,arima,residual),1):
            rows.append({"model":"PSO-ARIMA-GRU","city":origin.city,"meter":origin.meter,"split":"validation","forecast_origin":origin.forecast_origin,
             "horizon":h,"target_timestamp":ts,"actual":a,"prediction":b+r,"inference_time_ms":inference_ms,"run_id":run_id,
             "arima_prediction":b,"residual_correction":r,"hybrid_prediction":b+r,"seed":seed})
    pred=pd.DataFrame(rows); scaler={"input_residual_mean":x_mean,"input_residual_std":x_std,"target_correction_mean":y_mean,"target_correction_std":y_std,"fit_split":"train"}
    return model,pred,hist,train_seconds,inference_ms,scaler


def main():
    prepare_dirs(); protected=[]
    for folder in (INDIVIDUAL,HYBRID,ATTENTION):
        protected.extend(p for p in folder.rglob("*") if p.is_file())
    hashes_before={str(p):sha256(p) for p in protected}
    splits=load_research_splits({"train":TRAIN_PATH,"validation":VALIDATION_PATH})
    origins=pd.read_csv(REPORT_DIR/"validation_origins.csv",parse_dates=["forecast_origin","first_target_timestamp","last_target_timestamp"])
    residuals,_,X,y,metadata,models=build_training_residuals_and_windows(splits["train"],sorted(origins.meter.unique()))
    frozen=pd.read_csv(HYBRID/"residuals"/"train_oos_arima_residuals.csv",parse_dates=["timestamp","fit_end_timestamp"])
    pd.testing.assert_frame_equal(residuals.reset_index(drop=True),frozen.reset_index(drop=True),check_exact=False,rtol=1e-10,atol=1e-10)
    selected,best_fitness,search_seconds,unique=run_pso(X,y,metadata)
    arrays,val_meta,failures=build_validation_arrays(splits["train"],splits["validation"],origins,residuals,models); Xv,yv,base,actual=arrays
    seed_rows=[]; histories=[]; primary=None; checkpoints={}
    for seed in FINAL_SEEDS:
        model,pred,hist,seconds,inference,scaler=train_final(selected,X,y,Xv,yv,base,actual,val_meta,seed)
        pred.to_csv(PREDICTIONS/f"pso_arima_gru_seed{seed}_predictions.csv",index=False)
        row=evaluate_predictions(pred[PREDICTION_COLUMNS])["overall"].query("aggregation=='pooled_predictions'").iloc[0]
        seed_rows.append({"seed":seed,"MAE":row.MAE,"RMSE":row.RMSE,"sMAPE_pct":row.sMAPE_pct,"R2":row.R2,"NRMSE_mean":row.NRMSE_mean,
                          "Peak_Error":row.peak_load_error,"training_time_seconds":seconds,"inference_time_ms":inference,"selected_epoch":pd.DataFrame(hist).validation_MSE_scaled.idxmin()+1})
        histories.extend(hist); checkpoints[seed]={"model_state_dict":model.state_dict(),"scaler":scaler,"config":selected,"seed":seed}
        if seed==FINAL_SEEDS[0]: primary=pred
    primary.to_csv(PREDICTIONS/"pso_arima_gru_predictions.csv",index=False); pd.DataFrame(seed_rows).to_csv(TABLES/"pso_seed_variability.csv",index=False)
    pd.DataFrame(histories).to_csv(RUNS/"pso_final_training_history.csv",index=False); failures.to_csv(RUNS/"pso_validation_failures.csv",index=False)
    for seed,payload in checkpoints.items(): torch.save(payload,CHECKPOINTS/f"pso_arima_gru_seed{seed}_checkpoint.pt")
    for level,table in evaluate_predictions(primary[PREDICTION_COLUMNS]).items(): table.to_csv(METRICS/f"pso_arima_gru_{level}.csv",index=False)
    hashes_after={str(p):sha256(p) for p in protected}; manifest={"model":"PSO-ARIMA-GRU","split":"validation","research_eligible":True,
      "created_at_utc":datetime.now(timezone.utc).isoformat(),"train_data":str(TRAIN_PATH),"validation_data":str(VALIDATION_PATH),"test_data_loaded":False,
      "test_predictions_generated":False,"pso_fitness":"inner-validation RMSE","inner_split":"chronological 80/20 per meter within train windows",
      "screen_epochs":SCREEN_EPOCHS,"swarm_size":SWARM_SIZE,"iterations":ITERATIONS,"particle_evaluations":SWARM_SIZE*ITERATIONS,
      "unique_configurations_trained":unique,"pso_seed":PSO_SEED,"final_seeds":FINAL_SEEDS,"search_space":SPACE,"selected_configuration":selected,
      "best_inner_validation_RMSE":best_fitness,"search_runtime_seconds":search_seconds,"successful_validation_origins":len(val_meta),"meters":primary.meter.nunique(),
      "architecture":"plain unidirectional one-layer residual GRU, direct 24 output, no attention","residual_pipeline_matches_milestone4":True,
      "scaler_fit":"train only; inner-train only during PSO screening","baseline_outputs_untouched":hashes_before==hashes_after,
      "baseline_hashes_before":hashes_before,"baseline_hashes_after":hashes_after}
    (RUNS/"pso_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("particle_evaluations","unique_configurations_trained","best_inner_validation_RMSE","selected_configuration","search_runtime_seconds","baseline_outputs_untouched")},indent=2))


if __name__=="__main__": main()
