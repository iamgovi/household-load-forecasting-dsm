"""Leakage-safe ARIMA(3,0,0)-GRU residual hybrid for Review 3.

Only authoritative train/validation data are read. Training residuals are
strictly prequential: each residual is predicted before its target is added to
the expanding conditional least-squares AR(3) fit.
"""

from __future__ import annotations

import copy
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "review3" / "arima_gru_hybrid"
PREDICTIONS, TABLES, FIGURES = OUTPUT / "predictions", OUTPUT / "tables", OUTPUT / "figures"
METRICS, RUNS, CHECKPOINTS, RESIDUALS = OUTPUT / "metrics", OUTPUT / "runs", OUTPUT / "checkpoints", OUTPUT / "residuals"
AR_ORDER = 3
WARMUP_EQUATIONS = 168
TRAIN_ORIGIN_STRIDE = 24


def prepare_dirs():
    for path in (PREDICTIONS, TABLES, FIGURES, METRICS, RUNS, CHECKPOINTS, RESIDUALS):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def design(history: list[float]) -> np.ndarray:
    return np.array([1.0, *history[-AR_ORDER:][::-1]], dtype=float)


def coefficients(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    return np.linalg.solve(xtx + np.eye(AR_ORDER + 1) * 1e-8, xty)


def recursive_forecast(beta: np.ndarray, history: list[float]) -> np.ndarray:
    work = list(history); result = []
    for _ in range(HORIZON):
        value = float(design(work) @ beta)
        result.append(value); work.append(value)
    return np.asarray(result)


def build_training_residuals_and_windows(train: pd.DataFrame, meters: list[str]):
    """Return prequential one-step residuals and direct 24-step correction windows."""
    residual_rows, coverage_rows, X, y, metadata = [], [], [], [], []
    final_models = {}
    for meter in meters:
        frame = train[train.meter.eq(meter)].sort_values("x_Timestamp")
        series = frame.set_index("x_Timestamp").t_kWh.asfreq("h")
        city = str(frame.city.iloc[0]); values = series.to_numpy(float); timestamps = series.index
        xtx = np.zeros((AR_ORDER + 1, AR_ORDER + 1)); xty = np.zeros(AR_ORDER + 1)
        one_step = np.full(len(series), np.nan); equations = 0
        snapshots = {}
        eligible = generated = skipped_warmup = skipped_missing = 0
        for i in range(AR_ORDER, len(series)):
            lag_values = values[i-AR_ORDER:i]
            if not np.isfinite(values[i]) or not np.isfinite(lag_values).all():
                skipped_missing += 1; continue
            eligible += 1; x = np.array([1.0, *lag_values[::-1]])
            if equations >= WARMUP_EQUATIONS:
                beta_before = coefficients(xtx, xty)
                prediction = float(x @ beta_before)
                one_step[i] = values[i] - prediction
                residual_rows.append({
                    "city": city, "meter": meter, "timestamp": timestamps[i], "actual": values[i],
                    "arima_prediction": prediction, "residual": one_step[i],
                    "fit_end_timestamp": timestamps[i-1], "training_rows_before_prediction": equations,
                    "out_of_sample": True,
                })
                generated += 1
            else:
                skipped_warmup += 1
            xtx += np.outer(x, x); xty += x * values[i]; equations += 1
            snapshots[i] = coefficients(xtx, xty)
        residual_series = pd.Series(one_step, index=timestamps)
        # Daily-spaced train origins prevent near-duplicate overlapping targets.
        for i in range(LOOKBACK - 1, len(series) - HORIZON, TRAIN_ORIGIN_STRIDE):
            context = residual_series.iloc[i-LOOKBACK+1:i+1].to_numpy(float)
            target = values[i+1:i+1+HORIZON]
            beta = snapshots.get(i)
            if beta is None or not np.isfinite(context).all() or not np.isfinite(target).all():
                continue
            base = recursive_forecast(beta, values[:i+1].tolist())
            X.append(context[:, None]); y.append(target - base)
            metadata.append({"meter":meter,"forecast_origin":timestamps[i],
                             "residual_input_start":timestamps[i]-pd.Timedelta(hours=LOOKBACK-1),
                             "residual_input_end":timestamps[i],"first_target":timestamps[i]+pd.Timedelta(hours=1),
                             "last_target":timestamps[i]+pd.Timedelta(hours=HORIZON),
                             "arima_fit_end":timestamps[i],"targets_excluded_from_arima_fit":True})
        final_models[meter] = coefficients(xtx, xty)
        coverage_rows.append({
            "city": city, "meter": meter, "eligible_lag_target_rows": eligible,
            "residuals_generated": generated, "skipped_warmup": skipped_warmup,
            "skipped_missing_or_invalid_lags": skipped_missing,
            "usable_direct_24h_training_windows": sum(row["meter"] == meter for row in metadata),
            "warmup_equations": WARMUP_EQUATIONS, "origin_stride_hours": TRAIN_ORIGIN_STRIDE,
        })
    return (pd.DataFrame(residual_rows), pd.DataFrame(coverage_rows),
            np.asarray(X, np.float32), np.asarray(y, np.float32), metadata, final_models)


def build_validation_arrays(train, validation, origins, train_residuals, final_models):
    X, y, base_all, actual_all, metadata, failures = [], [], [], [], [], []
    residual_map = {
        meter: group.set_index("timestamp").residual.sort_index()
        for meter, group in train_residuals.groupby("meter")
    }
    combined = pd.concat([train, validation], ignore_index=True)
    series_map = {m: g.set_index("x_Timestamp").t_kWh.sort_index() for m, g in combined.groupby("meter")}
    val_map = {m: g.set_index("x_Timestamp").t_kWh.sort_index() for m, g in validation.groupby("meter")}
    # Validation one-step residuals use the final train-only AR coefficients.
    for meter in sorted(origins.meter.unique()):
        series = series_map[meter]; beta = final_models[meter]
        val_residuals = {}
        for timestamp, actual in val_map[meter].items():
            prior_times = pd.date_range(end=timestamp-pd.Timedelta(hours=1), periods=AR_ORDER, freq="h")
            prior = series.reindex(prior_times).to_numpy(float)
            if pd.notna(actual) and np.isfinite(prior).all():
                val_residuals[timestamp] = float(actual - design(prior.tolist()) @ beta)
        residual_map[meter] = pd.concat([residual_map[meter], pd.Series(val_residuals)]).sort_index()
    for origin in origins.sort_values(["meter", "forecast_origin"]).itertuples(index=False):
        history_times = pd.date_range(end=origin.forecast_origin, periods=LOOKBACK, freq="h")
        target_times = pd.date_range(origin.first_target_timestamp, periods=HORIZON, freq="h")
        residual_history = residual_map[origin.meter].reindex(history_times).to_numpy(float)
        actual = series_map[origin.meter].reindex(target_times).to_numpy(float)
        load_history = series_map[origin.meter].loc[:origin.forecast_origin].dropna().tolist()
        if not np.isfinite(residual_history).all():
            failures.append({"city": origin.city, "meter": origin.meter,
                             "forecast_origin": origin.forecast_origin, "reason": "invalid_residual_history"}); continue
        base = recursive_forecast(final_models[origin.meter], load_history)
        if not np.isfinite(actual).all():
            failures.append({"city": origin.city, "meter": origin.meter,
                             "forecast_origin": origin.forecast_origin, "reason": "invalid_target"}); continue
        X.append(residual_history[:, None]); y.append(actual-base); base_all.append(base); actual_all.append(actual); metadata.append(origin)
    return tuple(map(lambda z: np.asarray(z, np.float32), (X, y, base_all, actual_all))), metadata, pd.DataFrame(failures)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.output = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.output(hidden[-1])


def train_and_predict(X_train, y_train, X_val, y_val):
    x_mean, x_std = float(X_train.mean()), float(X_train.std()) or 1.0
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0
    Xs = (X_train-x_mean)/x_std; ys = (y_train-y_mean)/y_std
    Xv = (X_val-x_mean)/x_std; yv = (y_val-y_mean)/y_std
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)), batch_size=128,
                        shuffle=True, generator=generator)
    val_x, val_y = torch.tensor(Xv), torch.tensor(yv)
    model = ResidualGRU(32); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
    best_loss, best_state, stale, history = np.inf, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, 21):
        model.train(); losses=[]
        for xb, yb in loader:
            optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); losses.append(float(loss))
        model.eval()
        with torch.no_grad(): val_loss=float(loss_fn(model(val_x), val_y))
        history.append({"epoch":epoch,"train_residual_MSE_scaled":float(np.mean(losses)),
                        "validation_residual_MSE_scaled":val_loss})
        if val_loss < best_loss-1e-6:
            best_loss=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale += 1
            if stale >= 4: break
    training_seconds=time.perf_counter()-started; model.load_state_dict(best_state); model.eval()
    started=time.perf_counter()
    with torch.no_grad(): correction=model(val_x).numpy()*y_std+y_mean
    inference_ms=(time.perf_counter()-started)*1000/max(len(X_val),1)
    scaler={"input_residual_mean":x_mean,"input_residual_std":x_std,
            "target_correction_mean":y_mean,"target_correction_std":y_std,"fit_split":"train"}
    return model, correction, training_seconds, inference_ms, history, scaler


def prediction_records(metadata, actual, base, correction, inference_ms, run_id):
    rows=[]
    for origin, truth, arima, residual in zip(metadata,actual,base,correction):
        targets=pd.date_range(origin.first_target_timestamp,periods=HORIZON,freq="h")
        hybrid=arima+residual
        for h,(ts,a,b,r,p) in enumerate(zip(targets,truth,arima,residual,hybrid),1):
            rows.append({"model":"ARIMA-GRU","city":origin.city,"meter":origin.meter,"split":"validation",
                         "forecast_origin":origin.forecast_origin,"horizon":h,"target_timestamp":ts,
                         "actual":a,"prediction":p,"inference_time_ms":inference_ms,"run_id":run_id,
                         "arima_prediction":b,"residual_correction":r,"hybrid_prediction":p})
    return pd.DataFrame(rows)


def main():
    prepare_dirs(); seed_everything()
    splits=load_research_splits({"train":TRAIN_PATH,"validation":VALIDATION_PATH})
    origins=pd.read_csv(REPORT_DIR/"validation_origins.csv",parse_dates=["forecast_origin","first_target_timestamp","last_target_timestamp"])
    meters=sorted(origins.meter.unique()); started=time.perf_counter()
    residuals, coverage, X_train, y_train, training_index, models=build_training_residuals_and_windows(splits["train"],meters)
    residual_generation_seconds=time.perf_counter()-started
    residuals.to_csv(RESIDUALS/"train_oos_arima_residuals.csv",index=False)
    coverage.to_csv(TABLES/"residual_generation_coverage.csv",index=False)
    pd.DataFrame(training_index).to_csv(TABLES/"residual_training_window_index.csv",index=False)
    arrays, metadata, failures=build_validation_arrays(splits["train"],splits["validation"],origins,residuals,models)
    X_val,y_val,base,actual=arrays
    model,correction,train_seconds,inference_ms,history,scaler=train_and_predict(X_train,y_train,X_val,y_val)
    run_id=f"ARIMA-GRU_validation_{int(time.time())}"
    predictions=prediction_records(metadata,actual,base,correction,inference_ms,run_id)
    validate_prediction_schema(predictions[PREDICTION_COLUMNS])
    predictions.to_csv(PREDICTIONS/"arima_gru_predictions.csv",index=False)
    failures.to_csv(RUNS/"arima_gru_failures.csv",index=False)
    pd.DataFrame(history).to_csv(RUNS/"arima_gru_training_history.csv",index=False)
    for level,table in evaluate_predictions(predictions[PREDICTION_COLUMNS]).items():
        table.to_csv(METRICS/f"arima_gru_{level}.csv",index=False)
    checkpoint={"model_state_dict":model.state_dict(),"scaler":scaler,"ar_order":AR_ORDER,
                "lookback":LOOKBACK,"horizon":HORIZON,"hidden_size":32,"seed":SEED}
    torch.save(checkpoint,CHECKPOINTS/"arima_gru_validation_checkpoint.pt")
    manifest={
        "model":"ARIMA-GRU","research_eligible":True,"split":"validation","run_id":run_id,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"train_data":str(TRAIN_PATH),
        "validation_data":str(VALIDATION_PATH),"validation_origins":str(REPORT_DIR/"validation_origins.csv"),
        "test_data_loaded":False,"test_predictions_generated":False,"arima_specification":"conditional least-squares ARIMA(3,0,0)",
        "residual_generation":"expanding prequential OLS; predict before target update",
        "warmup_equations":WARMUP_EQUATIONS,"training_origin_stride_hours":TRAIN_ORIGIN_STRIDE,
        "training_residual_rows":len(residuals),"training_windows":len(X_train),
        "canonical_validation_origins":len(origins),"successful_validation_origins":len(metadata),
        "failed_validation_origins":len(failures),"meters":int(predictions.meter.nunique()),
        "residual_generation_seconds":residual_generation_seconds,"training_time_seconds":train_seconds,
        "inference_time_ms_per_origin":inference_ms,"architecture":"unidirectional 1-layer residual-only GRU, 32 hidden, direct 24 output",
        "parameter_count":sum(p.numel() for p in model.parameters()),"scaler":scaler,
        "early_stopping":{"split":"validation","patience":4,"max_epochs":20,"selected_epoch":int(pd.DataFrame(history).validation_residual_MSE_scaled.idxmin()+1)},
        "attention":False,"pso":False,
    }
    (RUNS/"arima_gru_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("training_residual_rows","training_windows","successful_validation_origins","failed_validation_origins","meters","training_time_seconds")},indent=2))


if __name__=="__main__": main()
