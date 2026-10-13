"""Train and evaluate individual models on the Milestone-3 contract.

This module uses validation only. It never reads the legacy root split files or
generates test predictions. Run from the repository root with ``python -m``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ClassicalExponentialSmoothing
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.comparison import common_origin_predictions
from evaluation.contract import (
    HORIZON, LOOKBACK, PREDICTION_COLUMNS, REPORT_DIR as EVALUATION_REPORT_DIR,
    SEED, TRAIN_PATH, VALIDATION_PATH,
)
from evaluation.metrics import evaluate_predictions, point_metrics, validate_prediction_schema
from evaluation.origins import build_canonical_origins, load_research_splits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "review3" / "individual_models"
PREDICTIONS = OUTPUT / "predictions"
METRICS = OUTPUT / "metrics"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
RUNS = OUTPUT / "runs"
CHECKPOINTS = OUTPUT / "checkpoints"
MODEL_ORDER = ["SeasonalNaive_Daily", "HoltWinters", "ARIMA", "LinearRegression", "XGBoost", "GRU"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_dirs() -> None:
    for path in (PREDICTIONS, METRICS, TABLES, FIGURES, RUNS, CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)


def load_contract_data():
    # Test is intentionally not loaded anywhere in the Review-3 model runner.
    paths = {"train": TRAIN_PATH, "validation": VALIDATION_PATH}
    splits = load_research_splits(paths)
    origins = pd.read_csv(
        EVALUATION_REPORT_DIR / "validation_origins.csv",
        parse_dates=["forecast_origin", "first_target_timestamp", "last_target_timestamp"],
    )
    return splits, origins


def series_by_meter(splits):
    full = pd.concat(list(splits.values()), ignore_index=True)
    return {
        meter: group.set_index("x_Timestamp")["t_kWh"].sort_index()
        for meter, group in full.groupby("meter")
    }


def prediction_records(model, origins, actual_by_meter, predict_one, run_id):
    rows, failures = [], []
    for origin in origins.sort_values(["meter", "forecast_origin"]).itertuples(index=False):
        targets = pd.date_range(origin.first_target_timestamp, periods=HORIZON, freq="h")
        try:
            started = time.perf_counter()
            prediction = np.asarray(predict_one(origin, targets), dtype=float)
            runtime_ms = (time.perf_counter() - started) * 1000
            actual = actual_by_meter[origin.meter].reindex(targets).to_numpy(float)
            if prediction.shape != (HORIZON,) or not np.isfinite(prediction).all():
                raise ValueError("forecast is not 24 finite values")
            for h, (timestamp, truth, estimate) in enumerate(zip(targets, actual, prediction), 1):
                rows.append({
                    "model": model, "city": origin.city, "meter": origin.meter,
                    "split": "validation", "forecast_origin": origin.forecast_origin,
                    "horizon": h, "target_timestamp": timestamp, "actual": truth,
                    "prediction": estimate, "inference_time_ms": runtime_ms, "run_id": run_id,
                })
        except Exception as exc:
            failures.append({"model": model, "city": origin.city, "meter": origin.meter,
                             "forecast_origin": origin.forecast_origin, "error": repr(exc)})
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if not frame.empty:
        validate_prediction_schema(frame)
    return frame, pd.DataFrame(failures)


def save_model_outputs(model, predictions, failures, training_seconds, config, history=None):
    slug = model.lower()
    predictions.to_csv(PREDICTIONS / f"{slug}_predictions.csv", index=False)
    failures.to_csv(RUNS / f"{slug}_failures.csv", index=False)
    evaluated = evaluate_predictions(predictions)
    for level, table in evaluated.items():
        table.to_csv(METRICS / f"{slug}_{level}.csv", index=False)
    manifest = {
        "model": model, "split": "validation", "research_eligible": True,
        "run_id": predictions.run_id.iloc[0], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_data": str(TRAIN_PATH), "validation_origins": str(EVALUATION_REPORT_DIR / "validation_origins.csv"),
        "test_predictions_generated": False, "lookback": LOOKBACK, "horizon": HORIZON,
        "training_time_seconds": training_seconds, "configuration": config,
        "successful_origins": int(predictions.groupby(["meter", "forecast_origin"]).ngroups),
        "failed_origins": int(len(failures)), "meters": int(predictions.meter.nunique()),
    }
    if history is not None:
        pd.DataFrame(history).to_csv(RUNS / f"{slug}_training_history.csv", index=False)
    (RUNS / f"{slug}_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def run_seasonal_naive(splits, origins):
    model = "SeasonalNaive_Daily"; run_id = f"{model}_{int(time.time())}"
    all_series = series_by_meter(splits)
    predict = lambda origin, targets: all_series[origin.meter].reindex(
        targets - pd.Timedelta(hours=24)).to_numpy()
    predictions, failures = prediction_records(model, origins, all_series, predict, run_id)
    save_model_outputs(model, predictions, failures, 0.0,
                       {"definition": "prediction(t+h)=actual(t+h-24)"})


def _fit_statistical(model, series, config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "HoltWinters":
            return ExponentialSmoothing(
                series, trend=config["trend"], damped_trend=config.get("damped_trend", False),
                seasonal=24, initialization_method="estimated",
            ).fit(disp=False, maxiter=100)
        if model == "ARIMA":
            return ARIMA(series, order=tuple(config["order"]), enforce_stationarity=False,
                         enforce_invertibility=False).fit(method_kwargs={"maxiter": 100})
        return SARIMAX(
            series, order=tuple(config["order"]), seasonal_order=tuple(config["seasonal_order"]),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=config.get("maxiter", 30))


def _statistical_forecasts(model, config, splits, origins, limit_meters=None):
    if model == "HoltWinters":
        return _holt_winters_forecasts(config, splits, origins, limit_meters)
    if model == "ARIMA":
        return _conditional_arima_forecasts(config, splits, origins, limit_meters)
    rows, failures = [], []
    meters = sorted(origins.meter.unique())
    if limit_meters:
        meters = meters[:limit_meters]
    fit_seconds = 0.0
    for meter in meters:
        meter_origins = origins[origins.meter.eq(meter)].sort_values("forecast_origin")
        train = splits["train"][splits["train"].meter.eq(meter)].set_index("x_Timestamp").t_kWh.asfreq("h")
        validation = splits["validation"][splits["validation"].meter.eq(meter)].set_index("x_Timestamp").t_kWh.asfreq("h")
        started = time.perf_counter()
        try:
            state = _fit_statistical(model, train, config)
        except Exception as exc:
            failures.extend({"model": model, "meter": meter, "forecast_origin": o,
                             "error": f"fit: {exc!r}"} for o in meter_origins.forecast_origin)
            continue
        fit_seconds += time.perf_counter() - started
        state_end = train.index.max()
        city = meter_origins.city.iloc[0]
        for origin in meter_origins.itertuples(index=False):
            try:
                new = validation.loc[(validation.index > state_end) & (validation.index <= origin.forecast_origin)]
                if len(new):
                    # ``extend`` preserves the final filtered state without
                    # retaining an ever-growing validation-results history.
                    state = state.extend(new)
                    state_end = new.index.max()
                forecast_started = time.perf_counter()
                forecast = np.asarray(state.forecast(HORIZON), dtype=float)
                runtime_ms = (time.perf_counter() - forecast_started) * 1000
                targets = pd.date_range(origin.first_target_timestamp, periods=HORIZON, freq="h")
                actual = validation.reindex(targets).to_numpy(float)
                if forecast.shape != (HORIZON,) or not np.isfinite(forecast).all():
                    raise ValueError("non-finite statistical forecast")
                run_id = f"{model}_validation"
                for h, (timestamp, truth, estimate) in enumerate(zip(targets, actual, forecast), 1):
                    rows.append({"model": model, "city": city, "meter": meter, "split": "validation",
                                 "forecast_origin": origin.forecast_origin, "horizon": h,
                                 "target_timestamp": timestamp, "actual": truth, "prediction": estimate,
                                 "inference_time_ms": runtime_ms, "run_id": run_id})
            except Exception as exc:
                failures.append({"model": model, "meter": meter,
                                 "forecast_origin": origin.forecast_origin, "error": repr(exc)})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), pd.DataFrame(failures), fit_seconds


def _conditional_arima_forecasts(config, splits, origins, limit_meters=None):
    """Conditional least-squares ARIMA(p,0,0), recursively forecast for 24h."""
    p=int(config["order"][0]); rows=[]; failures=[]; fit_seconds=0.0
    meters=sorted(origins.meter.unique())[:limit_meters or None]
    all_series=series_by_meter(splits)
    for meter in meters:
        train=splits["train"][splits["train"].meter.eq(meter)].set_index("x_Timestamp").t_kWh.sort_index()
        lagged=pd.concat({f"lag_{lag}":train.shift(lag) for lag in range(1,p+1)},axis=1)
        fit_frame=lagged.assign(target=train).dropna()
        meter_origins=origins[origins.meter.eq(meter)].sort_values("forecast_origin")
        if len(fit_frame)<100:
            failures.extend({"model":"ARIMA","meter":meter,"forecast_origin":o,"error":"insufficient valid lag pairs"}
                            for o in meter_origins.forecast_origin); continue
        estimator=LinearRegression(); started=time.perf_counter()
        estimator.fit(fit_frame.drop(columns="target").to_numpy(),fit_frame.target.to_numpy()); fit_seconds+=time.perf_counter()-started
        city=meter_origins.city.iloc[0]; series=all_series[meter]
        for origin in meter_origins.itertuples(index=False):
            history=series.loc[:origin.forecast_origin].dropna().tolist()
            started=time.perf_counter(); forecast=[]
            for _ in range(HORIZON):
                features=np.array(history[-p:][::-1],dtype=float).reshape(1,-1)
                estimate=float(estimator.predict(features)[0]); forecast.append(estimate); history.append(estimate)
            runtime_ms=(time.perf_counter()-started)*1000
            targets=pd.date_range(origin.first_target_timestamp,periods=HORIZON,freq="h")
            actual=series.reindex(targets).to_numpy(float)
            for h,(timestamp,truth,estimate) in enumerate(zip(targets,actual,forecast),1):
                rows.append({"model":"ARIMA","city":city,"meter":meter,"split":"validation",
                             "forecast_origin":origin.forecast_origin,"horizon":h,"target_timestamp":timestamp,
                             "actual":truth,"prediction":estimate,"inference_time_ms":runtime_ms,
                             "run_id":"ARIMA_validation"})
    return pd.DataFrame(rows,columns=PREDICTION_COLUMNS),pd.DataFrame(failures),fit_seconds


def _longest_valid_block(series: pd.Series) -> pd.Series:
    valid = series.notna()
    block_id = (valid != valid.shift(fill_value=False)).cumsum()
    blocks = [group for _, group in series[valid].groupby(block_id[valid])]
    return max(blocks, key=len)


def _holt_winters_forecasts(config, splits, origins, limit_meters=None):
    """Classical additive HW with train-fitted parameters and causal state updates.

    Missing observations cause a prediction-only state advance; they are never
    imputed. Parameters are estimated on the longest contiguous training block,
    while all later available observations update level/trend/seasonal states.
    """
    rows, failures, fit_seconds = [], [], 0.0
    meters = sorted(origins.meter.unique())[:limit_meters or None]
    for meter in meters:
        meter_origins = origins[origins.meter.eq(meter)].sort_values("forecast_origin")
        train = splits["train"][splits["train"].meter.eq(meter)].set_index("x_Timestamp").t_kWh.asfreq("h")
        validation = splits["validation"][splits["validation"].meter.eq(meter)].set_index("x_Timestamp").t_kWh.asfreq("h")
        block = _longest_valid_block(train)
        if len(block) < 72:
            failures.extend({"model":"HoltWinters","meter":meter,"forecast_origin":o,
                             "error":"no contiguous training block >=72h"} for o in meter_origins.forecast_origin)
            continue
        started=time.perf_counter()
        try:
            fitted=ClassicalExponentialSmoothing(
                block, trend="add" if config["trend"] else None, seasonal="add",
                seasonal_periods=24, initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        except Exception as exc:
            failures.extend({"model":"HoltWinters","meter":meter,"forecast_origin":o,
                             "error":f"fit: {exc!r}"} for o in meter_origins.forecast_origin)
            continue
        fit_seconds += time.perf_counter()-started
        alpha=float(fitted.params["smoothing_level"])
        beta=float(fitted.params.get("smoothing_trend") or 0.0)
        gamma=float(fitted.params["smoothing_seasonal"])
        level=float(fitted.level.iloc[-1]); trend=float(fitted.trend.iloc[-1]) if config["trend"] else 0.0
        seasons={timestamp.hour:float(value) for timestamp,value in fitted.season.iloc[-24:].items()}
        combined=pd.concat([train,validation]).sort_index()
        state_time=block.index[-1]
        city=meter_origins.city.iloc[0]
        for origin in meter_origins.itertuples(index=False):
            for timestamp,value in combined.loc[(combined.index>state_time)&(combined.index<=origin.forecast_origin)].items():
                old_level=level; old_season=seasons.get(timestamp.hour,0.0)
                if pd.isna(value):
                    level=level+trend
                else:
                    level=alpha*(float(value)-old_season)+(1-alpha)*(level+trend)
                    if config["trend"]:
                        trend=beta*(level-old_level)+(1-beta)*trend
                    seasons[timestamp.hour]=gamma*(float(value)-level)+(1-gamma)*old_season
                state_time=timestamp
            started=time.perf_counter()
            targets=pd.date_range(origin.first_target_timestamp,periods=HORIZON,freq="h")
            forecast=np.array([level+(h*trend if config["trend"] else 0)+seasons.get(ts.hour,0.0)
                               for h,ts in enumerate(targets,1)])
            runtime_ms=(time.perf_counter()-started)*1000
            actual=validation.reindex(targets).to_numpy(float)
            for h,(timestamp,truth,estimate) in enumerate(zip(targets,actual,forecast),1):
                rows.append({"model":"HoltWinters","city":city,"meter":meter,"split":"validation",
                             "forecast_origin":origin.forecast_origin,"horizon":h,"target_timestamp":timestamp,
                             "actual":truth,"prediction":estimate,"inference_time_ms":runtime_ms,
                             "run_id":"HoltWinters_validation"})
    return pd.DataFrame(rows,columns=PREDICTION_COLUMNS),pd.DataFrame(failures),fit_seconds


def choose_statistical_config(model, candidates, splits, origins):
    calibration_meters = sorted(origins.meter.unique())[:5]
    calibration = origins[origins.meter.isin(calibration_meters)].groupby("meter").head(2)
    scores = []
    for config in candidates:
        predictions, _, runtime = _statistical_forecasts(model, config, splits, calibration)
        rmse = point_metrics(predictions.actual, predictions.prediction)["RMSE"] if len(predictions) else np.inf
        scores.append({"config": config, "validation_RMSE": rmse, "fit_seconds": runtime})
    pd.DataFrame([{**row["config"], "validation_RMSE": row["validation_RMSE"],
                   "fit_seconds": row["fit_seconds"]} for row in scores]).to_csv(
        RUNS / f"{model.lower()}_candidate_screen.csv", index=False)
    return min(scores, key=lambda row: row["validation_RMSE"])["config"]


def run_statistical(model, candidates, splits, origins):
    selected = choose_statistical_config(model, candidates, splits, origins)
    predictions, failures, fit_seconds = _statistical_forecasts(model, selected, splits, origins)
    validate_prediction_schema(predictions)
    save_model_outputs(model, predictions, failures, fit_seconds,
                       {**selected, "selection": "lowest RMSE on first 5 meters, first 2 validation origins"})


def _origin_arrays(splits, origins, include_features=False):
    all_data = pd.concat(list(splits.values()), ignore_index=True)
    frames = {m: g.set_index("x_Timestamp").sort_index() for m, g in all_data.groupby("meter")}
    X, y, metadata = [], [], []
    for origin in origins.sort_values(["meter", "forecast_origin"]).itertuples(index=False):
        frame = frames[origin.meter]
        history_times = pd.date_range(end=origin.forecast_origin, periods=LOOKBACK, freq="h")
        target_times = pd.date_range(origin.first_target_timestamp, periods=HORIZON, freq="h")
        history = frame.reindex(history_times)
        target = frame.reindex(target_times).t_kWh.to_numpy(float)
        if history.t_kWh.isna().any() or np.isnan(target).any():
            continue
        loads = history.t_kWh.to_numpy(float)
        if include_features:
            features = np.column_stack([
                loads, history.hour_sin, history.hour_cos, history.dow_sin,
                history.dow_cos, history.is_weekend,
            ])
        else:
            target_hours = target_times.hour.to_numpy()
            target_dow = target_times.dayofweek.to_numpy()
            future_calendar = np.r_[np.sin(2*np.pi*target_hours/24), np.cos(2*np.pi*target_hours/24),
                                    np.sin(2*np.pi*target_dow/7), np.cos(2*np.pi*target_dow/7)]
            features = np.r_[loads, future_calendar]
        X.append(features); y.append(target); metadata.append(origin)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), metadata


def training_origins(splits):
    # Reuse the validated origin rules, temporarily naming train as validation.
    renamed = {"prior_context": splits["validation"], "validation": splits["train"]}
    origins, _, _ = build_canonical_origins(renamed, "validation")
    origins["split"] = "train"
    return origins


def _records_from_matrix(model, origins_metadata, actual, prediction, runtime_ms, run_id):
    rows = []
    per_origin_runtime = runtime_ms / max(len(origins_metadata), 1)
    for meta, truth, estimate in zip(origins_metadata, actual, prediction):
        targets = pd.date_range(meta.first_target_timestamp, periods=HORIZON, freq="h")
        for h, (timestamp, a, p) in enumerate(zip(targets, truth, estimate), 1):
            rows.append({"model": model, "city": meta.city, "meter": meta.meter, "split": "validation",
                         "forecast_origin": meta.forecast_origin, "horizon": h,
                         "target_timestamp": timestamp, "actual": a, "prediction": p,
                         "inference_time_ms": per_origin_runtime, "run_id": run_id})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def run_linear(splits, origins):
    train_origins = training_origins(splits)
    X_train, y_train, _ = _origin_arrays(splits, train_origins)
    X_val, y_val, meta = _origin_arrays(splits, origins)
    model = LinearRegression()
    started = time.perf_counter(); model.fit(X_train, y_train); train_seconds = time.perf_counter()-started
    started = time.perf_counter(); prediction = model.predict(X_val); runtime_ms=(time.perf_counter()-started)*1000
    records = _records_from_matrix("LinearRegression", meta, y_val, prediction, runtime_ms, "LinearRegression_validation")
    save_model_outputs("LinearRegression", records, pd.DataFrame(), train_seconds,
                       {"strategy": "direct multi-output", "features": "168 loads + target calendar"})


def run_xgboost(splits, origins):
    train_origins = training_origins(splits)
    X_train, y_train, _ = _origin_arrays(splits, train_origins)
    X_val, y_val, meta = _origin_arrays(splits, origins)
    candidates = [
        {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    ]
    screens, fitted = [], []
    total_training = 0.0
    for config in candidates:
        model = XGBRegressor(**config, objective="reg:squarederror", random_state=SEED,
                             n_jobs=8, tree_method="hist", multi_strategy="one_output_per_tree")
        started=time.perf_counter(); model.fit(X_train, y_train); elapsed=time.perf_counter()-started
        total_training += elapsed
        prediction=model.predict(X_val); rmse=point_metrics(y_val.ravel(), prediction.ravel())["RMSE"]
        screens.append({**config, "validation_RMSE": rmse, "training_seconds": elapsed}); fitted.append(model)
    pd.DataFrame(screens).to_csv(RUNS / "xgboost_candidate_screen.csv", index=False)
    selected_index=int(np.argmin([row["validation_RMSE"] for row in screens])); model=fitted[selected_index]
    started=time.perf_counter(); prediction=model.predict(X_val); runtime_ms=(time.perf_counter()-started)*1000
    records=_records_from_matrix("XGBoost", meta, y_val, prediction, runtime_ms, "XGBoost_validation")
    save_model_outputs("XGBoost", records, pd.DataFrame(), total_training,
                       {**candidates[selected_index], "strategy":"direct multi-output", "selection":"validation RMSE"})


class PlainGRU(nn.Module):
    def __init__(self, features=6, hidden=32):
        super().__init__(); self.gru=nn.GRU(features, hidden, batch_first=True); self.head=nn.Linear(hidden,HORIZON)
    def forward(self, x):
        _, hidden=self.gru(x); return self.head(hidden[-1])


def run_gru(splits, origins):
    train_origins=training_origins(splits)
    X_train,y_train,_=_origin_arrays(splits,train_origins,True)
    X_val,y_val,meta=_origin_arrays(splits,origins,True)
    x_mean=float(X_train[:,:,0].mean()); x_std=float(X_train[:,:,0].std()) or 1.0
    y_mean=float(y_train.mean()); y_std=float(y_train.std()) or 1.0
    X_train[:,:,0]=(X_train[:,:,0]-x_mean)/x_std; X_val[:,:,0]=(X_val[:,:,0]-x_mean)/x_std
    y_train_scaled=(y_train-y_mean)/y_std; y_val_scaled=(y_val-y_mean)/y_std
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(TensorDataset(torch.tensor(X_train),torch.tensor(y_train_scaled)),batch_size=128,
                      shuffle=True,generator=generator)
    val_x=torch.tensor(X_val); val_y=torch.tensor(y_val_scaled)
    model=PlainGRU(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.MSELoss()
    history=[]; best_loss=np.inf; best_state=None; patience=4; stale=0
    started=time.perf_counter()
    for epoch in range(1,21):
        model.train(); losses=[]
        for batch_x,batch_y in loader:
            optimizer.zero_grad(); loss=loss_fn(model(batch_x),batch_y); loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val_loss=float(loss_fn(model(val_x),val_y))
        history.append({"epoch":epoch,"train_MSE_scaled":float(np.mean(losses)),"validation_MSE_scaled":val_loss})
        if val_loss < best_loss-1e-6:
            best_loss=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    train_seconds=time.perf_counter()-started; model.load_state_dict(best_state)
    torch.save(best_state,CHECKPOINTS/"gru_validation_checkpoint.pt")
    model.eval(); started=time.perf_counter()
    with torch.no_grad(): prediction=model(val_x).numpy()*y_std+y_mean
    runtime_ms=(time.perf_counter()-started)*1000
    records=_records_from_matrix("GRU",meta,y_val,prediction,runtime_ms,"GRU_validation")
    save_model_outputs("GRU",records,pd.DataFrame(),train_seconds,
                       {"hidden_size":32,"layers":1,"learning_rate":1e-3,"batch_size":128,
                        "max_epochs":20,"patience":patience,"parameter_count":sum(p.numel() for p in model.parameters()),
                        "scaler_fit":"train only","architecture":"unidirectional direct 168-to-24"},history)


def attempt_sarima(splits, origins):
    config={"order":[1,0,0],"seasonal_order":[1,0,0,24],"maxiter":20}
    sample=origins[origins.meter.eq(sorted(origins.meter.unique())[0])].head(2)
    started=time.perf_counter(); predictions,failures,fit_seconds=_statistical_forecasts("SARIMA",config,splits,sample,1)
    elapsed=time.perf_counter()-started
    result={"status":"limited_runtime_attempt","configuration":config,"meters":1,
            "requested_origins":len(sample),"successful_origins":int(len(predictions)/24),
            "failures":len(failures),"fit_seconds":fit_seconds,"wall_seconds":elapsed,
            "included_in_master_comparison":False,
            "reason":"Full 62-meter run deferred if measured single-meter fit is excessive for Review 3."}
    (RUNS/"sarima_runtime_attempt.json").write_text(json.dumps(result,indent=2)+"\n")
    if len(predictions): predictions.to_csv(PREDICTIONS/"sarima_limited_predictions.csv",index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",nargs="+",default=["seasonal","holtwinters","arima","linear","xgboost","gru","sarima_attempt"])
    args=parser.parse_args(); prepare_dirs(); seed_everything(); splits,origins=load_contract_data()
    actions={
        "seasonal":lambda:run_seasonal_naive(splits,origins),
        "holtwinters":lambda:run_statistical("HoltWinters",[
            {"trend":False},{"trend":True}],splits,origins),
        "arima":lambda:run_statistical("ARIMA",[
            {"order":[1,0,0]},{"order":[2,0,0]},{"order":[3,0,0]}],splits,origins),
        "linear":lambda:run_linear(splits,origins), "xgboost":lambda:run_xgboost(splits,origins),
        "gru":lambda:run_gru(splits,origins), "sarima_attempt":lambda:attempt_sarima(splits,origins),
    }
    for name in args.models:
        print(f"Running {name}...",flush=True); actions[name]()


if __name__=="__main__": main()
