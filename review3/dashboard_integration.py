"""Testable integration helpers shared by the Review-3 Streamlit dashboard."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from evaluation.contract import HORIZON, LOOKBACK, TRAIN_PATH
from review3.demand_side_optimisation import (APPLIANCES,baseline_schedule,calculate_metrics,
 load_forecast,optimize_schedule,schedule_table,tariff_table)

ROOT=Path(__file__).resolve().parents[1]
DASHBOARD_OUTPUT=ROOT/"reports"/"review3"/"dashboard_integration"
MODEL_PATH=DASHBOARD_OUTPUT/"checkpoints"/"linear_regression_train_only.joblib"
MODEL_MANIFEST=DASHBOARD_OUTPUT/"runs"/"linear_regression_train_only_manifest.json"

def validate_uploaded_history(source):
    frame=pd.read_csv(source)
    if not {"timestamp","kwh"}.issubset(frame.columns): raise ValueError("CSV must contain columns named exactly 'timestamp' and 'kwh'.")
    try: frame["timestamp"]=pd.to_datetime(frame.timestamp,errors="raise")
    except Exception as exc: raise ValueError("The timestamp column contains an unparseable value.") from exc
    frame["kwh"]=pd.to_numeric(frame.kwh,errors="coerce")
    if frame.kwh.isna().any(): raise ValueError("The kwh column must contain valid numeric values.")
    if (frame.kwh<0).any(): raise ValueError("Negative kwh values are not accepted.")
    if frame.timestamp.duplicated().any(): raise ValueError("Duplicate timestamps are not allowed.")
    frame=frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame)<LOOKBACK: raise ValueError(f"At least {LOOKBACK} hourly observations are required; received {len(frame)}.")
    breaks=frame.timestamp.diff().fillna(pd.Timedelta(hours=1)).ne(pd.Timedelta(hours=1))
    frame["segment"]=breaks.cumsum()
    segments=[part for _,part in frame.groupby("segment",sort=True) if len(part)>=LOOKBACK]
    if not segments:
        longest=int(frame.groupby("segment").size().max())
        raise ValueError(f"No continuous {LOOKBACK}-hour block exists. The longest valid hourly block has {longest} rows; gaps are not interpolated.")
    chosen=segments[-1]
    history=chosen.set_index("timestamp").kwh.astype(float)
    history.attrs.update({"uploaded_rows":len(frame),"usable_segment_rows":len(chosen),
                          "discarded_rows":len(frame)-len(chosen),"gap_count":int(breaks.sum()),
                          "segment_start":chosen.timestamp.iloc[0],"segment_end":chosen.timestamp.iloc[-1]})
    return history

def _training_examples():
    """Reproduce the frozen direct Linear Regression training contract on TRAIN only."""
    columns=["x_Timestamp","t_kWh","meter"]
    train=pd.read_csv(TRAIN_PATH,usecols=columns,parse_dates=["x_Timestamp"])
    X,y=[],[]
    for _,part in train.groupby("meter",sort=True):
        series=part.set_index("x_Timestamp").t_kWh.sort_index()
        if series.index.has_duplicates: raise ValueError("Authoritative training data contains duplicate timestamps.")
        # Daily stride matches the research origin contract while requiring complete history/target windows.
        for first_target in series.index[::24]:
            history_times=pd.date_range(end=first_target-pd.Timedelta(hours=1),periods=LOOKBACK,freq="h")
            target_times=pd.date_range(start=first_target,periods=HORIZON,freq="h")
            history=series.reindex(history_times).to_numpy(float)
            target=series.reindex(target_times).to_numpy(float)
            if np.isnan(history).any() or np.isnan(target).any(): continue
            hours=target_times.hour.to_numpy(); dow=target_times.dayofweek.to_numpy()
            calendar=np.r_[np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24),
                           np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]
            X.append(np.r_[history,calendar]); y.append(target)
    if not X: raise RuntimeError("No valid training windows were generated.")
    return np.asarray(X,np.float32),np.asarray(y,np.float32)

def build_dashboard_forecaster(force=False):
    """Build the experimental global direct-24 model without validation/test data."""
    if MODEL_PATH.exists() and not force: return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True); MODEL_MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    X,y=_training_examples(); model=LinearRegression().fit(X,y)
    artifact={"model":model,"lookback":LOOKBACK,"horizon":HORIZON,
              "features":"168 hourly loads + 24 target-hour and target-weekday cyclical values",
              "trained_on":str(TRAIN_PATH),"training_windows":len(X)}
    joblib.dump(artifact,MODEL_PATH)
    manifest={"model":"LinearRegression","purpose":"experimental dashboard inference",
              "research_comparison_eligible":False,"training_data":str(TRAIN_PATH),
              "validation_data_used":False,"test_data_used":False,"lookback":LOOKBACK,
              "horizon":HORIZON,"training_windows":len(X),"created_at_utc":datetime.now(timezone.utc).isoformat()}
    MODEL_MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
    return MODEL_PATH

def forecast_uploaded_history(history,model_path=MODEL_PATH):
    """Forecast the next 24 hours from the latest 168 observed hourly values."""
    if len(history)<LOOKBACK: raise ValueError(f"Forecasting requires at least {LOOKBACK} continuous hourly observations.")
    if not model_path.exists(): build_dashboard_forecaster()
    artifact=joblib.load(model_path)
    recent=history.iloc[-LOOKBACK:]
    expected=pd.date_range(end=recent.index[-1],periods=LOOKBACK,freq="h")
    if not recent.index.equals(expected): raise ValueError("The selected forecasting context is not exactly 168 continuous hourly observations.")
    target_times=pd.date_range(recent.index[-1]+pd.Timedelta(hours=1),periods=HORIZON,freq="h")
    hours=target_times.hour.to_numpy(); dow=target_times.dayofweek.to_numpy()
    calendar=np.r_[np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24),
                   np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]
    features=np.r_[recent.to_numpy(float),calendar].reshape(1,-1)
    raw=np.asarray(artifact["model"].predict(features)[0],float)
    if raw.shape!=(HORIZON,) or not np.isfinite(raw).all(): raise RuntimeError("Uploaded-history forecast is not 24 finite values.")
    # Energy cannot be negative. Preserve raw values so this deployment safeguard is auditable.
    adjusted=np.maximum(raw,0.0)
    forecast=pd.DataFrame({"timestamp":target_times,"raw_prediction_kwh":raw,"predicted_kwh":adjusted})
    metadata={"model":"LinearRegression (experimental global inference)","forecast_origin":recent.index[-1],
              "horizon_hours":HORIZON,"context_start":recent.index[0],"context_end":recent.index[-1],
              "negative_predictions_clipped":int((raw<0).sum()),"training_data":"authoritative research train split only",
              "research_status":"Experimental deployment demonstration; not research-comparison evidence"}
    return forecast,metadata

def load_validated_dashboard_example():
    source=load_forecast()
    forecast=pd.DataFrame({"timestamp":source.target_timestamp,"predicted_kwh":source.prediction})
    metadata={"model":source.model.iloc[0],"meter":source.meter.iloc[0],"city":source.city.iloc[0],
              "forecast_origin":source.forecast_origin.iloc[0],"horizon_hours":len(source),
              "research_status":"Validation-stage saved Review-3 example"}
    return forecast,metadata

def run_validated_optimisation(forecast):
    if len(forecast)!=24: raise ValueError("Optimisation requires exactly 24 forecast hours.")
    timestamps=pd.DatetimeIndex(pd.to_datetime(forecast.timestamp)); base=np.asarray(forecast.predicted_kwh,float)
    tariff=tariff_table(timestamps); prices=tariff.price_inr_per_kwh.to_numpy(float)
    before=baseline_schedule(timestamps); after,result,_=optimize_schedule(base,prices,timestamps)
    bm,bf,bt=calculate_metrics(base,before,prices); am,af,at=calculate_metrics(base,after,prices)
    metrics={"cost_before":bm["electricity_cost_inr"],"cost_after":am["electricity_cost_inr"],"cost_saving":bm["electricity_cost_inr"]-am["electricity_cost_inr"],
      "cost_reduction_pct":100*(bm["electricity_cost_inr"]-am["electricity_cost_inr"])/bm["electricity_cost_inr"],
      "peak_before":bm["peak_load_kw_equivalent"],"peak_after":am["peak_load_kw_equivalent"],"peak_reduction_pct":100*(bm["peak_load_kw_equivalent"]-am["peak_load_kw_equivalent"])/bm["peak_load_kw_equivalent"],
      "par_before":bm["PAR"],"par_after":am["PAR"],"par_change_pct":100*(am["PAR"]-bm["PAR"])/bm["PAR"],
      "shifted_energy_kwh":.5*float(np.abs(af-bf).sum()),"flexible_before_kwh":bm["flexible_energy_kwh"],"flexible_after_kwh":am["flexible_energy_kwh"],
      "flexible_difference_kwh":am["flexible_energy_kwh"]-bm["flexible_energy_kwh"],"solver_status":"Optimal" if result.success else "Failed",
      "solver_message":result.message,"objective_value":result.fun}
    hourly=pd.DataFrame({"timestamp":timestamps,"clock_hour":timestamps.hour,"base_forecast":base,"tariff":prices,
      "flexible_before":bf,"flexible_after":af,"total_before":bt,"total_after":at})
    return {"metrics":metrics,"hourly":hourly,"appliance_schedule":schedule_table(before,after,timestamps),"tariff":tariff,"before_matrix":before,"after_matrix":after}
