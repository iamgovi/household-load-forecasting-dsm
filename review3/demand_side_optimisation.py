"""Simulation-based flexible appliance scheduling for the Review-3 demonstration.

The smart-meter data are aggregate. Appliance profiles and the tariff below
are explicitly illustrative simulation inputs, not measurements or official rates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"reports"/"review3"/"demand_side_optimisation"
TABLES,FIGURES,SCHEDULES,RUNS,CONFIGS=[OUTPUT/x for x in ("tables","figures","schedules","runs","configs")]
FORECAST_FILE=ROOT/"reports"/"review3"/"individual_models"/"predictions"/"all_models_common_origin_predictions.csv"
FORECAST_MODEL="LinearRegression"
HOUSEHOLD_MAX_POWER_KW=3.5
COST_WEIGHT=0.3
PEAK_WEIGHT=0.7

# All values are ILLUSTRATIVE / SIMULATED. latest_finish is exclusive.
APPLIANCES=[
 {"name":"Washing Machine","power_kw":0.8,"duration_h":2,"earliest_start":8,"latest_finish":20,"interruptible":False,"preferred_start":9,"source_class":"ILLUSTRATIVE / SIMULATED"},
 {"name":"Dishwasher","power_kw":1.0,"duration_h":2,"earliest_start":18,"latest_finish":24,"interruptible":False,"preferred_start":19,"source_class":"ILLUSTRATIVE / SIMULATED"},
 {"name":"Water Heater","power_kw":2.0,"duration_h":1,"earliest_start":5,"latest_finish":9,"interruptible":False,"preferred_start":7,"source_class":"ILLUSTRATIVE / SIMULATED"},
 {"name":"Water Pump","power_kw":1.0,"duration_h":1,"earliest_start":6,"latest_finish":18,"interruptible":False,"preferred_start":12,"source_class":"ILLUSTRATIVE / SIMULATED"},
]


def tariff_table(timestamps=None):
    clock_hours=list(range(24)) if timestamps is None else pd.DatetimeIndex(pd.to_datetime(timestamps)).hour.tolist()
    rows=[]
    for slot,hour in enumerate(clock_hours):
        if hour<6: category,price="Off-peak",4.0
        elif hour<18: category,price="Normal",6.0
        elif hour<22: category,price="Peak",9.0
        else: category,price="Off-peak",4.5
        rows.append({"schedule_slot":slot,"clock_hour":hour,"category":category,"price_inr_per_kwh":price,"source_class":"ILLUSTRATIVE INDIAN TIME-OF-USE SCENARIO"})
    return pd.DataFrame(rows)


def load_forecast():
    predictions=pd.read_csv(FORECAST_FILE,parse_dates=["forecast_origin","target_timestamp"])
    subset=predictions[predictions.model.eq(FORECAST_MODEL)].sort_values(["meter","forecast_origin","horizon"])
    first=subset[["city","meter","forecast_origin"]].drop_duplicates().iloc[0]
    selected=subset[(subset.city.eq(first.city))&(subset.meter.eq(first.meter))&(subset.forecast_origin.eq(first.forecast_origin))].sort_values("horizon")
    if len(selected)!=24 or selected.horizon.tolist()!=list(range(1,25)): raise ValueError("Selected corrected forecast is not a complete 24-hour profile")
    return selected.reset_index(drop=True)


def feasible_start_slots(appliance,timestamps):
    timestamps=pd.DatetimeIndex(timestamps); starts=[]
    for slot in range(24-appliance["duration_h"]+1):
        clock=timestamps[slot].hour
        if clock<appliance["earliest_start"] or clock+appliance["duration_h"]>appliance["latest_finish"]: continue
        expected=[clock+i for i in range(appliance["duration_h"])]
        if [timestamps[slot+i].hour for i in range(appliance["duration_h"])]==expected: starts.append(slot)
    return starts


def baseline_schedule(timestamps):
    schedule=np.zeros((len(APPLIANCES),24),dtype=int)
    for a,appliance in enumerate(APPLIANCES):
        candidates=feasible_start_slots(appliance,timestamps)
        matches=[slot for slot in candidates if pd.Timestamp(timestamps[slot]).hour==appliance["preferred_start"]]
        if len(matches)!=1: raise ValueError(f"Invalid preferred baseline for {appliance['name']}")
        slot=matches[0]; schedule[a,slot:slot+appliance["duration_h"]]=1
    return schedule


def profile(schedule):
    powers=np.array([a["power_kw"] for a in APPLIANCES])
    return powers@schedule


def optimize_schedule(base_load,tariff,timestamps):
    """Joint MILP with on/off x[a,t], start s[a,k], and continuous peak."""
    n_a,n_t=len(APPLIANCES),24; x_count=n_a*n_t
    starts=[]
    for a,ap in enumerate(APPLIANCES):
        for start in feasible_start_slots(ap,timestamps): starts.append((a,start))
    start_offset=x_count; peak_index=x_count+len(starts); n=peak_index+1
    c=np.zeros(n); baseline=baseline_schedule(timestamps); baseline_total=base_load+profile(baseline)
    baseline_cost=float(np.sum(baseline_total*tariff));
    for a,ap in enumerate(APPLIANCES):
        for t in range(n_t): c[a*n_t+t]=COST_WEIGHT*ap["power_kw"]*tariff[t]/baseline_cost
    c[peak_index]=PEAK_WEIGHT/HOUSEHOLD_MAX_POWER_KW
    rows=[]; lower=[]; upper=[]
    # Exactly one feasible start per non-interruptible appliance.
    for a in range(n_a):
        row={start_offset+i:1 for i,(aa,_) in enumerate(starts) if aa==a}; rows.append(row); lower.append(1); upper.append(1)
    # Link every on/off variable to covering start variables; this also forces zero outside the window.
    for a,ap in enumerate(APPLIANCES):
        for t in range(n_t):
            row={a*n_t+t:1}
            for i,(aa,start) in enumerate(starts):
                if aa==a and start<=t<start+ap["duration_h"]: row[start_offset+i]=row.get(start_offset+i,0)-1
            rows.append(row); lower.append(0); upper.append(0)
    # Total-load ceiling and peak epigraph.
    for t in range(n_t):
        load_terms={a*n_t+t:APPLIANCES[a]["power_kw"] for a in range(n_a)}
        rows.append(load_terms); lower.append(-np.inf); upper.append(HOUSEHOLD_MAX_POWER_KW-base_load[t])
        peak_row=dict(load_terms); peak_row[peak_index]=-1
        rows.append(peak_row); lower.append(-np.inf); upper.append(-base_load[t])
    A=lil_matrix((len(rows),n),dtype=float)
    for r,row in enumerate(rows):
        for col,value in row.items(): A[r,col]=value
    lb=np.zeros(n); ub=np.ones(n); ub[peak_index]=HOUSEHOLD_MAX_POWER_KW
    integrality=np.ones(n); integrality[peak_index]=0
    result=milp(c,integrality=integrality,bounds=Bounds(lb,ub),constraints=LinearConstraint(A.tocsr(),lower,upper),options={"disp":False})
    if not result.success: raise RuntimeError(f"MILP failed: {result.message}")
    schedule=np.rint(result.x[:x_count]).astype(int).reshape(n_a,n_t)
    return schedule,result,starts


def calculate_metrics(base,schedule,tariff):
    flexible=profile(schedule); total=base+flexible; cost=float(np.sum(total*tariff)); peak=float(total.max()); mean=float(total.mean())
    return {"electricity_cost_inr":cost,"peak_load_kw_equivalent":peak,"PAR":peak/mean,"flexible_energy_kwh":float(flexible.sum()),
            "base_load_energy_kwh":float(base.sum()),"total_energy_kwh":float(total.sum())},flexible,total


def schedule_table(before,after,timestamps):
    timestamps=pd.DatetimeIndex(timestamps)
    rows=[]
    for a,ap in enumerate(APPLIANCES):
        bh=np.flatnonzero(before[a]).tolist(); ah=np.flatnonzero(after[a]).tolist()
        rows.append({"Appliance":ap["name"],"Power_kW":ap["power_kw"],"Duration_h":ap["duration_h"],
          "Allowed_Window":f"{ap['earliest_start']:02d}:00–{ap['latest_finish']:02d}:00","Interruptible":ap["interruptible"],
          "Baseline_Start":timestamps[bh[0]].hour,"Optimised_Start":timestamps[ah[0]].hour,
          "Baseline_Hours":",".join(str(timestamps[h].hour) for h in bh),"Optimised_Hours":",".join(str(timestamps[h].hour) for h in ah),
          "Baseline_Slots":",".join(map(str,bh)),"Optimised_Slots":",".join(map(str,ah)),
          "Energy_kWh":ap["power_kw"]*ap["duration_h"],"Assumption_Source":ap["source_class"]})
    return pd.DataFrame(rows)


def main():
    for p in (TABLES,FIGURES,SCHEDULES,RUNS,CONFIGS): p.mkdir(parents=True,exist_ok=True)
    forecast=load_forecast(); base=forecast.prediction.to_numpy(float); timestamps=pd.DatetimeIndex(forecast.target_timestamp)
    tariff_df=tariff_table(timestamps); tariff=tariff_df.price_inr_per_kwh.to_numpy(float)
    before=baseline_schedule(timestamps); after,result,_=optimize_schedule(base,tariff,timestamps)
    before_metrics,before_flex,before_total=calculate_metrics(base,before,tariff)
    after_metrics,after_flex,after_total=calculate_metrics(base,after,tariff)
    shifted=.5*float(np.abs(after_flex-before_flex).sum())
    summary=pd.DataFrame([
      {"Metric":"Electricity Cost","Unit":"INR","Before":before_metrics["electricity_cost_inr"],"After":after_metrics["electricity_cost_inr"],"Change":after_metrics["electricity_cost_inr"]-before_metrics["electricity_cost_inr"],"Change_pct":100*(after_metrics["electricity_cost_inr"]-before_metrics["electricity_cost_inr"])/before_metrics["electricity_cost_inr"]},
      {"Metric":"Peak Load","Unit":"kW equivalent hourly average","Before":before_metrics["peak_load_kw_equivalent"],"After":after_metrics["peak_load_kw_equivalent"],"Change":after_metrics["peak_load_kw_equivalent"]-before_metrics["peak_load_kw_equivalent"],"Change_pct":100*(after_metrics["peak_load_kw_equivalent"]-before_metrics["peak_load_kw_equivalent"])/before_metrics["peak_load_kw_equivalent"]},
      {"Metric":"Peak-to-Average Ratio","Unit":"ratio","Before":before_metrics["PAR"],"After":after_metrics["PAR"],"Change":after_metrics["PAR"]-before_metrics["PAR"],"Change_pct":100*(after_metrics["PAR"]-before_metrics["PAR"])/before_metrics["PAR"]},
      {"Metric":"Flexible Energy","Unit":"kWh","Before":before_metrics["flexible_energy_kwh"],"After":after_metrics["flexible_energy_kwh"],"Change":after_metrics["flexible_energy_kwh"]-before_metrics["flexible_energy_kwh"],"Change_pct":0.0},
      {"Metric":"Shifted Energy","Unit":"kWh","Before":np.nan,"After":shifted,"Change":shifted,"Change_pct":np.nan},
    ]); summary.to_csv(TABLES/"optimisation_summary.csv",index=False)
    appliance=schedule_table(before,after,timestamps); appliance.to_csv(TABLES/"appliance_schedule.csv",index=False)
    profile_df=pd.DataFrame({"schedule_slot":range(24),"clock_hour":timestamps.hour,"target_timestamp":forecast.target_timestamp,"base_forecast_kwh":base,"tariff_inr_per_kwh":tariff,
      "flexible_before_kwh":before_flex,"flexible_after_kwh":after_flex,"total_before_kwh":before_total,"total_after_kwh":after_total})
    profile_df.to_csv(SCHEDULES/"hourly_before_after_schedule.csv",index=False); forecast.to_csv(SCHEDULES/"source_forecast_24h.csv",index=False); tariff_df.to_csv(CONFIGS/"illustrative_tariff.csv",index=False)
    (CONFIGS/"illustrative_appliances.json").write_text(json.dumps(APPLIANCES,indent=2)+"\n")
    manifest={"experiment_label":"Simulation-based flexible appliance scheduling","created_at_utc":datetime.now(timezone.utc).isoformat(),
      "forecast_model":FORECAST_MODEL,"forecast_file":str(FORECAST_FILE),"city":forecast.city.iloc[0],"meter":forecast.meter.iloc[0],
      "forecast_origin":str(forecast.forecast_origin.iloc[0]),"forecast_target_start":str(forecast.target_timestamp.iloc[0]),"forecast_target_end":str(forecast.target_timestamp.iloc[-1]),
      "base_load_assumption":"Saved aggregate-load forecast is treated as the counterfactual non-flexible background for this simulation; appliance separation is not measured.",
      "tariff_label":"Illustrative Indian Time-of-Use tariff scenario; not an official verified tariff","appliance_label":"ILLUSTRATIVE / SIMULATED",
      "solver":"SciPy milp using HiGHS","solver_status":"Optimal" if result.success else "Failed","solver_message":result.message,"solver_objective":result.fun,
      "objective":"0.3*(total cost/baseline total cost) + 0.7*(total household peak/3.5 kW)","household_max_power_kw":HOUSEHOLD_MAX_POWER_KW,
      "baseline_rule":"Each appliance starts at its illustrative preferred time, all within its allowed window.","cost_weight":COST_WEIGHT,"peak_weight":PEAK_WEIGHT,
      "base_load_energy_kwh":before_metrics["base_load_energy_kwh"],"flexible_energy_kwh":before_metrics["flexible_energy_kwh"],"total_energy_kwh":before_metrics["total_energy_kwh"],
      "shifted_energy_definition":"0.5 * sum(abs(flexible_after - flexible_before)) for 1-hour bins","renewable_metrics_included":False,"carbon_metrics_included":False,
      "time_alignment":"Every schedule slot, tariff value and appliance window is mapped to the actual saved forecast target timestamp/clock hour.",
      "test_data_used_for_model_selection":False}
    (RUNS/"optimisation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.savez_compressed(RUNS/"schedule_arrays.npz",before=before,after=after,base=base,tariff=tariff)
    print(summary.to_string(index=False)); print("\n",appliance.to_string(index=False))

if __name__=="__main__": main()
