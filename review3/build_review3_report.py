"""Build fair common-origin Review-3 tables, figures and narrative."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.comparison import common_origin_predictions
from evaluation.metrics import evaluate_predictions, per_origin_metrics, validate_prediction_schema
from review3.individual_models import FIGURES, METRICS, MODEL_ORDER, OUTPUT, PREDICTIONS, RUNS, TABLES


def load_predictions():
    frames=[]
    for model in MODEL_ORDER:
        path=PREDICTIONS/f"{model.lower()}_predictions.csv"
        frame=pd.read_csv(path,parse_dates=["forecast_origin","target_timestamp"])
        validate_prediction_schema(frame); frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def manifests():
    return {model:json.loads((RUNS/f"{model.lower()}_manifest.json").read_text()) for model in MODEL_ORDER}


def build_tables(all_predictions, run_info):
    common,availability=common_origin_predictions(all_predictions,MODEL_ORDER)
    common.to_csv(PREDICTIONS/"all_models_common_origin_predictions.csv",index=False)
    availability.to_csv(TABLES/"forecast_availability.csv",index=False)
    evaluated=evaluate_predictions(common)
    evaluated["per_meter"].to_csv(TABLES/"per_meter_comparison.csv",index=False)
    evaluated["per_horizon"].to_csv(TABLES/"per_horizon_metrics.csv",index=False)

    pooled=evaluated["overall"].query("aggregation == 'pooled_predictions'").copy()
    pooled["Train_Time_s"]=pooled.model.map({m:v["training_time_seconds"] for m,v in run_info.items()})
    pooled["Meters"]=pooled.model.map({m:v["meters"] for m,v in run_info.items()})
    pooled["Forecast_Origins"]=pooled.model.map({m:v["successful_origins"] for m,v in run_info.items()})
    comparison=pooled[["model","Meters","Forecast_Origins","MAE","RMSE","sMAPE_pct","R2","NRMSE_mean",
                       "peak_load_error","normalized_peak_error","peak_timing_error_hours",
                       "Train_Time_s","inference_time_ms"]].rename(columns={
        "model":"Model","sMAPE_pct":"sMAPE_pct","peak_load_error":"Peak_Error",
        "normalized_peak_error":"Normalized_Peak_Error","peak_timing_error_hours":"Peak_Timing_Error_h",
        "inference_time_ms":"Inference_Time_ms"})
    comparison["Model"]=pd.Categorical(comparison.Model,MODEL_ORDER,ordered=True)
    comparison=comparison.sort_values("Model"); comparison.to_csv(TABLES/"individual_model_comparison.csv",index=False)
    for model,row in comparison.groupby("Model",observed=True):
        row.to_csv(TABLES/f"{str(model).lower()}_summary.csv",index=False)

    coverage=[]
    for model in MODEL_ORDER:
        frame=all_predictions[all_predictions.model.eq(model)]
        coverage.append({"Model":model,"Meters":frame.meter.nunique(),
                         "Raw_Successful_Origins":frame.groupby(["meter","forecast_origin"]).ngroups,
                         "Failed_Origins":run_info[model]["failed_origins"]})
    common_origins=common[["city","meter","split","forecast_origin"]].drop_duplicates()
    coverage_df=pd.DataFrame(coverage); coverage_df["Common_Origins"]=len(common_origins)
    coverage_df["Common_Meters"]=common_origins.meter.nunique()
    coverage_df.to_csv(TABLES/"model_coverage.csv",index=False)
    return common,evaluated,comparison,coverage_df


def paired_differences(common):
    origin=per_origin_metrics(common)
    rows=[]
    keys=["city","meter","split","forecast_origin"]
    for a,b in combinations(MODEL_ORDER,2):
        left=origin[origin.model.eq(a)][keys+["MAE","RMSE"]]
        right=origin[origin.model.eq(b)][keys+["MAE","RMSE"]]
        paired=left.merge(right,on=keys,suffixes=("_A","_B"))
        rows.append({"Model_A":a,"Model_B":b,"Paired_Origins":len(paired),
                     "MAE_A":paired.MAE_A.mean(),"MAE_B":paired.MAE_B.mean(),
                     "Absolute_MAE_Difference_A_minus_B":(paired.MAE_A-paired.MAE_B).mean(),
                     "Relative_MAE_Improvement_B_vs_A_pct":100*(paired.MAE_A.mean()-paired.MAE_B.mean())/paired.MAE_A.mean(),
                     "RMSE_A":paired.RMSE_A.mean(),"RMSE_B":paired.RMSE_B.mean(),
                     "Absolute_RMSE_Difference_A_minus_B":(paired.RMSE_A-paired.RMSE_B).mean(),
                     "Relative_RMSE_Improvement_B_vs_A_pct":100*(paired.RMSE_A.mean()-paired.RMSE_B.mean())/paired.RMSE_A.mean()})
    pd.DataFrame(rows).to_csv(TABLES/"paired_model_differences.csv",index=False)


def save_profile_figures(common):
    origins=common[["city","meter","forecast_origin"]].drop_duplicates().sort_values(["meter","forecast_origin"])
    representative=origins.iloc[0]
    for model in MODEL_ORDER:
        frame=common[(common.model==model)&(common.meter==representative.meter)&
                     (common.forecast_origin==representative.forecast_origin)].sort_values("horizon")
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(frame.horizon,frame.actual,marker="o",label="Actual")
        ax.plot(frame.horizon,frame.prediction,marker="o",label="Predicted")
        ax.set(xlabel="Forecast horizon (hour)",ylabel="Hourly energy (kWh*)",
               title=f"{model}: {representative.meter}, origin {representative.forecast_origin}")
        ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/f"{model.lower()}_actual_vs_predicted.png",dpi=180); plt.close(fig)
    (RUNS/"representative_origin.json").write_text(json.dumps({
        "selection_rule":"lexicographically first meter and earliest origin in all-model intersection",
        "city":representative.city,"meter":representative.meter,
        "forecast_origin":str(representative.forecast_origin)},indent=2)+"\n")


def save_comparison_figures(comparison,evaluated):
    specs=[("MAE","MAE"),("RMSE","RMSE"),("sMAPE_pct","sMAPE (%)"),("R2","R²"),
           ("NRMSE_mean","NRMSE"),("Peak_Error","Peak error (kWh*)"),
           ("Train_Time_s","Training/selection time (s)"),("Inference_Time_ms","Inference time/origin (ms)")]
    names=comparison.Model.astype(str)
    for column,label in specs:
        fig,ax=plt.subplots(figsize=(9,5)); ax.bar(names,comparison[column]); ax.set_ylabel(label); ax.set_title(f"Validation {label} by model")
        ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(FIGURES/f"comparison_{column.lower()}.png",dpi=180); plt.close(fig)
    horizon=evaluated["per_horizon"]
    fig,ax=plt.subplots(figsize=(10,5))
    for model,group in horizon.groupby("model"):
        ax.plot(group.horizon,group.MAE,marker="o",label=model)
    ax.set(xlabel="Forecast horizon (hour)",ylabel="MAE (kWh*)",title="MAE by forecast horizon"); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES/"per_horizon_mae.png",dpi=180); plt.close(fig)
    meter=evaluated["per_meter"]
    values=[meter[meter.model.eq(model)].RMSE for model in MODEL_ORDER]
    fig,ax=plt.subplots(figsize=(10,5)); ax.boxplot(values,tick_labels=MODEL_ORDER,showfliers=False)
    ax.set(ylabel="Per-meter RMSE (kWh*)",title="Household-level validation RMSE distribution"); ax.tick_params(axis="x",rotation=30)
    fig.tight_layout(); fig.savefig(FIGURES/"per_meter_rmse_distribution.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,2.8)); ax.axis("off")
    display=comparison[["Model","MAE","RMSE","sMAPE_pct","R2","NRMSE_mean","Peak_Error","Train_Time_s","Inference_Time_ms"]].copy()
    for col in display.columns[1:]: display[col]=display[col].map(lambda x:f"{x:.4f}")
    table=ax.table(cellText=display.values,colLabels=display.columns,loc="center",cellLoc="center"); table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1,1.5)
    fig.tight_layout(); fig.savefig(FIGURES/"individual_model_comparison_table.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def write_summary(comparison,coverage):
    lowest_rmse=comparison.loc[comparison.RMSE.idxmin()]
    lowest_mae=comparison.loc[comparison.MAE.idxmin()]
    lowest_smape=comparison.loc[comparison.sMAPE_pct.idxmin()]
    sarima=json.loads((RUNS/"sarima_runtime_attempt.json").read_text())
    text=f"""# Individual models — Review 3

## Models implemented

Daily Seasonal Naive, Holt–Winters, conditional ARIMA, direct multi-output Linear Regression,
direct multi-output XGBoost, and a plain direct 168-to-24 GRU were evaluated. SARIMA was
implemented and attempted on one meter but excluded from the master comparison.

## Forecast protocol and validation data

Every included model predicts horizons 1--24 at the same {int(coverage.Common_Origins.iloc[0])}
canonical validation origins across {int(coverage.Common_Meters.iloc[0])} meters. No test predictions,
interpolation or future filling were used. The representative graph origin is selected before model
comparison as the lexicographically first common meter/origin.

## Main validation findings

{lowest_rmse.Model} achieved the lowest pooled validation RMSE ({lowest_rmse.RMSE:.4f}) among the
individual models under the current common protocol. {lowest_mae.Model} had the lowest MAE
({lowest_mae.MAE:.4f}), while {lowest_smape.Model} had the lowest sMAPE ({lowest_smape.sMAPE_pct:.2f}%).
These differing rankings show why no model should be selected from RMSE alone.

Per-horizon and per-meter tables/figures preserve forecast-distance behavior and household variability.
Training/selection runtime and per-origin inference runtime are included in the comparison table.

## SARIMA limitation

The SARIMA{tuple(sarima['configuration']['order'])} x {tuple(sarima['configuration']['seasonal_order'])}
attempt succeeded for {sarima['successful_origins']} origins on one meter, with {sarima['fit_seconds']:.2f}s
fit time. A full run was deferred for Review 3 and SARIMA is not mixed into the fair comparison.

## Current limitations and next step

Results are validation-stage preliminary evidence. `t_kWh` semantics remain authoritatively unverified.
ARIMA is specifically a conditional-least-squares ARIMA(3,0,0) selected from p=1,2,3—not a general
ARIMA order search. The next milestone may reconstruct the residual ARIMA-GRU hybrid, using these
individual benchmarks and the same origins/evaluator; test evaluation remains frozen until selection.
"""
    (OUTPUT/"INDIVIDUAL_MODELS_REVIEW3.md").write_text(text)


def main():
    all_predictions=load_predictions(); info=manifests()
    common,evaluated,comparison,coverage=build_tables(all_predictions,info)
    paired_differences(common); save_profile_figures(common); save_comparison_figures(comparison,evaluated)
    write_summary(comparison,coverage)
    print(comparison.to_string(index=False)); print(coverage.to_string(index=False))


if __name__=="__main__": main()
