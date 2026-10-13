"""Build common-origin tables, paired diagnostics and figures for ARIMA-GRU."""
from __future__ import annotations

import json
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.comparison import common_origin_predictions
from evaluation.metrics import evaluate_predictions, point_metrics
from review3.arima_gru_hybrid import FIGURES, METRICS, OUTPUT, PREDICTIONS, RUNS, TABLES

INDIVIDUAL = OUTPUT.parent / "individual_models"
MODELS = ["SeasonalNaive_Daily","ARIMA","GRU","LinearRegression","XGBoost","ARIMA-GRU"]


def load_predictions():
    frozen=pd.read_csv(INDIVIDUAL/"predictions"/"all_models_common_origin_predictions.csv",
                       parse_dates=["forecast_origin","target_timestamp"])
    hybrid=pd.read_csv(PREDICTIONS/"arima_gru_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    return pd.concat([frozen[frozen.model.isin(MODELS[:-1])],hybrid],ignore_index=True)


def bootstrap_ci(values, seed=42, draws=2000):
    values=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(draws)])
    return np.quantile(means,[.025,.975])


def build_tables(common, hybrid_full):
    manifest=json.loads((RUNS/"arima_gru_manifest.json").read_text())
    # The frozen presentation table is authoritative for individual-model
    # runtimes; legacy/stale jobs must not silently change this comparison.
    frozen_table=pd.read_csv(INDIVIDUAL/"tables"/"individual_model_comparison.csv")
    runtime=dict(zip(frozen_table.Model,frozen_table.Train_Time_s))
    runtime["ARIMA-GRU"]=manifest["training_time_seconds"]+manifest["residual_generation_seconds"]
    overall=[]; evaluated={}
    for model in MODELS:
        pred=common[common.model.eq(model)].copy(); result=evaluate_predictions(pred)
        evaluated[model]=result
        row=result["overall"].query("aggregation=='pooled_predictions'").iloc[0]
        overall.append({"Model":model,"Meters":pred.meter.nunique(),"Forecast_Origins":pred.groupby(["meter","forecast_origin"]).ngroups,
                        "MAE":row.MAE,"RMSE":row.RMSE,"sMAPE_pct":row.sMAPE_pct,"R2":row.R2,
                        "NRMSE_mean":row.NRMSE_mean,"Peak_Error":row.peak_load_error,
                        "Normalized_Peak_Error":row.normalized_peak_error,"Train_Time_s":runtime.get(model,np.nan),
                        "Inference_Time_ms":row.inference_time_ms})
    comparison=pd.DataFrame(overall); comparison.to_csv(TABLES/"hybrid_model_comparison.csv",index=False)
    pd.concat([v["per_meter"] for v in evaluated.values()],ignore_index=True).to_csv(TABLES/"hybrid_per_meter_comparison.csv",index=False)
    pd.concat([v["per_horizon"] for v in evaluated.values()],ignore_index=True).to_csv(TABLES/"hybrid_per_horizon_comparison.csv",index=False)
    pd.concat([v["per_origin"] for v in evaluated.values()],ignore_index=True).to_csv(TABLES/"hybrid_per_origin_comparison.csv",index=False)
    hybrid_eval=evaluated["ARIMA-GRU"]
    for level,table in hybrid_eval.items(): table.to_csv(METRICS/f"arima_gru_common_{level}.csv",index=False)
    base=comparison.set_index("Model"); h=base.loc["ARIMA-GRU"]
    improvements=[]
    for model in ["ARIMA","GRU","LinearRegression","XGBoost"]:
        b=base.loc[model]
        improvements.append({"Baseline":model,"Hybrid_MAE":h.MAE,"Baseline_MAE":b.MAE,
          "Absolute_MAE_Reduction":b.MAE-h.MAE,"Relative_MAE_Improvement_pct":100*(b.MAE-h.MAE)/b.MAE,
          "Hybrid_RMSE":h.RMSE,"Baseline_RMSE":b.RMSE,"Absolute_RMSE_Reduction":b.RMSE-h.RMSE,
          "Relative_RMSE_Improvement_pct":100*(b.RMSE-h.RMSE)/b.RMSE})
    pd.DataFrame(improvements).to_csv(TABLES/"hybrid_relative_improvements.csv",index=False)
    origin=pd.concat([evaluated[m]["per_origin"] for m in MODELS],ignore_index=True)
    paired=[]
    for model in ["ARIMA","GRU","LinearRegression","XGBoost"]:
        a=origin[origin.model.eq("ARIMA-GRU")].set_index(["city","meter","forecast_origin"])
        b=origin[origin.model.eq(model)].set_index(["city","meter","forecast_origin"])
        joined=a[["MAE","RMSE"]].join(b[["MAE","RMSE"]],lsuffix="_hybrid",rsuffix="_baseline",how="inner")
        for metric in ("MAE","RMSE"):
            diff=joined[f"{metric}_hybrid"]-joined[f"{metric}_baseline"]
            lo,hi=bootstrap_ci(diff)
            paired.append({"Baseline":model,"Metric":metric,"Origins":len(diff),"Mean_Difference_Hybrid_minus_Baseline":diff.mean(),
                           "Median_Difference":diff.median(),"Hybrid_Win_Fraction":float((diff<0).mean()),
                           "Bootstrap_95CI_Lower":lo,"Bootstrap_95CI_Upper":hi})
    pd.DataFrame(paired).to_csv(TABLES/"paired_origin_analysis.csv",index=False)
    # Per-meter win counts.
    pm=pd.read_csv(TABLES/"hybrid_per_meter_comparison.csv")
    pivot=pm.pivot(index="meter",columns="model",values=["MAE","RMSE"])
    rows=[]
    for model in ["ARIMA","GRU","LinearRegression","XGBoost"]:
        rows.append({"Baseline":model,"Common_Meters":len(pivot),
                     "Hybrid_lower_MAE_meters":int((pivot["MAE"]["ARIMA-GRU"]<pivot["MAE"][model]).sum()),
                     "Hybrid_lower_RMSE_meters":int((pivot["RMSE"]["ARIMA-GRU"]<pivot["RMSE"][model]).sum())})
    pd.DataFrame(rows).to_csv(TABLES/"per_meter_win_counts.csv",index=False)
    best=pivot["RMSE"].idxmin(axis=1).value_counts().rename_axis("Model").reset_index(name="Meters_with_lowest_RMSE")
    best.to_csv(TABLES/"per_meter_best_model_counts.csv",index=False)
    # Residual diagnostics before and after correction.
    hfull=hybrid_full.copy(); hfull["arima_error"]=hfull.actual-hfull.arima_prediction; hfull["hybrid_error"]=hfull.actual-hfull.prediction
    diag=[]
    for name,col in (("ARIMA_before_correction","arima_error"),("Hybrid_after_correction","hybrid_error")):
        e=hfull[col]
        diag.append({"Stage":name,"Mean_Error":e.mean(),"Variance":e.var(),"Lag1_Autocorrelation":e.autocorr(1),
                     "Lag24_Autocorrelation":e.autocorr(24),"MAE":e.abs().mean(),"RMSE":np.sqrt(np.mean(e**2))})
    pd.DataFrame(diag).to_csv(TABLES/"residual_diagnostics.csv",index=False)
    hfull.assign(hour=hfull.target_timestamp.dt.hour).groupby("hour")[["arima_error","hybrid_error"]].mean().reset_index().to_csv(
        TABLES/"residual_by_hour.csv",index=False)
    return comparison,evaluated


def figures(common, comparison, evaluated, hybrid_full):
    representative=common[common.model.eq("ARIMA-GRU")].sort_values(["meter","forecast_origin"]).iloc[0]
    meter,origin=representative.meter,representative.forecast_origin
    rep=common[(common.meter.eq(meter))&(common.forecast_origin.eq(origin))]
    h=hybrid_full[(hybrid_full.meter.eq(meter))&(hybrid_full.forecast_origin.eq(origin))].sort_values("horizon")
    actual=h.actual.to_numpy(); hours=np.arange(1,25)
    def save(lines,title,name,ylabel="Hourly energy (t_kWh, provisional)"):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for values,label,style in lines: ax.plot(hours,values,style,label=label,linewidth=2)
        ax.set(xlabel="Forecast horizon (hour)",ylabel=ylabel,title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/name,dpi=180); plt.close(fig)
    save([(actual,"Actual","o-"),(h.prediction,"ARIMA-GRU","s-")],f"ARIMA-GRU: {meter}, origin {origin}","arima_gru_actual_vs_predicted.png")
    ar=rep[rep.model.eq("ARIMA")].sort_values("horizon")
    save([(actual,"Actual","o-"),(ar.prediction,"ARIMA","--"),(h.prediction,"ARIMA-GRU","s-")],
         f"ARIMA versus ARIMA-GRU: {meter}, origin {origin}","arima_vs_hybrid_overlay.png")
    save([(actual,"Actual","o-"),(h.arima_prediction,"ARIMA base","--"),(h.residual_correction,"GRU residual correction",":"),(h.prediction,"Final hybrid","s-")],
         f"Hybrid decomposition: {meter}, origin {origin}","hybrid_decomposition.png")
    for metric,label in (("MAE","MAE"),("RMSE","RMSE")):
        fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(comparison.Model,comparison[metric]); ax.set(ylabel=label,title=f"Common-origin validation {label}")
        ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(FIGURES/f"comparison_{metric.lower()}.png",dpi=180); plt.close(fig)
    ph=pd.concat([evaluated[m]["per_horizon"] for m in ["ARIMA","GRU","LinearRegression","XGBoost","ARIMA-GRU"]])
    for metric in ("MAE","RMSE"):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for model,g in ph.groupby("model"): ax.plot(g.horizon,g[metric],label=model,linewidth=2)
        ax.set(xlabel="Forecast horizon",ylabel=metric,title=f"{metric} by forecast horizon"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(FIGURES/f"per_horizon_{metric.lower()}.png",dpi=180); plt.close(fig)
    pm=pd.concat([evaluated[m]["per_meter"] for m in ["ARIMA","GRU","LinearRegression","XGBoost","ARIMA-GRU"]])
    order=["ARIMA","GRU","LinearRegression","XGBoost","ARIMA-GRU"]
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.boxplot([pm[pm.model.eq(m)].RMSE for m in order],tick_labels=order,showfliers=False)
    ax.set(ylabel="Per-meter RMSE",title="Household-level validation variability"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(FIGURES/"per_meter_rmse_distribution.png",dpi=180); plt.close(fig)
    (RUNS/"representative_origin.json").write_text(json.dumps({"selection_rule":"lexicographically first common meter/origin","meter":meter,"forecast_origin":str(origin)},indent=2)+"\n")


def summary(comparison):
    imp=pd.read_csv(TABLES/"hybrid_relative_improvements.csv"); pair=pd.read_csv(TABLES/"paired_origin_analysis.csv")
    wins=pd.read_csv(TABLES/"per_meter_win_counts.csv"); diag=pd.read_csv(TABLES/"residual_diagnostics.csv")
    h=comparison.set_index("Model").loc["ARIMA-GRU"]
    arima=imp[imp.Baseline.eq("ARIMA")].iloc[0]; gru=imp[imp.Baseline.eq("GRU")].iloc[0]
    linear=imp[imp.Baseline.eq("LinearRegression")].iloc[0]; xgb=imp[imp.Baseline.eq("XGBoost")].iloc[0]
    pa=pair[(pair.Baseline.eq("ARIMA"))&(pair.Metric.eq("MAE"))].iloc[0]
    pg=pair[(pair.Baseline.eq("GRU"))&(pair.Metric.eq("MAE"))].iloc[0]
    winmap=wins.set_index("Baseline")
    before=diag[diag.Stage.eq("ARIMA_before_correction")].iloc[0]
    after=diag[diag.Stage.eq("Hybrid_after_correction")].iloc[0]
    text=f"""# ARIMA–GRU hybrid — Review 3

## Architecture and leakage-safe training

The base is the benchmark-compatible conditional ARIMA(3,0,0). A one-layer, 32-hidden-unit,
unidirectional GRU receives 168 chronological prequential residuals and directly predicts 24
residual corrections. No attention or PSO is present. Each training residual was predicted by an
expanding AR fit before its target updated that fit. Gaps were not interpolated or filled.

## Training protocol and residual coverage

After a 168-equation warm-up, 286,079 chronological out-of-sample one-step residuals were
generated. They yielded 8,289 daily-spaced direct 24-hour training windows. Residual and target
scalers were fitted only on these training data. Validation loss selected the checkpoint with
patience 4 and a maximum of 20 epochs.

## Validation protocol and main result

The hybrid produced 1,794 of 1,799 canonical validation forecasts across 58 meters. Every table
compares models on those same 1,794 origins. Its pooled validation MAE was {h.MAE:.4f} and RMSE
was {h.RMSE:.4f}. These are validation-stage findings, not final test evidence.

## Interpretation

Relative to ARIMA, hybridisation reduced MAE by {arima.Relative_MAE_Improvement_pct:.2f}% and RMSE
by {arima.Relative_RMSE_Improvement_pct:.2f}%. Relative to the standalone GRU, it reduced MAE by
{gru.Relative_MAE_Improvement_pct:.2f}% and RMSE by {gru.Relative_RMSE_Improvement_pct:.2f}%.
However, it was worse than Linear Regression by {-linear.Relative_MAE_Improvement_pct:.2f}% MAE
and {-linear.Relative_RMSE_Improvement_pct:.2f}% RMSE, and worse than XGBoost by
{-xgb.Relative_MAE_Improvement_pct:.2f}% MAE and {-xgb.Relative_RMSE_Improvement_pct:.2f}% RMSE.
Thus residual learning helped its two component baselines, but did not establish the hybrid as the
strongest individual/hybrid candidate. The test split was neither loaded nor evaluated.

## Paired-origin and per-meter findings

The hybrid had lower origin-level MAE than ARIMA for {pa.Hybrid_Win_Fraction:.1%} of origins
(mean hybrid-minus-ARIMA difference {pa.Mean_Difference_Hybrid_minus_Baseline:.4f}; bootstrap
95% CI [{pa.Bootstrap_95CI_Lower:.4f}, {pa.Bootstrap_95CI_Upper:.4f}]). It beat GRU on
{pg.Hybrid_Win_Fraction:.1%} of origins. At meter level it reduced RMSE for
{int(winmap.loc['ARIMA'].Hybrid_lower_RMSE_meters)}/58 meters relative to ARIMA and
{int(winmap.loc['GRU'].Hybrid_lower_RMSE_meters)}/58 relative to GRU, but only
{int(winmap.loc['LinearRegression'].Hybrid_lower_RMSE_meters)}/58 relative to Linear Regression.

## Per-horizon behavior

The saved horizon table shows hybrid MAE/RMSE of 0.0982/0.1910 at horizon 1 and
0.1728/0.2525 at horizon 24. The correction substantially reduces ARIMA's long-horizon error,
although Linear Regression and XGBoost remain stronger at horizon 24.

## Runtime and residual diagnostics

Residual generation plus GRU fitting required {h.Train_Time_s:.2f} seconds on CPU; inference
averaged {h.Inference_Time_ms:.3f} ms per origin. Error variance fell from {before.Variance:.4f}
before correction to {after.Variance:.4f} after correction. Lag-24 error autocorrelation fell from
{before.Lag24_Autocorrelation:.3f} to {after.Lag24_Autocorrelation:.3f}, indicating that the GRU
removed some—but not all—remaining temporal structure.

## Main conclusion and next step

ARIMA-GRU provides credible validation evidence that residual learning improves ARIMA and a plain
GRU. It does not outperform the simpler Linear Regression or XGBoost benchmarks. Attention may be
implemented next as a controlled ablation, but is not justified by a claim that the current hybrid
is already the leading model; it must demonstrate an additional reproducible gain under this same
protocol.

## Limitations

Five canonical origins lack a complete 168-hour residual history. `t_kWh` semantics remain
authoritatively unverified. The ARIMA component is conditional least-squares ARIMA(3,0,0), not a
broadly tuned state-space ARIMA. Validation early stopping makes this preliminary selection-stage
evidence. Temporal attention should only be attempted if the measured hybrid result justifies it.
"""
    (OUTPUT/"ARIMA_GRU_REVIEW3.md").write_text(text)


def main():
    all_predictions=load_predictions(); common,availability=common_origin_predictions(all_predictions,MODELS)
    availability.to_csv(TABLES/"common_origin_availability.csv",index=False)
    common.to_csv(PREDICTIONS/"hybrid_common_origin_predictions.csv",index=False)
    hybrid=pd.read_csv(PREDICTIONS/"arima_gru_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    comparison,evaluated=build_tables(common,hybrid); figures(common,comparison,evaluated,hybrid); summary(comparison)
    print(comparison.to_string(index=False))


if __name__=="__main__": main()
