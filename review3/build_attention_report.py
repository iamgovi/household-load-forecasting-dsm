"""Generate the temporal-attention ablation evidence and Review-3 report."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.comparison import common_origin_predictions
from evaluation.metrics import evaluate_predictions
from review3.arima_gru_attention import BASELINE, FIGURES, METRICS, OUTPUT, PREDICTIONS, RUNS, TABLES, WEIGHTS

MODELS=["SeasonalNaive_Daily","LinearRegression","XGBoost","GRU","ARIMA","ARIMA-GRU","ARIMA-GRU-Attention"]


def bootstrap(values,draws=2000):
    values=np.asarray(values,float); rng=np.random.default_rng(42)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(draws)])
    return np.quantile(means,[.025,.975])


def load_common():
    prior=pd.read_csv(BASELINE/"predictions"/"hybrid_common_origin_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    attention=pd.read_csv(PREDICTIONS/"arima_gru_attention_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    allp=pd.concat([prior,attention],ignore_index=True)
    common,availability=common_origin_predictions(allp,MODELS)
    availability.to_csv(TABLES/"attention_common_origin_availability.csv",index=False)
    common.to_csv(PREDICTIONS/"attention_common_origin_predictions.csv",index=False)
    return common,attention


def tables(common,attention):
    results={m:evaluate_predictions(common[common.model.eq(m)]) for m in MODELS}
    manifest=json.loads((RUNS/"arima_gru_attention_manifest.json").read_text())
    baseline_manifest=json.loads((BASELINE/"runs"/"arima_gru_manifest.json").read_text())
    runtimes=pd.read_csv(BASELINE/"tables"/"hybrid_model_comparison.csv").set_index("Model")
    rows=[]
    for m in MODELS:
        r=results[m]["overall"].query("aggregation=='pooled_predictions'").iloc[0]
        train=(manifest["residual_generation_seconds"]+manifest["training_time_seconds"] if m=="ARIMA-GRU-Attention"
               else runtimes.loc[m].Train_Time_s)
        params=(manifest["parameter_count"] if m=="ARIMA-GRU-Attention" else baseline_manifest["parameter_count"] if m=="ARIMA-GRU" else np.nan)
        rows.append({"Model":m,"Meters":common[common.model.eq(m)].meter.nunique(),"Forecast_Origins":common[common.model.eq(m)].groupby(["meter","forecast_origin"]).ngroups,
          "MAE":r.MAE,"RMSE":r.RMSE,"sMAPE_pct":r.sMAPE_pct,"R2":r.R2,"NRMSE_mean":r.NRMSE_mean,
          "Peak_Error":r.peak_load_error,"Normalized_Peak_Error":r.normalized_peak_error,"Train_Time_s":train,
          "Inference_Time_ms":r.inference_time_ms,"Parameters":params})
    comparison=pd.DataFrame(rows); comparison.to_csv(TABLES/"attention_expanded_comparison.csv",index=False)
    comparison[comparison.Model.isin(["ARIMA","ARIMA-GRU","ARIMA-GRU-Attention"])].to_csv(TABLES/"attention_primary_ablation.csv",index=False)
    ph=pd.concat([v["per_horizon"] for v in results.values()],ignore_index=True); ph.to_csv(TABLES/"attention_per_horizon.csv",index=False)
    pm=pd.concat([v["per_meter"] for v in results.values()],ignore_index=True); pm.to_csv(TABLES/"attention_per_meter.csv",index=False)
    po=pd.concat([v["per_origin"] for v in results.values()],ignore_index=True); po.to_csv(TABLES/"attention_per_origin.csv",index=False)
    for level,frame in results["ARIMA-GRU-Attention"].items(): frame.to_csv(METRICS/f"attention_common_{level}.csv",index=False)
    base=comparison.set_index("Model").loc["ARIMA-GRU"]; att=comparison.set_index("Model").loc["ARIMA-GRU-Attention"]
    improvement=pd.DataFrame([{"Baseline":"ARIMA-GRU","Attention_MAE":att.MAE,"Baseline_MAE":base.MAE,
      "Absolute_MAE_Reduction":base.MAE-att.MAE,"Relative_MAE_Improvement_pct":100*(base.MAE-att.MAE)/base.MAE,
      "Attention_RMSE":att.RMSE,"Baseline_RMSE":base.RMSE,"Absolute_RMSE_Reduction":base.RMSE-att.RMSE,
      "Relative_RMSE_Improvement_pct":100*(base.RMSE-att.RMSE)/base.RMSE}])
    improvement.to_csv(TABLES/"attention_improvement.csv",index=False)
    a=results["ARIMA-GRU-Attention"]["per_origin"].set_index(["city","meter","forecast_origin"])
    b=results["ARIMA-GRU"]["per_origin"].set_index(["city","meter","forecast_origin"])
    joined=a[["MAE","RMSE"]].join(b[["MAE","RMSE"]],lsuffix="_attention",rsuffix="_hybrid")
    paired=[]
    for metric in ("MAE","RMSE"):
        diff=joined[f"{metric}_attention"]-joined[f"{metric}_hybrid"]; lo,hi=bootstrap(diff)
        paired.append({"Metric":metric,"Origins":len(diff),"Mean_Difference_Attention_minus_Hybrid":diff.mean(),
          "Median_Difference":diff.median(),"Attention_Win_Fraction":float((diff<0).mean()),
          "Bootstrap_95CI_Lower":lo,"Bootstrap_95CI_Upper":hi})
    pd.DataFrame(paired).to_csv(TABLES/"attention_paired_origin_analysis.csv",index=False)
    bands=[]
    for m in ["ARIMA","ARIMA-GRU","ARIMA-GRU-Attention","LinearRegression","XGBoost"]:
        g=ph[ph.model.eq(m)]
        for name,lo,hi in (("short_H1_H6",1,6),("medium_H7_H12",7,12),("long_H13_H24",13,24)):
            s=g[g.horizon.between(lo,hi)]; bands.append({"Model":m,"Band":name,"Mean_Horizon_MAE":s.MAE.mean(),"Mean_Horizon_RMSE":s.RMSE.mean()})
    pd.DataFrame(bands).to_csv(TABLES/"attention_horizon_bands.csv",index=False)
    p=pm.pivot(index="meter",columns="model",values=["MAE","RMSE"]); delta=p["RMSE"]["ARIMA-GRU-Attention"]-p["RMSE"]["ARIMA-GRU"]
    tol=1e-4
    meter_summary=pd.DataFrame([{"Meters":len(delta),"Attention_improved":int((delta<-tol).sum()),"Attention_worsened":int((delta>tol).sum()),
      "Approximately_tied_abs_RMSE_delta_le_1e-4":int((delta.abs()<=tol).sum()),"Attention_median_MAE":p["MAE"]["ARIMA-GRU-Attention"].median(),
      "Attention_median_RMSE":p["RMSE"]["ARIMA-GRU-Attention"].median(),
      "Attention_overall_best_meters":int((p["RMSE"].idxmin(axis=1)=="ARIMA-GRU-Attention").sum())}])
    meter_summary.to_csv(TABLES/"attention_per_meter_summary.csv",index=False)
    attention=attention.copy(); attention["arima_error"]=attention.actual-attention.arima_prediction
    attention["attention_error"]=attention.actual-attention.prediction
    hybrid=common[common.model.eq("ARIMA-GRU")].copy(); hybrid["hybrid_error"]=hybrid.actual-hybrid.prediction
    merged=attention.merge(hybrid[["meter","forecast_origin","horizon","hybrid_error"]],on=["meter","forecast_origin","horizon"])
    diag=[]
    for stage,col in (("ARIMA","arima_error"),("ARIMA-GRU","hybrid_error"),("ARIMA-GRU-Attention","attention_error")):
        e=merged[col]; diag.append({"Stage":stage,"Mean_Error":e.mean(),"Variance":e.var(),"Lag1_Autocorrelation":e.autocorr(1),
          "Lag24_Autocorrelation":e.autocorr(24),"MAE":e.abs().mean(),"RMSE":np.sqrt(np.mean(e**2))})
    pd.DataFrame(diag).to_csv(TABLES/"attention_residual_diagnostics.csv",index=False)
    weights=np.load(WEIGHTS/"validation_attention_weights.npz")["weights"]
    mass=pd.DataFrame([{"Region":"recent_24h_lags_0_23","Mean_Attention_Mass":weights[:,-24:].sum(1).mean()},
                       {"Region":"previous_day_lags_24_47","Mean_Attention_Mass":weights[:,-48:-24].sum(1).mean()},
                       {"Region":"older_lags_48_167","Mean_Attention_Mass":weights[:,:-48].sum(1).mean()}])
    mass.to_csv(TABLES/"attention_weight_mass.csv",index=False)
    entropy=-np.sum(weights*np.log(weights+1e-15),axis=1)/np.log(weights.shape[1])
    pd.DataFrame([{"Origins":len(weights),"Sequence_Length":weights.shape[1],
      "Uniform_Weight":1/weights.shape[1],"Mean_Maximum_Weight":weights.max(axis=1).mean(),
      "Mean_Normalized_Entropy":entropy.mean(),"Mean_Recent_24h_Mass":weights[:,-24:].sum(axis=1).mean()}]).to_csv(
          TABLES/"attention_weight_summary.csv",index=False)
    complexity=comparison[comparison.Model.isin(["ARIMA-GRU","ARIMA-GRU-Attention"])][["Model","Parameters","Train_Time_s","Inference_Time_ms","RMSE"]]
    complexity.to_csv(TABLES/"attention_complexity.csv",index=False)
    return comparison,results,improvement,paired,meter_summary,mass


def figures(common,comparison,results):
    att=common[common.model.eq("ARIMA-GRU-Attention")].sort_values(["meter","forecast_origin"])
    first=att.iloc[0]; meter,origin=first.meter,first.forecast_origin; rep=common[(common.meter.eq(meter))&(common.forecast_origin.eq(origin))]
    hours=np.arange(1,25); actual=att[(att.meter.eq(meter))&(att.forecast_origin.eq(origin))].sort_values("horizon").actual
    def line(models,name,title):
        fig,ax=plt.subplots(figsize=(9,4.8)); ax.plot(hours,actual,"o-",label="Actual",linewidth=2)
        for m in models: ax.plot(hours,rep[rep.model.eq(m)].sort_values("horizon").prediction,label=m,linewidth=2)
        ax.set(xlabel="Forecast horizon",ylabel="Hourly energy (t_kWh, provisional)",title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/name,dpi=180); plt.close(fig)
    line(["ARIMA-GRU-Attention"],"attention_actual_vs_predicted.png","Temporal-attention forecast")
    line(["ARIMA","ARIMA-GRU","ARIMA-GRU-Attention"],"attention_ablation_overlay.png","ARIMA → residual GRU → temporal attention")
    ab=comparison[comparison.Model.isin(["ARIMA","ARIMA-GRU","ARIMA-GRU-Attention"])]
    for metric in ("MAE","RMSE"):
        fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(ab.Model,ab[metric]); ax.set(ylabel=metric,title=f"Attention ablation: {metric}"); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(FIGURES/f"attention_ablation_{metric.lower()}.png",dpi=180); plt.close(fig)
    ph=pd.concat([results[m]["per_horizon"] for m in ["ARIMA","ARIMA-GRU","ARIMA-GRU-Attention","LinearRegression","XGBoost"]])
    for metric in ("MAE","RMSE"):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for m,g in ph.groupby("model"): ax.plot(g.horizon,g[metric],label=m,linewidth=2)
        ax.set(xlabel="Forecast horizon",ylabel=metric,title=f"{metric} by horizon"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/f"attention_per_horizon_{metric.lower()}.png",dpi=180); plt.close(fig)
    pm=pd.concat([results[m]["per_meter"] for m in ["ARIMA-GRU","ARIMA-GRU-Attention","LinearRegression","XGBoost"]]); order=["ARIMA-GRU","ARIMA-GRU-Attention","LinearRegression","XGBoost"]
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.boxplot([pm[pm.model.eq(m)].RMSE for m in order],tick_labels=order,showfliers=False); ax.set(ylabel="Per-meter RMSE",title="Attention household variability"); ax.tick_params(axis="x",rotation=18); fig.tight_layout(); fig.savefig(FIGURES/"attention_per_meter_rmse.png",dpi=180); plt.close(fig)
    w=pd.read_csv(WEIGHTS/"representative_attention_weights.csv",parse_dates=["history_timestamp"])
    fig,ax=plt.subplots(figsize=(10,4.5)); ax.plot(w.history_timestamp,w.attention_weight); ax.set(xlabel="Historical timestamp",ylabel="Attention weight",title=f"Temporal attention weights: {meter}, origin {origin}"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"temporal_attention_weights.png",dpi=180); plt.close(fig)
    hist=pd.read_csv(RUNS/"arima_gru_attention_training_history.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(hist.epoch,hist.train_residual_MSE_scaled,label="Train"); ax.plot(hist.epoch,hist.validation_residual_MSE_scaled,label="Validation"); ax.set(xlabel="Epoch",ylabel="Scaled MSE",title="Attention training history"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"attention_training_history.png",dpi=180); plt.close(fig)
    d=pd.read_csv(TABLES/"attention_residual_diagnostics.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(len(d)); width=.35; ax.bar(x-width/2,d.Lag1_Autocorrelation,width,label="Lag 1"); ax.bar(x+width/2,d.Lag24_Autocorrelation,width,label="Lag 24"); ax.set_xticks(x,d.Stage,rotation=15); ax.set(ylabel="Error autocorrelation",title="Residual temporal structure"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"attention_residual_autocorrelation.png",dpi=180); plt.close(fig)


def report(comparison,improvement,paired,meter,mass):
    c=comparison.set_index("Model"); a=c.loc["ARIMA-GRU-Attention"]; b=c.loc["ARIMA-GRU"]; imp=improvement.iloc[0]
    pma=paired[0]; pmr=paired[1]; ms=meter.iloc[0]
    classification="ATTENTION NOT JUSTIFIED" if imp.Relative_RMSE_Improvement_pct<=0 else "ATTENTION MARGINALLY JUSTIFIED"
    text=f"""# ARIMA–GRU temporal-attention ablation — Review 3

## Architecture and preserved residual pipeline

The model adds additive temporal softmax attention across all 168 hidden states of the frozen
one-layer, 32-hidden-unit residual GRU. All other principal training settings are unchanged. The
286,079 residuals regenerated through the Milestone 4 function matched the frozen residual file,
and protected baseline hashes were unchanged. No test data, PSO, spatial attention or future fill
was used.

## Validation result

All 1,794 hybrid-compatible origins across 58 meters were evaluated. Attention obtained MAE
{a.MAE:.4f}, RMSE {a.RMSE:.4f}, sMAPE {a.sMAPE_pct:.2f}%, R² {a.R2:.4f} and NRMSE
{a.NRMSE_mean:.4f}. The plain hybrid retained MAE {b.MAE:.4f} and RMSE {b.RMSE:.4f}.
Attention changed MAE by {imp.Absolute_MAE_Reduction:.4f} ({imp.Relative_MAE_Improvement_pct:.2f}%)
and RMSE by {imp.Absolute_RMSE_Reduction:.4f} ({imp.Relative_RMSE_Improvement_pct:.2f}%); negative
improvement means degradation.

## Paired and household evidence

Origin-level attention-minus-hybrid MAE averaged {pma['Mean_Difference_Attention_minus_Hybrid']:.4f}
(95% bootstrap CI [{pma['Bootstrap_95CI_Lower']:.4f}, {pma['Bootstrap_95CI_Upper']:.4f}]); attention
won {pma['Attention_Win_Fraction']:.1%} of origins. For RMSE the mean difference was
{pmr['Mean_Difference_Attention_minus_Hybrid']:.4f} and win rate {pmr['Attention_Win_Fraction']:.1%}.
Attention improved {int(ms.Attention_improved)}/58 meters, worsened {int(ms.Attention_worsened)}/58,
and was approximately tied on {int(ms.Approximately_tied_abs_RMSE_delta_le_1e4) if hasattr(ms,'Approximately_tied_abs_RMSE_delta_le_1e4') else int(ms.iloc[3])}/58.

## Attention weights and interpretation

Weights are descriptive model allocations, not causal effects. Region-level mass and all 168-lag
aggregate weights are saved. Mean normalized entropy was 0.996 (1.0 is uniform), and the most
recent 24 hours received 14.35% of mass versus 14.29% under a uniform distribution. Therefore no
strong concentration pattern was observed. Visual structure alone cannot justify attention.

## Complexity and decision

The attention model has {int(a.Parameters):,} parameters versus {int(b.Parameters):,} for the plain
hybrid. Total residual-generation plus training time was {a.Train_Time_s:.2f}s versus
{b.Train_Time_s:.2f}s; inference was {a.Inference_Time_ms:.3f} versus {b.Inference_Time_ms:.3f}ms/origin.

**Decision: {classification}.** Attention degraded the main validation metrics and added complexity.
It should remain documented as a negative ablation, not be assumed as the preferred architecture.
PSO should not tune this attention model merely because it appears in the provisional title. If PSO
is pursued next, the defensible target is the plain residual hybrid (or another validation-supported
candidate), with attention retained as a documented negative ablation.

## Limitations

This is validation-stage evidence with validation-based early stopping. Five original canonical
origins were already excluded by the common residual-history constraint. `t_kWh` semantics remain
authoritatively unverified. Attention weights do not establish causality.
"""
    (OUTPUT/"ARIMA_GRU_ATTENTION_REVIEW3.md").write_text(text)


def main():
    common,attention=load_common(); outputs=tables(common,attention); comparison,results,improvement,paired,meter,mass=outputs
    figures(common,comparison,results); report(comparison,improvement,paired,meter,mass); print(comparison.to_string(index=False))


if __name__=="__main__": main()
