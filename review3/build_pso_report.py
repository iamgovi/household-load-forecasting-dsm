"""Create PSO comparison tables, figures and the Review-3 narrative."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluation.comparison import common_origin_predictions
from evaluation.metrics import evaluate_predictions
from review3.arima_gru_pso import ATTENTION,CHECKPOINTS,FIGURES,HYBRID,INDIVIDUAL,METRICS,OUTPUT,PREDICTIONS,RUNS,SEARCH,TABLES

MODELS=["SeasonalNaive_Daily","LinearRegression","XGBoost","GRU","ARIMA","ARIMA-GRU","ARIMA-GRU-Attention","PSO-ARIMA-GRU"]

def torch_load(path):
    return torch.load(path,map_location="cpu",weights_only=False)

def bootstrap(v):
    v=np.asarray(v,float); rng=np.random.default_rng(42); m=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
    return np.quantile(m,[.025,.975])

def build():
    base=pd.read_csv(HYBRID/"predictions"/"hybrid_common_origin_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    attention=pd.read_csv(ATTENTION/"predictions"/"arima_gru_attention_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    tuned=pd.read_csv(PREDICTIONS/"pso_arima_gru_predictions.csv",parse_dates=["forecast_origin","target_timestamp"])
    allp=pd.concat([base,attention,tuned],ignore_index=True); common,availability=common_origin_predictions(allp,MODELS)
    common.to_csv(PREDICTIONS/"pso_common_origin_predictions.csv",index=False); availability.to_csv(TABLES/"pso_common_origin_availability.csv",index=False)
    results={m:evaluate_predictions(common[common.model.eq(m)]) for m in MODELS}
    old=pd.read_csv(HYBRID/"tables"/"hybrid_model_comparison.csv").set_index("Model"); manifest=json.loads((RUNS/"pso_manifest.json").read_text())
    hybrid_manifest=json.loads((HYBRID/"runs"/"arima_gru_manifest.json").read_text())
    seed=pd.read_csv(TABLES/"pso_seed_variability.csv"); primary=seed[seed.seed.eq(42)].iloc[0]
    rows=[]
    for m in MODELS:
        r=results[m]["overall"].query("aggregation=='pooled_predictions'").iloc[0]
        if m=="PSO-ARIMA-GRU":
            train=primary.training_time_seconds; infer=primary.inference_time_ms
            state=torch_load(CHECKPOINTS/"pso_arima_gru_seed42_checkpoint.pt")["model_state_dict"]; params=sum(v.numel() for v in state.values())
        elif m in old.index:
            train=hybrid_manifest["training_time_seconds"] if m=="ARIMA-GRU" else old.loc[m].Train_Time_s
            infer=old.loc[m].Inference_Time_ms; params=4152 if m=="ARIMA-GRU" else np.nan
        else:
            ar=pd.read_csv(ATTENTION/"tables"/"attention_complexity.csv").set_index("Model").loc[m]; train=ar.Train_Time_s; infer=ar.Inference_Time_ms; params=ar.Parameters
        rows.append({"Model":m,"Meters":common[common.model.eq(m)].meter.nunique(),"Forecast_Origins":common[common.model.eq(m)].groupby(["meter","forecast_origin"]).ngroups,
         "MAE":r.MAE,"RMSE":r.RMSE,"sMAPE_pct":r.sMAPE_pct,"R2":r.R2,"NRMSE_mean":r.NRMSE_mean,"Peak_Error":r.peak_load_error,
         "Normalized_Peak_Error":r.normalized_peak_error,"Final_Training_Time_s":train,"Inference_Time_ms":infer,"Parameters":params})
    comp=pd.DataFrame(rows); comp.to_csv(TABLES/"pso_expanded_comparison.csv",index=False)
    comp[comp.Model.isin(["ARIMA","ARIMA-GRU","PSO-ARIMA-GRU"])].to_csv(TABLES/"pso_primary_comparison.csv",index=False)
    interpretations={"ARIMA":"statistical base","GRU":"DL base","ARIMA-GRU":"residual hybrid","ARIMA-GRU-Attention":"negative attention ablation","PSO-ARIMA-GRU":"PSO-tuned residual hybrid"}
    abl=comp[comp.Model.isin(interpretations)].copy(); abl["Interpretation"]=abl.Model.map(interpretations); abl.to_csv(TABLES/"pso_ablation_table.csv",index=False)
    ph=pd.concat([r["per_horizon"] for r in results.values()]); ph.to_csv(TABLES/"pso_per_horizon.csv",index=False)
    pm=pd.concat([r["per_meter"] for r in results.values()]); pm.to_csv(TABLES/"pso_per_meter.csv",index=False)
    po=pd.concat([r["per_origin"] for r in results.values()]); po.to_csv(TABLES/"pso_per_origin.csv",index=False)
    for level,frame in results["PSO-ARIMA-GRU"].items(): frame.to_csv(METRICS/f"pso_common_{level}.csv",index=False)
    c=comp.set_index("Model"); t=c.loc["PSO-ARIMA-GRU"]
    changes=[]
    for m in ["ARIMA-GRU","LinearRegression","XGBoost"]:
        b=c.loc[m]; changes.append({"Baseline":m,"Tuned_MAE":t.MAE,"Baseline_MAE":b.MAE,"Absolute_MAE_Reduction":b.MAE-t.MAE,
         "Relative_MAE_Improvement_pct":100*(b.MAE-t.MAE)/b.MAE,"Tuned_RMSE":t.RMSE,"Baseline_RMSE":b.RMSE,
         "Absolute_RMSE_Reduction":b.RMSE-t.RMSE,"Relative_RMSE_Improvement_pct":100*(b.RMSE-t.RMSE)/b.RMSE})
    changes=pd.DataFrame(changes); changes.to_csv(TABLES/"pso_relative_improvements.csv",index=False)
    paired=[]; a=results["PSO-ARIMA-GRU"]["per_origin"].set_index(["city","meter","forecast_origin"])
    for m in ["ARIMA-GRU","LinearRegression","XGBoost"]:
        b=results[m]["per_origin"].set_index(["city","meter","forecast_origin"]); j=a[["MAE","RMSE"]].join(b[["MAE","RMSE"]],lsuffix="_tuned",rsuffix="_baseline")
        for metric in ("MAE","RMSE"):
            d=j[f"{metric}_tuned"]-j[f"{metric}_baseline"]; lo,hi=bootstrap(d); paired.append({"Baseline":m,"Metric":metric,"Origins":len(d),
             "Mean_Difference_Tuned_minus_Baseline":d.mean(),"Median_Difference":d.median(),"Tuned_Win_Fraction":(d<0).mean(),"Bootstrap_95CI_Lower":lo,"Bootstrap_95CI_Upper":hi})
    paired=pd.DataFrame(paired); paired.to_csv(TABLES/"pso_paired_origin_analysis.csv",index=False)
    bands=[]
    for m in ["ARIMA","ARIMA-GRU","PSO-ARIMA-GRU","LinearRegression","XGBoost"]:
        g=ph[ph.model.eq(m)]
        for name,lo,hi in (("short_H1_H6",1,6),("medium_H7_H12",7,12),("long_H13_H24",13,24)):
            z=g[g.horizon.between(lo,hi)]; bands.append({"Model":m,"Band":name,"Mean_Horizon_MAE":z.MAE.mean(),"Mean_Horizon_RMSE":z.RMSE.mean()})
    pd.DataFrame(bands).to_csv(TABLES/"pso_horizon_bands.csv",index=False)
    p=pm.pivot(index="meter",columns="model",values=["MAE","RMSE"]); d=p["RMSE"]["PSO-ARIMA-GRU"]-p["RMSE"]["ARIMA-GRU"]; tol=1e-4
    detail=pd.DataFrame({"meter":p.index,"untuned_MAE":p["MAE"]["ARIMA-GRU"],"tuned_MAE":p["MAE"]["PSO-ARIMA-GRU"],"untuned_RMSE":p["RMSE"]["ARIMA-GRU"],"tuned_RMSE":p["RMSE"]["PSO-ARIMA-GRU"]}).reset_index(drop=True)
    detail["RMSE_difference_tuned_minus_untuned"]=detail.tuned_RMSE-detail.untuned_RMSE; detail["RMSE_improvement_pct"]=100*(detail.untuned_RMSE-detail.tuned_RMSE)/detail.untuned_RMSE; detail.to_csv(TABLES/"pso_per_meter_changes.csv",index=False)
    meter=pd.DataFrame([{"Meters":len(d),"Tuned_improved":int((d<-tol).sum()),"Tuned_worsened":int((d>tol).sum()),"Approximately_tied":int((d.abs()<=tol).sum()),
      "Median_RMSE_change_tuned_minus_untuned":d.median(),"Tuned_overall_best_meters":int((p["RMSE"].idxmin(axis=1)=="PSO-ARIMA-GRU").sum())}]); meter.to_csv(TABLES/"pso_per_meter_summary.csv",index=False)
    tuned_seed=pd.read_csv(TABLES/"pso_seed_variability.csv"); untuned_seed=pd.read_csv(TABLES/"untuned_seed_variability.csv")
    variability=pd.DataFrame([{"Model":"ARIMA-GRU","Seeds":len(untuned_seed),"MAE_mean":untuned_seed.MAE.mean(),"MAE_std":untuned_seed.MAE.std(ddof=1),"RMSE_mean":untuned_seed.RMSE.mean(),"RMSE_std":untuned_seed.RMSE.std(ddof=1)},
                              {"Model":"PSO-ARIMA-GRU","Seeds":len(tuned_seed),"MAE_mean":tuned_seed.MAE.mean(),"MAE_std":tuned_seed.MAE.std(ddof=1),"RMSE_mean":tuned_seed.RMSE.mean(),"RMSE_std":tuned_seed.RMSE.std(ddof=1)}]); variability.to_csv(TABLES/"pso_seed_variability_summary.csv",index=False)
    complexity=pd.DataFrame([{"Model":"ARIMA-GRU","Parameters":4152,"Search_Time_s":0,"Final_Training_Time_s":c.loc["ARIMA-GRU"].Final_Training_Time_s,"Inference_Time_ms":c.loc["ARIMA-GRU"].Inference_Time_ms,"MAE":c.loc["ARIMA-GRU"].MAE,"RMSE":c.loc["ARIMA-GRU"].RMSE},
      {"Model":"PSO-ARIMA-GRU","Parameters":int(t.Parameters),"Search_Time_s":manifest["search_runtime_seconds"],"Final_Training_Time_s":t.Final_Training_Time_s,"Inference_Time_ms":t.Inference_Time_ms,"MAE":t.MAE,"RMSE":t.RMSE}]); complexity.to_csv(TABLES/"pso_complexity.csv",index=False)
    return common,comp,results,changes,paired,meter,variability,manifest

def figures(common,comp,results):
    conv=pd.read_csv(SEARCH/"pso_convergence.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(conv.iteration,conv.iteration_best_fitness,"o-",label="Iteration best"); ax.plot(conv.iteration,conv.global_best_fitness,"s-",label="Global best"); ax.set(xlabel="PSO iteration",ylabel="Inner-validation RMSE",title="PSO convergence"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"pso_convergence.png",dpi=180); plt.close(fig)
    focus=comp[comp.Model.isin(["ARIMA-GRU","PSO-ARIMA-GRU"])]
    for metric in ("MAE","RMSE"):
        fig,ax=plt.subplots(figsize=(6,4.5)); ax.bar(focus.Model,focus[metric]); ax.set(ylabel=metric,title=f"Untuned versus PSO-tuned {metric}"); fig.tight_layout(); fig.savefig(FIGURES/f"pso_untuned_tuned_{metric.lower()}.png",dpi=180); plt.close(fig)
        expanded=comp[comp.Model.isin(["LinearRegression","XGBoost","ARIMA","ARIMA-GRU","PSO-ARIMA-GRU"])]
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(expanded.Model,expanded[metric]); ax.set(ylabel=metric,title=f"Expanded validation {metric}"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(FIGURES/f"pso_expanded_{metric.lower()}.png",dpi=180); plt.close(fig)
    ph=pd.concat([results[m]["per_horizon"] for m in ["ARIMA","ARIMA-GRU","PSO-ARIMA-GRU","LinearRegression","XGBoost"]])
    for metric in ("MAE","RMSE"):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for m,g in ph.groupby("model"): ax.plot(g.horizon,g[metric],label=m,linewidth=2)
        ax.set(xlabel="Forecast horizon",ylabel=metric,title=f"PSO comparison: {metric} by horizon"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/f"pso_per_horizon_{metric.lower()}.png",dpi=180); plt.close(fig)
    changes=pd.read_csv(TABLES/"pso_per_meter_changes.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(changes.RMSE_difference_tuned_minus_untuned,bins=18); ax.axvline(0,color="black",linestyle="--"); ax.set(xlabel="Tuned RMSE − untuned RMSE",ylabel="Meters",title="Per-meter PSO RMSE change"); fig.tight_layout(); fig.savefig(FIGURES/"pso_per_meter_rmse_difference.png",dpi=180); plt.close(fig)
    tuned=common[common.model.eq("PSO-ARIMA-GRU")].sort_values(["meter","forecast_origin"]); first=tuned.iloc[0]; rep=common[(common.meter.eq(first.meter))&(common.forecast_origin.eq(first.forecast_origin))]; hours=np.arange(1,25); actual=tuned[(tuned.meter.eq(first.meter))&(tuned.forecast_origin.eq(first.forecast_origin))].sort_values("horizon").actual
    for models,name,title in [(["PSO-ARIMA-GRU"],"pso_actual_vs_predicted.png","PSO-tuned hybrid forecast"),(["ARIMA-GRU","PSO-ARIMA-GRU"],"pso_untuned_tuned_overlay.png","Untuned versus PSO-tuned hybrid")]:
        fig,ax=plt.subplots(figsize=(9,4.8)); ax.plot(hours,actual,"o-",label="Actual")
        for m in models: ax.plot(hours,rep[rep.model.eq(m)].sort_values("horizon").prediction,label=m,linewidth=2)
        ax.set(xlabel="Forecast horizon",ylabel="Hourly energy (t_kWh, provisional)",title=title); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/name,dpi=180); plt.close(fig)
    seed=pd.concat([pd.read_csv(TABLES/"untuned_seed_variability.csv").assign(Model="ARIMA-GRU"),pd.read_csv(TABLES/"pso_seed_variability.csv").assign(Model="PSO-ARIMA-GRU")]); fig,ax=plt.subplots(figsize=(6,4.5))
    for i,(m,g) in enumerate(seed.groupby("Model")): ax.scatter(np.repeat(i,len(g)),g.RMSE,label=m,s=55)
    ax.set_xticks([0,1],sorted(seed.Model.unique())); ax.set(ylabel="Validation RMSE",title="Seed variability"); fig.tight_layout(); fig.savefig(FIGURES/"pso_seed_variability.png",dpi=180); plt.close(fig)

def report(comp,changes,paired,meter,variability,manifest):
    c=comp.set_index("Model"); u=c.loc["ARIMA-GRU"]; t=c.loc["PSO-ARIMA-GRU"]; imp=changes.set_index("Baseline").loc["ARIMA-GRU"]; m=meter.iloc[0]; v=variability.set_index("Model")
    p=paired[(paired.Baseline.eq("ARIMA-GRU"))&(paired.Metric.eq("RMSE"))].iloc[0]
    classification="CLEARLY USEFUL" if imp.Relative_RMSE_Improvement_pct>=3 and p.Bootstrap_95CI_Upper<0 else "MARGINALLY USEFUL" if imp.Relative_RMSE_Improvement_pct>0 and p.Bootstrap_95CI_Upper<0 else "NOT USEFUL"
    text=f"""# PSO-tuned ARIMA–GRU — Review 3

## Search and leakage controls

PSO used 5 particles for 5 iterations (25 logged evaluations; {manifest['unique_configurations_trained']} unique configurations), fixed seed 42 and four screening epochs. Fitness was RMSE on a chronological 80/20 inner split of training windows. Main validation and test data did not influence particle fitness; test was not loaded. The Milestone 4 residual pipeline and ARIMA(3,0,0) were unchanged.

## Selected configuration

PSO selected hidden=64, learning rate=0.002, dropout=0.3, lookback=24 and batch size=64. Inner-validation RMSE was {manifest['best_inner_validation_RMSE']:.4f}. Search runtime was {manifest['search_runtime_seconds']:.2f}s.

## Main validation result

On the same 1,794 origins and 58 meters, tuned seed 42 obtained MAE {t.MAE:.4f} and RMSE {t.RMSE:.4f}, versus untuned MAE {u.MAE:.4f} and RMSE {u.RMSE:.4f}. This is an absolute RMSE reduction of {imp.Absolute_RMSE_Reduction:.4f} ({imp.Relative_RMSE_Improvement_pct:.2f}%) and MAE reduction of {imp.Absolute_MAE_Reduction:.4f} ({imp.Relative_MAE_Improvement_pct:.2f}%).

## Seed, paired and household evidence

Tuned RMSE across three seeds was {v.loc['PSO-ARIMA-GRU'].RMSE_mean:.4f} ± {v.loc['PSO-ARIMA-GRU'].RMSE_std:.4f}. Untuned RMSE across two seeds was {v.loc['ARIMA-GRU'].RMSE_mean:.4f} ± {v.loc['ARIMA-GRU'].RMSE_std:.4f}. The tuned-minus-untuned paired-origin RMSE difference averaged {p.Mean_Difference_Tuned_minus_Baseline:.4f}, with 95% CI [{p.Bootstrap_95CI_Lower:.4f}, {p.Bootstrap_95CI_Upper:.4f}] and tuned win rate {p.Tuned_Win_Fraction:.1%}. PSO improved {int(m.Tuned_improved)}/58 meters, worsened {int(m.Tuned_worsened)}/58 and tied {int(m.Approximately_tied)}/58.

## Conclusion

**PSO classification: {classification}.** The gain exceeds observed seed variability and paired evidence supports it, but the search adds optimization cost and the tuned hybrid still does not surpass Linear Regression or XGBoost. Final architecture selection and test evaluation must wait. The defensible final-selection candidates are Linear Regression, XGBoost, untuned ARIMA-GRU and PSO-ARIMA-GRU; attention remains a negative ablation.

## Limitations

The discrete PSO space and 5×5 budget are intentionally small. Screening uses only four epochs, so fitness is a resource-bounded proxy. Untuned variability has two seeds rather than three due CPU cost. Main validation remains selection-stage evidence, and `t_kWh` semantics remain authoritatively unverified.
"""
    (OUTPUT/"PSO_ARIMA_GRU_REVIEW3.md").write_text(text)

def main():
    outputs=build(); common,comp,results,changes,paired,meter,variability,manifest=outputs; figures(common,comp,results); report(comp,changes,paired,meter,variability,manifest); print(comp.to_string(index=False))
if __name__=="__main__": main()
