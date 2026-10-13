"""Build Review-3 demand-side optimisation figures and reports."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from review3.demand_side_optimisation import APPLIANCES,CONFIGS,FIGURES,OUTPUT,RUNS,SCHEDULES,TABLES

LABEL="Simulation-based flexible appliance scheduling"

def save(fig,name):
    fig.text(.5,.01,LABEL,ha="center",fontsize=9,style="italic"); fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(FIGURES/name,dpi=180); plt.close(fig)

def figures():
    p=pd.read_csv(SCHEDULES/"hourly_before_after_schedule.csv"); slots=p.schedule_slot; labels=[f"{h:02d}" for h in p.clock_hour]
    fig,ax=plt.subplots(figsize=(8,4.4)); ax.plot(slots,p.base_forecast_kwh,"o-"); ax.set_xticks(slots[::2],labels[::2]); ax.set(xlabel="Forecast target clock hour",ylabel="Hourly energy (t_kWh, provisional)",title="Corrected 24-hour forecast used as simulated background load"); ax.grid(alpha=.25); save(fig,"base_load_forecast.png")
    fig,ax=plt.subplots(figsize=(8,4.4)); ax.step(slots,p.tariff_inr_per_kwh,where="mid"); ax.set_xticks(slots[::2],labels[::2]); ax.set(xlabel="Forecast target clock hour",ylabel="INR/kWh",title="Illustrative Indian Time-of-Use tariff scenario"); ax.grid(alpha=.25); save(fig,"illustrative_tariff.png")
    arrays=np.load(RUNS/"schedule_arrays.npz"); before,after=arrays["before"],arrays["after"]
    for data,name,title in ((before,"appliance_schedule_before.png","Flexible appliance schedule before optimisation"),(after,"appliance_schedule_after.png","Flexible appliance schedule after optimisation")):
        fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(data,aspect="auto",cmap="Blues",vmin=0,vmax=1); ax.set_yticks(range(len(APPLIANCES)),[a["name"] for a in APPLIANCES]); ax.set_xticks(range(24),labels); ax.set(xlabel="Forecast target clock hour",title=title); save(fig,name)
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.plot(slots,p.total_before_kwh,"o-",label="Before"); ax.plot(slots,p.total_after_kwh,"s-",label="After"); ax.plot(slots,p.base_forecast_kwh,"--",label="Forecast background"); ax.set_xticks(slots[::2],labels[::2]); ax.set(xlabel="Forecast target clock hour",ylabel="Hourly total load (kW-equivalent)",title="Total household load before versus after scheduling"); ax.legend(); ax.grid(alpha=.25); save(fig,"total_load_before_after.png")
    summary=pd.read_csv(TABLES/"optimisation_summary.csv").set_index("Metric")
    for metric,name,ylabel in (("Peak Load","peak_comparison.png","kW-equivalent"),("Electricity Cost","cost_comparison.png","INR")):
        row=summary.loc[metric]; fig,ax=plt.subplots(figsize=(5.5,4.3)); ax.bar(["Before","After"],[row.Before,row.After],color=["#8c96c6","#41ab5d"]); ax.set(ylabel=ylabel,title=f"{metric}: before versus after"); save(fig,name)
    table=pd.read_csv(TABLES/"appliance_schedule.csv"); fig,ax=plt.subplots(figsize=(10,4.5))
    colors={"Baseline":"#9e9ac8","Optimised":"#31a354"}
    for y,row in table.iterrows():
        for mode,col,offset in (("Baseline","Baseline_Hours",-.18),("Optimised","Optimised_Hours",.18)):
            h=[int(v) for v in str(row[col]).split(",")]; ax.broken_barh([(min(h),len(h))],(y+offset-.13,.26),facecolors=colors[mode],label=mode if y==0 else None)
    ax.set_yticks(range(len(table)),table.Appliance); ax.set_xlim(0,24); ax.set_xticks(range(0,25,2)); ax.set(xlabel="Hour",title="Appliance scheduling timeline"); ax.legend(); ax.grid(axis="x",alpha=.2); save(fig,"appliance_gantt.png")
    fig,ax=plt.subplots(figsize=(11,2.8)); ax.axis("off"); boxes=["Smart-meter data","Corrected 24-hour forecast","Illustrative appliances + tariff","Joint MILP schedule","Cost / peak / PAR metrics"]
    xs=np.linspace(.02,.82,len(boxes))
    for i,(x,text) in enumerate(zip(xs,boxes)):
        ax.add_patch(FancyBboxPatch((x,.35),.16,.3,boxstyle="round,pad=.02",facecolor="#deebf7",edgecolor="#3182bd")); ax.text(x+.08,.5,text,ha="center",va="center",fontsize=9,wrap=True)
        if i<len(boxes)-1: ax.annotate("",xy=(xs[i+1],.5),xytext=(x+.16,.5),arrowprops={"arrowstyle":"->","lw":1.5})
    ax.set_title("Forecast-to-demand-side-optimisation demonstration"); save(fig,"end_to_end_flow.png")

def reports():
    manifest=json.loads((RUNS/"optimisation_manifest.json").read_text()); summary=pd.read_csv(TABLES/"optimisation_summary.csv").set_index("Metric"); schedule=pd.read_csv(TABLES/"appliance_schedule.csv")
    audit="""# Existing optimisation audit

| Component | Current legacy behavior | Inputs/outputs | Scientific assessment | Review-3 disposition |
|---|---|---|---|---|
| `dashboard.py:tod_multiplier` | Three-period tariff multiplier over an illustrative INR base rate | Hour → tariff multiplier | Internally coherent and already labelled illustrative | Concept retained; explicit 24-row tariff created separately |
| `dashboard.py:APPLIANCES` | Five manually specified loads including EV and comfort-labelled AC | Hard-coded powers, durations, deadlines | Explicitly non-measured, but EV has no SOC and AC has no thermal model | Deprecated for research evidence; replaced with four simpler simulated loads |
| `dashboard.py:schedule_appliance` | Greedy, independent cheapest/solar-aware contiguous start | One appliance at a time | Not joint optimisation; ignores household peak and appliance concurrency | Replaced in research path by joint MILP |
| Deadline filter | Accepts `finish <= deadline OR start >= deadline` | Candidate starts | Incorrect: explicitly permits operation after deadline | Repaired with start variables whose complete duration lies inside allowed window |
| Baseline | Every appliance starts at 19:00 with modulo-24 wrap | Appliance assumptions | Violates several deadlines/windows and exaggerates savings | Replaced by deterministic feasible preferred-time schedule |
| Cost | Appliance-only tariff cost | Flexible load only | Arithmetic is locally correct but omits base-load cost | Repaired: same forecast background included before and after |
| Solar | Synthetic cosine curve independently credited to every appliance | Simulated solar and appliance power | Can double-count solar and is not real renewable evidence | Excluded from main experiment |
| “Flexible energy shifted” | Reports total flexible energy | Appliance power × duration | Incorrect shifted-energy definition | Replaced by `0.5*sum(abs(after-before))` |
| Sustainability | Reports simulated solar self-consumption | Synthetic curve | Not defensible as real renewable utilisation; no carbon factor | Disabled from Review-3 evidence; no renewable/carbon claim |
| Dashboard structure | Streamlit layout and tables | Legacy forecast/deployment path | Structurally useful for future integration | Preserved unchanged; dashboard redesign deferred |

The legacy dashboard consumes its own deployment forecast and root-level legacy data paths. It is not used by this research demonstration.
"""
    (OUTPUT/"EXISTING_OPTIMISATION_AUDIT.md").write_text(audit)
    cost=summary.loc["Electricity Cost"]; peak=summary.loc["Peak Load"]; par=summary.loc["Peak-to-Average Ratio"]; shifted=summary.loc["Shifted Energy","After"]
    movements=", ".join(f"{r.Appliance} {int(r.Baseline_Start):02d}→{int(r.Optimised_Start):02d}" for r in schedule.itertuples())
    report=f"""# Demand-side optimisation — Review 3

## Purpose and scope

This is **simulation-based demand-side optimisation**. It demonstrates how a corrected 24-hour
forecast can feed a joint flexible-appliance scheduler. It is not a claim of measured appliance-level
optimisation or a production home-energy-management system.

## Forecast input

The input is the deterministically selected first common Linear Regression validation forecast from
`{manifest['forecast_file']}`: meter `{manifest['meter']}`, city `{manifest['city']}`, forecast origin
`{manifest['forecast_origin']}`, targets `{manifest['forecast_target_start']}` through
`{manifest['forecast_target_end']}`. Linear Regression was used because it achieved the strongest
common-origin validation MAE/RMSE; the optimiser does not force the project architecture to win.

The saved aggregate forecast is treated as counterfactual non-flexible background load for this
simulation. The four appliance profiles are added as a separate synthetic flexible component. This
separation was not measured from the smart-meter dataset.

## Appliance and tariff assumptions

All appliance profiles are **ILLUSTRATIVE / SIMULATED**. Four non-interruptible loads are used:
Washing Machine, Dishwasher, Water Heater and Water Pump. Powers, durations, preferred times and
allowed windows are explicit in `illustrative_appliances.json`. The 24-price tariff is an
**illustrative Indian Time-of-Use tariff scenario**, not a verified official tariff.

## MILP formulation

Binary `x[a,t]` represents appliance on/off state and binary `s[a,k]` selects one feasible start.
Every appliance runs exactly its required duration, stays contiguous, and finishes within its window.
The continuous peak variable bounds total forecast-background plus flexible load at every hour. A
3.5 kW household ceiling is an illustrative parameter. The normalized objective is
`0.3*(total cost/baseline cost) + 0.7*(total peak/3.5)`.

SciPy's MILP interface with the HiGHS solver was used because PuLP is not installed locally. This is
a genuine mixed-integer linear model, not a greedy approximation.

## Before and after

The baseline starts each appliance at a reasonable predefined preferred time within its allowed
window. The resulting start-time changes were: {movements}.

| Metric | Before | After | Interpretation |
|---|---:|---:|---|
| Cost | INR {cost.Before:.2f} | INR {cost.After:.2f} | {-cost.Change_pct:.2f}% reduction |
| Total peak | {peak.Before:.3f} | {peak.After:.3f} kW-equivalent | {-peak.Change_pct:.2f}% reduction |
| PAR | {par.Before:.3f} | {par.After:.3f} | {-par.Change_pct:.2f}% reduction |
| Flexible energy | 6.60 kWh | 6.60 kWh | Conserved exactly |
| Shifted energy | N/A | {shifted:.2f} kWh | `0.5*sum(abs(flexible_after-flexible_before))` |

Base-load energy is {manifest['base_load_energy_kwh']:.3f} kWh and total simulated energy is
{manifest['total_energy_kwh']:.3f} kWh. Solver status: **{manifest['solver_status']}**.

The baseline already places the 2 kW Water Heater at the lowest-background-load feasible hour in
its window, so the minimum feasible total peak equals the baseline peak. The balanced optimiser
therefore preserves peak/PAR while reducing cost; it does not fabricate a peak reduction.

## Interpretation and limitations

The experiment demonstrates feasible load shifting, cost calculation, and total-profile peak/PAR
measurement under transparent assumptions. It does not demonstrate reduced energy consumption,
measured appliance separation, an official tariff, renewable utilisation, carbon reduction, EV SOC,
AC comfort dynamics, forecast-error robustness, or real-time control. Those claims are excluded.
"""
    (OUTPUT/"DEMAND_SIDE_OPTIMISATION_REVIEW3.md").write_text(report)

def main():
    FIGURES.mkdir(parents=True,exist_ok=True); figures(); reports()
if __name__=="__main__": main()
