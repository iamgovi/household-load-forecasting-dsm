"""Review-3 validated forecasting and demand-side optimisation dashboard."""
import pandas as pd
import streamlit as st
from review3.dashboard_integration import (forecast_uploaded_history,load_validated_dashboard_example,
 run_validated_optimisation,validate_uploaded_history)

st.set_page_config(page_title="Household Energy Dashboard",layout="wide")
st.title("Household Energy Management Dashboard")
st.subheader("Review-3 Validated Demonstration")
st.caption("Corrected next-24-hour research forecast → simulation-based flexible appliance MILP → cost, peak, PAR and shifted-energy evidence.")

mode=st.sidebar.radio("Demonstration mode",["Review-3 validated example","Upload household CSV (experimental)"],index=0)
if mode=="Upload household CSV (experimental)":
    st.warning("Experimental inference mode. Uploaded forecasts and optimisation results are deployment demonstrations, not official Review-3 model-comparison evidence.")
    st.sidebar.caption("Required: `timestamp,kwh`. The latest continuous hourly block with at least 168 rows is used; gaps are never interpolated.")
    uploaded=st.sidebar.file_uploader("Upload household history",type="csv")
    if uploaded is None:
        st.info("Upload a CSV to validate it, or choose the validated example for the panel demonstration."); st.stop()
    try: history=validate_uploaded_history(uploaded)
    except ValueError as exc: st.error(str(exc)); st.stop()
    details=history.attrs
    st.success(f"Using the latest valid continuous segment: {len(history)} hours ({details['segment_start']} to {details['segment_end']}).")
    if details["gap_count"]:
        st.info(f"Detected {details['gap_count']} historical gap(s); {details['discarded_rows']} earlier row(s) were excluded. No values were interpolated.")
    with st.expander("View uploaded usable history"): st.line_chart(history.rename("Uploaded kWh"))
    try:
        with st.spinner("Generating the direct next-24-hour forecast..."): forecast,metadata=forecast_uploaded_history(history)
    except Exception as exc: st.error(f"Forecast generation failed: {exc}"); st.stop()
    metadata.update({"meter":uploaded.name.rsplit(".",1)[0],"city":"User upload / not provided"})
else:
    forecast,metadata=load_validated_dashboard_example()
st.markdown("### Forecast Source")
c1,c2,c3,c4=st.columns(4)
c1.metric("Model",metadata["model"]); c2.metric("Meter",metadata["meter"]); c3.metric("City",metadata["city"]); c4.metric("Horizon","24 hours")
st.caption(f"Forecast origin: {metadata['forecast_origin']} · {metadata['research_status']}")
if mode=="Upload household CSV (experimental)":
    st.caption(f"Context: {metadata['context_start']} to {metadata['context_end']} · Train-only global model · Negative predictions clipped: {metadata['negative_predictions_clipped']}")

st.markdown("### Next 24-Hour Household Load Forecast")
st.line_chart(forecast.set_index("timestamp")[["predicted_kwh"]].rename(columns={"predicted_kwh":"Forecast background (kWh*)"}),height=300)
with st.expander("View 24-hour forecast values"): st.dataframe(forecast,width="stretch",hide_index=True)

st.markdown("### Simulation-Based Flexible Appliance Scheduling")
st.warning("Appliance profiles are illustrative and not measured from the smart-meter dataset. The aggregate forecast is treated as simulated background load; appliance separation is not measured.")
st.info("The tariff is an illustrative Indian Time-of-Use scenario, not an official tariff. No unsupported sustainability claim is made.")
if not st.button("Run 24-Hour Optimisation",type="primary"):
    st.caption("Run the validated joint MILP to produce the before/after schedule and metrics."); st.stop()
try: result=run_validated_optimisation(forecast)
except Exception as exc: st.error(f"Optimisation failed: {exc}"); st.stop()
metrics=result["metrics"]
if metrics["solver_status"]!="Optimal": st.error(f"Solver failed: {metrics['solver_message']}"); st.stop()
st.success(f"MILP solver status: {metrics['solver_status']}")

st.markdown("### Validated Before/After Metrics")
r1=st.columns(4)
r1[0].metric("Cost before",f"₹{metrics['cost_before']:.2f}")
r1[1].metric("Cost after",f"₹{metrics['cost_after']:.2f}",delta=f"−₹{metrics['cost_saving']:.2f} ({metrics['cost_reduction_pct']:.2f}%)")
r1[2].metric("Peak before",f"{metrics['peak_before']:.4f} kW*")
r1[3].metric("Peak after",f"{metrics['peak_after']:.4f} kW*",delta=f"{metrics['peak_reduction_pct']:.2f}% reduction")
r2=st.columns(4)
r2[0].metric("PAR before",f"{metrics['par_before']:.4f}")
r2[1].metric("PAR after",f"{metrics['par_after']:.4f}",delta=f"{metrics['par_change_pct']:.2f}% change")
r2[2].metric("Shifted energy",f"{metrics['shifted_energy_kwh']:.2f} kWh")
r2[3].metric("Flexible energy",f"{metrics['flexible_after_kwh']:.2f} kWh",delta=f"difference {metrics['flexible_difference_kwh']:.2e}")
if abs(metrics["peak_reduction_pct"])<1e-8: st.caption("The optimiser reduced electricity cost while preserving the already-minimal feasible peak for this example.")

st.markdown("### Appliance Schedule")
schedule=result["appliance_schedule"][["Appliance","Baseline_Start","Optimised_Start","Duration_h","Energy_kWh","Allowed_Window"]].copy()
schedule["Baseline_Start"]=schedule.Baseline_Start.map(lambda h:f"{int(h):02d}:00"); schedule["Optimised_Start"]=schedule.Optimised_Start.map(lambda h:f"{int(h):02d}:00")
st.dataframe(schedule,width="stretch",hide_index=True)

st.markdown("### Tariff Profile")
tariff=result["tariff"].copy(); tariff["timestamp"]=forecast.timestamp
st.line_chart(tariff.set_index("timestamp")[["price_inr_per_kwh"]],height=220)
with st.expander("View illustrative tariff values"): st.dataframe(tariff,width="stretch",hide_index=True)

st.markdown("### Flexible Schedules Before and After")
names=result["appliance_schedule"].Appliance.tolist(); clock=result["hourly"].clock_hour.tolist()
before=pd.DataFrame(result["before_matrix"],index=names,columns=[f"{h:02d}" for h in clock]); after=pd.DataFrame(result["after_matrix"],index=names,columns=[f"{h:02d}" for h in clock])
left,right=st.columns(2); left.caption("Before optimisation (1 = ON)"); left.dataframe(before,width="stretch"); right.caption("After optimisation (1 = ON)"); right.dataframe(after,width="stretch")

st.markdown("### Total Household Load Before vs After")
hourly=result["hourly"].set_index("timestamp"); st.line_chart(hourly[["total_before","total_after","base_forecast"]],height=320)
left,right=st.columns(2); left.markdown("#### Cost Comparison"); left.bar_chart(pd.DataFrame({"INR":[metrics["cost_before"],metrics["cost_after"]]},index=["Before","After"])); right.markdown("#### Peak Comparison"); right.bar_chart(pd.DataFrame({"kW equivalent":[metrics["peak_before"],metrics["peak_after"]]},index=["Before","After"]))

with st.expander("Methodology and limitations"):
    st.markdown("""
- Forecast values are a validation-stage saved research example, not final test evidence.
- Appliance demand, preferred times, tariff and household limit are illustrative simulation parameters.
- Scheduling shifts 6.6 kWh of flexible demand; it does not reduce total appliance energy.
- No measured appliance disaggregation, official tariff, forecast-error robustness, or real-time control is claimed.
""")
