# Demand-side optimisation — Review 3

## Purpose and scope

This is **simulation-based demand-side optimisation**. It demonstrates how a corrected 24-hour
forecast can feed a joint flexible-appliance scheduler. It is not a claim of measured appliance-level
optimisation or a production home-energy-management system.

## Forecast input

The input is the deterministically selected first common Linear Regression validation forecast from
`/home/stackdev_24/Downloads/arima_gru_results_per_meter/reports/review3/individual_models/predictions/all_models_common_origin_predictions.csv`: meter `BR02`, city `Bareilly`, forecast origin
`2020-09-06 03:00:00`, targets `2020-09-06 04:00:00` through
`2020-09-07 03:00:00`. Linear Regression was used because it achieved the strongest
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
window. The resulting start-time changes were: Washing Machine 09→16, Dishwasher 19→22, Water Heater 07→07, Water Pump 12→17.

| Metric | Before | After | Interpretation |
|---|---:|---:|---|
| Cost | INR 97.64 | INR 88.64 | 9.22% reduction |
| Total peak | 2.341 | 2.341 kW-equivalent | -0.00% reduction |
| PAR | 3.588 | 3.588 | -0.00% reduction |
| Flexible energy | 6.60 kWh | 6.60 kWh | Conserved exactly |
| Shifted energy | N/A | 4.60 kWh | `0.5*sum(abs(flexible_after-flexible_before))` |

Base-load energy is 9.055 kWh and total simulated energy is
15.655 kWh. Solver status: **Optimal**.

The baseline already places the 2 kW Water Heater at the lowest-background-load feasible hour in
its window, so the minimum feasible total peak equals the baseline peak. The balanced optimiser
therefore preserves peak/PAR while reducing cost; it does not fabricate a peak reduction.

## Interpretation and limitations

The experiment demonstrates feasible load shifting, cost calculation, and total-profile peak/PAR
measurement under transparent assumptions. It does not demonstrate reduced energy consumption,
measured appliance separation, an official tariff, renewable utilisation, carbon reduction, EV SOC,
AC comfort dynamics, forecast-error robustness, or real-time control. Those claims are excluded.
