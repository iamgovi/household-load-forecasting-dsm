# Individual models — Review 3

## Models implemented

Daily Seasonal Naive, Holt–Winters, conditional ARIMA, direct multi-output Linear Regression,
direct multi-output XGBoost, and a plain direct 168-to-24 GRU were evaluated. SARIMA was
implemented and attempted on one meter but excluded from the master comparison.

## Forecast protocol and validation data

Every included model predicts horizons 1--24 at the same 1799
canonical validation origins across 59 meters. No test predictions,
interpolation or future filling were used. The representative graph origin is selected before model
comparison as the lexicographically first common meter/origin.

## Main validation findings

LinearRegression achieved the lowest pooled validation RMSE (0.2523) among the
individual models under the current common protocol. LinearRegression had the lowest MAE
(0.1467), while SeasonalNaive_Daily had the lowest sMAPE (40.62%).
These differing rankings show why no model should be selected from RMSE alone.

Per-horizon and per-meter tables/figures preserve forecast-distance behavior and household variability.
Training/selection runtime and per-origin inference runtime are included in the comparison table.

## SARIMA limitation

The SARIMA(1, 0, 0) x (1, 0, 0, 24)
attempt succeeded for 2 origins on one meter, with 8.25s
fit time. A full run was deferred for Review 3 and SARIMA is not mixed into the fair comparison.

## Current limitations and next step

Results are validation-stage preliminary evidence. `t_kWh` semantics remain authoritatively unverified.
ARIMA is specifically a conditional-least-squares ARIMA(3,0,0) selected from p=1,2,3—not a general
ARIMA order search. The next milestone may reconstruct the residual ARIMA-GRU hybrid, using these
individual benchmarks and the same origins/evaluator; test evaluation remains frozen until selection.
