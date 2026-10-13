# Forecasting Model Registry

## 1. Purpose

This registry records forecasting candidates, their implementation status, and their intended role in the household electricity load forecasting project.

Registration indicates that a model is a candidate; it does not mean that the model is fully implemented, validated, or approved for final use.

## 2. Shared Forecasting Contract

All models must follow the project forecasting contract.

| Parameter | Required value |
|---|---|
| Frequency | 1 hour |
| Historical lookback | 168 observations |
| Forecast horizon | 24 observations |
| Origin stride | 24 hours |
| Default random seed | 42 |
| Prediction schema | Canonical evaluation schema |
| Missing-data policy | No silent imputation |
| Final test usage | Prohibited during preparation |

## 3. Candidate Models

| Registry ID | Category | Current status | Intended role |
|---|---|---|---|
| `seasonal_naive` | Baseline | Existing Review-3 runner; baseline forecaster implemented and tests passing | Simple daily seasonal benchmark |
| `holt_winters` | Statistical | Existing Review-3 runner; shared-interface integration pending | Level, trend, and optional seasonal patterns |
| `arima` | Statistical | Existing Review-3 runner; shared-interface integration pending | Autoregressive baseline |
| `sarima` | Statistical | Limited Review-3 runtime attempt; further verification required | Seasonal autoregressive baseline |
| `linear_regression` | Machine learning | Baseline forecaster implemented and tests passing | Strong, interpretable regression baseline |
| `xgboost` | Machine learning | Existing Review-3 runner; shared-interface integration pending | Nonlinear feature-based baseline |
| `gru` | Deep learning | Existing Review-3 runner; shared-interface integration pending | Recurrent sequence baseline |
| `lstm` | Deep learning | Implemented and registered; automated tests passing; research validation pending | Alternative recurrent sequence baseline |
| `arima_lstm` | Hybrid | **Not working reliably; not ready for controlled evaluation** | AR baseline plus learned LSTM residual correction |
| `arima_gru` | Hybrid | Existing Review-3 implementation; controlled integration verification pending | AR forecast plus GRU residual correction |
| `sarima_gru` | Hybrid | Registered candidate; implementation and verification pending | Seasonal AR model plus GRU residual correction |
| `pso_arima_gru` | Hybrid/optimisation | Existing Review-3 experiment; leakage and integration review pending | PSO-based GRU configuration search for the residual hybrid |

The attention-based ARIMA-GRU implementation is recorded separately as an ablation because it is not currently a standard registry entry.

**Important status note:** The full test suite last reported 63 passing tests, but that result does not establish that every registered model works. ARIMA-LSTM has had unresolved implementation and prediction-contract issues and must not be described as fully working or research-ready. Its implementation requires further debugging and testing before controlled evaluation.

## 4. Implementation References

| Component | Location |
|---|---|
| Registry definitions | `forecasting/registry.py` |
| Legacy implementation metadata | `forecasting/legacy.py` |
| Shared configuration | `forecasting/config.py` |
| Prediction validation | `forecasting/contracts.py` |
| Dataset normalization | `forecasting/data_adapter.py` |
| Evaluation-schema adapter | `forecasting/evaluation_adapter.py` |
| Common forecaster interface | `forecasting/interface.py` |
| Shared forecast runner | `forecasting/runner.py` |
| Run metadata utilities | `forecasting/run_metadata.py` |
| Baseline forecasters | `forecasting/baselines.py` |
| LSTM forecaster | `forecasting/lstm.py` |
| ARIMA-LSTM candidate | `forecasting/arima_lstm.py` |
| Hybrid helper functions | `forecasting/hybrid.py` |
| Existing individual models | `review3/individual_models.py` |
| ARIMA-GRU hybrid | `review3/arima_gru_hybrid.py` |
| PSO-ARIMA-GRU | `review3/arima_gru_pso.py` |
| Attention ablation | `review3/arima_gru_attention.py` |

## 5. Common Output Requirements

Every successfully generated forecast must use these columns:

`model, city, meter, split, forecast_origin, horizon, target_timestamp, actual, prediction, inference_time_ms, run_id`

Each complete forecast origin must contain horizons 1 through 24. Target timestamps must correspond to the forecast origin plus the respective horizon.

A run should record its configuration, random seed, data-partition identity, software/model version, runtime, failed origins, and output paths.

Model-specific output directories must not overwrite Review-3 artifacts. Failed forecasts must be recorded rather than silently omitted from evaluation coverage.

## 6. Evaluation Requirements

Candidate comparisons must use the same complete forecast origins.

Report:

- MAE and RMSE.
- sMAPE and R² where meaningful.
- Normalized RMSE.
- Peak-load error and peak-timing error.
- Inference runtime.
- Failed forecast origins and coverage.

Validation data may be used for model selection under the approved protocol. Final test evaluation is deferred until the dataset and evaluation protocol are approved.

No model should be declared the winner based solely on implementation status, test-suite success, or historical results from a different experimental setup.

## 7. Selection Policy

No final winner is selected by this registry.

Review-3 evidence suggests that Linear Regression and XGBoost were strong candidates and that the corrected ARIMA-GRU hybrid improved on its ARIMA and GRU baselines in the reported experiment. These findings are preliminary and must not be treated as final results for a future dataset.

Selection must be based on controlled comparisons under a common data split, forecast-origin set, and evaluation contract.

ARIMA-LSTM must remain excluded from controlled comparisons until its training, residual generation, prediction output, and interface behavior have been corrected and verified.

## 8. Completion Criteria

A candidate is ready for controlled evaluation only when:

1. Its implementation can be invoked through the shared forecasting interface.
2. Input and output schemas are validated.
3. It follows the forecast-origin and leakage-prevention rules.
4. Its run configuration and failures are recorded.
5. Its output location is isolated from legacy research artifacts.
6. Automated tests cover its interface and relevant edge cases.
7. Its implementation has been reviewed for methodological soundness.

A registered or documented candidate that fails these conditions remains incomplete.

## 9. Current Readiness Summary

The registry and shared forecasting infrastructure are under development. Seasonal Naive, Linear Regression, and LSTM have implementations with passing automated tests. ARIMA-LSTM remains a known unresolved candidate and is not ready for controlled evaluation.

The last reported full test run had 63 passing tests. That result should be updated after further code changes and must not be interpreted as evidence that every model is ready.

The dataset decision, model-specific integration, leakage review, run-isolation verification, and evaluation protocol approval remain outstanding. Final training and final test evaluation must not begin until the preparation exit criteria are satisfied.
