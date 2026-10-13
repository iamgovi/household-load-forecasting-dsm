# Hybrid Forecasting Design

## 1. Purpose

This document defines the proposed hybrid forecasting architecture for household electricity load forecasting. It records existing Review-3 implementations, planned extensions, and constraints that must be satisfied before final training.

## 2. Forecasting Contract

All candidate models must follow the shared forecasting contract:

- Historical context: 168 hourly observations.
- Forecast horizon: 24 hourly observations.
- Forecast origin stride: 24 hours.
- Frequency: hourly.
- Random seed: 42 unless a documented experiment requires additional seeds.
- Information available after the forecast origin must not be used to generate that forecast.
- Missing or invalid observations must not be silently imputed.
- Model comparisons must use the same complete forecast origins.

## 3. Candidate Architectures

### 3.1 ARIMA-GRU Residual Hybrid

**Status:** Existing Review-3 implementation.

The baseline AR component generates a forecast from autoregressive coefficients. A GRU learns residual patterns from strictly prequential training residuals and produces a 24-step correction.

The final prediction is:

`Hybrid forecast = AR forecast + GRU residual correction`

The existing implementation uses AR order 3, a 168-equation warm-up, and a GRU residual-correction model.

### 3.2 PSO-ARIMA-GRU

**Status:** Existing Review-3 experiment.

Particle Swarm Optimization searches GRU configuration parameters, including hidden units, learning rate, dropout, lookback, and batch size.

The existing experiment uses a small swarm and a limited number of iterations. Its inner chronological split must be reviewed for temporal leakage, including overlapping target windows, before reuse for final model selection.

### 3.3 Attention-Based ARIMA-GRU

**Status:** Existing Review-3 ablation.

This variant applies temporal attention to GRU hidden states before producing the residual correction. Review-3 found that attention worsened performance in the reported experiment. It remains an ablation candidate rather than an assumed improvement.

### 3.4 ARIMA-LSTM

**Status:** Planned extension.

The AR component produces the base forecast. An LSTM learns residual patterns and estimates a correction over the forecast horizon.

This architecture must use training-only fitted preprocessing and chronologically valid residual windows. It must not be described as implemented until its code and tests exist.

### 3.5 SARIMA-GRU

**Status:** Planned extension.

A seasonal autoregressive model captures repeating temporal patterns, while a GRU learns residual corrections. Seasonal order, convergence, runtime, and forecast-origin validity require explicit validation.

### 3.6 LSTM Standalone Model

**Status:** Planned extension.

The standalone LSTM provides a recurrent neural baseline against which residual hybrids can be compared. It must use the same input context, forecast horizon, data partitions, and prediction schema as other candidates.

## 4. Leakage Prevention

1. Fit scalers and other learned preprocessing only on the relevant training data.
2. Generate training residuals prequentially: predict each observation before incorporating its target into the fitted autoregressive model.
3. Keep validation-based model selection separate from final test evaluation.
4. Verify chronological boundaries and target-window overlap in internal tuning splits.
5. Do not use future observations to construct forecast features.
6. Record failed origins and compare models on the intersection of complete forecast origins.
7. Do not run final training while the dataset decision remains unresolved.

## 5. Shared Prediction Interface

Every candidate must emit the canonical prediction columns:

`model, city, meter, split, forecast_origin, horizon, target_timestamp, actual, prediction, inference_time_ms, run_id`

Each successful forecast origin must contain exactly one prediction for each horizon from 1 through 24. Target timestamps must agree with the forecast origin and horizon.

## 6. Evaluation

The existing evaluation package supports MAE, RMSE, sMAPE, R², normalized RMSE, peak-load error, normalized peak error, peak timing error, and inference time.

Comparisons must use equivalent forecast origins and report failures and runtime alongside accuracy. Validation is used for development and model selection. The final test set remains untouched until the dataset and experimental protocol are approved.

## 7. Current Evidence and Limitations

Review-3 reported that Linear Regression and XGBoost were strong candidates, the corrected ARIMA-GRU hybrid improved on ARIMA and GRU, temporal attention worsened the reported result, and PSO provided a modest improvement.

These findings are preliminary evidence from the existing experiments. They do not establish a final winner or guarantee performance on a subsequently selected dataset.

## 8. Implementation Priorities

1. Complete the dataset-agnostic input adapter and prediction contract.
2. Connect existing models through isolated wrappers.
3. Implement and test the standalone LSTM and ARIMA-LSTM extension.
4. Add SARIMA-GRU only after its seasonal baseline and residual pipeline are verified.
5. Add controlled runner configuration, run manifests, and failure accounting.
6. Test leakage safeguards and common-origin comparisons.
7. Complete the code audit and readiness report.
8. Retrain and evaluate only after the dataset and protocol are approved.
