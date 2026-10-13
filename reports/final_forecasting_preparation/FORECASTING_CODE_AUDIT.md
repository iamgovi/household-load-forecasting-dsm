# Forecasting Code Audit

## 1. Objective

This audit documents the existing forecasting implementation, identifies reusable components, and records the changes required to establish a dataset-agnostic forecasting package.

The audit is preparatory. It does not certify final model performance, approve a dataset, or authorize final test evaluation.

## 2. Existing Architecture

### 2.1 Evaluation Contract

**Location:** `evaluation/contract.py`

Defines shared forecasting parameters and paths, including:

- Historical lookback: 168 hours
- Forecast horizon: 24 hours
- Forecast-origin stride: 24 hours
- Sampling frequency: hourly
- Random seed: 42
- Canonical prediction columns
- Research train, validation, and test paths

The new forecasting package reuses these constants through `forecasting/config.py` to reduce configuration drift.

### 2.2 Forecast-Origin Construction

**Location:** `evaluation/origins.py`

Provides research-split loading and canonical forecast-origin construction.

The existing contract requires valid historical context and valid target observations. Missing observations must not be silently filled or interpolated.

The final test partition must remain untouched during the current preparation phase.

### 2.3 Evaluation and Comparison

**Locations:**
- `evaluation/metrics.py`
- `evaluation/comparison.py`

The metrics implementation validates prediction schemas and calculates point and grouped evaluation metrics. Available metrics include MAE, RMSE, sMAPE, R², normalized RMSE, peak-load errors, peak timing error, and inference time.

The comparison implementation supports comparison on common complete forecast origins and restricts prediction discovery using research-eligibility metadata.

These components should be reused rather than independently reimplemented.

## 3. Existing Individual Models

**Location:** `review3/individual_models.py`

The existing runners cover:

| Model | Existing status |
|---|---|
| Seasonal Naive | Implemented |
| Holt-Winters | Implemented |
| ARIMA | Implemented |
| SARIMA | Limited runtime attempt |
| Linear Regression | Implemented |
| XGBoost | Implemented |
| GRU | Implemented |
| LSTM | Requires implementation or verification |

The legacy script has a protected command-line entry point. Its runners can write prediction files, metrics, run records, and checkpoints. They must not be called casually from package imports or automated tests.

## 4. Existing Hybrid Implementations

| Implementation | Location | Audit finding |
|---|---|---|
| ARIMA-GRU residual hybrid | `review3/arima_gru_hybrid.py` | Existing prequential residual-generation approach |
| PSO-ARIMA-GRU | `review3/arima_gru_pso.py` | Existing limited hyperparameter-search experiment; chronological split requires further leakage review |
| Attention ARIMA-GRU | `review3/arima_gru_attention.py` | Existing attention ablation; not assumed to improve performance |
| ARIMA-LSTM | Planned | Requires implementation and tests |
| SARIMA-GRU | Planned | Requires implementation and tests |

Existing hybrid scripts create their own output directories and write Review-3 artifacts. Their `main()` functions must not be invoked by package imports.

## 5. New Forecasting Package

The following components have been added:

- `forecasting/config.py`: shared contract constants.
- `forecasting/contracts.py`: prediction schema, timestamp, horizon, duplicate, and completeness validation.
- `forecasting/data_adapter.py`: dataset-agnostic load-data normalization without imputation.
- `forecasting/registry.py`: central metadata for candidate forecasting models.
- `forecasting/evaluation_adapter.py`: conversion from canonical input data to the existing research evaluation schema.
- `forecasting/legacy.py`: metadata mapping for existing implementations without importing or running legacy training code.

Initial tests cover configuration, prediction validation, data normalization, registry behavior, evaluation-schema conversion, and legacy metadata.

## 6. Known Limitations

1. The registry describes candidates but does not yet provide a unified prediction interface for every model.
2. Some registered models have no implementation in the new package.
3. Legacy functions use different calling conventions and output locations.
4. Existing runners must be isolated from the new package's output paths.
5. The PSO inner chronological split needs additional review for overlapping target windows.
6. Dataset-specific assumptions must be checked before adapting legacy models to new source data.
7. No final dataset has been approved for final training.

## 7. Required Next Changes

1. Define a shared forecasting interface.
2. Add controlled adapters for existing implementations.
3. Separate model execution from import and registration.
4. Implement and test the missing LSTM and ARIMA-LSTM components.
5. Assess the requirements for a leakage-safe SARIMA-GRU implementation.
6. Add explicit run metadata, failure tracking, and output isolation.
7. Expand tests for forecast-origin validity and leakage prevention.
8. Complete the data interface and readiness reports.

## 8. Protection Rules

- Do not overwrite existing Review-3 outputs or reports.
- Do not load or evaluate the final test partition during preparation.
- Do not claim final performance or select a final winning model.
- Do not silently fill missing observations.
- Do not treat model registration as proof that the implementation is complete.
- Do not initiate final retraining before the dataset and protocol are approved.

## 9. Audit Conclusion

The repository contains reusable forecasting, evaluation, and hybrid-model code. The new package establishes initial configuration, data, validation, registry, and legacy metadata components.

The principal remaining engineering task is to connect model implementations through a controlled, tested interface while preserving the existing research artifacts and evaluation safeguards.
