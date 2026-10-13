# Legacy evaluation audit

These paths are historical/deployment code and are not evidence under the new
common research contract.

| Path | Current behavior | Contract mismatch |
|---|---|---|
| `preprocess/baseline_model.py` | Lag-based point baselines plus rolling 24-step statistical forecasts on root splits | Mixed one-step/24-step tasks; bidirectional test interpolation |
| `preprocess/ml_models.py` | Recursive 24-step forecasts | Bidirectional history interpolation; model-specific origins |
| `preprocess/ste5.py` | One scalar prediction at exactly `t+24` | Not a direct 24-output profile; validation unused |
| `preprocess/gru.py` | Direct 24-output GRU with 168-hour input | Legacy splits and model-specific window population |
| `preprocess/gru_improved.py` | Byte-identical copy of `gru.py` | Not an independent improved model |
| `preprocess/arima-gru.py` | Rolling ARIMA plus GRU residual forecasts | Bidirectional interpolation and validation-contaminated residual generation |
| `train_general_model.py` | General deployment GRU | Combines train/validation/test; randomly splits overlapping windows |
| `preprocess/generate_model.py` | Deployment forecast generation | Fits/trains with all splits and random overlapping windows |
| `forecast_model.py` | Live multi-day deployment forecast | Uses deployment artifact and bidirectional history interpolation |

Existing root-level result CSVs summarize unequal forecast definitions,
origins, household subsets and aggregation methods. They must not be entered
into the new research comparison tables.

`general_gru_model.pt` and `general_model_meta.joblib` are explicitly classified
as **DEPLOYMENT / NON-RESEARCH**. They are retained but excluded from research
result discovery.
