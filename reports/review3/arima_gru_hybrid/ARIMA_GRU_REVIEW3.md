# ARIMA–GRU hybrid — Review 3

## Architecture and leakage-safe training

The base is the benchmark-compatible conditional ARIMA(3,0,0). A one-layer, 32-hidden-unit,
unidirectional GRU receives 168 chronological prequential residuals and directly predicts 24
residual corrections. No attention or PSO is present. Each training residual was predicted by an
expanding AR fit before its target updated that fit. Gaps were not interpolated or filled.

## Training protocol and residual coverage

After a 168-equation warm-up, 286,079 chronological out-of-sample one-step residuals were
generated. They yielded 8,289 daily-spaced direct 24-hour training windows. Residual and target
scalers were fitted only on these training data. Validation loss selected the checkpoint with
patience 4 and a maximum of 20 epochs.

## Validation protocol and main result

The hybrid produced 1,794 of 1,799 canonical validation forecasts across 58 meters. Every table
compares models on those same 1,794 origins. Its pooled validation MAE was 0.1798 and RMSE
was 0.2869. These are validation-stage findings, not final test evidence.

## Interpretation

Relative to ARIMA, hybridisation reduced MAE by 17.37% and RMSE
by 13.80%. Relative to the standalone GRU, it reduced MAE by
8.38% and RMSE by 7.60%.
However, it was worse than Linear Regression by 22.63% MAE
and 13.76% RMSE, and worse than XGBoost by
21.74% MAE and 12.14% RMSE.
Thus residual learning helped its two component baselines, but did not establish the hybrid as the
strongest individual/hybrid candidate. The test split was neither loaded nor evaluated.

## Paired-origin and per-meter findings

The hybrid had lower origin-level MAE than ARIMA for 80.8% of origins
(mean hybrid-minus-ARIMA difference -0.0378; bootstrap
95% CI [-0.0403, -0.0354]). It beat GRU on
66.6% of origins. At meter level it reduced RMSE for
52/58 meters relative to ARIMA and
49/58 relative to GRU, but only
4/58 relative to Linear Regression.

## Per-horizon behavior

The saved horizon table shows hybrid MAE/RMSE of 0.0982/0.1910 at horizon 1 and
0.1728/0.2525 at horizon 24. The correction substantially reduces ARIMA's long-horizon error,
although Linear Regression and XGBoost remain stronger at horizon 24.

## Runtime and residual diagnostics

Residual generation plus GRU fitting required 230.08 seconds on CPU; inference
averaged 0.091 ms per origin. Error variance fell from 0.1106
before correction to 0.0820 after correction. Lag-24 error autocorrelation fell from
0.544 to 0.342, indicating that the GRU
removed some—but not all—remaining temporal structure.

## Main conclusion and next step

ARIMA-GRU provides credible validation evidence that residual learning improves ARIMA and a plain
GRU. It does not outperform the simpler Linear Regression or XGBoost benchmarks. Attention may be
implemented next as a controlled ablation, but is not justified by a claim that the current hybrid
is already the leading model; it must demonstrate an additional reproducible gain under this same
protocol.

## Limitations

Five canonical origins lack a complete 168-hour residual history. `t_kWh` semantics remain
authoritatively unverified. The ARIMA component is conditional least-squares ARIMA(3,0,0), not a
broadly tuned state-space ARIMA. Validation early stopping makes this preliminary selection-stage
evidence. Temporal attention should only be attempted if the measured hybrid result justifies it.
