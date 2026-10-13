# Leakage-safe preprocessing report

## Methodological decisions

- Retain meters with at least 60% raw completeness over their observed span.
- Define hourly coverage as valid readings / 20 expected readings.
- A complete hour has >= 20 readings; its interval energies are summed.
- A partial hour has 1--19 readings. Its observed sum is audit-only and `t_kWh` is NaN.
- A missing hour has zero readings. Its observed sum and `t_kWh` remain NaN.
- No forward fill, backward fill, interpolation or other energy imputation is used.
- Split every meter chronologically 70%/15%/15% on the full hourly grid.
- Compute only backward lags and `shift(1)` rolling mean/std/min/max.
- Count 168-input/24-target windows only inside one split and only when every load is usable.
- Apply no shared scaling; future model scalers must fit training data only.

## Results

- Retained meters: 62 ({'Bareilly': 43, 'Mathura': 19}).
- Hourly rows: 516,528.
- Complete usable hours: 447,312.
- Partial unusable hours: 0.
- Fully missing hours: 69,216.
- True observed zero-energy complete hours: 34,895.
- Split rows: {'train': 361533, 'validation': 77449, 'test': 77546}.
- Usable 168-to-24 windows: {'train': 211669, 'validation': 41818, 'test': 41219}.
- Windows dropped for missing inputs/targets: {'train': 138022, 'validation': 23919, 'test': 24614}.

## Legacy isolation

New splits are under `data/research`; root-level splits and all model artifacts/results remain unchanged.
Legacy statistical/hybrid evaluators still use bidirectional interpolation and are unsafe for new scientific
results until a later modelling milestone. `train_general_model.py` combines all splits and randomly partitions
overlapping windows; it is a deployment path, not valid research evaluation evidence.

## Unresolved assumptions

The authoritative `t_kWh` definition, source URL, licence and timezone remain UNVERIFIED. Hourly SUM remains
conditional on confirming interval kWh. The 100% hourly threshold is deliberately conservative.
