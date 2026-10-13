# Dataset audit and provenance report

Generated reproducibly by `preprocess/dataset_audit.py` from the two 2020 raw CSV files.

## Verified from files

- Cities encoded by filenames: Bareilly and Mathura.
- Raw meters: 84.
- Meters represented in current train/validation/test files: 62.
- Timestamp column: `x_Timestamp`; observed dominant interval: 180 seconds.
- Load/energy column: `t_kWh`; its values fluctuate rather than behave like a cumulative counter.
- Duplicate meter-timestamps: 0.
- Out-of-order timestamps: 0.
- Negative `t_kWh` values: 0.
- Estimated missing 3-minute intervals inside meter coverage spans: 3585120.
- Raw date extent across meters: 2020-01-01 00:00:00 to 2020-12-31 23:57:00.

## UNVERIFIED metadata

- Official dataset title and publisher/source: **UNVERIFIED**.
- Source URL/reference: **UNVERIFIED**.
- Licence and redistribution terms: **UNVERIFIED**.
- Timestamp timezone: **UNVERIFIED**.
- Authoritative definition and measurement process for `t_kWh`: **UNVERIFIED**.
- Whether each meter ID maps one-to-one to a household: **UNVERIFIED**.

## Interpretation of `t_kWh` and hourly aggregation

The column name identifies kWh and consecutive values frequently decrease
(29.93% of valid consecutive comparisons), so it is not a
monotonically increasing cumulative register. It is empirically consistent with energy recorded per
three-minute interval. On that interpretation, hourly **SUM** is appropriate because energy quantities
are additive. This conclusion remains conditional until the source data dictionary confirms the field.

## Current preprocessing issues

- Hourly resampling calls pandas `sum` without `min_count=1`. A completely empty hour can therefore
  become zero instead of missing, contrary to the pipeline's diagnostic message. Any missing values
  that remain are forward-filled for up to three hours, which repeats energy. Both behaviours require
  replacement by an explicit, documented gap policy before regenerating modelling datasets.
- Existing statistical/hybrid scripts use bidirectional interpolation, including on test data. That can
  use future observations and must not be used for research evaluation.
- Empty resampling bins previously lost `city`; the pipeline now assigns city as invariant meter metadata.
- The 60% retention rule is calculated only over each meter's observed first-to-last span and therefore
  does not penalize missing periods before installation or after final observation.

An asterisk on plot axes (`kWh*`) indicates that the unit comes from the column label but is not yet
confirmed by authoritative metadata.
