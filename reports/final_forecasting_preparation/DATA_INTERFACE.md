# Forecasting Data Interface

## 1. Purpose

The forecasting package accepts hourly household-load observations through a canonical data interface. Source-specific column names can be mapped without embedding assumptions about a particular dataset.

## 2. Canonical Input Schema

| Column | Requirement | Meaning |
|---|---|---|
| `timestamp` | Required | Observation timestamp |
| `meter_id` | Required | Meter identifier |
| `load` | Required | Numeric load observation |
| `location` | Optional | Geographic or administrative identifier |
| `quality` | Optional | Source-provided observation-quality metadata |

Source columns can be mapped using `forecasting.data_adapter.normalize_load_data`.

## 3. Input Validation

The adapter must reject missing required columns, missing or non-finite loads, and duplicate meter/timestamp observations. It sorts observations chronologically and must not modify the input DataFrame.

Normalization is not imputation. It must not silently interpolate, forward-fill, backward-fill, or invent missing load values.

## 4. Temporal Requirements

- Sampling frequency: hourly.
- Historical context: 168 consecutive hourly observations.
- Forecast horizon: 24 hours.
- Forecast-origin stride: 24 hours.
- Forecasts may use only observations available at or before their origin.

The common forecaster interface validates history length, timestamp order, hourly spacing, and finite load values.

## 5. Existing Evaluation Adapter

`forecasting.evaluation_adapter.to_research_schema` maps canonical fields to the current research schema:

| Canonical field | Research field |
|---|---|
| `timestamp` | `x_Timestamp` |
| `meter_id` | `meter` |
| `load` | `t_kWh` |
| `location` | `city` |

If location is unavailable, the adapter uses the placeholder `unspecified`. This is not a verified geographic label.

The adapter does not establish units, timezone semantics, or quality-flag meanings. These must be confirmed with the selected dataset owner before final training.

## 6. Partition and Leakage Rules

1. Preserve the authoritative train, validation, and test boundaries.
2. Fit learned preprocessing only on the appropriate training partition.
3. Use validation data for development and model selection under the approved protocol.
4. Keep final test observations and targets out of current preparation and model selection.
5. Track missing observations and invalid forecast origins rather than silently repairing them.
6. Compare models on the same complete forecast origins.

## 7. Dataset Approval Checklist

Before final training, document:

- Dataset source, licence, and approved version.
- Load units and any scaling or conversion.
- Timestamp timezone and daylight-saving treatment, if applicable.
- Meter and location identifiers.
- Sampling frequency and missing-data policy.
- Quality flags and exclusion rules.
- Train/validation/test boundaries.
- Whether sufficient consecutive hourly observations exist for the required context and horizon.

## 8. Current Limitation

The final dataset decision is pending. The interface is therefore dataset-agnostic, but its compatibility with the eventual dataset has not yet been established.
