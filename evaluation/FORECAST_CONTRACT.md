# Common 24-hour research forecast contract

At an hourly forecast origin `t`, a model may use only information timestamped
at or before `t` and must predict the same meter's loads at `t+1` through
`t+24`. The standard sequence context is the immediately preceding 168 hourly
loads, ending at `t`.

All 168 context loads and all 24 targets must be observed and valid. No value is
interpolated or filled. A context may include an earlier chronological split;
for example, the end of training may provide validation history. All 24 target
timestamps must be wholly inside the evaluated split.

Forecast origins use a fixed 24-hour stride starting at each meter's split
boundary. This produces one complete day-ahead forecast per day and avoids
counting the same target hour in up to 24 highly overlapping forecasts.

Canonical validation and test origin tables define the shared sample population.
A model failure does not remove an origin silently: comparisons use the explicit
intersection of complete model forecasts and report missing forecasts.

Predictions use the long schema defined in `evaluation.contract.PREDICTION_COLUMNS`.
Metrics use the formulas in `evaluation.contract.METRIC_DEFINITIONS`. Overall
results report both pooled errors and the fixed macro mean of per-meter metrics;
neither is selected opportunistically.

`general_gru_model.pt`, `general_model_meta.joblib`, `train_general_model.py`,
`preprocess/generate_model.py`, and `forecast_model.py` are deployment/non-research
artifacts and are excluded from scientific result discovery.
