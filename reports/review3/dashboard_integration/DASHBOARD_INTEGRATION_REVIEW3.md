# Review-3 Dashboard Integration

## Purpose

The Streamlit dashboard is a presentation layer over the frozen, corrected Review-3 forecasting and simulation-based demand-side optimisation artifacts. It does not retrain a forecasting model or implement a second scheduling algorithm.

## Default validated demonstration

- Forecast source: saved corrected validation predictions used by the demand-side optimisation milestone
- Model: Linear Regression
- Meter/city: BR02, Bareilly
- Forecast origin: 2020-09-06 03:00:00
- Horizon: the next 24 hourly targets
- Optimiser: the validated joint MILP implementation in `review3/demand_side_optimisation.py`
- Appliances and tariff: illustrative simulation parameters, not measured appliance data or an official tariff

The dashboard reproduces the frozen demand-side optimisation evidence:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Electricity cost | ₹97.643814 | ₹88.643814 | −₹9.000000 (−9.217174%) |
| Total household peak | 2.340568 | 2.340568 | 0.000000% |
| Peak-to-average ratio | 3.588325 | 3.588325 | 0.000000% |
| Flexible energy | 6.600000 kWh | 6.600000 kWh | 0.000000 kWh |
| Shifted energy | N/A | 4.600000 kWh | N/A |

The unchanged peak is reported rather than hidden: the deterministic baseline already achieves the minimum feasible peak for this example, while the optimiser shifts flexible energy to reduce cost.

## Integration decisions

The existing Streamlit page shell, wide layout, sidebar mode control, metric cards, charts, tables, and methodology expander were retained. Legacy research logic was removed from the default path: deployment-model loading, 30-day forecasts, synthetic solar, unsupported EV/AC claims, greedy appliance scheduling, invalid deadline logic, universal 19:00 baselines, appliance-only cost calculations, and ambiguous shifted-energy reporting.

`review3/dashboard_integration.py` is the testable adapter. It loads the saved forecast, calls the same tariff, baseline, optimiser, metric, and schedule functions used by the validated demand-side milestone, and returns display-ready records. There is no duplicated optimisation formulation in the dashboard.

## Upload mode

An experimental inference mode accepts CSV files with `timestamp,kwh`, unique timestamps, and numeric non-negative loads. It identifies contiguous hourly segments without interpolation and uses the latest segment containing at least 168 hours. Earlier segments separated by gaps are reported and excluded. A global direct-output Linear Regression model reproducing the frozen 168-hour-input/24-hour-output configuration is fitted on the authoritative research training split only and saved as a dashboard inference artifact. Its next-24-hour output is passed to the same validated MILP used by the frozen example. Uploaded results are deployment demonstrations and are not presented as official Review-3 model-comparison evidence.

## Validation

Automated checks verify the 24-hour saved forecast, exact reproduction of frozen optimisation metrics, optimal solver status, conservation of flexible energy, validated appliance scope, removal of legacy default logic and unsupported claims, safe upload processing, train-only forecasting, and an end-to-end upload-to-forecast-to-MILP run using `dashboard_BR02.csv`. A Streamlit application test also verifies both dashboard modes render without exceptions.

## Reproduction

```bash
cd /home/stackdev_24/Downloads/arima_gru_results_per_meter
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m review3.validate_dashboard_integration
.venv/bin/streamlit run dashboard.py
```

For the panel, leave the default `Review-3 validated example` selected and click **Run 24-Hour Optimisation**.

## Limitations

This is validation-stage evidence, not final test-set performance. The aggregate smart-meter forecast does not provide measured appliance separation. Appliance profiles, household power limit, and tariff are illustrative. Forecast-error robustness, official tariff validation, renewable/carbon effects, and arbitrary-household inference remain outside this milestone.
