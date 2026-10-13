# Household Energy Management Dashboard

> **Legacy / pre-correction notice:** root-level model result CSVs and the old
> claim that the hybrid was the "best result" predate the leakage-safe research
> pipeline and common-origin evaluator. They are retained as development history,
> not valid research evidence. Current Review-3 evidence is under
> `reports/review3/individual_models/`.

Forecasts a household's electricity usage (ARIMA+GRU model), then recommends
when to run appliances (AC, geyser, washing machine, water pump, EV charger)
based on time-of-day electricity pricing and simulated solar availability.

## Quick start (just want the app running)

1. Install Python 3.10+ if you don't have it.

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure these files are all in the SAME folder:
   - `train.csv`, `val.csv`, `test.csv`   (the processed household data)
   - `train_general_model.py`
   - `forecast_model.py`
   - `dashboard.py`

4. Train the model (only needs to be done once):
   ```
   python train_general_model.py
   ```
   This takes a few minutes. When it finishes, you'll see two new files:
   `general_gru_model.pt` and `general_model_meta.joblib`.

5. Run the app:
   ```
   streamlit run dashboard.py
   ```
   This opens in your browser automatically. Do NOT run it with
   `python dashboard.py` -- it must be `streamlit run`.

## What you can do in the app

- **Upload a CSV** of any household's hourly usage (needs columns named
  exactly `timestamp` and `kwh`), OR
- **Pick a sample household** from the existing research dataset
- See a 30-day forecast, and a recommended appliance schedule for tomorrow
  with cost savings and solar usage estimates

## Important honesty notes (mention these if presenting this to anyone)

- Appliance power ratings (how many kW a geyser/AC/etc. uses) are standard
  published numbers, not something the model learned -- there's no
  per-appliance sensor data in this project.
- The electricity pricing shown is ONE example Indian time-of-day pricing
  scenario for demonstration, not a live rate feed.
- Solar generation is a simulated daily curve, not real weather data.
- The forecast gets less reliable the further into the 30 days you look --
  day 1 is solid, day 30 is a rough trend estimate. Normal for any
  forecast this far out.

## The rest of the project (research/model comparison scripts)

The app uses one final model (ARIMA+GRU). Getting there involved comparing
several different models -- these scripts are the research trail, not
needed to just run the app, but useful for understanding/reproducing the
full study:

| Script | What it does |
|---|---|
| `step1_verify_clean.py` | Checks the raw Kaggle smart-meter data for problems |
| `step2_pipeline.py` | Cleans data, splits into train/val/test, builds features |
| `step3_statistical_baselines.py` | Tests Naive, ARIMA, Holt-Winters, etc. |
| `step4_ml_recursive.py` | Tests Random Forest/XGBoost/LightGBM (recursive forecasting) |
| `step5_ml_direct.py` | Same ML models, direct forecasting (this approach won) |
| `step6_smape_diagnostic.py` | Explains a metric discrepancy found in step 5 |
| `step7_gru_direct.py` | Tests a GRU deep learning model |
| `step8_arima_gru.py` | Tests the ARIMA+GRU hybrid (best result) |
| `train_general_model.py` | Trains the FINAL model used in the app (works on any household) |
| `forecast_model.py` | The forecasting logic the app calls |
| `dashboard.py` | The app itself |

## Data source

Household smart-meter data: Mathura and Bareilly, Uttar Pradesh, India,
reportedly sourced from CEEW's smart-metering study (via a public Kaggle
upload -- see project write-up for provenance caveats).
