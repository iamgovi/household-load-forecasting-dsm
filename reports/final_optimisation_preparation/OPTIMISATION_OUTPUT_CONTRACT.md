# Optimisation output contract

`optimise(forecast, tariff, appliances, household) → OptimisationResult`
(`optimisation/contracts.py`). This is what the dashboard will consume in the next milestone.

## Input contract (summary)

| Input | Format |
|---|---|
| Forecast | DataFrame/CSV, 24 consecutive hourly rows: `timestamp`, `predicted_kwh` (aliases `target_timestamp`, `prediction` accepted). No NaN, no negatives. Model internals are never imported. |
| Tariff | `Tariff` object or JSON (`TARIFF_INTERFACE.md`) |
| Appliances | JSON list: `appliance_id, name, power_kw, duration_h, earliest_start, latest_finish, interruptible, preferred_start, priority?, source_class, notes` |
| Household | JSON: `max_power_kw, w_cost, w_peak, lexicographic_tie_break, preferences?` |

## `result.hourly` (24 rows)

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | datetime | forecast target hour |
| `clock_hour` | 0–23 | |
| `tariff_rate` | ₹/kWh | scheduling signal for that hour |
| `base_load` | kWh (= avg kW) | forecast |
| `appliance_load_before` | kWh | flexible load, baseline schedule |
| `appliance_load_after` | kWh | flexible load, optimised schedule |
| `total_before` | kWh | base + appliances before |
| `total_after` | kWh | base + appliances after |
| `on_before__<id>`, `on_after__<id>` | 0/1 | per-appliance ON flags (for Gantt charts) |

## `result.summary`

| Key | Formula / meaning |
|---|---|
| `cost_before`, `cost_after` | `tariff.bill(total)` |
| `cost_reduction_pct` | `100·(before − after)/before` (positive = saving) |
| `peak_before`, `peak_after` | `max_t total_t` (hourly-average kW) |
| `peak_reduction_pct` | `100·(before − after)/before` |
| `par_before`, `par_after` | `peak / mean(total)` |
| `par_reduction_pct` | `100·(before − after)/before` |
| `shifted_energy` | `Σ_a P_a Σ_t max(0, x_after − x_before)` kWh, per appliance, never netted |
| `net_profile_shift_kwh` | legacy `0.5·Σ|flex_after − flex_before|` (comparison only) |
| `energy_before_kwh`, `energy_after_kwh`, `energy_conserved` | accounting check |
| `cost_responsive`, `cost_note` | whether the tariff lets shifting change cost |
| `baseline_exceeds_max_load` | baseline already above the cap |
| `solver_status` | `optimal` / `infeasible` / `limit_reached` / `error` |
| `runtime` | seconds (both stages, incl. model build) |
| `objective_value` | stage-1 objective `z*` |

If `solver_status != "optimal"`, metric fields are `None`, and `after_matrix` is all zeros.

## `result.appliance_schedule` (one row per appliance)

`appliance_id, name, power_kw, duration_h, energy_kwh, allowed_window, interruptible, start_before,
start_after, hours_before, hours_after, moved, shifted_energy_kwh, scheduling_cost_before,
scheduling_cost_after, source_class`

## `result.metadata`

Solver name and method, model size, stage statuses, tariff description (incl. `is_official`, `label`),
household settings, forecast start/end, appliance source classes, `renewable_metrics_included = false`,
`carbon_metrics_included = false`.

## Files

`result.save(dir)` writes `hourly.csv`, `appliance_schedule.csv`, `summary.json`.
Example outputs: `reports/final_optimisation_preparation/runs/<scenario>/`.

## Mapping from the Review-3 dashboard keys

| Review-3 (`dashboard_integration.py`) | New |
|---|---|
| `base_forecast` | `base_load` |
| `flexible_before` / `flexible_after` | `appliance_load_before` / `appliance_load_after` |
| `tariff` | `tariff_rate` |
| `shifted_energy_kwh` (aggregate, tie-inflated) | `shifted_energy` (per appliance, tie-broken) |
| `solver_message` | `metadata.solver.solver_message` |
| — | `runtime`, `cost_responsive`, `par_reduction_pct` |
