# Optimisation audit (pre-change)

Scope: `review3/demand_side_optimisation.py`, `review3/validate_demand_side_optimisation.py`,
`reports/review3/demand_side_optimisation/*`, the appliance config, the illustrative tariff,
the dashboard hooks (`review3/dashboard_integration.py`, `dashboard.py`) and the metric code.
Audit written before any edit. Commit audited: `db180f7` on `keshav/optimization`.

## 1. What is valid

| Item | Finding |
|---|---|
| Formulation | Genuine joint MILP: binary on/off `x[a,t]`, binary start `s[a,k]`, continuous peak `P`. One start per appliance and `x`–`s` linking are correct for non-interruptible loads. |
| Solver | `scipy.optimize.milp`, which calls **HiGHS** (manifest message: "HiGHS Status 7: Optimal"). The claim "SciPy milp using HiGHS" is **true**. PuLP/CBC is not used anywhere. |
| Peak logic | Epigraph `base_t + Σ P_a x[a,t] ≤ P` plus hard cap `≤ 3.5 kW` is correct. |
| Deadline / window | Start variables are only created when the whole run fits inside `[earliest_start, latest_finish)`; `latest_finish` is exclusive. Correct for windows that do not cross midnight. |
| Time alignment | Windows and tariff are mapped to each forecast timestamp's clock hour (horizon is 04:00 → 03:00). Correct. |
| Cost | Includes the base-load forecast before and after, so % savings are not inflated. Correct. |
| Baseline | Each appliance at a feasible preferred start (the old "everything at 19:00" baseline is gone). Correct. |
| Labelling | Appliances "ILLUSTRATIVE / SIMULATED", tariff "illustrative … not official". No renewable/carbon metrics. Correct. |
| Energy | Flexible energy conserved before/after (6.6 kWh). Correct. |

## 2. What is simulation (must stay labelled)

* Base load = the LinearRegression forecast for one meter (Bareilly BR02, origin 2020-09-06 03:00), treated as the non-flexible background. Appliance disaggregation is **not measured**.
* All appliance powers, durations, windows and preferred starts are assumptions.
* The ToU tariff (4.0 / 6.0 / 9.0 / 4.5 ₹/kWh) is an illustrative scenario, not an official tariff.
* 3.5 kW household cap and weights 0.3 / 0.7 are assumptions.

## 3. Defects

| # | Defect | Evidence | Severity |
|---|---|---|---|
| D1 | **Shifted energy inflated by tie moves.** Reported 4.6 kWh, but only the dishwasher move (19–20 → 22–23, 2.0 kWh) changes cost. Washing machine 9→16 and pump 12→17 stay in the same ₹6.0 "Normal" band and do not change the peak: the solver picked them arbitrarily among equal-objective schedules. | `tables/appliance_schedule.csv`, `schedules/hourly_before_after_schedule.csv` | High (misreported result) |
| D2 | **Flat-tariff trap not handled.** With a flat or slab tariff the cost term is constant; the same tie behaviour would show "shifted energy" with zero benefit, and nothing flags that cost cannot respond. | code inspection | High once official tariff arrives |
| D3 | **`interruptible` flag ignored.** Every appliance gets start variables + one contiguous block regardless of the flag. | `optimize_schedule` | Medium |
| D4 | **Shifted-energy formula nets across appliances.** `0.5·Σ|flex_after − flex_before|` uses the *aggregate* profile, so two appliances swapping hours count as 0 kWh moved. | formula | Medium |
| D5 | **Tariff hard-coded** in the optimiser module (`tariff_table`), ToU only; no flat/slab, no billing vs scheduling separation. | code | High (blocks official tariff) |
| D6 | Appliances, cap and weights are module constants; no input contract. | code | Medium |
| D7 | Forecast source hard-wired to the LinearRegression rows of a gitignored predictions CSV (first meter only). | `load_forecast` | Medium |
| D8 | **Validation fails on a fresh clone**: needs the gitignored predictions CSV and gitignored `schedule_arrays.npz` (`FAILED (errors=1)`, 0 tests run). | ran on clean checkout | Medium |
| D9 | Sign convention inconsistent: summary CSV `Change_pct` = (after − before)/before = **−9.22 %**; dashboard `cost_reduction_pct` = **+9.22 %**. | CSV vs `dashboard_integration.py` | Low |
| D10 | No solver runtime recorded; output column names differ from the requested contract. | manifest | Low |
| D11 | No pre-check that base load ≤ cap; an infeasible case only surfaces as a generic solver failure. | code | Low |
| D12 | Windows cannot cross midnight (an overnight EV 22:00–06:00 is impossible to express). | `feasible_start_slots` | Low |
| D13 | Manifest stores an absolute path from another machine (`/home/stackdev_24/...`); `scipy` missing from `requirements.txt`. | manifest, requirements | Low |

## 4. Peak / PAR result is genuine, not a bug

Peak 2.34 kW is at 07:00 = geyser 2.0 kW + 0.34 kW base. The geyser window (05–09) already sits at the lowest-base hour, so no feasible schedule has a lower peak. 0 % peak/PAR change is the correct answer for this case and must be reported as such.

## 5. Dashboard integration

`review3/dashboard_integration.py` imports `APPLIANCES, baseline_schedule, calculate_metrics, load_forecast, optimize_schedule, schedule_table, tariff_table`. Per milestone rules the dashboard was **not modified**. The legacy module was therefore kept as a frozen Review-3 record (only D8 was patched) and the new engine was built alongside in `optimisation/`. Switching the dashboard to the new engine is the next milestone.

## 6. Disposition

| Defect | Fixed in |
|---|---|
| D1, D2 | Lexicographic stage-2 tie-break + `cost_responsive` flag (`optimisation/milp.py`, `metrics.py`) |
| D3 | Interruptible appliances use `Σ x = d` only (`milp.py`) |
| D4 | Per-appliance shifted energy (`problem.py`, `metrics.py`) |
| D5 | Tariff adapter: flat / ToU / slab (`tariff.py`) |
| D6, D7 | Input contract (`contracts.py`, JSON configs) |
| D8 | Legacy loader + validator fall back to committed files; new tests are self-contained |
| D9, D10 | Single metric function, positive = improvement; `runtime`, contract columns |
| D11 | Pre-solve infeasibility check → `solver_status = "infeasible"` |
| D12 | Overnight windows supported, one occurrence per horizon |
| D13 | Relative paths in new outputs; `scipy>=1.9` added |
