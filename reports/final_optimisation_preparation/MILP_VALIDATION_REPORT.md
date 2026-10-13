# MILP validation report

Solver: HiGHS (branch-and-cut) via `scipy.optimize.milp`, SciPy 1.18.1, `mip_rel_gap = 0`.
Every tariff used here is **illustrative, not official**. Every appliance value is a **simulation assumption**.

## How to reproduce

```bash
pip install -r requirements.txt
python -m unittest discover -s optimisation/tests -t . -v      # 30 tests, new engine
python -m unittest review3.validate_demand_side_optimisation    # 10 tests, Review-3 record
python -m optimisation.run --scenarios                          # regenerates runs/
```

Result on a fresh clone: **30/30 and 10/10 pass.** Before this change the Review-3 validator could not run
on a fresh clone (`FAILED (errors=1)`, missing gitignored files).

## Test coverage vs the brief

| Brief requirement | Test(s) | Result |
|---|---|---|
| Each appliance runs required duration | `test_required_duration` | ✅ |
| Appliances stay inside allowed windows | `test_window_and_deadline`, `test_overnight_window`, `test_window_occurrence_respected_by_checker` | ✅ |
| Deadline respected | `test_window_and_deadline` (deadline = exclusive `latest_finish`), `test_infeasible_preferred_start_rejected` | ✅ |
| Non-interruptible stays consecutive | `test_non_interruptible_consecutive`, `test_non_interruptible_never_splits` | ✅ |
| Household maximum load never violated | `test_max_load_never_violated`, `test_base_load_above_cap_reports_infeasible` | ✅ |
| Binary variables valid | `test_binary` | ✅ |
| Before/after energy accounting consistent | `test_energy_accounting` | ✅ |
| Objective value matches schedule | `test_objective_matches_schedule` (solver `z*` = independently recomputed objective) | ✅ |
| Solver returns feasible/optimal | `test_solver_returns_optimal`, `test_feasibility_checker_agrees` | ✅ |
| *Extra:* true optimum | `test_matches_brute_force`: exhaustive enumeration of all start combinations equals MILP `z*` | ✅ |
| *Extra:* interruptible splitting | `test_interruptible_may_split` | ✅ |
| *Extra:* flat/slab never claim savings | `test_flat_tariff_claims_no_cost_saving`, `test_slab_bill_independent_of_schedule` | ✅ |
| *Extra:* tie-break keeps optimum, reduces moves | `test_tie_break_removes_pointless_moves` | ✅ |
| *Extra:* metric formulas | `test_metric_formulas`, `test_shifted_energy_not_netted` | ✅ |
| *Extra:* output contract | `test_output_contract_columns` | ✅ |
| *Extra:* Review-3 reproduction | `test_same_cost_result_fewer_pointless_moves` | ✅ |
| *Extra:* heuristic interface | `test_benchmark_uses_same_scoring` | ✅ |

Rule tests run on 3 synthetic forecasts × 3 tariffs (ToU, flat, slab) with all 5 appliances.

## Scenario results

Forecast: committed Review-3 example (LinearRegression, Bareilly BR02, 2020-09-06 04:00 → 2020-09-07 03:00).
Household: cap 3.5 kW, w_cost 0.3, w_peak 0.7. Outputs: `runs/<scenario>/`, summary `runs/scenario_summary.csv`.

| Scenario | Tariff | Cost ₹ before → after | Cost ↓ | Peak kW before → after | Peak ↓ | PAR before → after | Shifted kWh | Moved |
|---|---|---|---|---|---|---|---|---|
| A Review-3 reproduction (4 appl.) | ToU | 97.64 → 88.64 | 9.22 % | 2.34 → 2.34 | 0 % | 3.59 → 3.59 | **2.0** | dishwasher |
| B 5 appliances | ToU | 106.74 → 97.04 | 9.09 % | 2.34 → 2.34 | 0 % | 3.16 → 3.16 | 3.4 | dishwasher, EV |
| C 5 appliances | Flat ₹6 | 106.53 → 106.53 | 0 % | 2.34 → 2.34 | 0 % | 3.16 → 3.16 | 0 | none |
| D 5 appliances | Slab | 44.39 → 44.39 | 0 % | 2.34 → 2.34 | 0 % | 3.16 → 3.16 | 0 | none |
| E Morning cluster (geyser + pump at 07:00) | ToU | 106.74 → 97.04 | 9.09 % | 3.34 → 2.34 | **29.9 %** | 4.52 → 3.16 | 4.4 | dishwasher, pump, EV |
| F = A without tie-break | ToU | 97.64 → 88.64 | 9.22 % | 2.34 → 2.34 | 0 % | 3.59 → 3.59 | 4.6 | washer, dishwasher, pump |

Runtime: 0.03–0.09 s per scenario (both stages).

### Findings

1. **A reproduces the Review-3 cost result exactly** (₹97.64 → ₹88.64). Shifted energy falls from 4.6 to
   2.0 kWh. **F** (tie-break off) reproduces the Review-3 4.6 kWh, which confirms the 2.6 kWh difference came only from
   arbitrary equal-objective moves (washer 9→16, pump 12→17, both within the ₹6.0 band).
2. **Flat and slab tariffs (C, D):** no cost change, `cost_responsive = false`, and no spurious moves.
   With the current base load, peak is already minimal, so nothing moves.
3. **Peak reduction is real only when the baseline creates an avoidable peak (E):** geyser + pump at 07:00
   → pump moved to 08:00, peak −29.9 %, PAR 4.52 → 3.16. In A/B the peak is set by the geyser at the
   lowest-base hour and cannot be reduced. The 0 % is correct.
4. **Netting in the legacy shifted-energy formula (B):** legacy formula 2.0 kWh vs per-appliance 3.4 kWh.
   The dishwasher arriving at 22–23 and the EV leaving 22–23 cancel in the aggregate profile.
5. Every scenario: duration, window, deadline, contiguity, cap and binarity checks pass
   (`check_schedule` returns no violations); energy conserved.

## Limitations

* Single example day/meter. Multi-day evaluation should wait for the final dataset and forecast interface.
* All tariffs illustrative; final numbers depend on Govardhan's official tariff.
* Hourly resolution; appliance values assumed.
