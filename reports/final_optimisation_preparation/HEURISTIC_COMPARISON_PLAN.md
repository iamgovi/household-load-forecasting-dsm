# Heuristic comparison plan (GA / PSO vs MILP)

Status: **interface ready, heuristics not implemented.** The MILP stays the primary method. A heuristic is
only a benchmark.

Code: `optimisation/heuristic_interface.py`: `Scheduler` protocol, `evaluate`, `benchmark`,
`MILPScheduler`, `BaselineScheduler`, and placeholder `GeneticAlgorithmScheduler` / `ParticleSwarmScheduler`
classes that raise `NotImplementedError`.

## Fairness (enforced in code)

| Must be identical | How it is enforced |
|---|---|
| 24-h forecast | every scheduler receives the same `SchedulingProblem` |
| Tariff | same object → same `scheduling_rates` |
| Appliance set + constraints | same `problem.appliances`, `problem.allowed`, `feasible_starts` |
| Objective | all schedules scored by `problem.objective(schedule, baseline)` |
| Feasibility | all schedules checked by `problem.check_schedule`; infeasible results reported, not dropped |
| Baseline | same `problem.baseline_schedule()` |

## Planned encoding

* Non-interruptible appliance → one integer gene = index into `feasible_starts(a)` (always window-feasible).
* Interruptible appliance → 24-bit mask restricted to `allowed`, repaired to exactly `d_a` ones.
* Max-load violations → penalty `λ · Σ_t max(0, total_t − P_max)` during search; the *reported* objective is the un-penalised one plus a feasibility flag.
* PSO: continuous position per gene, rounded to nearest feasible index (binary PSO for masks).

## Protocol

1. Same scenarios as `MILP_VALIDATION_REPORT.md` (A–E), plus synthetic forecasts from several meters/days.
2. Heuristics: ≥ 30 independent seeds each; report best / mean / std of objective, % runs feasible,
   gap to MILP optimum `100·(z_heur − z*)/z*`, runtime.
3. MILP: one run (deterministic, proven optimal with gap 0).
4. Equal reporting of runtime on the same machine.
5. Scaling study (where heuristics might matter): 15-minute slots (96 slots) and 10–20 appliances.

## Expected outcome (to verify, not to claim in advance)

At 24 slots × 5 appliances the MILP solves to proven optimality in < 0.1 s, so a heuristic can at best
match it. The meaningful comparison is solution quality vs runtime as the problem grows.
