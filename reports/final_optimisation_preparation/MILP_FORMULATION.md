# MILP formulation

> We formulate residential appliance scheduling as a Mixed-Integer Linear Programming problem
> and solve it with **HiGHS** (branch-and-cut) through `scipy.optimize.milp`.

Code: `optimisation/milp.py` (model), `optimisation/problem.py` (rules, objective, feasibility checker).
Solver string recorded in every output: `HiGHS (branch-and-cut) via scipy.optimize.milp, SciPy <version>`.
`mip_rel_gap = 0` → results are proven optimal, not "good enough".

## Sets and data

| Symbol | Meaning |
|---|---|
| `t ∈ {0..23}` | hourly slot of the 24-h forecast horizon (aligned to real timestamps) |
| `a ∈ A` | flexible appliance; `A = A_NI ∪ A_I` (non-interruptible / interruptible) |
| `base_t` | forecast base load (kWh in slot t = average kW) |
| `P_a`, `d_a` | appliance power (kW), required duration (h) |
| `W_a ⊆ {0..23}` | allowed slots: one occurrence of `[earliest_start, latest_finish)`; `latest_finish` is the deadline (exclusive) |
| `K_a` | feasible starts: `{k : k..k+d_a−1 ⊆ W_a}` |
| `r_t` | scheduling price ₹/kWh from the tariff adapter |
| `P_max` | household maximum load (kW) |
| `b[a,t]` | baseline (un-optimised) schedule |
| `w_cost, w_peak` | objective weights |
| `C_base` | baseline cost `Σ_t r_t (base_t + Σ_a P_a b[a,t])` (normaliser) |

## Variables

| Variable | Domain | Count (5-appliance case) |
|---|---|---|
| `x[a,t]` ON/OFF | {0,1} | 120 |
| `s[a,k]` start, `a ∈ A_NI, k ∈ K_a` | {0,1} | 32 |
| `P` household peak | [0, P_max] | 1 |

## Constraints

| # | Constraint | Purpose |
|---|---|---|
| 1 | `Σ_{k∈K_a} s[a,k] = 1`  ∀a∈A_NI | exactly one start (non-interruptible) |
| 2 | `x[a,t] = Σ_{k∈K_a: k≤t<k+d_a} s[a,k]`  ∀a∈A_NI, t | ON exactly during the d_a-hour block → consecutive operation |
| 3 | `Σ_t x[a,t] = d_a`  ∀a∈A_I | required duration (may split) |
| 4 | `x[a,t] = 0`  ∀t∉W_a (variable bound) | earliest start + deadline |
| 5 | `base_t + Σ_a P_a x[a,t] ≤ P_max`  ∀t | household maximum load |
| 6 | `base_t + Σ_a P_a x[a,t] ≤ P`  ∀t | peak epigraph (P = max total load at optimum) |

For non-interruptible appliances, (1)+(2) imply duration and window automatically.

## Objective (lexicographic, two stages)

**Stage 1 (primary):**

```
min  z = w_cost · Σ_t r_t (base_t + Σ_a P_a x[a,t]) / C_base  +  w_peak · P / P_max
```

Both terms are dimensionless (cost relative to baseline, peak relative to the cap). Default
`w_cost = 0.3, w_peak = 0.7` (assumption, configurable in `household_default.json`).
No discomfort term is included (no defensible definition yet).

**Stage 2 (tie-break):**

```
min  Σ_a P_a Σ_t (1 − b[a,t]) x[a,t]          (= shifted energy)
s.t. stage-1 objective ≤ z* + 1e-7, constraints 1–6
```

Many schedules share the same `z*` (e.g. moving a washer between two equal-price hours). Stage 2 picks the
one that moves the least energy, so every reported move has a cost or peak reason. It never worsens `z*`
(test `test_tie_break_removes_pointless_moves`). It can be turned off with `lexicographic_tie_break: false`.

## Tariff behaviour

If `r_t` is constant (flat or slab), the cost term is a constant, the optimiser acts on peak only, and the
output carries `cost_responsive = false`. See `TARIFF_INTERFACE.md`.

## Size and runtime

5 appliances: 153 variables (152 binary), 149 constraints, ~0.03–0.09 s per solve including stage 2 (both stages).

## Changes vs Review-3 model

| | Review-3 | Now |
|---|---|---|
| Interruptible | ignored | constraint 3 |
| Overnight windows | not expressible | supported (one occurrence per horizon) |
| Ties | arbitrary schedule | stage-2 minimal-movement |
| Tariff | hard-coded ToU | adapter (flat / ToU / slab) |
| Infeasible base load | generic solver failure | pre-check, `solver_status = infeasible` |
| Optimality gap | HiGHS default | 0 |
| Formulation core (x, s, P, constraints 1, 2, 4–6, objective) | — | **retained** |
