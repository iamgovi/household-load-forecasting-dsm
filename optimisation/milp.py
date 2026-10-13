"""MILP formulation and solve (HiGHS via scipy.optimize.milp).

Variables
  x[a,t] ∈ {0,1}   appliance a ON in slot t           (all appliances)
  s[a,k] ∈ {0,1}   non-interruptible a STARTS at k    (only feasible starts k)
  P ≥ 0            household peak (continuous)

Constraints
  (1) Σ_k s[a,k] = 1                                   non-interruptible: exactly one start
  (2) x[a,t] = Σ_{k: k ≤ t < k+d_a} s[a,k]             non-interruptible: ON exactly in the block
  (3) Σ_t x[a,t] = d_a                                 interruptible: required duration
  (4) x[a,t] = 0 for t outside the allowed window      (variable upper bound; encodes earliest start + deadline)
  (5) base_t + Σ_a P_a x[a,t] ≤ P_max                  household maximum load
  (6) base_t + Σ_a P_a x[a,t] ≤ P                      peak epigraph

Objective
  Stage 1:  min  w_cost · Σ_t r_t (base_t + Σ_a P_a x[a,t]) / C_base  +  w_peak · P / P_max
  Stage 2 (lexicographic tie-break, optional):
            min  Σ_a P_a Σ_t (1 − b[a,t]) x[a,t]       (= shifted energy)
            s.t. stage-1 objective ≤ z1* + tol
  Stage 2 removes arbitrary moves between equally-good schedules, so reported shifted energy
  reflects only moves that actually improve cost or peak.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from optimisation.contracts import HORIZON, Appliance, Household, OptimisationResult
from optimisation.metrics import appliance_table, hourly_table, summary_metrics
from optimisation.problem import SchedulingProblem
from optimisation.tariff import Tariff

SOLVER_NAME = f"HiGHS (branch-and-cut) via scipy.optimize.milp, SciPy {scipy.__version__}"
_STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "error"}
STAGE2_TOL = 1e-7


def _build(problem: SchedulingProblem, baseline: np.ndarray):
    n_a, T = problem.n_appliances, HORIZON
    nx = n_a * T
    starts = [(a, k) for a, app in enumerate(problem.appliances) if not app.interruptible
              for k in problem.feasible_starts(a)]
    s0 = nx
    p_idx = nx + len(starts)
    n = p_idx + 1
    xi = lambda a, t: a * T + t  # noqa: E731

    rows: list[dict[int, float]] = []
    lo: list[float] = []
    hi: list[float] = []
    for a, app in enumerate(problem.appliances):
        if app.interruptible:
            rows.append({xi(a, t): 1.0 for t in range(T)}); lo.append(app.duration_h); hi.append(app.duration_h)
            continue
        mine = [(i, k) for i, (aa, k) in enumerate(starts) if aa == a]
        rows.append({s0 + i: 1.0 for i, _ in mine}); lo.append(1); hi.append(1)
        for t in range(T):
            row = {xi(a, t): 1.0}
            for i, k in mine:
                if k <= t < k + app.duration_h:
                    row[s0 + i] = -1.0
            rows.append(row); lo.append(0); hi.append(0)
    P = problem.powers
    h = problem.household
    for t in range(T):
        load = {xi(a, t): P[a] for a in range(n_a)}
        rows.append(load); lo.append(-np.inf); hi.append(h.max_power_kw - problem.base[t])
        peak = dict(load); peak[p_idx] = -1.0
        rows.append(peak); lo.append(-np.inf); hi.append(-problem.base[t])

    A = lil_matrix((len(rows), n))
    for r, row in enumerate(rows):
        for col, val in row.items():
            A[r, col] = val

    ub = np.ones(n)
    ub[:nx] = problem.allowed.reshape(-1).astype(float)  # constraint (4)
    ub[p_idx] = h.max_power_kw
    lb = np.zeros(n)
    integrality = np.ones(n)
    integrality[p_idx] = 0

    norm = problem.cost_normaliser(baseline)
    c1 = np.zeros(n)
    for a in range(n_a):
        for t in range(T):
            c1[xi(a, t)] = h.w_cost * P[a] * problem.rates[t] / norm
    c1[p_idx] = h.w_peak / h.max_power_kw
    c1_const = h.w_cost * float(np.sum(problem.base * problem.rates)) / norm

    c2 = np.zeros(n)
    for a in range(n_a):
        for t in range(T):
            c2[xi(a, t)] = P[a] * (1 - baseline[a, t])
    return dict(A=A.tocsr(), lo=np.array(lo), hi=np.array(hi), bounds=Bounds(lb, ub), integrality=integrality,
                c1=c1, c1_const=c1_const, c2=c2, nx=nx, n=n, starts=starts, n_rows=len(rows))


def solve_milp(problem: SchedulingProblem, baseline: np.ndarray | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Solve the scheduling MILP. Returns (schedule or None, solver info)."""
    t0 = time.perf_counter()
    baseline = problem.baseline_schedule() if baseline is None else baseline
    h = problem.household
    info: dict[str, Any] = {"solver": SOLVER_NAME, "method": "branch-and-cut (HiGHS MIP)"}

    over = np.flatnonzero(problem.base > h.max_power_kw + 1e-12)
    if len(over):
        info.update(solver_status="infeasible", solver_message=(
            f"Base load alone exceeds max_power_kw ({h.max_power_kw}) at slots {over.tolist()}"),
            runtime=time.perf_counter() - t0)
        return None, info
    for a, app in enumerate(problem.appliances):
        if not app.interruptible and not problem.feasible_starts(a):
            info.update(solver_status="infeasible", solver_message=f"{app.appliance_id}: no feasible start",
                        runtime=time.perf_counter() - t0)
            return None, info

    m = _build(problem, baseline)
    info.update(n_variables=m["n"], n_binary=m["n"] - 1, n_constraints=m["n_rows"])
    cons = LinearConstraint(m["A"], m["lo"], m["hi"])
    r1 = milp(m["c1"], integrality=m["integrality"], bounds=m["bounds"], constraints=cons,
              options={"disp": False, "mip_rel_gap": 0.0})
    info["stage1_status"] = _STATUS.get(r1.status, str(r1.status))
    info["stage1_message"] = r1.message
    if r1.x is None or r1.status != 0:
        info.update(solver_status=info["stage1_status"], solver_message=r1.message, runtime=time.perf_counter() - t0)
        return None, info
    z1 = float(r1.fun) + m["c1_const"]
    x_final = r1.x
    info["stage2_used"] = False
    if h.lexicographic_tie_break:
        tie = LinearConstraint(m["c1"].reshape(1, -1), -np.inf, float(r1.fun) + STAGE2_TOL)
        r2 = milp(m["c2"], integrality=m["integrality"], bounds=m["bounds"], constraints=[cons, tie],
                  options={"disp": False, "mip_rel_gap": 0.0})
        info["stage2_status"] = _STATUS.get(r2.status, str(r2.status))
        if r2.x is not None and r2.status == 0:
            x_final = r2.x
            info["stage2_used"] = True
    schedule = np.rint(x_final[:m["nx"]]).astype(int).reshape(problem.n_appliances, HORIZON)
    info.update(solver_status="optimal", solver_message=r1.message, objective_value=z1,
                objective_value_final_schedule=problem.objective(schedule, baseline),
                runtime=time.perf_counter() - t0)
    return schedule, info


def optimise(forecast, tariff: Tariff, appliances: list[Appliance], household: Household | None = None) -> OptimisationResult:
    """End-to-end: validate inputs → baseline → MILP → metrics → output contract."""
    problem = SchedulingProblem(forecast, tariff, appliances, household or Household())
    before = problem.baseline_schedule()
    after, info = solve_milp(problem, before)
    metadata = {"solver": info, "tariff": tariff.describe(), "household": problem.household.__dict__,
                "forecast_start": str(problem.timestamps[0]), "forecast_end": str(problem.timestamps[-1]),
                "appliance_source_classes": sorted({a.source_class for a in appliances}),
                "renewable_metrics_included": False, "carbon_metrics_included": False}
    if after is None:
        summary = {k: None for k in ("cost_before", "cost_after", "cost_reduction_pct", "peak_before", "peak_after",
                                     "peak_reduction_pct", "par_before", "par_after", "shifted_energy")}
        summary.update(solver_status=info["solver_status"], runtime=info["runtime"])
        empty = np.zeros_like(before)
        return OptimisationResult(hourly_table(problem, before, before), summary,
                                  appliance_table(problem, before, before), before, empty, metadata)
    summary = summary_metrics(problem, before, after)
    summary.update(solver_status=info["solver_status"], runtime=info["runtime"],
                   objective_value=info["objective_value"])
    return OptimisationResult(hourly_table(problem, before, after), summary,
                              appliance_table(problem, before, after), before, after, metadata)
