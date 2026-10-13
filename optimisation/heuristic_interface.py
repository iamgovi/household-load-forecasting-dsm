"""Interface for an OPTIONAL GA/PSO benchmark against the MILP. Not a replacement for the MILP.

Fairness is enforced by construction: every scheduler receives the same ``SchedulingProblem``
(same 24-hour forecast, tariff, appliance set, household limits) and is scored by the same
``evaluate`` function (same objective, same feasibility checker). See HEURISTIC_COMPARISON_PLAN.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from optimisation.milp import solve_milp
from optimisation.problem import SchedulingProblem


class Scheduler(Protocol):
    name: str

    def solve(self, problem: SchedulingProblem, baseline: np.ndarray) -> np.ndarray:
        """Return a binary (n_appliances x 24) schedule."""


@dataclass
class Evaluation:
    scheduler: str
    feasible: bool
    violations: list[str]
    objective: float
    shifted_energy: float
    runtime_s: float


def evaluate(problem: SchedulingProblem, schedule: np.ndarray, baseline: np.ndarray,
             scheduler: str = "", runtime_s: float = float("nan")) -> Evaluation:
    violations = problem.check_schedule(schedule)
    return Evaluation(scheduler=scheduler, feasible=not violations, violations=violations,
                      objective=problem.objective(schedule, baseline),
                      shifted_energy=float(problem.shifted_energy_by_appliance(baseline, schedule).sum()),
                      runtime_s=runtime_s)


def benchmark(problem: SchedulingProblem, schedulers: list[Scheduler]) -> list[Evaluation]:
    baseline = problem.baseline_schedule()
    out = []
    for s in schedulers:
        t0 = time.perf_counter()
        schedule = s.solve(problem, baseline)
        out.append(evaluate(problem, schedule, baseline, s.name, time.perf_counter() - t0))
    return out


class MILPScheduler:
    name = "MILP (HiGHS)"

    def solve(self, problem, baseline):
        schedule, info = solve_milp(problem, baseline)
        if schedule is None:
            raise RuntimeError(info["solver_message"])
        return schedule


class BaselineScheduler:
    """Reference point: the un-optimised preferred-start schedule."""
    name = "Baseline (no optimisation)"

    def solve(self, problem, baseline):
        return baseline.copy()


class GeneticAlgorithmScheduler:
    """Placeholder. Planned encoding: one integer gene per non-interruptible appliance (start slot index
    from ``problem.feasible_starts(a)``) and a 24-bit mask with repair for interruptible ones; infeasible
    individuals repaired or penalised; fitness = ``problem.objective``. Run >= 30 seeds, report mean/std/best."""
    name = "GA (not implemented)"

    def solve(self, problem, baseline):
        raise NotImplementedError("GA benchmark is planned, not implemented (see HEURISTIC_COMPARISON_PLAN.md)")


class ParticleSwarmScheduler:
    """Placeholder. Planned: continuous position per appliance start, rounded to nearest feasible start (binary PSO
    for interruptible masks); same fitness and repair as GA."""
    name = "PSO (not implemented)"

    def solve(self, problem, baseline):
        raise NotImplementedError("PSO benchmark is planned, not implemented (see HEURISTIC_COMPARISON_PLAN.md)")


def as_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    return [e.__dict__ for e in evaluations]
