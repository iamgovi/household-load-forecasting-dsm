"""Demand-side optimisation engine (owner: Keshav).

Residential appliance scheduling formulated as a Mixed-Integer Linear Program
and solved with HiGHS (branch-and-cut) through ``scipy.optimize.milp``.

Public entry point::

    from optimisation import optimise
    result = optimise(forecast_df, tariff, appliances, household)

The engine consumes only the standard 24-hour forecast table
(``timestamp``, ``predicted_kwh``). It never imports forecasting-model internals.
"""
from optimisation.contracts import (Appliance, Household, OptimisationResult, load_appliances,
                                    load_household, validate_forecast)
from optimisation.tariff import (FlatTariff, SlabTariff, Tariff, TouTariff, illustrative_tou_tariff,
                                 tariff_from_dict)
from optimisation.problem import SchedulingProblem
from optimisation.milp import optimise, solve_milp

__all__ = ["Appliance", "Household", "OptimisationResult", "load_appliances", "load_household",
           "validate_forecast", "Tariff", "FlatTariff", "TouTariff", "SlabTariff",
           "illustrative_tou_tariff", "tariff_from_dict", "SchedulingProblem", "optimise", "solve_milp"]
