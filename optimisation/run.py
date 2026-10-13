"""Command-line runner and the synthetic validation scenarios used in the reports.

    python -m optimisation.run --forecast path/to/forecast.csv            # illustrative ToU, default configs
    python -m optimisation.run --forecast f.csv --tariff tariff.json --out outdir
    python -m optimisation.run --scenarios                                # regenerate report evidence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from optimisation.contracts import Household, _json_default, load_appliances, load_household
from optimisation.milp import optimise
from optimisation.tariff import FlatTariff, SlabTariff, illustrative_tou_tariff, tariff_from_json

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"
REPORT_DIR = ROOT / "reports" / "final_optimisation_preparation"
# Committed copy of the Review-3 example forecast (LinearRegression, Bareilly BR02, origin 2020-09-06 03:00).
EXAMPLE_FORECAST = ROOT / "reports" / "review3" / "demand_side_optimisation" / "schedules" / "source_forecast_24h.csv"


def scenarios():
    forecast = pd.read_csv(EXAMPLE_FORECAST)
    full = load_appliances(CONFIGS / "appliances_illustrative.json")
    legacy = load_appliances(CONFIGS / "appliances_review3_legacy.json")
    house = load_household(CONFIGS / "household_default.json")
    tou = illustrative_tou_tariff()
    flat = FlatTariff(rate=6.0, label="ILLUSTRATIVE FLAT 6.0 INR/kWh - NOT OFFICIAL - synthetic validation only")
    slab = SlabTariff(slabs=[{"upto_kwh": 100, "rate": 0.0}, {"upto_kwh": 200, "rate": 2.5},
                             {"upto_kwh": 400, "rate": 4.5}, {"upto_kwh": None, "rate": 6.5}],
                      prior_cycle_kwh=150.0, billing_cycle_days=60,
                      label="ILLUSTRATIVE SLAB - NOT OFFICIAL - synthetic validation only")
    clustered = [a.__class__(**{**a.__dict__, "preferred_start": p})
                 for a, p in zip(full, [8, 19, 7, 7, 22])]  # geyser + pump + washer together in the morning
    return {
        "A_review3_reproduction": (forecast, tou, legacy, house),
        "B_five_appliances_tou": (forecast, tou, full, house),
        "C_flat_tariff": (forecast, flat, full, house),
        "D_slab_tariff": (forecast, slab, full, house),
        "E_morning_cluster_tou": (forecast, tou, clustered, house),
        "F_no_tie_break_review3": (forecast, tou, legacy, Household(**{**house.__dict__, "lexicographic_tie_break": False})),
    }


def run_scenarios(out_dir: Path = REPORT_DIR / "runs") -> pd.DataFrame:
    rows = []
    for name, (forecast, tariff, appliances, house) in scenarios().items():
        result = optimise(forecast, tariff, appliances, house)
        result.save(out_dir / name)
        moved = result.appliance_schedule.loc[result.appliance_schedule.moved, "appliance_id"].tolist()
        rows.append({"scenario": name, "tariff": tariff.tariff_type, "n_appliances": len(appliances),
                     **{k: result.summary.get(k) for k in ("cost_before", "cost_after", "cost_reduction_pct",
                        "peak_before", "peak_after", "peak_reduction_pct", "par_before", "par_after",
                        "shifted_energy", "net_profile_shift_kwh", "cost_responsive", "solver_status", "runtime")},
                     "moved_appliances": ",".join(moved)})
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "scenario_summary.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecast", type=Path, help="CSV with timestamp,predicted_kwh (24 rows)")
    p.add_argument("--tariff", type=Path, help="tariff JSON (default: illustrative ToU)")
    p.add_argument("--appliances", type=Path, default=CONFIGS / "appliances_illustrative.json")
    p.add_argument("--household", type=Path, default=CONFIGS / "household_default.json")
    p.add_argument("--out", type=Path, default=Path("optimisation_output"))
    p.add_argument("--scenarios", action="store_true", help="regenerate the report validation scenarios")
    args = p.parse_args()
    if args.scenarios:
        print(run_scenarios().to_string(index=False))
        return
    if args.forecast is None:
        p.error("--forecast is required unless --scenarios is used")
    tariff = tariff_from_json(args.tariff) if args.tariff else illustrative_tou_tariff()
    result = optimise(pd.read_csv(args.forecast), tariff, load_appliances(args.appliances), load_household(args.household))
    result.save(args.out)
    print(json.dumps(result.summary, indent=2, default=_json_default))
    print(result.appliance_schedule[["appliance_id", "start_before", "start_after", "hours_after", "shifted_energy_kwh"]].to_string(index=False))


if __name__ == "__main__":
    main()
