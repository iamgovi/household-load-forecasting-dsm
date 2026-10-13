# Appliance assumptions

**Status of every value below: SIMULATION / LITERATURE-BASED ASSUMPTION.**
None of them is measured from the household smart-meter dataset, which records aggregate load only.
They are plausible typical values for an Indian household, chosen so the scheduling problem is
realistic; they are not claimed as measurements.

Config file: `optimisation/configs/appliances_illustrative.json`.

| ID | Appliance | Power (kW) | Duration (h) | Window (clock) | Deadline | Interruptible | Baseline start | Rationale |
|---|---|---|---|---|---|---|---|---|
| `washing_machine` | Washing machine | 0.8 | 2 | 08:00–20:00 | off by 20:00 | No | 09:00 | Average over a cycle with water heating; a cycle cannot pause |
| `dishwasher` | Dishwasher | 1.0 | 2 | 18:00–24:00 | off by 24:00 | No | 19:00 | After-dinner run |
| `water_heater` | Water heater (geyser) | 2.0 | 1 | 05:00–09:00 | off by 09:00 | No | 07:00 | 2 kW storage element; hot water needed before 09:00 |
| `water_pump` | Water pump | 1.0 | 1 | 06:00–18:00 | off by 18:00 | No | 12:00 | ~1 HP motor filling overhead tank |
| `ev_2w_charger` | EV two-wheeler charging | 0.7 | 3 | 22:00–06:00 (overnight) | off by 06:00 | **Yes** | 22:00 | Fixed energy need (2.1 kWh); no state-of-charge model |

The first four are identical to the Review-3 set (`appliances_review3_legacy.json`) so results remain
comparable. The EV charger was added to exercise the interruptible and overnight-window constraints.

## Modelling simplifications (state these in the thesis)

* Constant power while ON (no cycle-level power profile).
* Hourly resolution: durations are whole hours.
* EV: energy requirement only; no SOC, no charging-rate curve.
* Geyser: no thermal model; "1 h before 09:00" stands in for comfort.
* Baseline = user's preferred start; it represents un-optimised behaviour, not observed behaviour.
* Non-flexible base load = full aggregate forecast (appliance energy is added on top, not separated out).

## How to change them

Edit the JSON or pass another file: `python -m optimisation.run --forecast f.csv --appliances my.json`.
Validation (`Appliance.__post_init__`) rejects non-positive power, non-integer durations, windows outside
0–24 and durations longer than the window. If verified appliance measurements become available,
change `source_class` and cite the source.
