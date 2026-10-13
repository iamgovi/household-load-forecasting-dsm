# Tariff interface

Code: `optimisation/tariff.py`. **No official tariff is implemented.** The final tariff will be
supplied by Govardhan as JSON in the format below.

## Separation of concerns

| | Method | Used for |
|---|---|---|
| A. Billing | `bill(hourly_kwh, timestamps)` = energy charge + `fixed_charge_per_horizon` | `cost_before`, `cost_after`, `cost_reduction_pct` |
| B. Scheduling signal | `scheduling_rates(timestamps)` → 24 ₹/kWh values | MILP cost term |
| Cost responsiveness | `is_time_varying(timestamps)` | `cost_responsive` flag in output |

## Supported types

| Type | Class | Scheduling signal | Can shifting change the bill? |
|---|---|---|---|
| Flat | `FlatTariff(rate)` | constant | **No** |
| Time-of-use | `TouTariff(hourly_rates)` / `TouTariff.from_periods(...)` | varies by clock hour | Yes |
| Slab | `SlabTariff(slabs, telescopic, prior_cycle_kwh, billing_cycle_days)` | constant (marginal rate at current cycle position) | **No**: the bill depends on total kWh, and shifting keeps total kWh constant |

Slab billing charges the *increment* this horizon adds to the billing-cycle bill, starting from
`prior_cycle_kwh`. Both telescopic (rate per slab band) and non-telescopic (whole consumption at the
final slab's rate) are supported.

## If the official tariff is flat or slab

The optimiser will **not** claim cost savings: `cost_reduction_pct = 0`, `cost_responsive = false`,
and `cost_note` says so. Valid claims remain peak and PAR reduction (scenario E: −29.9 % peak).
A ToU result may still be shown as a **separately labelled research scenario** using
`illustrative_tou_tariff()`.

## JSON format (template: `optimisation/configs/tariff_TEMPLATE_for_govardhan.json`)

```json
{"tariff_type": "slab", "label": "<DISCOM> domestic, order <no.>", "is_official": true,
 "source": "<URL / order / page>", "effective_from": "YYYY-MM-DD", "effective_to": null,
 "fixed_charge_per_horizon": 0.0,
 "slabs": [{"upto_kwh": 100, "rate": 0.0}, {"upto_kwh": 200, "rate": 0.0}, {"upto_kwh": null, "rate": 0.0}],
 "telescopic": true, "prior_cycle_kwh": 0, "billing_cycle_days": 60}
```

ToU: `{"tariff_type":"tou","periods":[{"name":"Peak","start_hour":18,"end_hour":22,"rate":...}, ...]}`
(must cover all 24 h, no overlap) or `"hourly_rates": [24 values]`.
Flat: `{"tariff_type":"flat","rate":...}`.

Safeguards: `is_official: true` without a `source` is rejected; negative rates are rejected; ToU gaps or overlaps are rejected.

## Current illustrative tariff (tests / synthetic validation only)

`illustrative_tou_tariff()`: 00–06 ₹4.0, 06–18 ₹6.0, 18–22 ₹9.0, 22–24 ₹4.5. Label:
"ILLUSTRATIVE TARIFF – NOT OFFICIAL – TEST/SYNTHETIC VALIDATION ONLY". The slab and flat values used in
scenarios C and D are also illustrative.

## Not yet supported (raise with Govardhan if needed)

* ToU surcharge on top of slab rates (combined structure).
* Demand/connected-load charges based on kW.
* Taxes/electricity duty (can be folded into `fixed_charge_per_horizon` or rates).
