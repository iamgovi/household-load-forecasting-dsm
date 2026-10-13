# Existing optimisation audit

| Component | Current legacy behavior | Inputs/outputs | Scientific assessment | Review-3 disposition |
|---|---|---|---|---|
| `dashboard.py:tod_multiplier` | Three-period tariff multiplier over an illustrative INR base rate | Hour → tariff multiplier | Internally coherent and already labelled illustrative | Concept retained; explicit 24-row tariff created separately |
| `dashboard.py:APPLIANCES` | Five manually specified loads including EV and comfort-labelled AC | Hard-coded powers, durations, deadlines | Explicitly non-measured, but EV has no SOC and AC has no thermal model | Deprecated for research evidence; replaced with four simpler simulated loads |
| `dashboard.py:schedule_appliance` | Greedy, independent cheapest/solar-aware contiguous start | One appliance at a time | Not joint optimisation; ignores household peak and appliance concurrency | Replaced in research path by joint MILP |
| Deadline filter | Accepts `finish <= deadline OR start >= deadline` | Candidate starts | Incorrect: explicitly permits operation after deadline | Repaired with start variables whose complete duration lies inside allowed window |
| Baseline | Every appliance starts at 19:00 with modulo-24 wrap | Appliance assumptions | Violates several deadlines/windows and exaggerates savings | Replaced by deterministic feasible preferred-time schedule |
| Cost | Appliance-only tariff cost | Flexible load only | Arithmetic is locally correct but omits base-load cost | Repaired: same forecast background included before and after |
| Solar | Synthetic cosine curve independently credited to every appliance | Simulated solar and appliance power | Can double-count solar and is not real renewable evidence | Excluded from main experiment |
| “Flexible energy shifted” | Reports total flexible energy | Appliance power × duration | Incorrect shifted-energy definition | Replaced by `0.5*sum(abs(after-before))` |
| Sustainability | Reports simulated solar self-consumption | Synthetic curve | Not defensible as real renewable utilisation; no carbon factor | Disabled from Review-3 evidence; no renewable/carbon claim |
| Dashboard structure | Streamlit layout and tables | Legacy forecast/deployment path | Structurally useful for future integration | Preserved unchanged; dashboard redesign deferred |

The legacy dashboard consumes its own deployment forecast and root-level legacy data paths. It is not used by this research demonstration.
