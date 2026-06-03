# Datacenter on-site power sourcing model

## Goal
A datacenter has constant demand of **1 GW for all 8760 h/yr**. By law a fraction
`x` of demand must be produced **on-site**; up to `(1 - x)` may be imported from
the grid at an exogenous hourly market price (DK weighted-average spot, converted
DKK→EUR at 7.46).

We compare two on-site options on **total annualised cost** and pick the cheapest:

1. **KK** — a constant-output SMR (nuclear) reactor.
2. **VE** — solar + wind + battery, with all three capacities jointly optimised.

## Conventions
- Python, NumPy, SciPy. `dataclasses` for parameter containers, type hints throughout.
- Units: capacity **MW**, energy **MWh**, prices **€/MWh**, capex **€/MW**
  (battery energy **€/MWh**). 1 GW = 1000 MW.
- Costs are annualised via CRF (discount rate × lifetime) + fixed O&M + variable.
- Keep it the simplest thing that answers the question. Surgical edits, no rewrites.
  No preamble in responses; explain the "why" of any non-obvious modelling choice.

## Layout
- `model.py` — `DatacenterModel`, `Tech`, `Battery`, `Result`, `DatacenterDemand`,
  `GridSupply`, `KKSupply`, `VESupply`.
- `assumptions.py` — all cost/technical parameters; import from here, never hardcode.
- `variation_patterns/` — three 8760-row txt files (dot-decimal, one value per line):
  - `PV_VE_2025_2026.txt` — raw Danish solar fleet production (MWh/h); divide by
    4955.5428 MW to get capacity factor.
  - `WL_VE_2025_2026.txt` — raw Danish onshore wind production (MWh/h); divide by
    4878.483 MW to get capacity factor.
  - `wp_2025_2026.txt` — hourly DK weighted-average spot price (DKK/MWh); divide by
    7.46 for EUR.
- `1_VE.ipynb` — fetches and saves the variation pattern files.
- `2_model.ipynb` — loads inputs, builds and runs the model, shows results.

## KK sizing logic
KK output is constant, so total cost is **linear** in installed capacity. The optimum
is always a corner solution: either `floor_mw = x·P` (minimum legal) or `demand_mw`
(full on-site, zero grid). Decision rule: compare KK marginal cost per MW-yr against
`mean_price × 8760`. Break-even ≈ 67 €/MWh at current assumptions.

`capacity_factor` (currently 0.9) enters only the fuel-cost calculation; dispatch
stays constant at the installed capacity (100% availability assumed).

## VE optimisation
Three decision variables: `C_solar` (MW), `C_wind` (MW), `P_batt` (MW battery power).
Battery energy = `P_batt × storage_hours` (currently 4 h, fixed).

**Nested structure:**
- **Inner bisection** — for given `(C_solar, C_wind)`, finds minimum `P_batt` that
  makes every hour feasible (hourly CFE matching). Returns `None` if infeasible even
  at `bisect_upper_batt` (50 000 MW).
- **Outer Nelder-Mead** — minimises total annual cost over `(C_solar, C_wind)`. Grid
  purchase cost is included in the objective when `prices` is passed to `VESupply`.

Battery dispatch is greedy: charges from VE surplus above `floor_mw`, discharges to
cover shortfalls. SOC initialised at 0 (conservative/worst-case for bisection).

## Feasibility note
With hourly CFE matching, strict Dunkelflaute periods (multi-day low wind + no solar)
require very large battery energy. The joint optimisation trades off VE capacity
(reduces per-hour shortfall) against battery size. Expect the optimizer to push
heavily toward wind over solar, as wind provides generation at night.

## Fixed modelling decisions
- Battery cannot charge from the grid (only from VE surplus).
- Grid import cap: `(1 - x)·P` per hour; grid cannot backfill on-site CFE shortfalls.
- Variable O&M charged on gross generation (before curtailment).
- Excess VE above `demand_mw` is curtailed (free disposal).

## Open decisions / next steps
- **KK capacity** — set to `demand_mw` (1000 MW, full load, zero grid imports).
  This is the optimal corner: mean spot price exceeds the KK break-even price.
- **Solar:wind ratio** — now endogenised; Nelder-Mead chooses freely.
- **Battery duration** — `storage_hours = 4` is fixed; could be added as a third
  outer decision variable.
- **Annual vs hourly matching** — hourly is the current default; annual would make
  VE substantially cheaper.
- **SMR fixed O&M** — currently 0; needs an ET assumption.
