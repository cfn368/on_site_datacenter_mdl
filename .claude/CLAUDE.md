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
- `3_VE.ipynb` — VE dispatch visualisation: hourly, daily, weekly, monthly stacked
  area plots of how demand is met (PV, wind, battery, grid) plus exports below zero.

## Cost assumptions (DEA 2030, all in assumptions.py)
- Discount rate: 4%
- **SMR**: 8 M€/MW capex, opex_var ≈ 29.74 €/MWh (fuel 9.32 + O&M 20.42),
  opex_fixed = 0 (not provided by ET — understates KK cost), lifetime 60 yr [ET/FOA]
- **Solar**: 450 k€/MW capex, 10,400 €/MW/yr opex_fixed, lifetime 40 yr [DEA 2030]
- **Wind**: 1,150 k€/MW capex, 16,663 €/MW/yr opex_fixed, 1.98 €/MWh opex_var,
  lifetime 30 yr [DEA 2030]
- **Battery**: 80 k€/MW power capex, 200 k€/MWh energy capex, 8,100 €/MW/yr
  opex_fixed, lifetime 20 yr [DEA 2030]. Power and energy sized independently.

## KK sizing logic
KK output is constant, so total cost is **linear** in installed capacity. The optimum
is always a corner solution: either `floor_mw = x·P` (minimum legal) or `demand_mw`
(full on-site, zero grid). Decision rule: compare KK marginal cost per MW-yr against
`mean_price × 8760`. Break-even ≈ 67 €/MWh at current assumptions.

`capacity_factor = 0.9` enters **both** sizing and fuel cost:
`capacity_mw = demand_mw / capacity_factor = 1111 MW` — oversized so that at 90%
average availability it delivers the full 1000 MW in expectation. Fuel cost is charged
on `1111 × 0.9 × 8760 MWh`.

Current result (x = 0.5): **LCOE 74.60 €/MWh, total 653.5 M€/yr**, zero grid imports.
KK is at the full-demand corner because mean spot (82 €/MWh) > break-even (67 €/MWh).

## VE optimisation
Four effective degrees of freedom: `C_solar`, `C_wind`, `P_batt`, `E_batt`.
Battery power and energy are **decoupled** — each has its own cost and is sized
independently.

**Nested structure:**
- **Outer Nelder-Mead** — minimises total annual cost over `(C_solar, C_wind, P_batt)`.
  Grid purchase cost minus export revenue included in objective.
- **Inner bisection** — for given `(C_solar, C_wind, P_batt)`, finds minimum `E_batt`
  (MWh) that makes every hour feasible (hourly CFE matching). Returns `None` if
  infeasible at `bisect_upper_energy = 1 000 000 MWh`.

Battery dispatch is greedy: charges from VE surplus above `floor_mw`, discharges to
cover shortfalls. SOC initialised at 0 (conservative/worst-case for bisection).

**Export**: VE surplus above `demand_mw` (after battery charging) is sold to the grid
at `max(spot_price, 0)` — curtailed at negative prices. Export capped at `grid_cap_mw`
(same physical connection as imports). Export revenue is subtracted from total cost.

Current result (x = 0.5): **4 620 MW solar + 3 539 MW wind + 500 MW / 30 036 MWh
battery, LCOE 116.89 €/MWh, total 1 024.0 M€/yr** (on-site 910.5 M€ + grid 203.6 M€
− export 90.1 M€).

## Feasibility note
With hourly CFE matching, strict Dunkelflaute periods (multi-day low wind + no solar)
require very large battery energy. The joint optimisation trades off VE capacity
(reduces per-hour shortfall) against battery energy. P_batt converges to floor_mw
(500 MW) — the minimum needed to discharge at the floor rate. Dunkelflaute drives
E_batt to ~30 000 MWh.

## Fixed modelling decisions
- Battery cannot charge from the grid (only from VE surplus).
- Grid import cap: `(1 - x)·P` per hour; grid cannot backfill on-site CFE shortfalls.
- Grid export cap: same `(1 - x)·P` per hour (same physical connection).
- Variable O&M charged on gross generation (before curtailment/export).
- Excess VE above `demand_mw + export_cap` is curtailed (free disposal).

## KK vs VE gap and break-even
At x = 0.5 with current assumptions: **KK 653.5 M€ vs VE 1 024.0 M€ — gap 370 M€**.
No crossover exists for x ∈ [0, 1]: KK LCOE (74.6 €/MWh) < mean spot (81.6 €/MWh),
so KK serving full demand always beats any VE + grid combination. VE → 715 M€ (pure
grid) as x → 0, still above KK.

## Open decisions / next steps
- **SMR fixed O&M** — currently 0; needs an ET assumption (understates KK cost).
- **Battery dispatch** — greedy is a heuristic; optimal dispatch (sell vs charge
  decision by hour) would reduce VE cost further.
- **Annual vs hourly CFE matching** — hourly is current default; annual would make
  VE substantially cheaper by removing Dunkelflaute battery requirement.
- **Battery duration sensitivity** — `storage_hours` is gone; E_batt is now free.
  Seasonal storage (alternative technology, lower €/MWh) is a different model.
- **KK fixed O&M** — once ET provides a number, add to `smr_tech.opex_fixed`.
