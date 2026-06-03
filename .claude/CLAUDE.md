# Datacenter on-site power sourcing model

## Goal
A datacenter has constant demand of **1 GW for all 8760 h/yr**. By law a fraction
`x` of demand must be produced **on-site**; up to `(1 - x)` may be imported from
the grid at an exogenous hourly market price (`prices.txt`, length 8760).

We compare two on-site options on **total annualised cost** and pick the cheapest:

1. **KK** — a constant-output SMR (nuclear) reactor.
2. **VE** — solar + wind + battery (solar:wind installed capacity fixed **1:1**
   for now; to be endogenised later).

For each option, find the **minimal capacity** that meets the on-site requirement,
then cost it.

## Conventions
- Python, NumPy. `dataclasses` for parameter containers, type hints throughout.
- Units: capacity **MW**, energy **MWh**, prices **€/MWh**, capex **€/MW**
  (battery energy **€/MWh**). 1 GW = 1000 MW.
- Costs are annualised via CRF (discount rate × lifetime) + fixed O&M + variable.
- Keep it the simplest thing that answers the question. Surgical edits, no rewrites.
  No preamble in responses; explain the "why" of any non-obvious modelling choice.

## Layout
- `model.py` — `DatacenterModel`, `Tech`, `Battery`, `Result`.
- `prices.txt`, `solar.txt`, `wind.txt` — length-8760 inputs (VE files are
  normalised capacity factors in [0, 1]).

## Open modelling decisions (confirm before relying on results)
- **Matching basis** — must on-site ≥ `x·P` *every hour* (24/7 CFE-style, drives
  heavy storage) or only on an *annual energy* basis? Current default: `hourly`.
- **1:1 solar:wind** — equal installed MW (current assumption) or equal annual energy?
- **Excess VE** — curtail (free disposal, default) or sell to grid at market price?
- **Battery** — cannot charge from the grid in v0; power = `x·P`, energy =
  `storage_hours · power` (not yet co-optimised with VE capacity).
- **SMR** — assumed 100% availability; no planned-outage capacity factor yet.
- **Grid import cap** — `(1 - x)·P` per hour, so the grid cannot backfill VE
  shortfalls beyond that.

## Known v0 limitations / next steps
- VE sizing is a bisection on a single capacity with a fixed-duration battery.
  Replace with a joint LP over (VE, battery power, battery energy).
- Battery dispatch is greedy (dispatch to the on-site floor), not cost-optimal.
- Variable O&M is charged on gross generation, not curtailment-adjusted.