# KK Datacenter — On-site power sourcing model

Developed by Linus Lindquist for [Erhvervslivets Tænketank](https://www.etank.dk) as part of Kernekraftprojektet.

A 1 GW datacenter must source a legally mandated fraction of its load from on-site generation. The model compares two technologies on annualised cost: a small modular nuclear reactor (KK) that delivers constant output, and a solar-wind-battery portfolio (VE) that dispatches optimally against hourly spot prices. The comparison is run across the full range of on-site fractions and storage durations.

## Data sources

| Source | Content | File |
|--------|---------|------|
| [Energi Data Service](https://www.energidataservice.dk) | Hourly DK weighted-average spot price (DKK/MWh) | `variation_patterns/wp_2025_2026.txt` |
| Energi Data Service | Hourly Danish solar fleet production (MWh/h) | `variation_patterns/PV_VE_2025_2026.txt` |
| Energi Data Service | Hourly Danish onshore wind fleet production (MWh/h) | `variation_patterns/WL_VE_2025_2026.txt` |
| DEA Technology Data 2030 | Capital and O&M cost assumptions for solar, wind, battery, and SMR | `assumptions.py` |

Raw production series are divided by fleet capacity (PV: 4 955.5 MW, wind: 4 878.5 MW) to obtain hourly capacity factors. Spot prices are converted from DKK to EUR at 7.46.

## Repository structure

```
KK_datacentre/
├── model.py              # Core model: DatacenterDemand, GridSupply, KKSupply, VESupply, DatacenterModel
├── assumptions.py        # All cost and technical parameters — import from here, never hardcode
│
├── pylib/
│   ├── setup.py          # Notebook preamble: autoreload, AEJ style, standard imports
│   └── ve_dispatch.py    # Dispatch detail, aggregation, and plotting for 3_VE.ipynb
│
├── 1_input.ipynb         # Fetches and saves variation pattern files via ET-eds-api
├── 2_model.ipynb         # Builds and runs the model, prints results, saves VE solution
├── 3_VE.ipynb            # VE dispatch visualisation: stacked area and battery figures
│
├── variation_patterns/   # 8760-row input files (one value per line, dot-decimal)
├── runs/                 # ve_solution.json — cached VE optimisation result
└── figures/              # Output figures
```

## The operator problems

### KK

The KK operator's problem is simple, because nuclear output is constant. The reactor either runs at the minimum legal output or at full demand — there is no intermediate optimum. Building more capacity adds cost linearly, and importing from the grid is cheaper only if the spot price falls below the reactor's variable cost. The decision rule reduces to a single break-even comparison: if the mean spot price exceeds the reactor's marginal cost (~67 €/MWh at current assumptions), the operator builds to full demand and imports nothing. At the mean Danish spot price of ~82 €/MWh, KK serves the full 1 GW with zero grid imports.

### VE

The VE operator faces a harder problem. Solar and wind are intermittent, the battery has limited storage, and the on-site floor must be met every hour regardless. The operator chooses how much solar, wind, and battery power to install, then dispatches them optimally given perfect foresight of prices and weather.

Dispatch is solved as a linear programme over all 8760 hours simultaneously (HiGHS via `scipy.optimize.linprog`). The battery is fully bidirectional: it charges from VE surplus or cheap grid electricity and discharges to the datacenter or the grid. The on-site fraction requirement is enforced by penalising excess grid imports at 1 M€/MWh, making the constraint effectively hard. The outer optimisation (Nelder-Mead over three capacity variables) then minimises total annualised cost — capital, O&M, net grid cost — trading off higher build-out against better CFE compliance and arbitrage revenue.

At current assumptions (`x = 0.5`, 12-hour battery, DEA 2030 costs) the VE optimum involves a large solar overbuild (~6× demand) that charges the battery cheaply and generates substantial export revenue. VE LCOE is approximately 86 €/MWh versus KK at 75 €/MWh.

## How to run

All notebooks open with:

```python
from pylib.setup import *
setup_notebook()
```

Run in order:

1. `1_input.ipynb` — fetches variation patterns from Energi Data Service and writes them to `variation_patterns/`. Requires internet access and the `ET-eds-api` package.
2. `2_model.ipynb` — loads inputs, runs both optimisations, prints the KK vs VE comparison, and saves the VE solution to `runs/ve_solution.json`. The VE optimisation takes 5–10 minutes.
3. `3_VE.ipynb` — loads the cached solution and produces dispatch figures. Runs in seconds.

To change the on-site fraction or battery duration, edit `x` and `storage_hours` in `2_model.ipynb`. All cost assumptions are in `assumptions.py`.
