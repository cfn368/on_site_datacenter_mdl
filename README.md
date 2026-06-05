# KK Datacenter — On-site power sourcing model

Developed by Linus Lindquist for [Erhvervslivets Tænketank](https://www.etank.dk) as part of Kernekraftprojektet.

A 1 GW datacenter must source a legally mandated fraction of its load from on-site generation. The model compares two technologies on annualised cost: a small modular nuclear reactor (KK) that delivers constant output interrupted by a planned annual maintenance window, and a solar-wind-battery portfolio (VE) that dispatches optimally against hourly spot prices. The comparison is run across multiple years and on-site fractions.

## Data sources

| Source | Content | Files |
|--------|---------|-------|
| [Energi Data Service](https://www.energidataservice.dk) | Hourly DK weighted-average spot price (DKK/MWh) | `variation_patterns/wp_{Y}_{Y+1}.txt` |
| Energi Data Service | Hourly Danish solar fleet production (MWh/h) | `variation_patterns/PV_VE_{Y}_{Y+1}.txt` |
| Energi Data Service | Hourly Danish onshore wind fleet production (MWh/h) | `variation_patterns/WL_VE_{Y}_{Y+1}.txt` |
| DEA Technology Data 2030 | Capital and O&M cost assumptions for solar, wind, battery, and SMR | `assumptions.py` |

Raw production series are divided by fleet capacity to obtain hourly capacity factors. Spot prices are converted from DKK to EUR at 7.46. Data is available for 2022–2025.

## Repository structure

```
KK_datacentre/
├── model.py              # Core model: DatacenterDemand, GridSupply, KKSupply, VESupply, DatacenterModel
├── assumptions.py        # All cost and technical parameters — import from here, never hardcode
├── lp_model.tex          # Full LP documentation: variables, objective, constraints, economic intuition
│
├── pylib/
│   ├── setup.py          # Notebook preamble: autoreload, AEJ style, standard imports + mdates
│   └── ve_dispatch.py    # Dispatch detail, aggregation, and plotting
│                         # (DISPATCH_COLORS, MAANED_DK, MAANED_EN, plot_dispatch, plot_battery)
│
├── 1_input.ipynb         # Fetches and saves variation pattern files via ET-eds-api
├── 2_model.ipynb         # Builds and runs the model, prints results, saves VE solution
├── 3_time_series.ipynb   # Time series figures: VE dispatch/battery, KK profile, curtailment
├── 4_cases.ipynb         # Multi-year, multi-x results table
│
├── variation_patterns/   # 8760-row input files (one value per line, dot-decimal)
├── runs/                 # ve_solution.json, ve_lp_arrays.npz, lp_arrays/ — cached VE solution
└── figures/              # Output figures
```

## The operator problems

### KK

The KK operator installs 1,000 MW and runs at full output for 90% of the year. The remaining 10% (876 hours) is a contiguous planned maintenance window, placed tactically at the cheapest consecutive hours in the price series to minimise grid purchase cost during downtime. During the outage the datacenter imports the full 1,000 MW from the grid; the hourly on-site fraction requirement is waived for planned maintenance.

Total KK cost has three components: (1) annualised reactor capex and O&M (~31 €/MWh variable, 60-year lifetime); (2) spot cost of grid imports during the 876-hour outage plus an 8.7 øre/kWh consumption tariff; (3) annualised grid connection fee (tilslutningsbidrag, treated as a perpetuity).

### VE

The VE operator faces a harder problem. Solar and wind are intermittent, the battery has limited storage, and the on-site floor must be met every hour. The operator chooses how much solar, wind, and battery power to install, then dispatches them optimally against hourly spot prices.

The entire problem — capacities and all 8,760-hour dispatch decisions — is solved as a single linear programme (HiGHS via `scipy.optimize.linprog`). The battery is fully bidirectional: it charges from VE surplus or cheap grid electricity and discharges to the datacenter or the grid. The on-site fraction requirement is enforced hourly. Grid purchases carry an 8.7 øre/kWh consumption tariff; grid sales carry a 1.15 øre/kWh production tariff; both enter the LP objective so the optimiser internalises them.

Surplus VE is tracked separately as solar curtailment and wind curtailment. Wind carries a 1.98 €/MWh variable O&M on dispatched output, so curtailing wind saves money; the LP therefore always curtails wind before solar — this is an outcome of the cost structure, not a rule imposed externally.

Additional fixed costs: solar land rent (4,800 €/MW/yr), grid connection fee (tilslutningsbidrag, scales with `(1−x)·P`, perpetuity).

At low `x` the VE optimum involves a large solar overbuild that generates substantial export revenue, making VE cheaper than KK. At high `x` the on-site floor tightens, the export cap shrinks, and KK's stable output becomes the cheaper option.

## How to run

All notebooks open with:

```python
from pylib.setup import *
setup_notebook()
```

Run in order:

1. `1_input.ipynb` — fetches variation patterns from Energi Data Service and writes them to `variation_patterns/`. Requires internet access and the `ET-eds-api` package.
2. `2_model.ipynb` — loads inputs, runs the single LP, prints the KK vs VE comparison, and saves the VE solution to `runs/`. Solves in seconds.
3. `3_time_series.ipynb` — loads the cached solution and produces dispatch and battery figures for VE, the KK hourly profile showing the planned outage window, and a weekly curtailment plot split by wind and solar.
4. `4_cases.ipynb` — runs both models across years (2022–2025) and on-site fractions (25/50/75 %) and prints the results table.

To change the on-site fraction or battery duration, edit `x` and `storage_hours` in `2_model.ipynb` and `4_cases.ipynb`. All cost assumptions are in `assumptions.py`.
