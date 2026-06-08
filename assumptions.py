# assumptions.py
"""
Cost and technical assumptions for the KK_datacentre model.
Sources tagged inline:
  [ET]  — Erhvervslivets Tænketank
  [DEA] — Danish Energy Agency, Technology Data (2030-column values for VE)
"""
from model import Tech, Battery, DatacenterDemand

# ── common ────────────────────────────────────────────────────────────────────

DISCOUNT_RATE = 0.04
DKK_EUR      = 7.46

# ── SMR (KK) ──────────────────────────────────────────────────────────────────

# Fuel price already accounts for thermal efficiency (€/GJ_el output).
# Convert to €/MWh_el: multiply by 3.6 GJ/MWh only.
_smr_fuel_eur_gj   = 259 / 100          # €/GJ_el  [ET]
_smr_opex_var_fuel  = _smr_fuel_eur_gj * 3.6   # €/MWh_el ≈ 9.32
_smr_opex_var_om    = 20.42              # €/MWh  [ET]
_smr_opex_var_extra = 1.0 / 100 / 7.46 * 1_000   # 1 øre/kWh → EUR/MWh ≈ 1.34  [ET]

smr_tech = Tech(
    capex         = 8_000_000,                              # €/MW  [ET]
    opex_fixed    = 0,                                      # €/MW/yr
    opex_var      = _smr_opex_var_fuel + _smr_opex_var_om + _smr_opex_var_extra, # €/MWh ≈ 31.08
    lifetime      = 60,                                     # years  [FOA]
    discount_rate = DISCOUNT_RATE,
)

SMR_DOWNTIME = 0.10   # planned maintenance fraction of year (one contiguous block)  [ET]

# ── solar PV (utility-scale) ──────────────────────────────────────────────────

solar_tech = Tech(
    capex         = 450_000,   # €/MW  [DEA 2030]
    opex_fixed    = 15_200,    # €/MW/yr  [DEA 2030: 10,400 + 4,800 land rent]
    opex_var      = 0,         # €/MWh
    lifetime      = 40,        # years
    discount_rate = DISCOUNT_RATE,
)

# ── onshore wind ──────────────────────────────────────────────────────────────

wind_tech = Tech(
    capex         = 1_150_000,   # €/MW  [DEA 2030]
    opex_fixed    = 16_663,      # €/MW/yr  [DEA 2030]
    opex_var      = 1.98,        # €/MWh  [DEA 2030]
    lifetime      = 30,          # years
    discount_rate = DISCOUNT_RATE,
)

# ── battery BESS ─────────────────────────────────────────────────────────────
# Power (MW) and energy (MWh) are sized independently by the optimiser.

battery = Battery(
    capex_power    = 80_000,    # €/MW  (power component)  [DEA 2030]
    capex_energy   = 200_000,   # €/MWh (energy + other project costs)  [DEA 2030]
    opex_fixed     = 8_100,     # €/MW/yr  [DEA 2030]
    lifetime       = 20,        # years  [DEA 2030]
    discount_rate  = DISCOUNT_RATE,
    eta_charge     = 0.98,      # DC charge efficiency  [DEA 2030]
    eta_discharge  = 0.97,      # DC discharge efficiency  [DEA 2030]
)

# ── demand ────────────────────────────────────────────────────────────────────

demand = DatacenterDemand(demand_mw=1_000.0, x=0.50)

# ── grid connection (tilslutningsbidrag) — VE only ────────────────────────────
# One-time capex for grid tie-in, annualised at DISCOUNT_RATE.
# Lifetime not specified by ET — assumed 20 yr.

GRID_CONNECT_FIXED_DKK  = 18_200_000   # DKK  (fixed component)
GRID_CONNECT_VAR_DKK_MW =    663_000   # DKK/MW of grid connection capacity


def grid_connect_annual(grid_cap_mw: float) -> float:
    """Annualised tilslutningsbidrag (€/yr) for a VE grid connection of grid_cap_mw.
    Treated as perpetuity: CRF = r."""
    total_eur = (GRID_CONNECT_FIXED_DKK + GRID_CONNECT_VAR_DKK_MW * grid_cap_mw) / DKK_EUR
    return total_eur * DISCOUNT_RATE


# ── grid running tariffs — VE only ───────────────────────────────────────────

GRID_TARIFF_BUY  = 8.7  / 100 / DKK_EUR * 1_000   # øre/kWh → EUR/MWh ≈ 11.66
GRID_TARIFF_SELL = 1.15 / 100 / DKK_EUR * 1_000   # øre/kWh → EUR/MWh ≈  1.54
