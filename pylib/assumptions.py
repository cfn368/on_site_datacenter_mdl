"""
Cost and technical assumptions for the KK_datacentre model.
Sources tagged inline:
  [ET]  — Erhvervslivets Tænketank
  [DEA] — Danish Energy Agency, Technology Data (2030-column values for VE)
"""

from pylib.model import Tech, Battery, DatacenterDemand

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

SOLAR_LAND_HA_PER_MW   = 1.593   # ha/MW  [DEA 2030: 15.93 × 1000 m²/MW_e]
SOLAR_LAND_RENT_DKK_HA = 3_581   # DKK/ha/yr  [DEA 2030] — gives 4,800 €/MW/yr at 10 ha/MW

_solar_land_eur_mw_yr = SOLAR_LAND_HA_PER_MW * SOLAR_LAND_RENT_DKK_HA / DKK_EUR  # ≈ 4,800

solar_tech = Tech(
    capex         = 450_000,   # €/MW  [DEA 2030]
    opex_fixed    = 10_400 + _solar_land_eur_mw_yr,  # €/MW/yr  [DEA 2030: 10,400 + land rent ≈ 15,200]
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

# ── gas turbine CCGT (steam extraction) ──────────────────────────────────────
# DEA Technology Data 2030, sheet "05 Gas turb. CC, steam extract."
# Fuel: green gas at 100 DKK/GJ. Efficiency 58% (DEA 2030 annual average).
# Gas tariff: 25 DKK/MWh_el.

GAS_EFFICIENCY     = 0.45         # effective efficiency incl. startup losses [ET, DEA 2030 = 0.58]
GAS_PRICE_DKK_GJ   = 100.0        # DKK/GJ green gas [ET]
GAS_TARIFF_DKK_MWH = 25.0         # DKK/MWh_el [ET]
GAS_MIN_LOAD       = 0.40         # minimum stable load fraction [DEA 2030]

# fuel cost per MWh_el: (100 DKK/GJ × 3.6 GJ/MWh) / 7.46 / 0.58
_gas_fuel_eur_mwh_el = GAS_PRICE_DKK_GJ * 3.6 / DKK_EUR / GAS_EFFICIENCY   # ≈ 83.2 €/MWh_el
_gas_tariff_eur_mwh  = GAS_TARIFF_DKK_MWH / DKK_EUR                        # ≈  3.4 €/MWh_el

gas_tech = Tech(
    capex         = 882_599,    # €/MW  [DEA 2030: 0.88259936 MEUR/MW_e]
    opex_fixed    = 29_562,     # €/MW/yr  [DEA 2030]
    opex_var      = 4.466 + _gas_fuel_eur_mwh_el + _gas_tariff_eur_mwh,  # €/MWh_el ≈ 91.1
    lifetime      = 25,         # years  [DEA 2030]
    discount_rate = DISCOUNT_RATE,
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
