# assumptions.py
"""
Cost and technical assumptions for the KK_datacentre model.
Sources tagged inline:
  [ET]  — Erhvervslivets Tænketank
  [FOA] — Finansiel Omstilling i Atomsektoren (or similar ET reference)
  [DEA] — Danish Energy Agency, Technology Data 2024 (2025-column values)
"""
from model import Tech, Battery, DatacenterDemand

# ── common ────────────────────────────────────────────────────────────────────

DISCOUNT_RATE = 0.04

# ── SMR (KK) ──────────────────────────────────────────────────────────────────

# Fuel price already accounts for thermal efficiency (€/GJ_el output).
# Convert to €/MWh_el: multiply by 3.6 GJ/MWh only.
_smr_fuel_eur_gj   = 259 / 100          # €/GJ_el  [ET]
_smr_opex_var_fuel = _smr_fuel_eur_gj * 3.6   # €/MWh_el ≈ 9.32
_smr_opex_var_om   = 20.42              # €/MWh  [ET]

smr_tech = Tech(
    capex         = 8_000_000,                              # €/MW  [ET]
    opex_fixed    = 0,                                      # €/MW/yr
    opex_var      = _smr_opex_var_fuel + _smr_opex_var_om, # €/MWh ≈ 29.74
    lifetime      = 60,                                     # years  [FOA]
    discount_rate = DISCOUNT_RATE,
)

SMR_CF = 9 / 10   # technical capacity factor; affects fuel cost only in v0  [ET]

# ── solar PV (utility-scale) ──────────────────────────────────────────────────

solar_tech = Tech(
    capex         = 620_000,   # €/MW  [DEA 2024, 2025]
    opex_fixed    = 9_000,     # €/MW/yr  [DEA 2024]
    opex_var      = 0,         # €/MWh
    lifetime      = 30,        # years
    discount_rate = DISCOUNT_RATE,
)

# ── onshore wind ──────────────────────────────────────────────────────────────

wind_tech = Tech(
    capex         = 1_150_000,   # €/MW  [DEA 2024, 2025]
    opex_fixed    = 18_000,      # €/MW/yr  [DEA 2024]
    opex_var      = 2.5,         # €/MWh  [DEA 2024]
    lifetime      = 30,          # years
    discount_rate = DISCOUNT_RATE,
)

# ── battery BESS (4 h duration) ───────────────────────────────────────────────

battery = Battery(
    capex_power   = 100_000,   # €/MW  (inverter + BOS)  [DEA 2024, 2025]
    capex_energy  = 200_000,   # €/MWh (cells)  [DEA 2024, 2025]
    opex_fixed    = 7_000,     # €/MW/yr  [DEA 2024]
    lifetime      = 15,        # years
    discount_rate = DISCOUNT_RATE,
    storage_hours = 24,
)

# ── demand ────────────────────────────────────────────────────────────────────

demand = DatacenterDemand(demand_mw=1_000.0, x=0.50)
