# model.py
"""
Datacenter on-site power sourcing model: KK (SMR) vs VE (solar+wind+battery).

Layer 1 — DatacenterDemand : demand and on-site fraction.
Layer 2 — GridSupply       : exogenous spot prices, import schedule and cost.
Layer 3 — KKSupply         : constant-output SMR on-site option.
          VESupply         : solar + wind + battery on-site option.
Layer 4 — DatacenterModel  : assembles layers, runs comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import pathlib
import numpy as np


# ── parameter containers ──────────────────────────────────────────────────────

@dataclass
class Tech:
    """Annualised cost for a generation technology."""
    capex:         float   # €/MW
    opex_fixed:    float   # €/MW/yr
    opex_var:      float   # €/MWh (charged on gross generation)
    lifetime:      int     # years
    discount_rate: float

    @property
    def crf(self) -> float:
        r, n = self.discount_rate, self.lifetime
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def annual_cost(self, capacity_mw: float, generation_mwh: float) -> float:
        return (
            self.capex * self.crf * capacity_mw
            + self.opex_fixed * capacity_mw
            + self.opex_var * generation_mwh
        )


@dataclass
class Battery:
    """BESS storage parameters. Power and energy are sized independently."""
    capex_power:   float   # €/MW
    capex_energy:  float   # €/MWh
    opex_fixed:    float   # €/MW/yr on power rating
    lifetime:      int
    discount_rate: float
    eta_charge:    float = 1.0   # DC charge efficiency [0, 1]
    eta_discharge: float = 1.0   # DC discharge efficiency [0, 1]

    @property
    def crf(self) -> float:
        r, n = self.discount_rate, self.lifetime
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def annual_cost(self, power_mw: float, energy_mwh: float) -> float:
        return (
            (self.capex_power * power_mw + self.capex_energy * energy_mwh) * self.crf
            + self.opex_fixed * power_mw
        )


@dataclass
class Result:
    label:               str
    annual_onsite_cost:  float         # €/yr
    annual_grid_cost:    float         # €/yr
    annual_total_cost:   float         # €/yr
    lcoe:                float         # €/MWh (total / annual demand)
    grid_import_mwh:     float         # MWh/yr
    onsite_capacity_mw:  float = 0.0  # KK: installed MW
    c_solar_mw:          float = 0.0  # VE only
    c_wind_mw:           float = 0.0  # VE only
    batt_power_mw:       float = 0.0  # VE only
    batt_energy_mwh:         float = 0.0  # VE only
    annual_export_revenue:   float = 0.0  # VE only (€/yr, spot price × exported MWh)
    cfe_shortfall_mwh:       float = 0.0  # VE only (MWh/yr floor unmet; 0 = fully feasible)
    annual_tariff_cost:      float = 0.0  # VE only (grid consumption + production tariffs)
    annual_grid_connect:     float = 0.0  # VE only (annualised tilslutningsbidrag)
    annual_inv_cost:         float = 0.0  # capex × CRF (season-prorated)
    annual_om_cost:          float = 0.0  # opex_fixed + opex_var (season-prorated fixed, actual variable)
    c_gas_mw:                float = 0.0  # VEGAS only: gas turbine capacity (MW)

    def __repr__(self) -> str:
        if self.label == "KK":
            cap = f"{self.onsite_capacity_mw:.0f} MW SMR"
        elif self.label == "VEGAS":
            cap = (
                f"{self.c_solar_mw:.0f} MW PV + "
                f"{self.c_wind_mw:.0f} MW wind + "
                f"{self.batt_power_mw:.0f} MW / {self.batt_energy_mwh:.0f} MWh batt + "
                f"{self.c_gas_mw:.0f} MW gas"
            )
        else:
            cap = (
                f"{self.c_solar_mw:.0f} MW PV + "
                f"{self.c_wind_mw:.0f} MW wind + "
                f"{self.batt_power_mw:.0f} MW / {self.batt_energy_mwh:.0f} MWh batt"
            )
        s = (
            f"{self.label} [{cap}] | "
            f"LCOE {self.lcoe:.2f} €/MWh | "
            f"total {self.annual_total_cost / 1e6:.1f} M€/yr"
        )
        if self.cfe_shortfall_mwh > 0:
            s += f" | CFE shortfall {self.cfe_shortfall_mwh:,.0f} MWh/yr"
        return s


_FULL_YEAR_HOURS = 8760   # reference for season_frac = demand.HOURS / _FULL_YEAR_HOURS


# ── layer 1: demand ───────────────────────────────────────────────────────────

class DatacenterDemand:
    """
    Flat-load datacenter. x is the legally required on-site fraction [0, 1].
    All sizing and cost comparisons are relative to this demand.
    hours defaults to 8760 (full year); pass a seasonal T_sub for sub-annual runs.
    """
    HOURS: int = 8760   # class-level default; overridden as instance attr when hours≠8760

    def __init__(self, demand_mw: float = 1_000.0, x: float = 0.50, hours: int = 8760):
        self.demand_mw = demand_mw
        self.x = x
        self.HOURS = hours

    @property
    def floor_mw(self) -> float:
        """Minimum on-site production every hour."""
        return self.x * self.demand_mw

    @property
    def grid_cap_mw(self) -> float:
        """Maximum grid import per hour."""
        return (1.0 - self.x) * self.demand_mw

    @property
    def annual_mwh(self) -> float:
        return self.demand_mw * self.HOURS


# ── layer 2: grid ─────────────────────────────────────────────────────────────

class GridSupply:
    """
    Spot-price grid connection. Imports fill the gap between on-site and total
    demand, capped at grid_cap_mw. Cannot backfill on-site CFE shortfalls.
    """
    def __init__(self, prices: np.ndarray, demand: DatacenterDemand):
        if len(prices) != demand.HOURS:
            raise ValueError(f"prices must be length {demand.HOURS}, got {len(prices)}")
        self.prices = prices
        self.demand = demand

    def imports(self, onsite_mw: np.ndarray) -> np.ndarray:
        """Hourly grid imports (MW) given an on-site dispatch profile."""
        return np.clip(self.demand.demand_mw - onsite_mw, 0.0, self.demand.grid_cap_mw)

    def cost(self, onsite_mw: np.ndarray) -> float:
        """Annual grid purchase cost (€)."""
        return float((self.imports(onsite_mw) * self.prices).sum())


# ── layer 3a: KK on-site ─────────────────────────────────────────────────────

class KKSupply:
    """
    SMR nuclear: constant output at floor_mw meets the hourly CFE requirement
    with zero excess — minimum feasible capacity by construction.

    capacity_factor affects fuel cost (annual generation) only; dispatch stays
    constant at floor_mw (100% availability assumed in v0).
    """
    def __init__(self, tech: Tech, demand: DatacenterDemand, prices: np.ndarray,
                 downtime_fraction: float = 0.10, buy_tariff: float = 0.0,
                 grid_connect_annual: float = 0.0):
        self.tech                = tech
        self.demand              = demand
        self.prices              = prices
        self.downtime_fraction   = downtime_fraction
        self.buy_tariff          = buy_tariff
        self.grid_connect_annual = grid_connect_annual
        self._outage: np.ndarray | None = None

    @property
    def capacity_mw(self) -> float:
        return self.demand.demand_mw

    @property
    def downtime_hours(self) -> int:
        return round(self.downtime_fraction * self.demand.HOURS)

    def _outage_mask(self) -> np.ndarray:
        """Contiguous downtime window placed at the cheapest hours (min spot + tariff)."""
        if self._outage is None:
            n = self.downtime_hours
            cum = np.concatenate([[0.0], np.cumsum(self.prices + self.buy_tariff)])
            start = int(np.argmin(cum[n:] - cum[:-n]))
            self._outage = np.zeros(self.demand.HOURS, dtype=bool)
            self._outage[start : start + n] = True
        return self._outage

    def dispatch(self) -> np.ndarray:
        d = np.full(self.demand.HOURS, self.capacity_mw)
        d[self._outage_mask()] = 0.0
        return d

    def annual_cost(self) -> float:
        sf             = self.demand.HOURS / _FULL_YEAR_HOURS
        generation_mwh = self.capacity_mw * (self.demand.HOURS - self.downtime_hours)
        fixed = (self.tech.capex * self.tech.crf + self.tech.opex_fixed) * self.capacity_mw * sf
        return fixed + self.tech.opex_var * generation_mwh

    def result(self, grid: GridSupply) -> Result:
        outage          = self._outage_mask()
        onsite_cost     = self.annual_cost()
        grid_import_mwh = float(self.capacity_mw * outage.sum())
        grid_cost       = float((self.prices[outage] * self.capacity_mw).sum())
        tariff_cost     = self.buy_tariff * grid_import_mwh
        total           = onsite_cost + grid_cost + tariff_cost + self.grid_connect_annual
        sf              = self.demand.HOURS / _FULL_YEAR_HOURS
        generation_mwh  = self.capacity_mw * (self.demand.HOURS - self.downtime_hours)
        inv_cost        = self.tech.capex * self.tech.crf * self.capacity_mw * sf
        om_cost         = self.tech.opex_fixed * self.capacity_mw * sf + self.tech.opex_var * generation_mwh
        return Result(
            label               = "KK",
            onsite_capacity_mw  = self.capacity_mw,
            annual_onsite_cost  = onsite_cost,
            annual_grid_cost    = grid_cost,
            annual_total_cost   = total,
            lcoe                = total / self.demand.annual_mwh,
            grid_import_mwh     = grid_import_mwh,
            annual_tariff_cost  = tariff_cost,
            annual_grid_connect = self.grid_connect_annual,
            annual_inv_cost     = inv_cost,
            annual_om_cost      = om_cost,
        )


# ── layer 3b: VE on-site ─────────────────────────────────────────────────────

class VESupply:
    """
    Solar + wind + battery. Jointly optimises (C_solar, C_wind, P_batt [, E_batt]).

    When prices are provided:
      Single LP over all capacity and dispatch variables simultaneously (scipy HiGHS).
      Battery is fully bidirectional. CFE floor enforced as grid_buy[t] ≤ grid_cap_mw.
      storage_hours=None makes E_batt a free LP variable; set → E_batt = sh × P_batt.

    Without prices (no-price mode):
      Greedy dispatch. Nelder-Mead outer over (C_solar, C_wind, P_batt); inner
      bisection on E_batt for feasibility.
    """
    def __init__(
        self,
        solar_cf:            np.ndarray,
        wind_cf:             np.ndarray,
        solar_tech:          Tech,
        wind_tech:           Tech,
        battery:             Battery,
        demand:              DatacenterDemand,
        prices:              np.ndarray | None = None,
        storage_hours:       float | None = None,   # None → free E_batt (bisection); set → fixed ratio
        buy_tariff:          float = 0.0,            # €/MWh added to grid_buy cost in LP
        sell_tariff:         float = 0.0,            # €/MWh deducted from grid_sell revenue in LP
        grid_connect_annual: float = 0.0,            # €/yr annualised tilslutningsbidrag
        cfe_penalty:         float = 1e6,            # €/MWh penalty for CFE shortfall (storage_hours mode)
        bisect_upper_energy: float = 1_000_000.0,   # MWh — ceiling for inner bisection
        tol_energy:          float = 1.0,           # MWh — inner bisection tolerance
        tol_ve:              float = 1.0,           # MW  — outer Nelder-Mead tolerance
        max_solar_mw:        float | None = None,   # upper bound on c_solar (e.g. area cap)
        max_wind_mw:         float | None = None,   # upper bound on c_wind
    ):
        if len(solar_cf) != demand.HOURS or len(wind_cf) != demand.HOURS:
            raise ValueError(f"Capacity factor arrays must match demand.HOURS ({demand.HOURS})")
        if prices is not None and len(prices) != demand.HOURS:
            raise ValueError("prices must be length 8760")
        self.solar_cf            = solar_cf
        self.wind_cf             = wind_cf
        self.solar_tech          = solar_tech
        self.wind_tech           = wind_tech
        self.battery             = battery
        self.demand              = demand
        self.prices              = prices
        self.buy_tariff          = buy_tariff
        self.sell_tariff         = sell_tariff
        self.grid_connect_annual = grid_connect_annual
        self.storage_hours       = storage_hours
        self.cfe_penalty         = cfe_penalty
        self.bisect_upper_energy = bisect_upper_energy
        self.tol_energy          = tol_energy
        self.tol_ve              = tol_ve
        self.max_solar_mw        = max_solar_mw
        self.max_wind_mw         = max_wind_mw
        self._solution: tuple[float, float, float, float] | None = None  # (c_solar, c_wind, batt_power, batt_energy)
        self._lp_cache: dict | None = None

    # ── simulation ────────────────────────────────────────────────────────────

    def _simulate(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float,
        soc_init: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Greedy dispatch. Battery charges from VE surplus above floor; discharges to
        cover shortfalls. Any VE above demand_mw after charging is exported.
        Returns (onsite MW, exported MW, cfe_shortfall_mwh, final_soc).
        """
        floor         = self.demand.floor_mw
        avail         = c_solar * self.solar_cf + c_wind * self.wind_cf
        onsite        = np.empty(self.demand.HOURS)
        sold          = np.zeros(self.demand.HOURS)
        soc           = soc_init
        shortfall_mwh = 0.0
        eta_c         = self.battery.eta_charge
        eta_d         = self.battery.eta_discharge

        for t in range(self.demand.HOURS):
            a = avail[t]
            if a >= floor:
                surplus    = a - floor
                charge     = min(surplus, batt_power, (batt_energy - soc) / eta_c)
                soc       += eta_c * charge
                consume    = min(a - charge, self.demand.demand_mw)
                sold[t]    = min(max(0.0, a - charge - self.demand.demand_mw), self.demand.grid_cap_mw)
                onsite[t]  = consume
            else:
                deficit    = floor - a
                discharge  = min(deficit, batt_power, soc * eta_d)
                soc       -= discharge / eta_d
                onsite[t]  = a + discharge
                shortfall_mwh += max(0.0, floor - onsite[t])
        return onsite, sold, shortfall_mwh, soc

    def _simulate_cyclic(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Two-pass greedy: first pass finds end-of-year SOC; second pass starts there."""
        _, _, _, soc_end = self._simulate(c_solar, c_wind, batt_power, batt_energy)
        return self._simulate(c_solar, c_wind, batt_power, batt_energy, soc_end)

    # ── LP dispatch (perfect foresight) ──────────────────────────────────────

    def _lp_solve(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float
    ) -> dict | None:
        """
        Solve the perfect-foresight LP for given capacities.

        Variable layout (7 blocks × T):
          0=charge  1=discharge  2=soc  3=grid_buy  4=grid_sell  5=curtail  6=cfe_excess

        grid_buy is bounded at demand_mw (not grid_cap_mw) so the LP is always feasible.
        cfe_excess[t] = max(0, grid_buy[t] - grid_cap_mw) tracks CFE violations; penalised
        at cfe_penalty in the objective so the LP internalises the constraint.

        Returns dict of arrays + 'lp_obj' (res.fun), or None if HiGHS fails unexpectedly.
        """
        from scipy.sparse import eye as speye, diags, hstack, vstack, csr_matrix
        from scipy.optimize import linprog

        T  = self.demand.HOURS
        d  = self.demand.demand_mw
        g  = self.demand.grid_cap_mw
        p  = self.prices
        M  = self.cfe_penalty
        ve = c_solar * self.solar_cf + c_wind * self.wind_cf

        I = speye(T, format='csr')
        Z = csr_matrix((T, T))
        # Cyclic L: L[0, T-1] = -1 makes the t=0 row use soc[T-1] as the prior SOC
        # instead of implicitly assuming soc_init = 0.
        L = (diags([np.ones(T), -np.ones(T - 1)], [0, -1], shape=(T, T), format='csr')
             + csr_matrix(([-1.0], ([0], [T - 1])), shape=(T, T)))

        # objective: min Σ [(price+buy_tariff)*grid_buy - (price-sell_tariff)*grid_sell + M*cfe_excess]
        # curtail_wl has cost -opex_var: LP saves wind O&M by curtailing wind before solar
        c_obj = np.concatenate([np.zeros(3 * T), p + self.buy_tariff, -p + self.sell_tariff,
                                 np.zeros(T), np.full(T, -self.wind_tech.opex_var), np.full(T, M)])

        # energy balance (equality): charge - discharge - grid_buy + grid_sell + curtail_pv + curtail_wl = ve - d
        A_en  = hstack([I, -I, Z, -I, I, I, I, Z], format='csr')
        b_en  = ve - d

        # SOC recurrence (equality): -η_c·charge + (1/η_d)·discharge + L·soc = 0
        eta_c = self.battery.eta_charge
        eta_d = self.battery.eta_discharge
        A_soc = hstack([-eta_c * I, (1.0 / eta_d) * I, L, Z, Z, Z, Z, Z], format='csr')
        b_soc = np.zeros(T)

        A_eq = vstack([A_en, A_soc], format='csr')
        b_eq = np.concatenate([b_en, b_soc])

        # CFE inequality: grid_buy - cfe_excess ≤ grid_cap_mw
        A_cfe = hstack([Z, Z, Z, I, Z, Z, Z, -I], format='csr')
        b_cfe = np.full(T, g)

        bounds = (
              [(0.0, batt_power)]  * T                                     # charge
            + [(0.0, batt_power)]  * T                                     # discharge
            + [(0.0, batt_energy)] * T                                     # soc
            + [(0.0, d)]           * T                                     # grid_buy
            + [(0.0, g)]           * T                                     # grid_sell
            + list(zip([0.0]*T, c_solar * self.solar_cf))                  # curtail_pv ≤ pv_gen
            + list(zip([0.0]*T, c_wind  * self.wind_cf))                   # curtail_wl ≤ wl_gen
            + [(0.0, None)]        * T                                     # cfe_excess
        )

        res = linprog(c_obj, A_ub=A_cfe, b_ub=b_cfe,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'disp': False})
        if res.status != 0:
            return None
        x = res.x
        return {
            'charge':     x[0 * T : 1 * T],
            'discharge':  x[1 * T : 2 * T],
            'soc':        x[2 * T : 3 * T],
            'grid_buy':   x[3 * T : 4 * T],
            'grid_sell':  x[4 * T : 5 * T],
            'curtail_pv': x[5 * T : 6 * T],
            'curtail_wl': x[6 * T : 7 * T],
            'curtail':    x[5 * T : 6 * T] + x[6 * T : 7 * T],
            'cfe_excess': x[7 * T : 8 * T],
            'lp_obj':     float(res.fun),
        }

    def _single_lp(self) -> dict | None:
        """
        Single LP over capacity variables + full dispatch. Replaces Nelder-Mead + inner LP.

        Variable layout: [c_solar, c_wind, batt_power (, batt_energy),
                          charge(T), discharge(T), soc(T),
                          grid_buy(T), grid_sell(T), curtail(T), cfe_excess(T)]

        storage_hours=None adds batt_energy as a 4th free capacity variable.
        Returns dict with capacity scalars, all 7 dispatch arrays, pv_gen, wl_gen, lp_obj.
        """
        from scipy.sparse import eye as speye, diags, hstack, vstack, csr_matrix
        from scipy.optimize import linprog

        T      = self.demand.HOURS
        d      = self.demand.demand_mw
        g      = self.demand.grid_cap_mw
        p      = self.prices
        M      = self.cfe_penalty
        sh     = self.storage_hours
        free_e = (sh is None)
        N_cap  = 4 if free_e else 3

        # ── capacity cost coefficients ────────────────────────────────────────
        # Fixed components (capex×CRF + opex_fixed) are pro-rated by the fraction of the
        # full year covered by demand.HOURS — so seasonal runs bear only their share of
        # annual capital costs. Variable O&M is already seasonal (charged on actual dispatch).

        sf           = self.demand.HOURS / _FULL_YEAR_HOURS
        c_solar_coef = ((self.solar_tech.capex * self.solar_tech.crf + self.solar_tech.opex_fixed) * sf
                        + self.solar_tech.opex_var * float(self.solar_cf.sum()))
        c_wind_coef  = ((self.wind_tech.capex * self.wind_tech.crf + self.wind_tech.opex_fixed) * sf
                        + self.wind_tech.opex_var * float(self.wind_cf.sum()))
        if free_e:
            cap_costs = [c_solar_coef, c_wind_coef,
                         (self.battery.capex_power * self.battery.crf + self.battery.opex_fixed) * sf,
                         self.battery.capex_energy * self.battery.crf * sf]
        else:
            cap_costs = [c_solar_coef, c_wind_coef,
                         ((self.battery.capex_power + self.battery.capex_energy * sh) * self.battery.crf
                          + self.battery.opex_fixed) * sf]

        c_obj = np.concatenate([cap_costs, np.zeros(3 * T), p + self.buy_tariff, -p + self.sell_tariff,
                                 np.zeros(T), np.full(T, -self.wind_tech.opex_var), np.full(T, M)])

        # ── sparse blocks ─────────────────────────────────────────────────────

        I  = speye(T, format='csr')
        Z  = csr_matrix((T, T))
        # Cyclic L: soc[T-1] acts as prior SOC for t=0 (true annual cycle).
        L  = (diags([np.ones(T), -np.ones(T - 1)], [0, -1], shape=(T, T), format='csr')
              + csr_matrix(([-1.0], ([0], [T - 1])), shape=(T, T)))
        Zc = csr_matrix((T, N_cap))

        # ── energy balance (equality, T rows) ─────────────────────────────────
        # −c_solar·solar_cf − c_wind·wind_cf [−0·batt_power [−0·batt_energy]]
        # + charge − discharge − grid_buy + grid_sell + curtail = −demand_mw

        cf_cols = [-self.solar_cf, -self.wind_cf] + [np.zeros(T)] * (N_cap - 2)
        A_en = hstack([csr_matrix(np.column_stack(cf_cols)),
                       I, -I, Z, -I, I, I, I, Z], format='csr')

        # ── SOC recurrence (equality, T rows) ─────────────────────────────────

        eta_c    = self.battery.eta_charge
        eta_d    = self.battery.eta_discharge
        A_soc_eq = hstack([Zc, -eta_c * I, (1.0 / eta_d) * I, L, Z, Z, Z, Z, Z], format='csr')

        A_eq = vstack([A_en, A_soc_eq], format='csr')
        b_eq = np.concatenate([np.full(T, -d), np.zeros(T)])

        # ── CFE inequality (T rows): grid_buy − cfe_excess ≤ grid_cap_mw ─────

        A_cfe = hstack([Zc, Z, Z, Z, I, Z, Z, Z, -I], format='csr')

        # ── capacity bound inequalities (3T rows) ─────────────────────────────
        # charge[t]    ≤ batt_power                  → col 2 coeff = −1
        # discharge[t] ≤ batt_power                  → col 2 coeff = −1
        # soc[t]       ≤ sh·batt_power  (or E_batt)  → col 2 coeff = −sh (or col 3 = −1)

        pw = np.zeros((T, N_cap)); pw[:, 2] = -1.0
        en = np.zeros((T, N_cap))
        en[:, 3 if free_e else 2] = -1.0 if free_e else -sh

        pv_cap = np.zeros((T, N_cap)); pv_cap[:, 0] = -self.solar_cf
        wl_cap = np.zeros((T, N_cap)); wl_cap[:, 1] = -self.wind_cf

        A_ub = vstack([
            A_cfe,
            hstack([csr_matrix(pw),     I, Z, Z, Z, Z, Z, Z, Z], format='csr'),
            hstack([csr_matrix(pw),     Z, I, Z, Z, Z, Z, Z, Z], format='csr'),
            hstack([csr_matrix(en),     Z, Z, I, Z, Z, Z, Z, Z], format='csr'),
            hstack([csr_matrix(pv_cap), Z, Z, Z, Z, Z, I, Z, Z], format='csr'),
            hstack([csr_matrix(wl_cap), Z, Z, Z, Z, Z, Z, I, Z], format='csr'),
        ], format='csr')
        b_ub = np.concatenate([np.full(T, g), np.zeros(5 * T)])

        # ── bounds ────────────────────────────────────────────────────────────

        cap_bounds = (
            [(0.0, self.max_solar_mw),   # c_solar
             (0.0, self.max_wind_mw)]    # c_wind
            + [(0.0, None)] * (N_cap - 2)
        )
        bounds = (
            cap_bounds
            + [(0.0, None)] * (3 * T)    # charge, discharge, soc (upper via inequality)
            + [(0.0, d)]    * T          # grid_buy ≤ demand_mw
            + [(0.0, g)]    * T          # grid_sell ≤ grid_cap_mw
            + [(0.0, None)] * (3 * T)    # curtail_pv, curtail_wl, cfe_excess
        )

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'disp': False})
        if res.status != 0:
            return None

        x   = res.x
        c_s = float(x[0])
        c_w = float(x[1])
        b_p = float(x[2])
        b_e = float(x[3]) if free_e else sh * b_p
        off = N_cap

        return {
            'c_solar':     c_s,
            'c_wind':      c_w,
            'batt_power':  b_p,
            'batt_energy': b_e,
            'charge':      x[off + 0*T : off + 1*T],
            'discharge':   x[off + 1*T : off + 2*T],
            'soc':         x[off + 2*T : off + 3*T],
            'grid_buy':    x[off + 3*T : off + 4*T],
            'grid_sell':   x[off + 4*T : off + 5*T],
            'curtail_pv':  x[off + 5*T : off + 6*T],
            'curtail_wl':  x[off + 6*T : off + 7*T],
            'curtail':     x[off + 5*T : off + 6*T] + x[off + 6*T : off + 7*T],
            'cfe_excess':  x[off + 7*T : off + 8*T],
            'lp_obj':      float(res.fun),
            'pv_gen':      c_s * self.solar_cf,
            'wl_gen':      c_w * self.wind_cf,
        }

    def _lp_dispatch(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (onsite_mw, grid_sell_mw, cfe_shortfall_mwh).
        cfe_shortfall_mwh = sum(cfe_excess); should be ~0 at the optimum.
        """
        sol = self._lp_solve(c_solar, c_wind, batt_power, batt_energy)
        if sol is None:
            T = self.demand.HOURS
            return np.full(T, self.demand.demand_mw), np.zeros(T), 1e9
        onsite = np.full(self.demand.HOURS, self.demand.demand_mw) - sol['grid_buy']
        return onsite, sol['grid_sell'], float(sol['cfe_excess'].sum())

    # ── inner bisection (battery energy) ─────────────────────────────────────

    def _bisect_energy(self, c_solar: float, c_wind: float, batt_power: float) -> float | None:
        """
        Minimum E_batt (MWh) that makes (c_solar, c_wind, batt_power) hourly-feasible.
        Returns None if infeasible even at bisect_upper_energy.
        """
        lo, hi = 0.0, self.bisect_upper_energy
        if self._simulate_cyclic(c_solar, c_wind, batt_power, hi)[2] > 0:
            return None
        while hi - lo > self.tol_energy:
            mid = (lo + hi) / 2.0
            if self._simulate_cyclic(c_solar, c_wind, batt_power, mid)[2] == 0:
                hi = mid
            else:
                lo = mid
        return hi

    # ── outer optimiser (VE capacities + battery power) ───────────────────────

    def _onsite_cost(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float
    ) -> float:
        sf         = self.demand.HOURS / _FULL_YEAR_HOURS
        solar_mwh  = float(c_solar * self.solar_cf.sum())
        wind_mwh   = float(c_wind  * self.wind_cf.sum())
        solar_cost = ((self.solar_tech.capex * self.solar_tech.crf + self.solar_tech.opex_fixed) * c_solar * sf
                      + self.solar_tech.opex_var * solar_mwh)
        wind_cost  = ((self.wind_tech.capex  * self.wind_tech.crf  + self.wind_tech.opex_fixed)  * c_wind  * sf
                      + self.wind_tech.opex_var  * wind_mwh)
        batt_cost  = ((self.battery.capex_power * batt_power + self.battery.capex_energy * batt_energy) * self.battery.crf
                      + self.battery.opex_fixed * batt_power) * sf
        return solar_cost + wind_cost + batt_cost

    def _objective(self, params: np.ndarray) -> float:
        c_solar    = max(0.0, float(params[0]))
        c_wind     = max(0.0, float(params[1]))
        batt_power = max(0.0, float(params[2]))

        if self.storage_hours is not None:
            batt_energy = self.storage_hours * batt_power
        else:
            batt_energy = self._bisect_energy(c_solar, c_wind, batt_power)
            if batt_energy is None:
                return 1e15

        if self.prices is not None:
            # LP objective = grid_cost_net + M*cfe_excess (penalty already baked in)
            sol = self._lp_solve(c_solar, c_wind, batt_power, batt_energy)
            if sol is None:
                return 1e15
            return self._onsite_cost(c_solar, c_wind, batt_power, batt_energy) + sol['lp_obj']

        onsite, sold, shortfall_mwh, _ = self._simulate_cyclic(c_solar, c_wind, batt_power, batt_energy)
        cost  = self._onsite_cost(c_solar, c_wind, batt_power, batt_energy)
        cost += shortfall_mwh * self.cfe_penalty
        return cost

    def _x0_batt(self) -> float:
        """Starting battery power for Nelder-Mead: enough to cover the worst shortfall run at x0 VE."""
        if self.storage_hours is None:
            return 500.0
        avail    = 1_000.0 * self.solar_cf + 3_000.0 * self.wind_cf
        deficit  = np.maximum(self.demand.floor_mw - avail, 0.0)
        # sliding window: find the maximum energy deficit over any storage_hours-length window
        win      = max(1, int(self.storage_hours))
        cum      = np.cumsum(np.concatenate([[0.0], deficit]))
        max_need = float(np.max(cum[win:] - cum[:-win]))
        # P_batt must deliver max_need MWh in storage_hours hours
        return max(500.0, max_need / self.storage_hours)

    def _optimise(self) -> tuple[float, float, float, float]:
        from scipy.optimize import minimize
        res = minimize(
            self._objective,
            x0      = np.array([1_000.0, 3_000.0, self._x0_batt()]),
            method  = 'Nelder-Mead',
            options = {'xatol': self.tol_ve, 'fatol': 1e6, 'maxiter': 10_000, 'adaptive': True},
        )
        c_solar    = max(0.0, float(res.x[0]))
        c_wind     = max(0.0, float(res.x[1]))
        batt_power = max(0.0, float(res.x[2]))
        if self.storage_hours is not None:
            batt_energy = self.storage_hours * batt_power
        else:
            batt_energy = self._bisect_energy(c_solar, c_wind, batt_power)
            if batt_energy is None:
                raise RuntimeError("Optimised VE solution is infeasible — raise bisect_upper_energy.")
        return c_solar, c_wind, batt_power, batt_energy

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | pathlib.Path = 'runs/ve_solution.json') -> None:
        """Save optimised solution to JSON (runs optimiser first if needed)."""
        c_solar, c_wind, batt_power, batt_energy = self.solution
        p = pathlib.Path(path)
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({
            'c_solar':     c_solar,
            'c_wind':      c_wind,
            'batt_power':  batt_power,
            'batt_energy': batt_energy,
        }, indent=2))

    def load(self, path: str | pathlib.Path = 'runs/ve_solution.json') -> None:
        """Load a previously saved solution, bypassing the optimiser."""
        d = json.loads(pathlib.Path(path).read_text())
        self._solution = (d['c_solar'], d['c_wind'], d['batt_power'], d['batt_energy'])

    def save_lp_arrays(self, path: str | pathlib.Path = 'runs/ve_lp_arrays.npz') -> None:
        """Save LP arrays (capacities + dispatch) to npz."""
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("No prices set — LP arrays not available.")
        p = pathlib.Path(path)
        p.parent.mkdir(exist_ok=True)
        save_keys = ['c_solar', 'c_wind', 'batt_power', 'batt_energy',
                     'charge', 'discharge', 'soc', 'grid_buy', 'grid_sell',
                     'curtail', 'curtail_pv', 'curtail_wl', 'cfe_excess', 'pv_gen', 'wl_gen']
        np.savez(str(p), **{k: lp[k] for k in save_keys if k in lp})

    def load_lp_arrays(self, path: str | pathlib.Path = 'runs/ve_lp_arrays.npz') -> None:
        """Load LP arrays from npz, bypassing the LP solve."""
        data  = np.load(str(pathlib.Path(path)))
        cache = {k: data[k] for k in data.files}
        for k in ('c_solar', 'c_wind', 'batt_power', 'batt_energy'):
            if k in cache:
                cache[k] = float(cache[k])
        self._lp_cache = cache

    def save_lp_txt(self, directory: str | pathlib.Path = 'runs/lp_arrays') -> None:
        """Save LP dispatch arrays as individual txt files (one value per line)."""
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("No prices set — LP arrays not available.")
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_keys = ['charge', 'discharge', 'soc', 'grid_buy', 'grid_sell',
                     'curtail', 'curtail_pv', 'curtail_wl', 'cfe_excess', 'pv_gen', 'wl_gen']
        for k in save_keys:
            np.savetxt(str(d / f'{k}.txt'), lp[k], fmt='%.4f')

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def solution(self) -> tuple[float, float, float, float]:
        """(c_solar, c_wind, batt_power, batt_energy) — optimised and cached."""
        if self._solution is None:
            if self.prices is not None:
                lp = self.lp_detail()   # single LP path
                if lp is None:
                    raise RuntimeError("Single LP failed — check inputs.")
                self._solution = (lp['c_solar'], lp['c_wind'],
                                  lp['batt_power'], lp['batt_energy'])
            else:
                self._solution = self._optimise()   # greedy NM path
        return self._solution

    @property
    def c_solar(self) -> float:
        return self.solution[0]

    @property
    def c_wind(self) -> float:
        return self.solution[1]

    @property
    def batt_power_mw(self) -> float:
        return self.solution[2]

    @property
    def batt_energy_mwh(self) -> float:
        return self.solution[3]

    def dispatch(self) -> np.ndarray:
        if self.prices is not None:
            lp = self.lp_detail()
            return np.full(self.demand.HOURS, self.demand.demand_mw) - lp['grid_buy']
        c_solar, c_wind, batt_power, batt_energy = self.solution
        return self._simulate_cyclic(c_solar, c_wind, batt_power, batt_energy)[0]

    def lp_detail(self) -> dict | None:
        """All LP arrays for the optimised solution (single LP). None if no prices."""
        if self._lp_cache is not None:
            return self._lp_cache
        if self.prices is None:
            return None
        self._lp_cache = self._single_lp()
        return self._lp_cache

    def annual_onsite_cost(self) -> float:
        c_solar, c_wind, batt_power, batt_energy = self.solution
        return self._onsite_cost(c_solar, c_wind, batt_power, batt_energy)

    def result(self, grid: GridSupply) -> Result:
        c_solar, c_wind, batt_power, batt_energy = self.solution
        onsite_cost = self.annual_onsite_cost()

        if self.prices is not None:
            sol             = self.lp_detail()
            grid_buy        = sol['grid_buy']
            grid_sell       = sol['grid_sell']
            grid_cost       = float((grid_buy * grid.prices).sum())
            export_revenue  = float((grid_sell * np.maximum(grid.prices, 0.0)).sum())
            shortfall_mwh   = float(sol['cfe_excess'].sum())
            grid_import_mwh = float(grid_buy.sum())
            grid_sell_mwh   = float(grid_sell.sum())
        else:
            onsite, sold, shortfall_mwh, _ = self._simulate_cyclic(c_solar, c_wind, batt_power, batt_energy)
            grid_cost       = grid.cost(onsite)
            export_revenue  = float((sold * np.maximum(grid.prices, 0.0)).sum())
            grid_import_mwh = float(grid.imports(onsite).sum())
            grid_sell_mwh   = float(sold.sum())

        tariff_cost = self.buy_tariff * grid_import_mwh + self.sell_tariff * grid_sell_mwh
        total = onsite_cost + grid_cost - export_revenue + tariff_cost + self.grid_connect_annual
        sf      = self.demand.HOURS / _FULL_YEAR_HOURS
        inv_cost = (
            self.solar_tech.capex * self.solar_tech.crf * c_solar
            + self.wind_tech.capex  * self.wind_tech.crf  * c_wind
            + (self.battery.capex_power * batt_power + self.battery.capex_energy * batt_energy) * self.battery.crf
        ) * sf
        om_cost = (
            self.solar_tech.opex_fixed * c_solar
            + self.wind_tech.opex_fixed  * c_wind
            + self.battery.opex_fixed   * batt_power
        ) * sf + self.wind_tech.opex_var * float(c_wind * self.wind_cf.sum())
        return Result(
            label                  = "VE",
            c_solar_mw             = c_solar,
            c_wind_mw              = c_wind,
            batt_power_mw          = batt_power,
            batt_energy_mwh        = batt_energy,
            onsite_capacity_mw     = c_solar + c_wind,
            annual_onsite_cost     = onsite_cost,
            annual_grid_cost       = grid_cost,
            annual_export_revenue  = export_revenue,
            annual_total_cost      = total,
            lcoe                   = total / self.demand.annual_mwh,
            grid_import_mwh        = grid_import_mwh,
            cfe_shortfall_mwh      = shortfall_mwh,
            annual_tariff_cost     = tariff_cost,
            annual_grid_connect    = self.grid_connect_annual,
            annual_inv_cost        = inv_cost,
            annual_om_cost         = om_cost,
        )


# ── layer 3c: VE + gas turbine on-site ───────────────────────────────────────

class VEGasSupply:
    """
    Solar + wind + battery + gas turbine. LP-only (prices required).

    Gas turbine is last in the merit order by cost: the LP dispatches it only when
    VE + battery cannot meet the CFE floor and grid import is capped. Variable cost
    includes fuel + tariff, so gas_tech.opex_var ≈ 91 €/MWh.

    Variable layout: [c_solar, c_wind, batt_power (, batt_energy), c_gas,
                      charge(T), discharge(T), soc(T), gas_gen(T),
                      grid_buy(T), grid_sell(T), curtail_pv(T), curtail_wl(T), cfe_excess(T)]
    """
    def __init__(
        self,
        solar_cf:            np.ndarray,
        wind_cf:             np.ndarray,
        solar_tech:          Tech,
        wind_tech:           Tech,
        battery:             Battery,
        gas_tech:            Tech,
        demand:              DatacenterDemand,
        prices:              np.ndarray,
        storage_hours:       float | None = None,
        buy_tariff:          float = 0.0,
        sell_tariff:         float = 0.0,
        grid_connect_annual: float = 0.0,
        cfe_penalty:         float = 1e6,
        min_load:            float = 0.0,   # minimum stable load fraction; triggers MILP dispatch when > 0
        max_solar_mw:        float | None = None,   # upper bound on c_solar (e.g. area cap)
        max_wind_mw:         float | None = None,   # upper bound on c_wind
    ):
        if len(solar_cf) != demand.HOURS or len(wind_cf) != demand.HOURS:
            raise ValueError(f"CF arrays must match demand.HOURS ({demand.HOURS})")
        if len(prices) != demand.HOURS:
            raise ValueError("prices must be length demand.HOURS")
        self.solar_cf            = solar_cf
        self.wind_cf             = wind_cf
        self.solar_tech          = solar_tech
        self.wind_tech           = wind_tech
        self.battery             = battery
        self.gas_tech            = gas_tech
        self.demand              = demand
        self.prices              = prices
        self.storage_hours       = storage_hours
        self.buy_tariff          = buy_tariff
        self.sell_tariff         = sell_tariff
        self.grid_connect_annual = grid_connect_annual
        self.cfe_penalty         = cfe_penalty
        self.min_load            = min_load
        self.max_solar_mw        = max_solar_mw
        self.max_wind_mw         = max_wind_mw
        self._lp_cache: dict | None = None

    def _single_lp(self) -> dict | None:
        from scipy.sparse import eye as speye, diags, hstack, vstack, csr_matrix
        from scipy.optimize import linprog

        T      = self.demand.HOURS
        d      = self.demand.demand_mw
        g      = self.demand.grid_cap_mw
        p      = self.prices
        M      = self.cfe_penalty
        sh     = self.storage_hours
        free_e = (sh is None)
        N_cap_ve = 4 if free_e else 3
        N_cap    = N_cap_ve + 1   # + c_gas (always last capacity variable)

        sf = self.demand.HOURS / _FULL_YEAR_HOURS

        # ── capacity cost coefficients ─────────────────────────────────────────
        c_solar_coef = ((self.solar_tech.capex * self.solar_tech.crf + self.solar_tech.opex_fixed) * sf
                        + self.solar_tech.opex_var * float(self.solar_cf.sum()))
        c_wind_coef  = ((self.wind_tech.capex * self.wind_tech.crf + self.wind_tech.opex_fixed) * sf
                        + self.wind_tech.opex_var * float(self.wind_cf.sum()))
        c_gas_coef   = (self.gas_tech.capex * self.gas_tech.crf + self.gas_tech.opex_fixed) * sf

        if free_e:
            cap_costs = [c_solar_coef, c_wind_coef,
                         (self.battery.capex_power * self.battery.crf + self.battery.opex_fixed) * sf,
                         self.battery.capex_energy * self.battery.crf * sf,
                         c_gas_coef]
        else:
            cap_costs = [c_solar_coef, c_wind_coef,
                         ((self.battery.capex_power + self.battery.capex_energy * sh) * self.battery.crf
                          + self.battery.opex_fixed) * sf,
                         c_gas_coef]

        # dispatch order: charge, discharge, soc, gas_gen, grid_buy, grid_sell,
        #                 curtail_pv, curtail_wl, cfe_excess  (9 blocks × T)
        c_obj = np.concatenate([
            cap_costs,
            np.zeros(3 * T),                              # charge, discharge, soc
            np.full(T, self.gas_tech.opex_var),           # gas_gen (fuel + O&M + tariff)
            p + self.buy_tariff,                          # grid_buy
            -p + self.sell_tariff,                        # grid_sell
            np.zeros(T),                                  # curtail_pv
            np.full(T, -self.wind_tech.opex_var),         # curtail_wl (saves wind O&M)
            np.full(T, M),                                # cfe_excess
        ])

        # ── sparse blocks ──────────────────────────────────────────────────────
        I  = speye(T, format='csr')
        Z  = csr_matrix((T, T))
        L  = (diags([np.ones(T), -np.ones(T - 1)], [0, -1], shape=(T, T), format='csr')
              + csr_matrix(([-1.0], ([0], [T - 1])), shape=(T, T)))
        Zc = csr_matrix((T, N_cap))

        # energy balance (T equalities):
        # sources (discharge, gas_gen, grid_buy) have -1; sinks (charge, grid_sell, curtail) have +1
        # -c_solar·cf - c_wind·cf + charge - discharge - gas_gen - grid_buy + grid_sell + curtail = -d
        cf_cols = [-self.solar_cf, -self.wind_cf] + [np.zeros(T)] * (N_cap - 2)
        A_en = hstack([csr_matrix(np.column_stack(cf_cols)),
                       I, -I, Z, -I, -I, I, I, I, Z], format='csr')

        # SOC recurrence (T equalities): gas_gen is independent of battery state
        eta_c = self.battery.eta_charge
        eta_d = self.battery.eta_discharge
        A_soc = hstack([Zc, -eta_c * I, (1.0 / eta_d) * I, L, Z, Z, Z, Z, Z, Z], format='csr')

        A_eq = vstack([A_en, A_soc], format='csr')
        b_eq = np.concatenate([np.full(T, -d), np.zeros(T)])

        # CFE inequality: grid_buy - cfe_excess ≤ grid_cap_mw
        A_cfe = hstack([Zc, Z, Z, Z, Z, I, Z, Z, Z, -I], format='csr')

        # capacity bound inequalities
        pw      = np.zeros((T, N_cap)); pw[:, 2]   = -1.0            # batt_power col
        en      = np.zeros((T, N_cap))
        en[:, 3 if free_e else 2] = -1.0 if free_e else -sh          # batt_energy or sh×batt_power
        pv_cap  = np.zeros((T, N_cap)); pv_cap[:, 0]  = -self.solar_cf
        wl_cap  = np.zeros((T, N_cap)); wl_cap[:, 1]  = -self.wind_cf
        gas_cap = np.zeros((T, N_cap)); gas_cap[:, -1] = -1.0        # c_gas col (last)

        A_ub = vstack([
            A_cfe,
            hstack([csr_matrix(pw),      I, Z, Z, Z, Z, Z, Z, Z, Z], format='csr'),  # charge ≤ batt_power
            hstack([csr_matrix(pw),      Z, I, Z, Z, Z, Z, Z, Z, Z], format='csr'),  # discharge ≤ batt_power
            hstack([csr_matrix(en),      Z, Z, I, Z, Z, Z, Z, Z, Z], format='csr'),  # soc ≤ E_batt
            hstack([csr_matrix(gas_cap), Z, Z, Z, I, Z, Z, Z, Z, Z], format='csr'),  # gas_gen ≤ c_gas
            hstack([csr_matrix(pv_cap),  Z, Z, Z, Z, Z, Z, I, Z, Z], format='csr'),  # curtail_pv ≤ pv_gen
            hstack([csr_matrix(wl_cap),  Z, Z, Z, Z, Z, Z, Z, I, Z], format='csr'),  # curtail_wl ≤ wl_gen
        ], format='csr')
        b_ub = np.concatenate([np.full(T, g), np.zeros(6 * T)])

        bounds = (
            [(0.0, self.max_solar_mw),   # c_solar
             (0.0, self.max_wind_mw)]    # c_wind
            + [(0.0, None)] * (N_cap - 2)
            + [(0.0, None)] * (3 * T)   # charge, discharge, soc
            + [(0.0, d)]    * T         # gas_gen (loose; tight via gas capacity constraint)
            + [(0.0, d)]    * T         # grid_buy ≤ demand_mw
            + [(0.0, g)]    * T         # grid_sell ≤ grid_cap_mw
            + [(0.0, None)] * (3 * T)   # curtail_pv, curtail_wl, cfe_excess
        )

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'disp': False})
        if res.status != 0:
            return None

        x   = res.x
        c_s = float(x[0])
        c_w = float(x[1])
        b_p = float(x[2])
        b_e = float(x[3]) if free_e else sh * b_p
        c_g = float(x[N_cap - 1])
        off = N_cap

        return {
            'c_solar':     c_s,
            'c_wind':      c_w,
            'batt_power':  b_p,
            'batt_energy': b_e,
            'c_gas':       c_g,
            'charge':      x[off + 0*T : off + 1*T],
            'discharge':   x[off + 1*T : off + 2*T],
            'soc':         x[off + 2*T : off + 3*T],
            'gas_gen':     x[off + 3*T : off + 4*T],
            'grid_buy':    x[off + 4*T : off + 5*T],
            'grid_sell':   x[off + 5*T : off + 6*T],
            'curtail_pv':  x[off + 6*T : off + 7*T],
            'curtail_wl':  x[off + 7*T : off + 8*T],
            'curtail':     x[off + 6*T : off + 7*T] + x[off + 7*T : off + 8*T],
            'cfe_excess':  x[off + 8*T : off + 9*T],
            'lp_obj':      float(res.fun),
            'pv_gen':      c_s * self.solar_cf,
            'wl_gen':      c_w * self.wind_cf,
        }

    def _dispatch_milp(self, c_solar: float, c_wind: float,
                       batt_power: float, batt_energy: float, c_gas: float) -> dict:
        """
        MILP dispatch with minimum-load constraint, given fixed capacities from the LP.

        Adds T binary on[t] variables. Constraint: gas_gen[t] ∈ {0} ∪ [min_load*c_gas, c_gas].
        Linearised as: gas_gen ≤ c_gas*on  AND  gas_gen ≥ min_load*c_gas*on.
        (c_gas is a scalar here so both constraints are linear in on[t] and gas_gen[t].)

        Variable layout (10T): on charge discharge soc gas_gen grid_buy grid_sell
                                curtail_pv curtail_wl cfe_excess
        """
        from scipy.sparse import eye as speye, diags, hstack, vstack, csr_matrix
        from scipy.optimize import milp, LinearConstraint, Bounds

        T     = self.demand.HOURS
        d     = self.demand.demand_mw
        g     = self.demand.grid_cap_mw
        p     = self.prices
        M     = self.cfe_penalty
        eta_c = self.battery.eta_charge
        eta_d = self.battery.eta_discharge

        pv_gen = c_solar * self.solar_cf
        wl_gen = c_wind  * self.wind_cf

        I = speye(T, format='csr')
        Z = csr_matrix((T, T))
        L = (diags([np.ones(T), -np.ones(T - 1)], [0, -1], shape=(T, T), format='csr')
             + csr_matrix(([-1.0], ([0], [T - 1])), shape=(T, T)))

        c_obj = np.concatenate([
            np.zeros(T),                              # on
            np.zeros(3 * T),                          # charge, discharge, soc
            np.full(T, self.gas_tech.opex_var),       # gas_gen
            p + self.buy_tariff,                      # grid_buy
            -p + self.sell_tariff,                    # grid_sell
            np.zeros(T),                              # curtail_pv
            np.full(T, -self.wind_tech.opex_var),     # curtail_wl
            np.full(T, M),                            # cfe_excess
        ])

        lb = np.zeros(10 * T)
        ub = np.empty(10 * T)
        ub[0*T:1*T] = 1.0;         ub[1*T:2*T] = batt_power
        ub[2*T:3*T] = batt_power;  ub[3*T:4*T] = batt_energy
        ub[4*T:5*T] = c_gas;       ub[5*T:6*T] = d
        ub[6*T:7*T] = g;           ub[7*T:8*T] = pv_gen
        ub[8*T:9*T] = wl_gen;      ub[9*T:]    = np.inf

        integrality = np.zeros(10 * T); integrality[:T] = 1   # on[t] binary

        # Energy balance (T, equality)
        A_bal    = hstack([Z, I, -I, Z, -I, -I, I, I, I, Z], format='csr')
        rhs_bal  = -d + pv_gen + wl_gen
        # SOC recurrence (T, equality)
        A_soc    = hstack([Z, -eta_c * I, (1.0 / eta_d) * I, L, Z, Z, Z, Z, Z, Z], format='csr')
        # CFE (T, ≤ g)
        A_cfe    = hstack([Z, Z, Z, Z, Z, I, Z, Z, Z, -I], format='csr')
        # Gas on/off upper (T, ≤ 0): gas_gen ≤ c_gas*on
        A_gas_up = hstack([-c_gas * I, Z, Z, Z, I, Z, Z, Z, Z, Z], format='csr')
        # Gas min load (T, ≤ 0): gas_gen ≥ min_load*c_gas*on
        A_gas_lo = hstack([self.min_load * c_gas * I, Z, Z, Z, -I, Z, Z, Z, Z, Z], format='csr')

        A      = vstack([A_bal, A_soc, A_cfe, A_gas_up, A_gas_lo], format='csr')
        lb_con = np.concatenate([rhs_bal, np.zeros(T), np.full(3 * T, -np.inf)])
        ub_con = np.concatenate([rhs_bal, np.zeros(T),
                                 np.full(T, g), np.zeros(T), np.zeros(T)])

        res = milp(c_obj, constraints=LinearConstraint(A, lb_con, ub_con),
                   integrality=integrality, bounds=Bounds(lb, ub))
        if res.status != 0:
            raise RuntimeError(f"MILP dispatch failed (status {res.status}): {res.message}")

        x = res.x
        return {
            'on':         x[0*T:1*T],
            'charge':     x[1*T:2*T],
            'discharge':  x[2*T:3*T],
            'soc':        x[3*T:4*T],
            'gas_gen':    x[4*T:5*T],
            'grid_buy':   x[5*T:6*T],
            'grid_sell':  x[6*T:7*T],
            'curtail_pv': x[7*T:8*T],
            'curtail_wl': x[8*T:9*T],
            'curtail':    x[7*T:8*T] + x[8*T:9*T],
            'cfe_excess': x[9*T:10*T],
            'milp_obj':   float(res.fun),
        }

    def lp_detail(self) -> dict | None:
        """Capacities from LP; if min_load > 0, dispatch is resolved as MILP."""
        if self._lp_cache is None:
            lp = self._single_lp()
            if lp is not None and self.min_load > 0:
                milp_disp = self._dispatch_milp(
                    lp['c_solar'], lp['c_wind'], lp['batt_power'], lp['batt_energy'], lp['c_gas']
                )
                lp.update(milp_disp)   # overwrites dispatch arrays; keeps LP capacities
            self._lp_cache = lp
        return self._lp_cache

    @property
    def solution(self) -> tuple[float, float, float, float]:
        """(c_solar, c_wind, batt_power, batt_energy) — compatible with VESupply interface."""
        lp = self.lp_detail()
        return lp['c_solar'], lp['c_wind'], lp['batt_power'], lp['batt_energy']

    def save_lp_arrays(self, path: str | pathlib.Path = 'runs/vegas_lp_arrays.npz') -> None:
        """Save LP arrays (capacities + dispatch) to npz."""
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("LP not solved.")
        p = pathlib.Path(path)
        p.parent.mkdir(exist_ok=True)
        save_keys = ['c_solar', 'c_wind', 'batt_power', 'batt_energy', 'c_gas',
                     'charge', 'discharge', 'soc', 'gas_gen', 'on',
                     'grid_buy', 'grid_sell',
                     'curtail', 'curtail_pv', 'curtail_wl', 'cfe_excess', 'pv_gen', 'wl_gen']
        np.savez(str(p), **{k: lp[k] for k in save_keys if k in lp})

    def load_lp_arrays(self, path: str | pathlib.Path = 'runs/vegas_lp_arrays.npz') -> None:
        """Load LP arrays from npz, bypassing the LP solve."""
        data  = np.load(str(pathlib.Path(path)))
        cache = {k: data[k] for k in data.files}
        for k in ('c_solar', 'c_wind', 'batt_power', 'batt_energy', 'c_gas'):
            if k in cache:
                cache[k] = float(cache[k])
        self._lp_cache = cache

    def result(self, grid: GridSupply) -> Result:
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("VEGasSupply LP failed — check inputs.")

        c_solar     = lp['c_solar']
        c_wind      = lp['c_wind']
        batt_power  = lp['batt_power']
        batt_energy = lp['batt_energy']
        c_gas       = lp['c_gas']
        gas_gen     = lp['gas_gen']
        grid_buy    = lp['grid_buy']
        grid_sell   = lp['grid_sell']

        sf = self.demand.HOURS / _FULL_YEAR_HOURS

        solar_cost   = ((self.solar_tech.capex * self.solar_tech.crf + self.solar_tech.opex_fixed) * c_solar * sf
                        + self.solar_tech.opex_var * float(c_solar * self.solar_cf.sum()))
        wind_cost    = ((self.wind_tech.capex * self.wind_tech.crf + self.wind_tech.opex_fixed) * c_wind * sf
                        + self.wind_tech.opex_var * float(c_wind * self.wind_cf.sum()))
        batt_cost    = ((self.battery.capex_power * batt_power + self.battery.capex_energy * batt_energy) * self.battery.crf
                        + self.battery.opex_fixed * batt_power) * sf
        gas_cap_cost = (self.gas_tech.capex * self.gas_tech.crf + self.gas_tech.opex_fixed) * c_gas * sf
        gas_var_cost = self.gas_tech.opex_var * float(gas_gen.sum())
        onsite_cost  = solar_cost + wind_cost + batt_cost + gas_cap_cost + gas_var_cost

        grid_cost       = float((grid_buy * grid.prices).sum())
        export_revenue  = float((grid_sell * np.maximum(grid.prices, 0.0)).sum())
        shortfall_mwh   = float(lp['cfe_excess'].sum())
        grid_import_mwh = float(grid_buy.sum())
        grid_sell_mwh   = float(grid_sell.sum())

        tariff_cost = self.buy_tariff * grid_import_mwh + self.sell_tariff * grid_sell_mwh
        total = onsite_cost + grid_cost - export_revenue + tariff_cost + self.grid_connect_annual

        inv_cost = (
            self.solar_tech.capex * self.solar_tech.crf * c_solar
            + self.wind_tech.capex  * self.wind_tech.crf  * c_wind
            + (self.battery.capex_power * batt_power + self.battery.capex_energy * batt_energy) * self.battery.crf
            + self.gas_tech.capex * self.gas_tech.crf * c_gas
        ) * sf
        om_cost = (
            self.solar_tech.opex_fixed * c_solar
            + self.wind_tech.opex_fixed  * c_wind
            + self.battery.opex_fixed   * batt_power
            + self.gas_tech.opex_fixed  * c_gas
        ) * sf + self.wind_tech.opex_var * float(c_wind * self.wind_cf.sum()) + gas_var_cost

        return Result(
            label                 = "VEGAS",
            c_solar_mw            = c_solar,
            c_wind_mw             = c_wind,
            batt_power_mw         = batt_power,
            batt_energy_mwh       = batt_energy,
            c_gas_mw              = c_gas,
            onsite_capacity_mw    = c_solar + c_wind + c_gas,
            annual_onsite_cost    = onsite_cost,
            annual_grid_cost      = grid_cost,
            annual_export_revenue = export_revenue,
            annual_total_cost     = total,
            lcoe                  = total / self.demand.annual_mwh,
            grid_import_mwh       = grid_import_mwh,
            cfe_shortfall_mwh     = shortfall_mwh,
            annual_tariff_cost    = tariff_cost,
            annual_grid_connect   = self.grid_connect_annual,
            annual_inv_cost       = inv_cost,
            annual_om_cost        = om_cost,
        )


# ── layer 4: model ────────────────────────────────────────────────────────────

class DatacenterModel:
    """Assembles all layers and runs the KK vs VE comparison."""
    def __init__(
        self,
        demand: DatacenterDemand,
        grid:   GridSupply,
        kk:     KKSupply,
        ve:     VESupply,
    ):
        self.demand = demand
        self.grid   = grid
        self.kk     = kk
        self.ve     = ve

    def run(self) -> dict[str, Result]:
        return {
            "KK": self.kk.result(self.grid),
            "VE": self.ve.result(self.grid),
        }
