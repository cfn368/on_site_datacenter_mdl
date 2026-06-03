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
    """BESS storage parameters. Power rating is set externally (= floor_mw)."""
    capex_power:   float   # €/MW
    capex_energy:  float   # €/MWh
    opex_fixed:    float   # €/MW/yr on power rating
    lifetime:      int
    discount_rate: float
    storage_hours: float   # duration: MWh per MW of power

    @property
    def crf(self) -> float:
        r, n = self.discount_rate, self.lifetime
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def annual_cost(self, power_mw: float) -> float:
        energy_mwh = power_mw * self.storage_hours
        return (
            (self.capex_power * power_mw + self.capex_energy * energy_mwh) * self.crf
            + self.opex_fixed * power_mw
        )


@dataclass
class Result:
    label:               str
    onsite_capacity_mw:  float   # installed on-site capacity (MW per tech for VE)
    annual_onsite_cost:  float   # €/yr
    annual_grid_cost:    float   # €/yr
    annual_total_cost:   float   # €/yr
    lcoe:                float   # €/MWh (total / annual demand)
    grid_import_mwh:     float   # MWh/yr

    def __repr__(self) -> str:
        return (
            f"{self.label}: {self.onsite_capacity_mw:.0f} MW on-site | "
            f"LCOE {self.lcoe:.2f} €/MWh | "
            f"total {self.annual_total_cost / 1e6:.1f} M€/yr"
        )


# ── layer 1: demand ───────────────────────────────────────────────────────────

class DatacenterDemand:
    """
    Flat-load datacenter. x is the legally required on-site fraction [0, 1].
    All sizing and cost comparisons are relative to this demand.
    """
    HOURS: int = 8760

    def __init__(self, demand_mw: float = 1_000.0, x: float = 0.50):
        self.demand_mw = demand_mw
        self.x = x

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
    def __init__(self, tech: Tech, demand: DatacenterDemand, capacity_factor: float = 1.0):
        self.tech = tech
        self.demand = demand
        self.capacity_factor = capacity_factor

    @property
    def capacity_mw(self) -> float:
        return self.demand.floor_mw

    def dispatch(self) -> np.ndarray:
        return np.full(self.demand.HOURS, self.capacity_mw)

    def annual_cost(self) -> float:
        generation_mwh = self.capacity_mw * self.capacity_factor * self.demand.HOURS
        return self.tech.annual_cost(self.capacity_mw, generation_mwh)

    def result(self, grid: GridSupply) -> Result:
        d = self.dispatch()
        onsite_cost = self.annual_cost()
        grid_cost   = grid.cost(d)
        total       = onsite_cost + grid_cost
        return Result(
            label               = "KK",
            onsite_capacity_mw  = self.capacity_mw,
            annual_onsite_cost  = onsite_cost,
            annual_grid_cost    = grid_cost,
            annual_total_cost   = total,
            lcoe                = total / self.demand.annual_mwh,
            grid_import_mwh     = float(grid.imports(d).sum()),
        )


# ── layer 3b: VE on-site ─────────────────────────────────────────────────────

class VESupply:
    """
    Solar + wind (1:1 installed MW) + BESS.

    Battery power = floor_mw; energy = storage_hours × power (fixed, v0).
    Dispatch: battery charges from VE surplus above floor_mw, discharges to cover
    shortfalls. Residual surplus (above battery limit) directly reduces grid imports.
    Minimum per-technology capacity found by bisection.
    """
    def __init__(
        self,
        solar_cf:        np.ndarray,
        wind_cf:         np.ndarray,
        solar_tech:      Tech,
        wind_tech:       Tech,
        battery:         Battery,
        demand:          DatacenterDemand,
        bisect_upper_mw: float = 5_000.0,
        tol_mw:          float = 0.1,
    ):
        if len(solar_cf) != demand.HOURS or len(wind_cf) != demand.HOURS:
            raise ValueError("Capacity factor arrays must be length 8760")
        self.solar_cf        = solar_cf
        self.wind_cf         = wind_cf
        self.solar_tech      = solar_tech
        self.wind_tech       = wind_tech
        self.battery         = battery
        self.demand          = demand
        self.bisect_upper_mw = bisect_upper_mw
        self.tol_mw          = tol_mw
        self._capacity_mw: float | None = None

    def _simulate(self, c_mw: float) -> tuple[np.ndarray, bool]:
        """
        Greedy dispatch for per-technology VE capacity c_mw (MW).
        SOC initialised at 0 (worst-case for the first low-VE window).
        Returns (onsite profile MW, feasible).
        """
        floor      = self.demand.floor_mw
        batt_power = floor
        batt_cap   = self.battery.storage_hours * batt_power
        avail      = c_mw * (self.solar_cf + self.wind_cf)
        onsite     = np.empty(self.demand.HOURS)
        soc        = 0.0

        for t in range(self.demand.HOURS):
            a = avail[t]
            if a >= floor:
                surplus   = a - floor
                charge    = min(surplus, batt_power, batt_cap - soc)
                soc      += charge
                # Remaining surplus after battery charging still serves the datacenter
                onsite[t] = min(a - charge, self.demand.demand_mw)
            else:
                shortfall  = floor - a
                discharge  = min(shortfall, batt_power, soc)
                soc       -= discharge
                onsite[t]  = a + discharge
                if onsite[t] < floor - 1e-6:
                    return onsite, False

        return onsite, True

    @property
    def capacity_mw(self) -> float:
        """Minimum per-technology capacity (MW) to meet hourly on-site floor. Cached."""
        if self._capacity_mw is None:
            self._capacity_mw = self._bisect()
        return self._capacity_mw

    def _bisect(self) -> float:
        lo, hi = 0.0, self.bisect_upper_mw
        if not self._simulate(hi)[1]:
            raise ValueError(f"bisect_upper_mw={hi} MW is infeasible; increase it.")
        while hi - lo > self.tol_mw:
            mid = (lo + hi) / 2.0
            _, ok = self._simulate(mid)
            if ok:
                hi = mid
            else:
                lo = mid
        return hi

    def dispatch(self) -> np.ndarray:
        return self._simulate(self.capacity_mw)[0]

    def annual_cost(self) -> float:
        c = self.capacity_mw
        # Variable O&M on gross generation (before curtailment), per v0 convention
        solar_mwh = float(c * self.solar_cf.sum())
        wind_mwh  = float(c * self.wind_cf.sum())
        return (
            self.solar_tech.annual_cost(c, solar_mwh)
            + self.wind_tech.annual_cost(c, wind_mwh)
            + self.battery.annual_cost(self.demand.floor_mw)
        )

    def result(self, grid: GridSupply) -> Result:
        d           = self.dispatch()
        onsite_cost = self.annual_cost()
        grid_cost   = grid.cost(d)
        total       = onsite_cost + grid_cost
        return Result(
            label               = "VE",
            onsite_capacity_mw  = self.capacity_mw,
            annual_onsite_cost  = onsite_cost,
            annual_grid_cost    = grid_cost,
            annual_total_cost   = total,
            lcoe                = total / self.demand.annual_mwh,
            grid_import_mwh     = float(grid.imports(d).sum()),
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
