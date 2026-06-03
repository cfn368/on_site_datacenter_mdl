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
    annual_onsite_cost:  float         # €/yr
    annual_grid_cost:    float         # €/yr
    annual_total_cost:   float         # €/yr
    lcoe:                float         # €/MWh (total / annual demand)
    grid_import_mwh:     float         # MWh/yr
    onsite_capacity_mw:  float = 0.0  # KK: installed MW
    c_solar_mw:          float = 0.0  # VE only
    c_wind_mw:           float = 0.0  # VE only
    batt_power_mw:       float = 0.0  # VE only

    def __repr__(self) -> str:
        if self.label == "KK":
            cap = f"{self.onsite_capacity_mw:.0f} MW SMR"
        else:
            cap = (
                f"{self.c_solar_mw:.0f} MW PV + "
                f"{self.c_wind_mw:.0f} MW wind + "
                f"{self.batt_power_mw:.0f} MW batt"
            )
        return (
            f"{self.label} [{cap}] | "
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
        return self.demand.demand_mw / self.capacity_factor

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
            label              = "KK",
            onsite_capacity_mw = self.capacity_mw,
            annual_onsite_cost = onsite_cost,
            annual_grid_cost   = grid_cost,
            annual_total_cost  = total,
            lcoe               = total / self.demand.annual_mwh,
            grid_import_mwh    = float(grid.imports(d).sum()),
        )


# ── layer 3b: VE on-site ─────────────────────────────────────────────────────

class VESupply:
    """
    Solar + wind + battery. Jointly optimises (C_solar, C_wind, P_batt) where
    battery energy = storage_hours × P_batt.

    Outer: Nelder-Mead over (C_solar, C_wind), minimising total annual cost.
    Inner: bisection on P_batt — minimum battery power that makes each
           (C_solar, C_wind) pair feasible for the hourly on-site floor.

    Providing prices includes grid purchase cost in the outer objective so the
    optimizer trades off larger VE against cheaper grid fills.
    """
    def __init__(
        self,
        solar_cf:           np.ndarray,
        wind_cf:            np.ndarray,
        solar_tech:         Tech,
        wind_tech:          Tech,
        battery:            Battery,
        demand:             DatacenterDemand,
        prices:             np.ndarray | None = None,
        bisect_upper_batt:  float = 50_000.0,   # MW — ceiling for inner bisection
        tol_batt:           float = 1.0,         # MW — inner bisection tolerance
        tol_ve:             float = 1.0,         # MW — outer Nelder-Mead tolerance
    ):
        if len(solar_cf) != demand.HOURS or len(wind_cf) != demand.HOURS:
            raise ValueError("Capacity factor arrays must be length 8760")
        if prices is not None and len(prices) != demand.HOURS:
            raise ValueError("prices must be length 8760")
        self.solar_cf          = solar_cf
        self.wind_cf           = wind_cf
        self.solar_tech        = solar_tech
        self.wind_tech         = wind_tech
        self.battery           = battery
        self.demand            = demand
        self.prices            = prices
        self.bisect_upper_batt = bisect_upper_batt
        self.tol_batt          = tol_batt
        self.tol_ve            = tol_ve
        self._solution: tuple[float, float, float] | None = None  # (c_solar, c_wind, batt_power)

    # ── simulation ────────────────────────────────────────────────────────────

    def _simulate(self, c_solar: float, c_wind: float, batt_power: float) -> tuple[np.ndarray, bool]:
        """
        Greedy dispatch. Battery charges from VE surplus above floor; discharges to
        cover shortfalls. SOC initialised at 0 (worst-case for opening hours).
        Returns (onsite profile MW, feasible).
        """
        floor    = self.demand.floor_mw
        batt_cap = batt_power * self.battery.storage_hours
        avail    = c_solar * self.solar_cf + c_wind * self.wind_cf
        onsite   = np.empty(self.demand.HOURS)
        soc      = 0.0

        for t in range(self.demand.HOURS):
            a = avail[t]
            if a >= floor:
                surplus   = a - floor
                charge    = min(surplus, batt_power, batt_cap - soc)
                soc      += charge
                onsite[t] = min(a - charge, self.demand.demand_mw)
            else:
                shortfall  = floor - a
                discharge  = min(shortfall, batt_power, soc)
                soc       -= discharge
                onsite[t]  = a + discharge
                if onsite[t] < floor - 1e-6:
                    return onsite, False
        return onsite, True

    # ── inner bisection (battery power) ───────────────────────────────────────

    def _bisect_battery(self, c_solar: float, c_wind: float) -> float | None:
        """
        Minimum P_batt that makes (c_solar, c_wind) hourly-feasible.
        Returns None if infeasible even at bisect_upper_batt.
        """
        lo, hi = 0.0, self.bisect_upper_batt
        if not self._simulate(c_solar, c_wind, hi)[1]:
            return None
        while hi - lo > self.tol_batt:
            mid = (lo + hi) / 2.0
            if self._simulate(c_solar, c_wind, mid)[1]:
                hi = mid
            else:
                lo = mid
        return hi

    # ── outer optimiser (VE capacities) ───────────────────────────────────────

    def _onsite_cost(self, c_solar: float, c_wind: float, batt_power: float) -> float:
        solar_mwh = float(c_solar * self.solar_cf.sum())
        wind_mwh  = float(c_wind  * self.wind_cf.sum())
        return (
            self.solar_tech.annual_cost(c_solar, solar_mwh)
            + self.wind_tech.annual_cost(c_wind, wind_mwh)
            + self.battery.annual_cost(batt_power)
        )

    def _objective(self, params: np.ndarray) -> float:
        c_solar = max(0.0, float(params[0]))
        c_wind  = max(0.0, float(params[1]))
        batt_power = self._bisect_battery(c_solar, c_wind)
        if batt_power is None:
            return 1e15
        cost = self._onsite_cost(c_solar, c_wind, batt_power)
        if self.prices is not None:
            onsite, _  = self._simulate(c_solar, c_wind, batt_power)
            imports    = np.clip(self.demand.demand_mw - onsite, 0.0, self.demand.grid_cap_mw)
            cost      += float((imports * self.prices).sum())
        return cost

    def _optimise(self) -> tuple[float, float, float]:
        from scipy.optimize import minimize
        res = minimize(
            self._objective,
            x0      = np.array([1_000.0, 3_000.0]),
            method  = 'Nelder-Mead',
            options = {'xatol': self.tol_ve, 'fatol': 1e6, 'maxiter': 10_000, 'adaptive': True},
        )
        c_solar    = max(0.0, float(res.x[0]))
        c_wind     = max(0.0, float(res.x[1]))
        batt_power = self._bisect_battery(c_solar, c_wind)
        if batt_power is None:
            raise RuntimeError("Optimised VE solution is infeasible — raise bisect_upper_batt.")
        return c_solar, c_wind, batt_power

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def solution(self) -> tuple[float, float, float]:
        """(c_solar, c_wind, batt_power) MW — optimised and cached."""
        if self._solution is None:
            self._solution = self._optimise()
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

    def dispatch(self) -> np.ndarray:
        c_solar, c_wind, batt_power = self.solution
        return self._simulate(c_solar, c_wind, batt_power)[0]

    def annual_onsite_cost(self) -> float:
        c_solar, c_wind, batt_power = self.solution
        return self._onsite_cost(c_solar, c_wind, batt_power)

    def result(self, grid: GridSupply) -> Result:
        c_solar, c_wind, batt_power = self.solution
        d           = self.dispatch()
        onsite_cost = self.annual_onsite_cost()
        grid_cost   = grid.cost(d)
        total       = onsite_cost + grid_cost
        return Result(
            label              = "VE",
            c_solar_mw         = c_solar,
            c_wind_mw          = c_wind,
            batt_power_mw      = batt_power,
            onsite_capacity_mw = c_solar + c_wind,
            annual_onsite_cost = onsite_cost,
            annual_grid_cost   = grid_cost,
            annual_total_cost  = total,
            lcoe               = total / self.demand.annual_mwh,
            grid_import_mwh    = float(grid.imports(d).sum()),
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
