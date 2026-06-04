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

    def __repr__(self) -> str:
        if self.label == "KK":
            cap = f"{self.onsite_capacity_mw:.0f} MW SMR"
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
    Solar + wind + battery. Jointly optimises (C_solar, C_wind, P_batt).

    When prices are provided:
      Outer: Nelder-Mead over (C_solar, C_wind, P_batt).
      Inner: perfect-foresight LP over all 8760 hours (scipy HiGHS). Battery is
             fully bidirectional — can charge from grid and discharge to grid.
             CFE floor enforced as grid_buy[t] ≤ grid_cap_mw per hour.

    Without prices (no-price mode):
      Greedy dispatch. Inner bisection on E_batt for feasibility.

    storage_hours fixes E_batt = storage_hours × P_batt (removes inner bisection).
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
        cfe_penalty:         float = 1e6,           # €/MWh penalty for CFE shortfall (storage_hours mode)
        bisect_upper_energy: float = 1_000_000.0,  # MWh — ceiling for inner bisection
        tol_energy:          float = 1.0,           # MWh — inner bisection tolerance
        tol_ve:              float = 1.0,           # MW  — outer Nelder-Mead tolerance
    ):
        if len(solar_cf) != demand.HOURS or len(wind_cf) != demand.HOURS:
            raise ValueError("Capacity factor arrays must be length 8760")
        if prices is not None and len(prices) != demand.HOURS:
            raise ValueError("prices must be length 8760")
        self.solar_cf            = solar_cf
        self.wind_cf             = wind_cf
        self.solar_tech          = solar_tech
        self.wind_tech           = wind_tech
        self.battery             = battery
        self.demand              = demand
        self.prices              = prices
        self.storage_hours       = storage_hours
        self.cfe_penalty         = cfe_penalty
        self.bisect_upper_energy = bisect_upper_energy
        self.tol_energy          = tol_energy
        self.tol_ve              = tol_ve
        self._solution: tuple[float, float, float, float] | None = None  # (c_solar, c_wind, batt_power, batt_energy)
        self._lp_cache: dict | None = None

    # ── simulation ────────────────────────────────────────────────────────────

    def _simulate(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Greedy dispatch. Battery charges from VE surplus above floor; discharges to
        cover shortfalls. Any VE above demand_mw after charging is exported.
        SOC initialised at 0 (worst-case for opening hours).
        Returns (onsite MW, exported MW, cfe_shortfall_mwh).
        cfe_shortfall_mwh = 0 means fully feasible.
        """
        floor         = self.demand.floor_mw
        avail         = c_solar * self.solar_cf + c_wind * self.wind_cf
        onsite        = np.empty(self.demand.HOURS)
        sold          = np.zeros(self.demand.HOURS)
        soc           = 0.0
        shortfall_mwh = 0.0

        for t in range(self.demand.HOURS):
            a = avail[t]
            if a >= floor:
                surplus    = a - floor
                charge     = min(surplus, batt_power, batt_energy - soc)
                soc       += charge
                consume    = min(a - charge, self.demand.demand_mw)
                sold[t]    = min(max(0.0, a - charge - self.demand.demand_mw), self.demand.grid_cap_mw)
                onsite[t]  = consume
            else:
                deficit    = floor - a
                discharge  = min(deficit, batt_power, soc)
                soc       -= discharge
                onsite[t]  = a + discharge
                shortfall_mwh += max(0.0, floor - onsite[t])
        return onsite, sold, shortfall_mwh

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
        L = diags([np.ones(T), -np.ones(T - 1)], [0, -1], shape=(T, T), format='csr')

        # objective: min Σ [price*(grid_buy - grid_sell) + M*cfe_excess]
        c_obj = np.concatenate([np.zeros(3 * T), p, -p, np.zeros(T), np.full(T, M)])

        # energy balance (equality): charge - discharge - grid_buy + grid_sell + curtail = ve - d
        A_en  = hstack([I, -I, Z, -I, I, I, Z], format='csr')
        b_en  = ve - d

        # SOC recurrence (equality): -charge + discharge + L*soc = 0
        A_soc = hstack([-I, I, L, Z, Z, Z, Z], format='csr')
        b_soc = np.zeros(T)

        A_eq = vstack([A_en, A_soc], format='csr')
        b_eq = np.concatenate([b_en, b_soc])

        # CFE inequality: grid_buy - cfe_excess ≤ grid_cap_mw
        A_cfe = hstack([Z, Z, Z, I, Z, Z, -I], format='csr')
        b_cfe = np.full(T, g)

        bounds = (
              [(0.0, batt_power)]  * T   # charge (from VE or grid)
            + [(0.0, batt_power)]  * T   # discharge (to demand or grid)
            + [(0.0, batt_energy)] * T   # soc
            + [(0.0, d)]           * T   # grid_buy ≤ demand_mw (always feasible)
            + [(0.0, g)]           * T   # grid_sell ≤ grid_cap_mw
            + [(0.0, None)]        * T   # curtail
            + [(0.0, None)]        * T   # cfe_excess
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
            'curtail':    x[5 * T : 6 * T],
            'cfe_excess': x[6 * T : 7 * T],
            'lp_obj':     float(res.fun),
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
        if self._simulate(c_solar, c_wind, batt_power, hi)[2] > 0:
            return None
        while hi - lo > self.tol_energy:
            mid = (lo + hi) / 2.0
            if self._simulate(c_solar, c_wind, batt_power, mid)[2] == 0:
                hi = mid
            else:
                lo = mid
        return hi

    # ── outer optimiser (VE capacities + battery power) ───────────────────────

    def _onsite_cost(
        self, c_solar: float, c_wind: float, batt_power: float, batt_energy: float
    ) -> float:
        solar_mwh = float(c_solar * self.solar_cf.sum())
        wind_mwh  = float(c_wind  * self.wind_cf.sum())
        return (
            self.solar_tech.annual_cost(c_solar, solar_mwh)
            + self.wind_tech.annual_cost(c_wind, wind_mwh)
            + self.battery.annual_cost(batt_power, batt_energy)
        )

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

        onsite, sold, shortfall_mwh = self._simulate(c_solar, c_wind, batt_power, batt_energy)
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
        """Save LP dispatch arrays to npz. Runs the LP if not already cached."""
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("No prices set — LP arrays not available.")
        p = pathlib.Path(path)
        p.parent.mkdir(exist_ok=True)
        save_keys = ['charge', 'discharge', 'soc', 'grid_buy', 'grid_sell',
                     'curtail', 'cfe_excess', 'pv_gen', 'wl_gen']
        np.savez(str(p), **{k: lp[k] for k in save_keys})

    def load_lp_arrays(self, path: str | pathlib.Path = 'runs/ve_lp_arrays.npz') -> None:
        """Load LP dispatch arrays from npz, bypassing the LP re-solve."""
        data = np.load(str(pathlib.Path(path)))
        self._lp_cache = {k: data[k] for k in data.files}

    def save_lp_txt(self, directory: str | pathlib.Path = 'runs/lp_arrays') -> None:
        """Save LP dispatch arrays as individual txt files (one value per line)."""
        lp = self.lp_detail()
        if lp is None:
            raise RuntimeError("No prices set — LP arrays not available.")
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_keys = ['charge', 'discharge', 'soc', 'grid_buy', 'grid_sell',
                     'curtail', 'cfe_excess', 'pv_gen', 'wl_gen']
        for k in save_keys:
            np.savetxt(str(d / f'{k}.txt'), lp[k], fmt='%.4f')

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def solution(self) -> tuple[float, float, float, float]:
        """(c_solar, c_wind, batt_power, batt_energy) — optimised and cached."""
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

    @property
    def batt_energy_mwh(self) -> float:
        return self.solution[3]

    def dispatch(self) -> np.ndarray:
        c_solar, c_wind, batt_power, batt_energy = self.solution
        if self.prices is not None:
            return self._lp_dispatch(c_solar, c_wind, batt_power, batt_energy)[0]
        return self._simulate(c_solar, c_wind, batt_power, batt_energy)[0]

    def lp_detail(self) -> dict | None:
        """All LP dispatch arrays for the optimised solution. None if no prices."""
        if self._lp_cache is not None:
            return self._lp_cache
        if self.prices is None:
            return None
        c_solar, c_wind, batt_power, batt_energy = self.solution
        result = self._lp_solve(c_solar, c_wind, batt_power, batt_energy)
        if result is not None:
            result['pv_gen'] = c_solar * self.solar_cf
            result['wl_gen'] = c_wind * self.wind_cf
        self._lp_cache = result
        return result

    def annual_onsite_cost(self) -> float:
        c_solar, c_wind, batt_power, batt_energy = self.solution
        return self._onsite_cost(c_solar, c_wind, batt_power, batt_energy)

    def result(self, grid: GridSupply) -> Result:
        c_solar, c_wind, batt_power, batt_energy = self.solution
        onsite_cost = self.annual_onsite_cost()

        if self.prices is not None:
            sol            = self._lp_solve(c_solar, c_wind, batt_power, batt_energy)
            grid_buy       = sol['grid_buy']
            grid_sell      = sol['grid_sell']
            grid_cost      = float((grid_buy * grid.prices).sum())
            export_revenue = float((grid_sell * np.maximum(grid.prices, 0.0)).sum())
            shortfall_mwh  = float(sol['cfe_excess'].sum())
            grid_import_mwh = float(grid_buy.sum())
        else:
            onsite, sold, shortfall_mwh = self._simulate(c_solar, c_wind, batt_power, batt_energy)
            grid_cost       = grid.cost(onsite)
            export_revenue  = float((sold * np.maximum(grid.prices, 0.0)).sum())
            grid_import_mwh = float(grid.imports(onsite).sum())

        total = onsite_cost + grid_cost - export_revenue
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
