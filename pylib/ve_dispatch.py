"""
ve_dispatch — VE dispatch detail, aggregation, and plotting
============================================================

Extracts hour-by-hour component breakdown from a VESupply solution
and provides aggregation and stacked-area plotting utilities.
Used by 3_VE.ipynb.
"""

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    'MAANED_DK', 'DISPATCH_COLORS',
    'dispatch_detail', 'aggregate_dispatch', 'plot_dispatch',
    'battery_detail', 'plot_battery',
]


# ==================== ==================== ==================== ====================
# 0. constants

MAANED_DK = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun',
             'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']

DISPATCH_COLORS = {
    'grid':     '#C9C0B8',   # warm light grey  — grid import
    'battery':  '#C0504D',   # brick red         — battery discharge
    'wind':     '#5B9BD5',   # cornflower blue   — wind
    'pv':       '#F2C94C',   # amber gold        — solar
    'exported': '#8C7B72',   # warm medium grey  — grid export (same family as import)
}


# ==================== ==================== ==================== ====================
# 1. dispatch

def _dispatch_from_lp(lp: dict, demand_mw: float) -> dict:
    """LP-consistent demand attribution from cached LP arrays."""
    ve_gen         = lp['pv_gen'] + lp['wl_gen']
    discharge_grid = np.minimum(lp['discharge'], lp['grid_sell'])
    batt_to_demand = lp['discharge'] - discharge_grid
    grid_to_demand = lp['grid_buy']
    ve_to_demand   = demand_mw - batt_to_demand - grid_to_demand
    frac_pv        = np.where(ve_gen > 1e-9, lp['pv_gen'] / ve_gen, 0.0)
    return dict(
        pv       = ve_to_demand * frac_pv,
        wind     = ve_to_demand * (1.0 - frac_pv),
        battery  = batt_to_demand,
        grid     = grid_to_demand,
        exported = lp['grid_sell'],
    )


def dispatch_detail(ve, solar_cf, wind_cf):
    """
    Hour-by-hour component breakdown: pv, wind, battery, grid, exported (MW).
    Uses LP dispatch arrays when ve.lp_detail() is cached (LP path); falls back
    to greedy replay on the optimised capacities otherwise.
    """
    lp = ve.lp_detail() if ve.prices is not None else None
    if lp is not None:
        return _dispatch_from_lp(lp, ve.demand.demand_mw)

    # greedy fallback
    c_solar, c_wind, batt_power, batt_energy = ve.solution
    floor     = ve.demand.floor_mw
    demand_mw = ve.demand.demand_mw
    grid_cap  = ve.demand.grid_cap_mw
    H         = ve.demand.HOURS

    pv_gen = c_solar * solar_cf
    wl_gen = c_wind  * wind_cf
    avail  = pv_gen + wl_gen

    batt_charge    = np.zeros(H)
    batt_discharge = np.zeros(H)
    exported       = np.zeros(H)
    soc = 0.0

    for t in range(H):
        a = avail[t]
        if a >= floor:
            surplus = a - floor
            charge  = min(surplus, batt_power, batt_energy - soc)
            soc    += charge
            batt_charge[t] = charge
            sell = min(max(0.0, a - charge - demand_mw), grid_cap)
            exported[t] = sell
        else:
            shortfall = floor - a
            discharge = min(shortfall, batt_power, soc)
            soc      -= discharge
            batt_discharge[t] = discharge

    ve_consumed = np.where(
        avail >= floor,
        np.minimum(avail - batt_charge, demand_mw),
        avail,
    )
    frac_pv     = np.where(avail > 1e-9, pv_gen / avail, 0.0)
    pv_demand   = ve_consumed * frac_pv
    wl_demand   = ve_consumed * (1.0 - frac_pv)
    grid_import = np.clip(demand_mw - ve_consumed - batt_discharge, 0.0, grid_cap)

    return dict(pv=pv_demand, wind=wl_demand, battery=batt_discharge,
                grid=grid_import, exported=exported)


def aggregate_dispatch(d, dates, freq):
    """Resample hourly dispatch dict to mean MW at pandas frequency ('D', 'W', 'ME')."""
    agg = pd.DataFrame(d, index=dates).resample(freq).mean()
    return {k: agg[k].values for k in d}, agg.index


def battery_detail(ve, solar_cf, wind_cf):
    """
    Hour-by-hour battery flows and end-of-hour state of charge.
    Returns dict: charge, discharge_dc, discharge_grid (MW), soc (MWh).

    When ve.prices is set, uses LP dispatch arrays directly (battery is
    bidirectional; discharge_grid = portion of discharge exported to grid).
    Otherwise replays greedy dispatch.
    """
    c_solar, c_wind, batt_power, batt_energy = ve.solution
    H = ve.demand.HOURS

    lp = ve.lp_detail() if ve.prices is not None else None

    if lp is not None:
        charge         = lp['charge']
        discharge      = lp['discharge']
        soc_arr        = lp['soc']
        grid_sell      = lp['grid_sell']
        # battery drives exports up to its own discharge; VE may also export
        discharge_grid = np.minimum(discharge, grid_sell)
        discharge_dc   = discharge - discharge_grid
        return dict(charge=charge, discharge_dc=discharge_dc,
                    discharge_grid=discharge_grid, soc=soc_arr)

    # greedy replay
    floor = ve.demand.floor_mw
    avail = c_solar * solar_cf + c_wind * wind_cf

    charge         = np.zeros(H)
    discharge_dc   = np.zeros(H)
    discharge_grid = np.zeros(H)
    soc_arr        = np.zeros(H)
    soc = 0.0

    for t in range(H):
        a = avail[t]
        if a >= floor:
            ch        = min(a - floor, batt_power, batt_energy - soc)
            soc      += ch
            charge[t] = ch
        else:
            dis             = min(floor - a, batt_power, soc)
            soc            -= dis
            discharge_dc[t] = dis
        soc_arr[t] = soc

    return dict(charge=charge, discharge_dc=discharge_dc,
                discharge_grid=discharge_grid, soc=soc_arr)


# ==================== ==================== ==================== ====================
# 2. plotting

def _align_zeros(ax1, ax2):
    """Expand lower limits so zero sits at the same fractional height on both axes."""
    lo1, hi1 = ax1.get_ylim()
    lo2, hi2 = ax2.get_ylim()
    f1 = (0 - lo1) / (hi1 - lo1) if hi1 != lo1 else 0.5
    f2 = (0 - lo2) / (hi2 - lo2) if hi2 != lo2 else 0.5
    f  = max(f1, f2)
    if 0 < f < 1:
        ax1.set_ylim(-f * hi1 / (1 - f), hi1)
        ax2.set_ylim(-f * hi2 / (1 - f), hi2)


def _month_ticks(idx):
    """Midpoint of each calendar month's first contiguous block, with Danish labels."""
    ticks, labels = [], []
    for m in range(1, 13):
        pos = np.where(idx.month == m)[0]
        if not len(pos):
            continue
        # drop year-boundary tail (weekly data: Jan appears at both ends of the array)
        gaps = np.where(np.diff(pos) > 1)[0]
        if len(gaps):
            pos = pos[:gaps[0] + 1]
        ticks.append((pos[0] + pos[-1]) // 2)
        labels.append(MAANED_DK[m - 1])
    return ticks, labels


def plot_dispatch(d_agg, idx, ylabel, save_path=None):
    """
    Stacked area dispatch plot for a VE scenario.
    Positive stacks: grid, battery, wind, pv. Negative fill: export.
    """
    x = np.arange(len(idx))
    C = DISPATCH_COLORS
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.stackplot(
        x,
        d_agg['grid'], d_agg['battery'], d_agg['wind'], d_agg['pv'],
        labels=['Netimport', 'Batteri', 'Vindkraft', 'Solkraft'],
        colors=[C['grid'], C['battery'], C['wind'], C['pv']],
        linewidth=0,
    )
    ax.fill_between(x, 0, -d_agg['exported'],
                    label='Eksport', color=C['exported'], linewidth=0)

    ax.set_xlim(0, len(idx) - 1)
    ax.set_ylabel(ylabel)

    ticks, labs = _month_ticks(idx)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(labels))

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return fig, ax


def plot_battery(b, idx, save_path=None):
    """
    Single-figure battery plot with twin y-axes.
    Left: state of charge (MWh, blue fill). Right: charge/discharge (MW, lines).
    """
    C = DISPATCH_COLORS
    x = np.arange(len(idx))

    fig, ax_soc = plt.subplots(figsize=(12, 6))
    ax_flow = ax_soc.twinx()

    ax_soc.fill_between(x, 0, b['soc'] / 1e3,
                        color=C['wind'], alpha=0.35, linewidth=0, label='SOC')
    ax_soc.set_ylabel('GWh stored')
    ax_soc.set_xlim(0, len(idx) - 1)

    ax_flow.plot(x, b['charge'],         color='#E8A09E', linewidth=1.5, label='Charge')
    ax_flow.plot(x, b['discharge_dc'],   color='#C0504D', linewidth=1.5, label='Discharge — datacenter')
    ax_flow.plot(x, b['discharge_grid'], color='#1A1A1A', linewidth=1.5, label='Discharge — grid')
    ax_flow.set_ylabel('MW')

    ticks, labs = _month_ticks(idx)
    ax_soc.set_xticks(ticks)
    ax_soc.set_xticklabels(labs)

    h1, l1 = ax_soc.get_legend_handles_labels()
    h2, l2 = ax_flow.get_legend_handles_labels()
    ax_soc.legend(h1 + h2[::-1], l1 + l2[::-1],
                  loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=len(h1) + len(h2))

    _align_zeros(ax_soc, ax_flow)
    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return fig, (ax_soc, ax_flow)
