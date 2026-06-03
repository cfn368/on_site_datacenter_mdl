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
]


# ==================== ==================== ==================== ====================
# 0. constants

MAANED_DK = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun',
             'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']

DISPATCH_COLORS = {
    'grid':     '#CCCCCC',
    'battery':  '#DE7626',   # orange
    'wind':     '#38BDF8',   # sky blue
    'pv':       '#F5C518',   # golden
    'exported': '#34BA5B',   # green
}


# ==================== ==================== ==================== ====================
# 1. dispatch

def dispatch_detail(ve, solar_cf, wind_cf):
    """
    Hour-by-hour component breakdown: pv, wind, battery, grid, exported (MW).
    Replays VESupply greedy dispatch to decompose how demand is met each hour.
    """
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


# ==================== ==================== ==================== ====================
# 2. plotting

def _month_ticks(idx):
    """First position of each calendar month in idx, with Danish labels."""
    ticks, labels = [], []
    for m in range(1, 13):
        pos = np.where(idx.month == m)[0]
        if len(pos):
            ticks.append(pos[0])
            labels.append(MAANED_DK[m - 1])
    return ticks, labels


def plot_dispatch(d_agg, idx, ylabel, save_path=None):
    """
    Stacked area dispatch plot for a VE scenario.
    Positive stacks: grid, battery, wind, pv. Negative fill: export.
    """
    x = np.arange(len(idx))
    C = DISPATCH_COLORS
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.stackplot(
        x,
        d_agg['grid'], d_agg['battery'], d_agg['wind'], d_agg['pv'],
        labels=['Netimport', 'Batteri', 'Vindkraft', 'Solkraft'],
        colors=[C['grid'], C['battery'], C['wind'], C['pv']],
        linewidth=0,
    )
    ax.fill_between(x, 0, -d_agg['exported'],
                    label='Eksport', color=C['exported'], linewidth=0)

    ax.axhline(0, color='0.2', lw=0.8, ls='--')
    ax.set_xlim(0, len(idx) - 1)
    ax.set_ylabel(ylabel)
    ax.grid(linewidth=0.6, alpha=0.35)

    ticks, labs = _month_ticks(idx)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left', frameon=True)

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax
