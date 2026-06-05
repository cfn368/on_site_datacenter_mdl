"""
setup — Notebook preamble: imports, autoreload, figure style, pylib re-exports
===============================================================================

Call `from pylib.setup import *` then `setup_notebook()` at the top of every
notebook. Standard names (np, pd, plt, pathlib, time) land in the caller's
namespace; autoreload is enabled; pylib modules are (re)loaded fresh.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

import pylib.ve_dispatch
from pylib.ve_dispatch import *   # noqa: F401, F403

# Dark shade of the figure facecolor (#EAF1F2, HSL≈188° 23% 93%)
# — same hue, much lower lightness; used as the global text colour.
TEXT_COLOR = "#3F6469"


def enable_autoreload(mode: int = 2) -> None:
    try:
        from IPython import get_ipython
    except Exception:
        return
    ip = get_ipython()
    if ip is None:
        return
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", str(int(mode)))


def set_aej(**kwargs) -> None:
    mpl.rcParams.update({
        "font.family":          "serif",
        "font.style":           "italic",
        "font.size":            15,
        "figure.dpi":           150,
        "figure.facecolor":     "#EAF1F2",
        "axes.facecolor":       "#EAF1F2",
        "axes.linewidth":       1.0,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     False,
        "axes.spines.bottom":   False,
        "text.color":           TEXT_COLOR,
        "axes.labelcolor":      TEXT_COLOR,
        "xtick.color":          TEXT_COLOR,
        "ytick.color":          TEXT_COLOR,
        "lines.linewidth":      1.2,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "legend.frameon":       False,
        "legend.fancybox":      False,
        "legend.borderaxespad": 0.4,
        "legend.handlelength":  2.0,
        "legend.handletextpad": 0.6,
        "legend.labelspacing":  0.35,
        "savefig.bbox":         "tight",
        "savefig.dpi":          300,
        **kwargs,
    })


def setup_notebook(*, autoreload: int = 2, aej: bool = True, **aej_kwargs) -> None:
    """Enable autoreload, set figure style, reload pylib modules, inject names into caller."""
    import inspect

    enable_autoreload(autoreload)

    for name in ("pylib.ve_dispatch",):
        if name in sys.modules:
            importlib.reload(sys.modules[name])

    caller_globals = inspect.stack()[1][0].f_globals
    for mod in (pylib.ve_dispatch,):
        caller_globals.update(
            {k: getattr(mod, k) for k in vars(mod) if not k.startswith("_")}
        )
    caller_globals.update({
        'np': np, 'pd': pd, 'plt': plt,
        'mpl': mpl, 'mticker': mticker, 'mdates': mdates,
        'pathlib': pathlib, 'time': time, 'sys': sys,
        'TEXT_COLOR': TEXT_COLOR,
        'fig_title': fig_title,
    })

    if aej:
        set_aej(**aej_kwargs)
