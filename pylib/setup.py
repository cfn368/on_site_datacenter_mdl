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

import pylib.ve_dispatch
from pylib.ve_dispatch import *   # noqa: F401, F403


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


def set_style() -> None:
    mpl.rcParams.update({
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'font.size':         10,
        'axes.labelsize':    10,
        'legend.fontsize':   9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


def setup_notebook(*, autoreload: int = 2) -> None:
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
        'mpl': mpl, 'mticker': mticker,
        'pathlib': pathlib, 'time': time, 'sys': sys,
    })

    set_style()
