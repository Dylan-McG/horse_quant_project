# src/hqp/plots/ages.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

_DPI: int = 120
_FIGSIZE: Tuple[int, int] = (8, 4)


def _rc() -> dict[str, object]:
    return {
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }


def _ensure_parent(out_path: Optional[Path]) -> None:
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_age_histogram(df: pd.DataFrame, age_col: str, out_path: Optional[Path] = None) -> Figure:
    """
    Plot an integer-binned age histogram (1-year bins from min to max).

    Returns a matplotlib Figure. If out_path is provided, also saves the figure.
    """
    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if ages.empty:
        with plt.rc_context(_rc()):
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.set_axis_off()
            ax.set_title(f"No valid ages in column '{age_col}'")
    else:
        amin = int(np.floor(ages.min()))
        amax = int(np.ceil(ages.max()))
        # Build float edges, then convert to a concrete List[float] for typing compatibility
        bins_edges = np.arange(amin, amax + 2, 1, dtype=float)
        bins_list = [float(x) for x in bins_edges]

        with plt.rc_context(_rc()):
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.hist(ages.to_numpy(), bins=bins_list, align="left", rwidth=0.9)
            ax.set_xlabel(age_col)
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution (1-year bins)")
            ax.set_xticks(np.arange(amin, amax + 1, 1))
            ax.set_xlim(amin - 0.5, amax + 0.5)

    fig.tight_layout()
    if out_path is not None:
        _ensure_parent(out_path)
        fig.savefig(str(out_path))
    return fig
