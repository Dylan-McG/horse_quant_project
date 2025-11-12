# src/hqp/plots/market.py
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

_DPI: int = 120
_FIGSIZE: tuple[float, float] = (9.0, 4.5)


def _rc() -> dict[str, object]:
    return {
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }


def _ensure_parent(out_path: Path | None) -> None:
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_market_overround(
    df: pd.DataFrame, group_col: str, implied_col: str, out_path: Path | None = None
) -> Figure:
    if group_col not in df.columns or implied_col not in df.columns:
        with plt.rc_context(_rc()):
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.set_axis_off()
            ax.set_title("Required columns not present")
            return fig

    grp = df.groupby(group_col, dropna=False)[implied_col].sum().sort_index()
    overround = grp * 100.0

    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        idx = overround.index.astype("string").to_list()
        ax.bar(idx, overround.to_list())
        ax.axhline(100.0)
        ax.set_ylabel("Overround (%)")
        ax.set_xlabel(group_col)
        ax.set_title(f"Overround by {group_col}")
        ax.tick_params(axis="x", rotation=45)

        if overround.size > 0:
            mu = float(overround.mean())
            ax.text(0.01, 0.98, f"mean={mu:.2f}%", transform=ax.transAxes, ha="left", va="top")

    fig.tight_layout()
    if out_path is not None:
        _ensure_parent(out_path)
        fig.savefig(str(out_path))
    return fig
