# src/hqp/plots/eda.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

_DPI: int = 120
_FIGSIZE_SINGLE: tuple[int, int] = (7, 4)


def _rc() -> dict[str, object]:
    return {
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }


def _nrows_ncols(n: int) -> tuple[int, int]:
    if n <= 0:
        return (1, 1)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    return (nrows, ncols)


def _ensure_parent(out_path: Path | None) -> None:
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def plot_feature_distributions(
    df: pd.DataFrame, features: Sequence[str], out_path: Path | None = None
) -> Figure:
    feats: list[str] = [f for f in features if f in df.columns]
    if not feats:
        with plt.rc_context(_rc()):
            fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
            ax.set_axis_off()
            ax.set_title("No valid features provided")
    else:
        nrows, ncols = _nrows_ncols(len(feats))
        with plt.rc_context(_rc()):
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 3.5 * nrows))
        if isinstance(axes, Axes):
            axes_arr: list[Axes] = [axes]
        else:
            axes_arr = [ax for row in np.atleast_2d(axes) for ax in row]

        for ax in axes_arr[len(feats) :]:
            ax.set_visible(False)

        for ax, col in zip(axes_arr, feats, strict=False):
            s = df[col]
            if _is_numeric(s):
                vals = s.dropna().to_numpy()
                ax.hist(vals, bins=40)
                ax.set_ylabel("Frequency")
            else:
                vc = s.astype("string").fillna("<NA>").value_counts().head(20)
                ax.bar(vc.index.to_list(), vc.to_list())
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
            ax.set_title(col)
            ax.set_xlabel(col)

    fig.tight_layout()
    if out_path is not None:
        _ensure_parent(out_path)
        fig.savefig(str(out_path))
    return fig
