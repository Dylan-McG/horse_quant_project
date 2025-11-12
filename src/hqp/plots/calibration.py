# src/hqp/plots/calibration.py
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

_DPI: int = 120


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


def _bin_stats(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, edges, right=True) - 1
    inds = np.clip(inds, 0, n_bins - 1)

    sums_pred = np.zeros(n_bins, dtype=float)
    sums_true = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i, p, t in zip(inds, y_prob, y_true, strict=False):
        sums_pred[i] += float(p)
        sums_true[i] += float(t)
        counts[i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_pred = np.where(counts > 0, sums_pred / counts, np.nan)
        avg_true = np.where(counts > 0, sums_true / counts, np.nan)

    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, avg_pred, avg_true


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, out_path: Path | None = None
) -> Figure:
    y_true_c = np.asarray(y_true, dtype=float).copy()
    y_prob_c = np.clip(np.asarray(y_prob, dtype=float).copy(), 0.0, 1.0)

    _centers, avg_pred, avg_true = _bin_stats(y_true_c, y_prob_c, n_bins)

    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.plot(avg_pred, avg_true, marker="o")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability Curve (bins={n_bins})")

        for xp, yt in zip(avg_pred, avg_true, strict=False):
            if not (np.isnan(xp) or np.isnan(yt)):
                ax.vlines(xp, ymin=min(xp, yt), ymax=max(xp, yt))

    fig.tight_layout()
    if out_path is not None:
        _ensure_parent(out_path)
        fig.savefig(str(out_path))
    return fig
