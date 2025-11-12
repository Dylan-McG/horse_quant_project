# tests/test_plots_calibration.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

from hqp.plots.calibration import plot_calibration_curve


def test_plot_calibration_curve_smoke_and_bins_partition() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000).astype(float)
    y_prob = rng.uniform(0.0, 1.0, size=1000)
    n_bins = 12

    with TemporaryDirectory() as td:
        out = Path(td) / "calib.png"
        fig = plot_calibration_curve(y_true, y_prob, n_bins=n_bins, out_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()

        # Diagonal line present
        ax = fig.axes[0]
        lines = ax.get_lines()
        diagonal_present = any(
            np.allclose(line.get_xdata(), [0.0, 1.0]) and np.allclose(line.get_ydata(), [0.0, 1.0])
            for line in lines
        )
        assert diagonal_present

        # Bins cover [0,1] and counts sum to N
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        inds = np.digitize(np.clip(y_prob, 0, 1), edges, right=True) - 1
        inds = np.clip(inds, 0, n_bins - 1)
        counts = np.bincount(inds, minlength=n_bins)
        assert int(counts.sum()) == len(y_prob)
