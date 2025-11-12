# # tests/test_plots_ages.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from hqp.plots.ages import plot_age_histogram


def test_plot_age_histogram_integer_bins_and_labels() -> None:
    ages = np.array([2, 3, 3, 4, 5, 5, 5, 7, 8, 10])
    df = pd.DataFrame({"age": ages})
    with TemporaryDirectory() as td:
        out = Path(td) / "age_hist.png"
        fig = plot_age_histogram(df, "age", out_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()

        # Expect one bar per integer from min..max
        amin, amax = int(ages.min()), int(ages.max())
        expected_bins = amax - amin + 1

        ax = fig.axes[0]
        # Count only Rectangle bars (hist patches are Rectangles)
        bars_count = sum(1 for p in ax.patches if isinstance(p, Rectangle) and p.get_height() >= 0)
        assert bars_count == expected_bins

        # Check x-ticks cover the integer range
        xticks = [int(t) for t in ax.get_xticks() if amin <= t <= amax]
        assert (amin in xticks) and (amax in xticks)
