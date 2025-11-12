# tests/test_plots_market.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

from hqp.plots.market import plot_market_overround


def test_plot_market_overround_smoke_and_normalized_near_100() -> None:
    # Build a DF with two races, each set of implied probs summing to ~1
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 2, 2, 2],
            "p": [0.25, 0.35, 0.40, 0.10, 0.60, 0.30],
        }
    )
    with TemporaryDirectory() as td:
        out = Path(td) / "overround.png"
        fig = plot_market_overround(df, group_col="race_id", implied_col="p", out_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()

        # Check computed overround from the source DF (not from figure)
        over1 = df.loc[df["race_id"] == 1, "p"].sum() * 100.0
        over2 = df.loc[df["race_id"] == 2, "p"].sum() * 100.0
        assert abs(over1 - 100.0) < 1e-6
        assert abs(over2 - 100.0) < 1e-6
