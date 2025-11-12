# tests/test_plots_eda.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

from hqp.plots.eda import plot_feature_distributions


def test_plot_feature_distributions_smoke() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 300),
            "num2": rng.exponential(1.0, 300),
            "cat": rng.choice(["A", "B", "C"], size=300),
        }
    )
    with TemporaryDirectory() as td:
        out = Path(td) / "eda_feats.png"
        fig = plot_feature_distributions(df, ["num1", "num2", "cat", "missing"], out_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()
