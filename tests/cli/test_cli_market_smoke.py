# tests/cli/test_cli_market_smoke.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hqp.cli import app

runner = CliRunner()


def _gen_uniform_odds(base_p: Path, out_p: Path) -> None:
    df = pd.read_parquet(base_p)[["race_id", "horse_id"]].drop_duplicates()
    n = df.groupby("race_id")["horse_id"].transform("count").astype(float)
    p = 1.0 / n
    odds = (1.0 / p).round(2)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.assign(decimal_odds=odds).to_parquet(out_p, index=False)


def test_cli_market_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    # minimal setup
    assert runner.invoke(app, ["ingest", "--dry-run"]).exit_code == 0
    assert runner.invoke(app, ["ratings"]).exit_code == 0
    assert runner.invoke(app, ["features"]).exit_code == 0

    # toy odds
    _gen_uniform_odds(Path("data/interim/base.parquet"), Path("data/market/odds.parquet"))

    # run market
    res = runner.invoke(
        app,
        [
            "market",
            "--features",
            "data/features/features.parquet",
            "--odds-path",
            "data/market/odds.parquet",
            "--out",
            "data/market/market_join.parquet",
        ],
    )
    assert res.exit_code == 0
    out_p = Path("data/market/market_join.parquet")
    assert out_p.exists()

    dfm = pd.read_parquet(out_p)
    assert "mkt_implied" in dfm.columns, "missing implied probability column"
    # per-race sums approximately 1 (ignore NaNs)
    sums = dfm.groupby("race_id", sort=False)["mkt_implied"].sum().to_numpy(dtype=float)
    assert np.allclose(sums, 1.0, atol=1e-9)
