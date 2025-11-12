# tests/test_market_attach.py

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hqp.cli import app
from hqp.market.ingest import attach_market


def _toy_features() -> pd.DataFrame:
    rows = [
        # race A
        {"race_id": "A", "horse_id": "H1"},
        {"race_id": "A", "horse_id": "H2"},
        {"race_id": "A", "horse_id": "H3"},
        # race B
        {"race_id": "B", "horse_id": "J1"},
        {"race_id": "B", "horse_id": "J2"},
    ]
    return pd.DataFrame(rows)


def _toy_odds_decimal() -> pd.DataFrame:
    # many-to-one mapping (unique keys)
    rows = [
        {"race_id": "A", "horse_id": "H1", "decimal_odds": "2.5"},
        {"race_id": "A", "horse_id": "H2", "decimal_odds": "5.0"},
        {"race_id": "A", "horse_id": "H3", "decimal_odds": "10"},
        {"race_id": "B", "horse_id": "J1", "decimal_odds": "EVS"},
        {"race_id": "B", "horse_id": "J2", "decimal_odds": "3.0"},
    ]
    return pd.DataFrame(rows)


def _toy_odds_duplicates() -> pd.DataFrame:
    # duplicate odds rows for ("A","H1") -> should raise
    rows = [
        {"race_id": "A", "horse_id": "H1", "decimal_odds": "2.5"},
        {"race_id": "A", "horse_id": "H1", "decimal_odds": "2.6"},
    ]
    return pd.DataFrame(rows)


def test_attach_market_happy_path() -> None:
    feats = _toy_features()
    odds = _toy_odds_decimal()

    out = attach_market(
        feats,
        odds,
        race_key="race_id",
        horse_key="horse_id",
        decimal_col="decimal_odds",
        fractional_col=None,
        prefix="mkt",
    )

    # Row count preserved
    assert len(out) == len(feats)

    # Columns exist and have float dtype
    for c in ["mkt_odds_decimal", "mkt_implied_raw", "mkt_implied"]:
        assert c in out.columns
        assert pd.api.types.is_float_dtype(out[c])

    # Per-race probabilities sum ~ 1
    for _, grp in out.groupby("race_id", sort=False):
        p = grp["mkt_implied"].to_numpy(dtype=float)
        m = np.isfinite(p)
        if m.any():
            s = float(p[m].sum())
            assert np.isfinite(s) and abs(s - 1.0) <= 1e-6


def test_attach_market_duplicate_rows_raise() -> None:
    feats = _toy_features()
    odds = _toy_odds_duplicates()
    with pytest.raises(ValueError):
        attach_market(
            feats,
            odds,
            race_key="race_id",
            horse_key="horse_id",
            decimal_col="decimal_odds",
            fractional_col=None,
            prefix="mkt",
        )


def _finite_sum(s: pd.Series) -> float:
    arr = s.to_numpy(dtype=float)
    mask = np.isfinite(arr)
    return float(arr[mask].sum()) if mask.any() else float("nan")


def test_cli_market_smoke(tmp_path: Path) -> None:
    # write toy features and odds
    feats = _toy_features()
    odds = _toy_odds_decimal()
    fpath = tmp_path / "features.parquet"
    opath = tmp_path / "odds.parquet"
    outp = tmp_path / "out.parquet"
    feats.to_parquet(fpath, index=False)
    odds.to_parquet(opath, index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "market",
            "--features",
            str(fpath),
            "--odds-path",
            str(opath),
            "--out",
            str(outp),
            "--decimal-col",
            "decimal_odds",
        ],
    )
    assert res.exit_code == 0, res.output

    joined = pd.read_parquet(outp)
    for c in ["mkt_odds_decimal", "mkt_implied_raw", "mkt_implied"]:
        assert c in joined.columns
        assert pd.api.types.is_float_dtype(joined[c])

    sums = joined.groupby("race_id")["mkt_implied"].apply(_finite_sum).dropna()
    assert not sums.empty
    assert ((sums - 1.0).abs() <= 1e-6).all()
