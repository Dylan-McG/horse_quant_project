# src/hqp/eval/market.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Market Snapshot Evaluation
#
# Purpose
# -------
# Provide quick sanity checks on a market parquet by reporting:
#   - counts of rows/races and availability of odds/implied probabilities
#   - overround summary (mean, p10, p90)
#   - races with zero observed odds entries
#
# Output
# ------
# A timestamped reports/market/<TS>/ directory with:
#   - summary.json
#   - sample.parquet (up to 1000 rows of selected columns for inspection)
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
import json
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class MarketSummary:
    rows: int
    races: int
    have_decimal: int
    have_implied: int
    overround_mean: float
    overround_p10: float
    overround_p90: float
    zero_odds_races: int
    ts: str


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def evaluate(market_path: Path) -> Path:
    """
    Read a market parquet and emit a compact health report.
    Requires columns: race_id, horse_id, mkt_odds_decimal, mkt_implied.
    """
    df = pd.read_parquet(market_path)
    req = ["race_id", "horse_id", "mkt_odds_decimal", "mkt_implied"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"evaluate-market: missing cols {missing}")

    rows = len(df)
    races = df["race_id"].nunique()

    have_decimal = int(df["mkt_odds_decimal"].notna().sum())
    have_implied = int(df["mkt_implied"].notna().sum())

    grp = df.groupby("race_id", sort=False, observed=True)
    sums = grp["mkt_implied"].sum(min_count=1)
    overround = sums  # already normalized to 1.0 if perfect; >1 implies overround
    overround_mean = float(np.nanmean(overround))
    overround_p10 = (
        float(np.nanpercentile(overround.dropna(), 10)) if overround.notna().any() else float("nan")
    )
    overround_p90 = (
        float(np.nanpercentile(overround.dropna(), 90)) if overround.notna().any() else float("nan")
    )
    zero_odds_races = int((grp["mkt_odds_decimal"].count() == 0).sum())

    out_dir = Path(f"reports/market/{_ts()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = MarketSummary(
        rows=rows,
        races=races,
        have_decimal=have_decimal,
        have_implied=have_implied,
        overround_mean=overround_mean,
        overround_p10=overround_p10,
        overround_p90=overround_p90,
        zero_odds_races=zero_odds_races,
        ts=_ts(),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    # Also save a small sample for quick inspection
    sample_cols = [
        c for c in ["race_id", "horse_id", "mkt_odds_decimal", "mkt_implied"] if c in df.columns
    ]
    df.sample(min(1000, len(df)), random_state=1)[sample_cols].to_parquet(
        out_dir / "sample.parquet", index=False
    )

    # Friendly console output when called directly via CLI wrapper
    print(
        f"[evaluate-market] rows={rows} races={races} overround(mean)={overround_mean:.3f} -> {out_dir}"
    )
    return out_dir
