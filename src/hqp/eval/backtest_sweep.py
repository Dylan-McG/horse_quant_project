# src/hqp/eval/backtest_sweep.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Backtest Grid Sweep
#
# Purpose
# -------
# Sweep a grid over (edge_threshold × max_odds) and collect the resulting
# backtest metrics in a single parquet. Optionally write a quick ROI plot
# (best-effort; plotting errors are ignored).
#
# Interface
# ---------
# sweep(edges_path, edge_thresholds, max_odds_grid, ..., out_root="reports/backtest")
#   -> Path to reports/backtest/sweep_<YYYYmmdd_HHMMSS>.parquet
#
# Notes
# -----
# - We import the actual backtest runner lazily (`hqp.eval.backtest.run`) to
#   avoid import-time dependencies when this module is imported for docs/tests.
# - The function is defensive: missing summary.json fields are treated as None.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json
import math
import pandas as pd


@dataclass
class SweepPoint:
    """
    A single parameter point in the sweep grid.
    """

    edge_threshold: float
    max_odds: float
    per_race_max_bets: int
    stake: float
    kelly_fraction: float
    try_join_market: bool


@dataclass
class SweepResult:
    """
    Recorded result for a single sweep point, including the written report_dir.
    All numeric fields are Optional and may be None if the backtest did not
    produce a summary field for any reason.
    """

    edge_threshold: float
    max_odds: float
    per_race_max_bets: int
    stake: float
    kelly_fraction: float
    try_join_market: bool
    bets: Optional[int]
    roi: Optional[float]
    hit_rate: Optional[float]
    avg_odds: Optional[float]
    pnl: Optional[float]
    report_dir: Optional[str]


def _now_ts() -> str:
    """Compact timestamp for artifact naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_backtest_summary(report_dir: Path) -> Dict[str, Any]:
    """
    Read summary.json that the backtest writes.
    Return {} if missing/unreadable (we'll still record the report_dir).
    """
    s_p = report_dir / "summary.json"
    if not s_p.exists():
        return {}
    try:
        with s_p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_one(edges_path: Path, cfg: Dict[str, Any]) -> SweepResult:
    """
    Invoke the real backtest function used by the CLI (local import).
    """
    from hqp.eval.backtest import (
        run as _bt_run,
    )  # local import to avoid hard dependency at import time

    out_dir: Path = _bt_run(edges_path, cfg)  # should create summary.json
    s = _read_backtest_summary(out_dir)

    # Extract with robust defaults (do not raise if keys are missing)
    bets = int(s.get("bets", 0)) if "bets" in s else None
    roi = float(s.get("roi", 0.0)) if "roi" in s else None
    hit_rate = float(s.get("hit_rate", 0.0)) if "hit_rate" in s else None
    avg_odds = float(s.get("avg_odds", 0.0)) if "avg_odds" in s else None
    pnl = float(s.get("pnl", 0.0)) if "pnl" in s else None

    return SweepResult(
        edge_threshold=float(cfg.get("edge_threshold", math.nan)),
        max_odds=float(cfg.get("max_odds", math.nan)),
        per_race_max_bets=int(cfg.get("per_race_max_bets", 1)),
        stake=float(cfg.get("stake", 1.0)),
        kelly_fraction=float(cfg.get("kelly_fraction", 0.0)),
        try_join_market=bool(cfg.get("try_join_market", True)),
        bets=bets,
        roi=roi,
        hit_rate=hit_rate,
        avg_odds=avg_odds,
        pnl=pnl,
        report_dir=str(out_dir).replace("\\", "/"),
    )


def sweep(
    edges_path: str | Path,
    edge_thresholds: Sequence[float],
    max_odds_grid: Sequence[float],
    *,
    per_race_max_bets: int = 1,
    stake: float = 1.0,
    kelly_fraction: float = 0.0,
    try_join_market: bool = True,
    out_root: str | Path = "reports/backtest",
    make_plot: bool = True,
) -> Path:
    """
    Run a grid of backtests over (edge_threshold x max_odds), collect metrics, save parquet
    and (optionally) a simple ROI plot.

    Returns
    -------
    Path
        Path to the written parquet: reports/backtest/sweep_<ts>.parquet
    """
    edges_p = Path(edges_path)
    if not edges_p.exists():
        raise FileNotFoundError(f"Edges parquet not found at: {edges_p}")

    ts = _now_ts()
    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    out_parquet = outdir / f"sweep_{ts}.parquet"

    results: List[SweepResult] = []
    for et in edge_thresholds:
        for mo in max_odds_grid:
            cfg: Dict[str, Any] = {
                "edge_threshold": float(et),
                "max_odds": float(mo),
                "per_race_max_bets": int(per_race_max_bets),
                "stake": float(stake),
                "kelly_fraction": float(kelly_fraction),
                "try_join_market": bool(try_join_market),
                # Keep extra knobs extensible if your backtest uses them:
                # "min_samples_per_bucket": 1,
                # "allow_ties": False,
            }
            r = _run_one(edges_p, cfg)
            results.append(r)

    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_parquet(out_parquet, index=False)

    # Optional: basic ROI plot (ROI vs edge_threshold, one line per max_odds)
    if make_plot:
        try:
            import matplotlib.pyplot as plt  # local import; optional

            plot_path = outdir / f"sweep_{ts}.png"
            # Pivot to shape [edge_threshold x max_odds] -> ROI (mean if duplicates)
            piv = df.pivot_table(
                index="edge_threshold",
                columns="max_odds",
                values="roi",
                aggfunc="mean",
            ).sort_index()
            plt.figure()
            for mo in sorted(piv.columns):
                plt.plot(piv.index, piv[mo], label=f"max_odds={mo}")
            plt.xlabel("edge_threshold")
            plt.ylabel("ROI")
            plt.title("ROI vs edge_threshold by max_odds")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
        except Exception:
            # Plotting is a convenience; ignore any errors silently.
            pass

    return out_parquet
