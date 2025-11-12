# src/hqp/eval/backtest.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Simple Backtest Engine
#
# Purpose
# -------
# Consume an edges parquet and simulate a staking strategy to produce:
#   - bets placed (filtered by edge threshold, odds cap, per-race cap)
#   - PnL / ROI / hit rate / average odds
#   - a timestamped report directory with:
#       * summary.json  (scalar metrics + counts)
#       * bets.parquet  (selected bets with stakes and outcomes)
#
# Inputs (required columns in edges parquet)
# -----------------------------------------
#   race_id, horse_id, edge, model_prob, mkt_implied, mkt_odds_decimal
#
# Labels
# ------
# If a 0/1 winner column is already in the edges parquet under any of:
#   ['obs__is_winner', 'won', 'is_win', 'position', 'obs__Place']
# we coerce it into a 'won' column (0/1).
# Otherwise, if `BTConfig.try_join_market=True`, we attempt to attach outcomes
# from a sibling market parquet:
#   <edges_dir>/market_join.parquet           (default)
#   or a custom path supplied via BTConfig.market_path
#
# Staking
# -------
# Flat stake by default (stake=1.0). If kelly_fraction > 0, we use fractional Kelly:
#   q = model probability (clipped to [1e-6, 1-1e-6])
#   b = odds - 1
#   Kelly(q, b) = (q * b - (1 - q)) / b, floored at 0
#   stake = kelly_fraction * Kelly(q, b)
#
# Console Output
# --------------
# Prints a single summary line formatted for parsers:
#   [backtest] bets=<int> roi=<float> pnl=<float> -> reports/backtest/<TS>
# (The compare tool keys off "bets=... roi=... pnl=..." and the trailing path.)
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List  # removed unused Set, added cast

import json
import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------- Config ----------------------------------
@dataclass
class BTConfig:
    edge_threshold: float = 0.02
    max_odds: float = 50.0
    stake: float = 1.0
    per_race_max_bets: int = 3
    kelly_fraction: float = 0.0  # 0 => flat stake
    try_join_market: bool = True
    market_path: Optional[str] = None

    # Bankroll-mode knobs (off by default unless bankroll_init is set)
    bankroll_init: Optional[float] = None  # e.g. 1000.0 to enable bankroll mode
    stake_min: float = 0.0  # floor per bet in bankroll mode
    stake_max: Optional[float] = None  # cap per bet in bankroll mode


# --------------------------------- Result ----------------------------------
@dataclass
class BTResult:
    rows: int
    bets: int
    races_bet: int
    pnl: float
    roi: float
    hit_rate: float
    avg_odds: float
    ts: str


_LABEL_CANDIDATES: List[str] = [
    "obs__is_winner",
    "won",
    "is_win",
    "position",
    "obs__Place",
]


def _find_label_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in _LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _coerce_label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    s = df[label_col]

    if label_col in ("obs__is_winner", "won", "is_win"):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)

    if label_col == "position":
        pos = pd.to_numeric(s, errors="coerce")
        return (pos == 1).astype(int)

    if label_col == "obs__Place":
        if pd.api.types.is_string_dtype(s):
            return s.fillna("").str.strip().str.lower().eq("win").astype(int)
        num = pd.to_numeric(s, errors="coerce")
        return (num == 1).astype(int)

    return pd.Series(0, index=df.index, dtype=int)


def _ensure_labels(edges: pd.DataFrame, edges_path: Path, cfg: BTConfig) -> pd.DataFrame:
    lbl = _find_label_col(edges)
    if lbl:
        edges = edges.copy()
        edges["won"] = _coerce_label_series(edges, lbl)
        return edges

    if not cfg.try_join_market:
        raise ValueError(
            "No outcome column in edges and try_join_market=False. "
            f"Expected one of {_LABEL_CANDIDATES}."
        )

    mkt_path = (
        Path(cfg.market_path) if cfg.market_path else edges_path.with_name("market_join.parquet")
    )
    if not mkt_path.exists():
        raise FileNotFoundError(f"Edges has no labels and market parquet not found at {mkt_path}.")

    want_cols: List[str] = ["race_id", "horse_id"] + _LABEL_CANDIDATES
    sel_cols: List[str] = []

    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.ParquetFile(str(mkt_path)).schema  # type: ignore[arg-type]
        names_list: List[str] = [str(x) for x in getattr(schema, "names", [])]  # type: ignore[attr-defined]
        sel_cols = [c for c in want_cols if c in set(names_list)]
    except Exception:
        tmp = pd.read_parquet(mkt_path)
        sel_cols = [c for c in want_cols if c in list(tmp.columns)]
        del tmp

    if not (
        {"race_id", "horse_id"}.issubset(sel_cols) and any(c in sel_cols for c in _LABEL_CANDIDATES)
    ):
        raise ValueError(
            f"Market file {mkt_path} missing keys/labels. Need race_id,horse_id and one of {_LABEL_CANDIDATES}."
        )

    mkt = pd.read_parquet(mkt_path, columns=sel_cols)
    merged = edges.merge(mkt, on=["race_id", "horse_id"], how="left", copy=False)

    lbl2 = _find_label_col(merged)
    if not lbl2:
        raise ValueError("Could not attach an outcome column from market parquet.")

    merged = merged.copy()
    merged["won"] = _coerce_label_series(merged, lbl2)
    return merged


# ---------------------------------- Main -----------------------------------
def run(edges_path: Path, cfg: Optional[Dict[str, Any]]) -> Path:
    edges_path = Path(edges_path)
    df = pd.read_parquet(edges_path)
    need = ["race_id", "horse_id", "edge", "model_prob", "mkt_implied", "mkt_odds_decimal"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[backtest] edges missing columns {miss}")

    cfg = cfg or {}
    valid_keys = set(BTConfig.__annotations__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    c = BTConfig(**filtered)

    df = _ensure_labels(df, edges_path, c)

    work = df.copy()
    work["odds"] = pd.to_numeric(work["mkt_odds_decimal"], errors="coerce").astype(float)
    work = work[
        (pd.to_numeric(work["edge"], errors="coerce") >= float(c.edge_threshold))
        & (work["odds"] > 1.0)
        & (work["odds"] <= float(c.max_odds))
    ]

    work = work.sort_values(["race_id", "edge"], ascending=[True, False], kind="mergesort")
    work["rank_in_race"] = work.groupby("race_id", sort=False).cumcount()
    work = work[work["rank_in_race"] < int(c.per_race_max_bets)].reset_index(drop=True)

    # Core quantities for staking
    q = pd.to_numeric(work["model_prob"], errors="coerce").clip(1e-6, 1 - 1e-6)
    b = work["odds"] - 1.0
    k_frac = ((q * b - (1 - q)) / b).clip(lower=0.0)  # Kelly fraction of bankroll for each bet
    hit = pd.to_numeric(work["won"], errors="coerce").fillna(0).astype(int)

    # --- Legacy linear mode (ROI invariant to scale) ---
    if not c.kelly_fraction or c.kelly_fraction <= 0 or c.bankroll_init is None:
        if c.kelly_fraction and c.kelly_fraction > 0:
            work["stake"] = (k_frac * float(c.kelly_fraction)).astype(float)
        else:
            work["stake"] = float(c.stake)

        ret = hit * (work["odds"] - 1.0) * work["stake"] - (1 - hit) * work["stake"]
        pnl = float(ret.sum())
        total_stake = float(work["stake"].sum())
        bets = int((work["stake"] > 0).sum())
        races_bet = int(work["race_id"].nunique())
        roi = float(pnl / total_stake) if bets > 0 and total_stake > 0 else 0.0
        hit_rate = float(hit.mean()) if len(hit) else 0.0
        avg_odds = float(work["odds"].mean()) if bets else 0.0

    # --- Bankroll mode (path-dependent; ROI varies with fraction) ---
    else:
        # Make the Optional[float] non-optional for the type-checker
        assert c.bankroll_init is not None
        init = float(c.bankroll_init)
        bankroll = init
        stakes: List[float] = []
        rets: List[float] = []

        s_min = float(c.stake_min)  # already a float in the dataclass
        s_max = float(c.stake_max) if c.stake_max is not None else float("inf")

        for i in range(len(work)):
            k = float(k_frac.iloc[i])
            raw = bankroll * float(c.kelly_fraction) * k
            stake_i = max(s_min, min(s_max, raw))
            stakes.append(stake_i)

            odds_i = float(work["odds"].iloc[i])
            win_i = int(hit.iloc[i])
            ret_i = win_i * (odds_i - 1.0) * stake_i - (1 - win_i) * stake_i
            rets.append(ret_i)
            bankroll += ret_i

        work["stake"] = pd.Series(stakes, index=work.index)
        pnl = bankroll - init
        bets = int((work["stake"] > 0).sum())
        races_bet = int(work["race_id"].nunique())
        roi = float(pnl / init) if init > 0 else 0.0  # no None comparison
        hit_rate = float(hit.mean()) if len(hit) else 0.0
        avg_odds = float(work["odds"].mean()) if bets else 0.0

    out_dir = Path(f"reports/backtest/{_ts()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = BTResult(
        rows=int(len(df)),
        bets=bets,
        races_bet=races_bet,
        pnl=float(pnl),
        roi=float(roi),
        hit_rate=hit_rate,
        avg_odds=avg_odds,
        ts=_ts(),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")

    work_out = work[["race_id", "horse_id", "edge", "model_prob", "odds", "stake", "won"]]
    work_out.to_parquet(out_dir / "bets.parquet", index=False)

    print(f"[backtest] bets={bets} roi={roi:.3f} pnl={pnl:.2f} -> {out_dir}")
    return out_dir
