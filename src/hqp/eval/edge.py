# src/hqp/eval/edge.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Edge Construction (Simple EB Baseline)
#
# Purpose
# -------
# Build a simple, reproducible "edge" parquet directly from a market join file.
# This does *not* depend on saved model artifacts; instead, it uses three
# empirical-Bayes style features (horse/jockey/trainer) and a fixed blend:
#
#   p_model_raw  = 0.70*eb_horse_rt + 0.15*eb_jockey_rt + 0.15*eb_trainer_rt
#   p_model_norm = p_model_raw / sum_race(p_model_raw)
#   edge         = p_model_norm - mkt_implied
#
# Output columns:
#   race_id, horse_id, model_prob, mkt_implied, mkt_odds_decimal, edge
#
# Notes
# -----
# - Accepts either `eb_*_rt` OR `rating_*_rt` columns. If only `rating_*_rt`
#   exist, they are copied to `eb_*_rt` automatically.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _safe_prob(x: pd.Series) -> pd.Series:
    """Clip to [1e-6, 1-1e-6] to avoid degenerate probabilities."""
    return x.clip(1e-6, 1 - 1e-6)


def write_eb_predictions(market_path: Path, out_path: Path) -> None:
    """
    Produce EB-blended model probabilities from a market-join parquet.

    Output schema:
      race_id (string), horse_id (string), model_prob (float)
    """
    df = pd.read_parquet(market_path)

    # Accept either eb_*_rt or rating_*_rt (auto-alias)
    for eb_col, rating_col in [
        ("eb_horse_rt", "rating_horse_rt"),
        ("eb_jockey_rt", "rating_jockey_rt"),
        ("eb_trainer_rt", "rating_trainer_rt"),
    ]:
        if eb_col not in df.columns and rating_col in df.columns:
            df[eb_col] = df[rating_col]

    need = ["race_id", "horse_id", "eb_horse_rt", "eb_jockey_rt", "eb_trainer_rt"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[predict-eb] market parquet missing required columns: {miss}")

    raw = (
        0.70 * pd.to_numeric(df["eb_horse_rt"], errors="coerce")
        + 0.15 * pd.to_numeric(df["eb_jockey_rt"], errors="coerce")
        + 0.15 * pd.to_numeric(df["eb_trainer_rt"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0) + 1e-9

    race_sum = raw.groupby(df["race_id"], sort=False).transform("sum")
    p = (raw / race_sum).fillna(0.0).clip(1e-6, 1 - 1e-6)

    out = pd.DataFrame(
        {
            "race_id": df["race_id"].astype("string"),
            "horse_id": df["horse_id"].astype("string"),
            "model_prob": p.astype(float),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def compute(market_path: Path, run_dir: Path, out_path: Path) -> None:
    """
    Minimal real edge computation that doesn't depend on saved model artifacts.
    We form a simple model probability per runner from EB features and normalize per race:

      p_model_raw = 0.70*eb_horse_rt + 0.15*eb_jockey_rt + 0.15*eb_trainer_rt
      p_model_norm = p_model_raw / sum_race(p_model_raw)

    Edge = p_model_norm - mkt_implied

    Output columns:
      race_id, horse_id, model_prob, mkt_implied, mkt_odds_decimal, edge

    Notes
    -----
    - `run_dir` is accepted for a consistent signature with other evaluators,
      but not used in this simple baseline (kept for CLI uniformity).
    """
    df = pd.read_parquet(market_path)

    # Required market fields
    base_needed = ["race_id", "horse_id", "mkt_implied", "mkt_odds_decimal"]
    miss_base = [c for c in base_needed if c not in df.columns]
    if miss_base:
        raise ValueError(f"[edge] missing required columns: {miss_base}")

    # Ensure expected EB rating columns exist; if not, alias from rating_*_rt
    pairs = [
        ("eb_horse_rt", "rating_horse_rt"),
        ("eb_jockey_rt", "rating_jockey_rt"),
        ("eb_trainer_rt", "rating_trainer_rt"),
    ]
    to_report_missing: list[str] = []
    for eb_col, rating_col in pairs:
        if eb_col not in df.columns:
            if rating_col in df.columns:
                df[eb_col] = df[rating_col]
            else:
                to_report_missing.append(f"'{eb_col}' or '{rating_col}'")
    if to_report_missing:
        raise ValueError(f"[edge] missing required columns: [{', '.join(to_report_missing)}]")

    # Raw score from EB rates (already ~win-rate style 0..1)
    raw = (
        0.70 * pd.to_numeric(df["eb_horse_rt"], errors="coerce")
        + 0.15 * pd.to_numeric(df["eb_jockey_rt"], errors="coerce")
        + 0.15 * pd.to_numeric(df["eb_trainer_rt"], errors="coerce")
    )
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0) + 1e-9

    # Normalize to probability per race (avoid unnamed series pitfalls)
    s = pd.Series(raw.values, index=df.index, name="_raw")
    race_sum = s.groupby(df["race_id"], sort=False).transform("sum")
    p_model = (s / race_sum).fillna(0.0)
    p_model = _safe_prob(p_model)

    # Edge vs market
    mkt_p = _safe_prob(pd.to_numeric(df["mkt_implied"], errors="coerce").fillna(0.0))
    edge = (p_model - mkt_p).astype(float)

    out = pd.DataFrame(
        {
            "race_id": df["race_id"],
            "horse_id": df["horse_id"],
            "model_prob": p_model.astype(float),
            "mkt_implied": mkt_p.astype(float),
            "mkt_odds_decimal": pd.to_numeric(df["mkt_odds_decimal"], errors="coerce").astype(
                float
            ),
            "edge": edge,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    # intentionally no print/echo here
