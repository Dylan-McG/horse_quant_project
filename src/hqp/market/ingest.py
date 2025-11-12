# src/hqp/market/ingest.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Market Attachment
#
# attach_market(features_df, odds_df, ...)
# ----------------------------------------
# Left-join parsed/normalized market information onto a features table:
#   - <prefix>_odds_decimal : cleaned decimal odds (>1.0 or NaN)
#   - <prefix>_implied_raw  : 1 / decimal_odds (un-normalized)
#   - <prefix>_implied      : normalized within-race implied probability
#
# Keys must be present in both: (race_key, horse_key). Duplicates on the odds
# side are rejected to avoid ambiguous joins (many-to-one is allowed).
# Fractional odds can be provided optionally via `fractional_col`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalise import implied_raw, normalize_per_race
from .parse import coalesce_decimal_fractional, parse_decimal_series, parse_fractional_series


def _require_columns(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns {missing}")


def _assert_no_duplicates(df: pd.DataFrame, keys: list[str], ctx: str, max_show: int = 5) -> None:
    dup_mask = df.duplicated(subset=keys, keep=False)
    if bool(dup_mask.any()):
        sample = df.loc[dup_mask, keys].head(max_show)
        raise ValueError(
            f"{ctx}: found duplicate rows on keys {keys}. Example duplicates:\n"
            f"{sample.to_dict(orient='records')}"
        )


def attach_market(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    race_key: str = "race_id",
    horse_key: str = "horse_id",
    decimal_col: str = "decimal_odds",
    fractional_col: str | None = None,
    prefix: str = "mkt",
) -> pd.DataFrame:
    """
    Left-join market odds onto features and produce normalized implied probs.

    Adds three columns:
      {prefix}_odds_decimal
      {prefix}_implied_raw
      {prefix}_implied

    Notes
    -----
    - If both decimal and fractional columns are provided, decimal dominates and
      fractional is used only when decimal is missing/invalid.
    - Within-race normalization removes the overround by construction.
    """
    _require_columns(features_df, [race_key, horse_key], "attach_market(features_df)")
    _require_columns(odds_df, [race_key, horse_key], "attach_market(odds_df)")
    _assert_no_duplicates(odds_df, [race_key, horse_key], "attach_market(odds_df)")

    # Parse odds sources (NaN-safe, >1.0 enforced)
    dec_parsed = pd.Series(np.nan, index=odds_df.index, dtype="float64")
    if decimal_col in odds_df.columns:
        dec_parsed = parse_decimal_series(odds_df[decimal_col])

    frac_parsed = None
    if fractional_col is not None:
        if fractional_col not in odds_df.columns:
            raise KeyError(f"attach_market(odds_df): missing '{fractional_col}'")
        frac_parsed = parse_fractional_series(odds_df[fractional_col])

    odds_decimal = coalesce_decimal_fractional(dec_parsed, frac_parsed)

    # Compose minimal tidy odds frame
    tidy = odds_df[[race_key, horse_key]].copy()
    dec_col_out = f"{prefix}_odds_decimal"
    raw_col = f"{prefix}_implied_raw"
    imp_col = f"{prefix}_implied"

    tidy[dec_col_out] = odds_decimal
    tidy[raw_col] = implied_raw(odds_decimal)

    # Normalize within race; include decimal so it survives to the merge
    tmp = normalize_per_race(
        tidy[[race_key, horse_key, dec_col_out, raw_col]],
        prob_col=raw_col,
        group_col=race_key,
        out_col=imp_col,
        check=True,
        tol=1e-6,
    )

    merged = features_df.merge(
        tmp[[race_key, horse_key, dec_col_out, raw_col, imp_col]],
        on=[race_key, horse_key],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Strongly type final numeric columns
    for c in (dec_col_out, raw_col, imp_col):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")

    return merged
