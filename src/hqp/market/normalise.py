# src/hqp/market/normalise.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Probability Normalization Utilities
#
# implied_raw(decimal_odds)       : 1 / decimal odds (invalid → NaN)
# normalize_per_race(df, ...)     : rescale probabilities within group to sum≈1
#
# Design notes
# ------------
# - We do not mutate input frames in place; we copy before writing outputs.
# - Summation checks are optional (enabled by default) and throw with examples
#   when group sums deviate by more than `tol`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_raw(dec_odds: pd.Series) -> pd.Series:
    """
    Implied probability = 1 / decimal odds.
    Invalid or <= 0 → NaN.

    Parameters
    ----------
    dec_odds : pd.Series
        Decimal back odds (>1 for valid back odds).

    Returns
    -------
    pd.Series (float64)
        Implied probabilities in (0, 1], NaN where invalid.
    """
    x = pd.to_numeric(dec_odds, errors="coerce").astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / x
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    return out.astype("float64")


def normalize_per_race(
    df: pd.DataFrame,
    prob_col: str,
    group_col: str = "race_id",
    out_col: str = "mkt_implied",
    *,
    check: bool = True,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Scale `prob_col` within each `group_col` so group sums (over non-NaNs) ≈ 1.
    Groups with zero valid probabilities remain NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing at least [group_col, prob_col].
    prob_col : str
        Column name containing raw probability mass (e.g., 1/odds).
    group_col : str
        Race grouping column (default 'race_id').
    out_col : str
        Destination column for normalized probabilities.
    check : bool
        If True, assert group sums are ~1 within tolerance.
    tol : float
        Absolute tolerance for group sum checks.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with an additional column `out_col`.
    """
    g = df.copy()
    if prob_col not in g.columns:
        raise KeyError(f"normalize_per_race: missing column '{prob_col}'.")
    if group_col not in g.columns:
        raise KeyError(f"normalize_per_race: missing column '{group_col}'.")

    p = pd.to_numeric(g[prob_col], errors="coerce").astype("float64")
    sums = p.groupby(g[group_col]).transform("sum")

    good = sums > 0.0
    g[out_col] = p.copy()
    g.loc[good, out_col] = p.loc[good] / sums.loc[good]

    if check:
        sums_after = g.groupby(group_col, sort=False)[out_col].sum(min_count=1).astype("float64")
        check_mask = sums_after.notna()
        bad = check_mask & (np.abs(sums_after - 1.0) > tol)
        if bool(bad.any()):
            offenders = sums_after[bad].head(5)
            raise AssertionError(
                "normalize_per_race: probability sums not ≈ 1 for groups: "
                + ", ".join(f"{k}→{float(v):.12f}" for k, v in offenders.items())
            )
    return g
