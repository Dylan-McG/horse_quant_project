# src/hqp/market/parse.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Odds Parsing
#
# parse_decimal_series(series)      : parse decimal odds (supports EVS/EVENS/EVEN)
# parse_fractional_series(series)   : parse fractional "A/B" to decimal (1+A/B)
# coalesce_decimal_fractional(d,f)  : choose decimal if present else fractional
#
# Behavior
# --------
# - All outputs are float64, strictly > 1.0 (invalid and non-sensical → NaN).
# - Non-numerics like "SP" or "—" become NaN.
# - Fractions require positive integers; "EVS|EVENS|EVEN" map to 2.0.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

EVENS_TOKENS = {"evs", "evens", "even"}  # common spellings mapped to 2.0
_DECIMAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
_FRACTION_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def _to_float_or_nan(token: str) -> float:
    """
    Try parse a plain decimal number from a string; return NaN on failure.
    """
    m = _DECIMAL_RE.match(token)
    if not m:
        return math.nan
    try:
        return float(m.group(1))
    except ValueError:
        return math.nan


def parse_decimal_series(s: pd.Series) -> pd.Series:
    """
    Parse a series that may contain decimal odds as numbers or strings.

    Rules:
      - numeric values are cast to float
      - strings "EVS" | "EVENS" | "EVEN" (case-insensitive) -> 2.0
      - non-numeric tokens like "SP" -> NaN
      - odds <= 1.0 are invalid -> NaN (back odds must exceed 1.0)

    Returns
    -------
    pd.Series (float64)
    """
    if is_numeric_dtype(s):
        out = s.astype("float64")
    else:
        vals = s.astype("string")
        lower = vals.str.lower()

        is_evens = lower.isin(EVENS_TOKENS)
        numeric_mask = lower.str.match(_DECIMAL_RE).fillna(False)

        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[is_evens] = 2.0

        to_parse = lower[numeric_mask].fillna("")
        parsed = to_parse.map(_to_float_or_nan)
        out.loc[parsed.index] = parsed.astype("float64").values
        # others remain NaN

    out = out.astype("float64")
    out[(out <= 1.0) | ~np.isfinite(out)] = np.nan
    return out


def parse_fractional_series(s: pd.Series) -> pd.Series:
    """
    Parse a series of fractional odds strings like "7/2", "100/30", or EVS variants.

    Rules:
      - "A/B" -> 1 + (A/B), where A,B are positive integers
      - "EVS" | "EVENS" | "EVEN" (case-insensitive) -> 2.0
      - anything else -> NaN

    Returns
    -------
    pd.Series (float64)
    """
    vals = s.astype("string")
    lower = vals.str.lower()

    out = pd.Series(np.nan, index=s.index, dtype="float64")

    # EVENS → 2.0
    is_evens = lower.isin(EVENS_TOKENS)
    out.loc[is_evens] = 2.0

    # Fractions (only apply parser to true "A/B" tokens to avoid overwriting EVS rows)
    frac_mask = lower.str.match(_FRACTION_RE).fillna(False)

    def _parse_fraction_token(tok: str) -> float:
        # tok should be a string matching A/B at this point
        m = _FRACTION_RE.match(tok)
        if not m:
            return math.nan
        try:
            num = int(m.group(1))
            den = int(m.group(2))
            if den <= 0:
                return math.nan
            dec = 1.0 + (num / den)
            return dec if dec > 1.0 else math.nan
        except Exception:
            return math.nan

    # Only map the subset that are fractions
    to_parse = lower[frac_mask].astype("string")
    parsed = to_parse.map(_parse_fraction_token)

    out.loc[parsed.index] = parsed.astype("float64").values

    out[(out <= 1.0) | ~np.isfinite(out)] = np.nan
    return out


def coalesce_decimal_fractional(
    decimal_odds: pd.Series, fractional_odds: pd.Series | None = None
) -> pd.Series:
    """
    Prefer parsed decimal odds when present; otherwise fallback to parsed fractional odds.

    Returns
    -------
    pd.Series (float64)
        Decimal odds, strictly > 1.0 or NaN.
    """
    # Ensure we run through the same parsing logic for both numeric and string inputs
    dec = parse_decimal_series(decimal_odds)
    if fractional_odds is None:
        return dec

    frac = parse_fractional_series(fractional_odds)
    out = dec.copy()
    need = ~np.isfinite(out)
    out.loc[need] = frac.loc[need].astype("float64")
    out[(out <= 1.0) | ~np.isfinite(out)] = np.nan
    return out.astype("float64")
