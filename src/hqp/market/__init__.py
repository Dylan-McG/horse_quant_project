# src/hqp/market/__init__.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Market Subpackage
#
# Purpose
# -------
# Helpers to parse raw odds (decimal / fractional), convert to implied
# probabilities, normalize within a race, and attach market columns to a
# features DataFrame keyed by (race_id, horse_id).
#
# Public API
# ----------
# - attach_market(...)                : left-join decimal odds + implied probs
# - implied_raw(series)               : 1 / decimal_odds (NaN-safe)
# - normalize_per_race(df, ...)       : re-scale probs to sum≈1 within race
# - parse_decimal_series(series)      : parse decimal odds (EVS/Evens supported)
# - parse_fractional_series(series)   : parse fractional odds like "7/2"
# - coalesce_decimal_fractional(d,f)  : prefer decimal, fallback to fractional
# -----------------------------------------------------------------------------

from __future__ import annotations

from .ingest import attach_market
from .normalise import implied_raw, normalize_per_race
from .parse import (
    coalesce_decimal_fractional,
    parse_decimal_series,
    parse_fractional_series,
)

__all__ = [
    "attach_market",
    "implied_raw",
    "normalize_per_race",
    "coalesce_decimal_fractional",
    "parse_decimal_series",
    "parse_fractional_series",
]
