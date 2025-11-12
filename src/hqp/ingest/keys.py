# src/hqp/ingest/keys.py
# -----------------------------------------------------------------------------
# Canonicalization and validation of composite keys used across the pipeline.
#
# Contract:
#   - All tables are keyed by (race_id, horse_id).
#   - We standardize both to pandas' nullable StringDtype to avoid accidental
#     coercions ('nan' strings) and preserve NA safely through joins/groupbys.
#   - We upper-case and strip whitespace so sources with mixed case / padding
#     still join deterministically.
#
# Why StringDtype():
#   Using pandas' nullable string avoids the pitfalls of 'object' with mixed
#   types and preserves NA semantics in joins (no accidental bucket creation).
# -----------------------------------------------------------------------------
from __future__ import annotations

import pandas as pd
from pandas import StringDtype

KEY_RACE = "race_id"
KEY_HORSE = "horse_id"


def _require_key_columns(df: pd.DataFrame) -> None:
    """Raise a ValueError if either key column is missing."""
    missing = [c for c in (KEY_RACE, KEY_HORSE) if c not in df.columns]
    if missing:
        raise ValueError(f"Expected key columns {KEY_RACE!r} and {KEY_HORSE!r}; missing={missing}")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy where key columns are canonicalized:

    - Ensures presence of 'race_id' and 'horse_id'.
    - Casts both to pandas StringDtype() (nullable string).
    - Trims surrounding whitespace and uppercases (stable join keys).

    Notes
    -----
    We intentionally do not touch non-key columns here; callers may choose to
    normalize additional IDs (e.g., jockey_id) in their own modules.
    """
    _require_key_columns(df)
    out = df.copy()
    out[KEY_RACE] = out[KEY_RACE].astype(StringDtype()).str.strip().str.upper()
    out[KEY_HORSE] = out[KEY_HORSE].astype(StringDtype()).str.strip().str.upper()
    return out


def check_unique_keys(df: pd.DataFrame) -> tuple[int, int]:
    """
    Validate uniqueness of (race_id, horse_id) pairs.

    Returns
    -------
    (n_rows, n_unique_pairs) : tuple[int, int]

    Raises
    ------
    ValueError
        If duplicates exist. The error includes a duplicate count.

    Notes
    -----
    We drop NA pairs when computing uniqueness (NA cannot form a valid key).
    This function is meant to be called *after* a deduplication step.
    """
    _require_key_columns(df)
    n_rows = int(len(df))
    n_unique = int(df[[KEY_RACE, KEY_HORSE]].dropna().drop_duplicates().shape[0])
    if n_unique != n_rows:
        dupes = int(df.duplicated(subset=[KEY_RACE, KEY_HORSE], keep=False).sum())
        raise ValueError(f"Duplicate (race_id, horse_id) keys found: {dupes} duplicates")
    return n_rows, n_unique
