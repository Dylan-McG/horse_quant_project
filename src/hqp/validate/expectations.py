# src/hqp/validate/expectations.py
# pyright: reportMissingImports=false
# -----------------------------------------------------------------------------
# Validation utilities for:
#   (A) Probability tables (race-normalized probabilities).
#   (B) Canonical runner-level parquet (schema sanity + uniqueness + labels).
#
# Design:
# - Pandera is optional. If installed, we use a Pandera schema for probabilities.
#   If not installed, we raise on `validate_probabilities`; callers can rely on
#   the hard guards (from hqp.common.checks) instead.
# - `validate_canonical` performs lightweight, dependency-free checks on the
#   canonical parquet produced by ingest (runner grain).
#
# CLI note:
# - The CLI calls `hqp.validate.run(...)` if available; we do not define it here.
#   Therefore the CLI will fall back to a minimal key check (see cli.validate).
#   This file is intended for test/ad-hoc validation, not the CLI hot path.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

# Optional dependency: Pandera
try:  # pragma: no cover
    import pandera as pa
except Exception:  # pragma: no cover
    pa = None  # type: ignore[assignment]

from hqp.common.checks import assert_no_na_inf, assert_prob_sum, assert_unique_keys


def _finite_check() -> Callable[[pd.Series], bool]:
    """Return a callable that asserts finiteness on a float-like Series."""

    def _fn(s: pd.Series) -> bool:
        return bool(np.isfinite(s.to_numpy(dtype=float, copy=False)).all())

    return _fn


# Lazily built Pandera schema (avoid import/type-analysis churn when pandera missing)
_probs_schema: Any = None


def _build_probabilities_schema() -> Any:
    """Construct a Pandera schema for a normalized probability table."""
    if pa is None:
        raise ImportError(
            "Pandera is required for schema validation but is not installed. "
            "Install with `poetry add pandera` (or pip) and re-run."
        )
    _pa: Any = cast(Any, pa)
    return _pa.DataFrameSchema(
        {
            # keys are kept generic/object to avoid over-constraining dtype
            "race_id": _pa.Column(_pa.Object, nullable=False),
            "horse_id": _pa.Column(_pa.Object, nullable=False),
            "p": _pa.Column(
                _pa.Float,
                nullable=False,
                checks=[
                    _pa.Check.ge(0.0),
                    _pa.Check.le(1.0),
                    _pa.Check(_finite_check(), error="p must be finite"),
                ],
            ),
        },
        strict=False,  # allow extra diagnostic columns (e.g., logits)
        coerce=True,  # coerce types where possible
        name="probabilities_table",
    )


def probabilities_schema() -> Any:
    """Get (and cache) the Pandera schema for probabilities."""
    global _probs_schema
    if _probs_schema is None:
        _probs_schema = _build_probabilities_schema()
    return _probs_schema


def validate_probabilities(
    df_probs: pd.DataFrame,
    *,
    race_col: str = "race_id",
    horse_col: str = "horse_id",
    prob_col: str = "p",
    atol: float = 1e-9,
) -> pd.DataFrame:
    """
    Validate a per-runner probability table:

      - Pandera schema: keys present, 0 <= p <= 1, finite.
      - Hard guards: no NA/Inf, composite-key uniqueness, per-race sum-to-one.

    Returns
    -------
    The validated (and possibly coerced) DataFrame.

    Raises
    ------
    ImportError   if Pandera is not installed.
    ValueError    if constraints are violated.
    """
    schema = probabilities_schema()
    df_valid: pd.DataFrame = cast(pd.DataFrame, schema.validate(df_probs, lazy=False))

    # Additional hard guards (independent of Pandera)
    assert_no_na_inf(df_valid, subset=[race_col, horse_col, prob_col])
    assert_unique_keys(df_valid, keys=[race_col, horse_col])
    assert_prob_sum(df_valid, race_col=race_col, prob_col=prob_col, atol=atol)
    return df_valid


def validate_canonical(parquet_path: Path) -> None:
    """
    Validate canonical runner-level table produced by ingest:

      - Required columns exist: ['race_id', 'horse_id', 'race_datetime'].
      - 'race_datetime' is timezone-aware (UTC expected).
      - Composite-key uniqueness: (race_id, horse_id) is unique.
      - Optional label bounds: obs__is_winner ∈ {0,1}.
      - Optional race consistency: n_runners constant within a race_id.

    Raises
    ------
    ValueError for any contract violation.
    """
    df = pd.read_parquet(parquet_path)

    # Presence of core fields
    required = ["race_id", "horse_id", "race_datetime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in canonical data: {missing}")

    # tz-aware check (ingest should normalize; we assert here defensively)
    if getattr(df["race_datetime"].dt, "tz", None) is None:
        raise ValueError("race_datetime must be timezone-aware (UTC).")

    # Composite-key uniqueness
    assert_unique_keys(df, keys=["race_id", "horse_id"])

    # Optional label bounds
    if "obs__is_winner" in df.columns:
        s = df["obs__is_winner"]
        if (s < 0).any() or (s > 1).any():
            raise ValueError("obs__is_winner must be in {0,1}.")

    # Optional: within-race consistency of n_runners
    if "n_runners" in df.columns:
        if df.groupby("race_id")["n_runners"].nunique().gt(1).any():
            raise ValueError("n_runners varies within a race_id.")
