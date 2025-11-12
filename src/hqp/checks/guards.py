# src/hqp/checks/guards.py
# -----------------------------------------------------------------------------
# Purpose
# -------
# Reusable *guards* that enforce global modeling policies:
#   1) No current-race observation leakage (ban obs__* predictors).
#   2) Per-race probability normalization: sum_runners p = 1.
#   3) Chronological split integrity: train ≤ test (optional buffer gap).
#
# These are light, dependency-free checks that can be called from:
#   - feature engineering (to strip/ban obs__*),
#   - model post-processing (probability normalization),
#   - CV/split code (temporal leakage checks),
#   - tests (unit/integration).
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

__all__ = [
    "list_current_obs",
    "ensure_no_current_obs",
    "ensure_prob_sums",
    "ensure_chrono_no_leak",
]


def list_current_obs(columns: Iterable[str], prefix: str = "obs__") -> list[str]:
    """
    Return column names that look like *current-race observational* features.

    Policy background
    -----------------
    - Columns prefixed 'obs__' are post-race or in-race observations (e.g., BSP,
      finish position, time/speed ratings) and must not be used to *predict the
      same race*.
    - Historical uses are fine if re-namespaced (e.g., 'hist__', 'rat__') and
      strictly based on past races.

    Parameters
    ----------
    columns : iterable of column names to scan.
    prefix  : disallowed prefix (default 'obs__').

    Returns
    -------
    list[str] of offending column names (may be empty).
    """
    return [c for c in columns if c.startswith(prefix)]


def ensure_no_current_obs(columns: Iterable[str], prefix: str = "obs__") -> None:
    """
    Enforce “no current-race obs__*” in the predictor set.

    Raises
    ------
    ValueError
        If any column starts with the disallowed prefix.
    """
    offenders = list_current_obs(columns, prefix=prefix)
    if offenders:
        raise ValueError(
            "Current-race observational predictors are not permitted. "
            f"Found {len(offenders)} columns with prefix '{prefix}': {sorted(offenders)}"
        )


def ensure_prob_sums(
    df: pd.DataFrame,
    race_col: str,
    prob_col: str,
    *,
    atol: float = 1e-9,
) -> None:
    """
    Assert that per-race probabilities sum to 1 within tolerance.

    This guard is useful when the model outputs independent probabilities that
    must be renormalized per race (e.g., via softmax or division by sum).

    Parameters
    ----------
    df       : DataFrame containing {race_col, prob_col}.
    race_col : Name of race key column.
    prob_col : Name of probability column to sum.
    atol     : Absolute tolerance for floating-point comparisons.

    Raises
    ------
    KeyError
        If required columns are missing.
    TypeError
        If prob_col is not numeric.
    ValueError
        If any race violates the sum-to-one constraint beyond tolerance.
    """
    if race_col not in df.columns:
        raise KeyError(f"Missing race_col '{race_col}'.")
    if prob_col not in df.columns:
        raise KeyError(f"Missing prob_col '{prob_col}'.")

    if not is_numeric_dtype(df[prob_col].dtype):
        raise TypeError(f"Probability column '{prob_col}' must be numeric.")

    grouped = df.groupby(race_col, sort=False)[prob_col].sum()
    diffs = (grouped - 1.0).abs()
    bad = diffs[diffs > atol]
    if not bad.empty:
        examples = bad.head(10).to_dict()
        raise ValueError(
            f"Per-race probabilities must sum to 1 ± {atol}. "
            f"Found violations for races (showing up to 10): {examples}"
        )


def ensure_chrono_no_leak(
    df: pd.DataFrame,
    time_col: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    gap: pd.Timedelta | None = None,
) -> None:
    """
    Assert no temporal leakage between train and test partitions.

    Enforced conditions
    -------------------
    1) 'time_col' is datetime-like (timezone-aware or naive).
    2) max(time[train]) <= min(time[test])  (strict forward chronology).
    3) Optional 'gap': min(time[test]) - max(time[train]) >= gap.

    Why this matters
    ----------------
    Any overlap or inversion (train peeking into the future) inflates validation
    metrics and biases model selection/hyperparameters.

    Raises
    ------
    KeyError, TypeError, AssertionError
        For schema issues or chronological violations.
    """
    if time_col not in df.columns:
        raise KeyError(f"Missing time_col '{time_col}'.")
    if not is_datetime64_any_dtype(df[time_col].dtype):
        raise TypeError(f"Column '{time_col}' must be datetime64 dtype.")
    if train_mask.dtype != bool or test_mask.dtype != bool:
        raise TypeError("train_mask and test_mask must be boolean arrays.")
    if train_mask.shape[0] != len(df) or test_mask.shape[0] != len(df):
        raise AssertionError("Masks must align with DataFrame length.")

    # Normalize to UTC for stable comparisons
    time_series = df[time_col]
    if time_series.dt.tz is None:
        time_series = time_series.dt.tz_localize("UTC")
    else:
        time_series = time_series.dt.tz_convert("UTC")

    train_times = time_series[train_mask]
    test_times = time_series[test_mask]

    # Empty partitions are considered OK (e.g., during CV folds creation).
    if train_times.empty or test_times.empty:
        return

    train_latest = train_times.max()
    test_earliest = test_times.min()

    if not (train_latest <= test_earliest):
        raise AssertionError("Chronology violated: max(train.time) > min(test.time).")

    if gap is not None and (test_earliest - train_latest) < gap:
        raise AssertionError(
            f"Chronology gap violated: required gap {gap}, "
            f"observed {(test_earliest - train_latest)}."
        )
