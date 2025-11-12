# src/hqp/common/checks.py
# -----------------------------------------------------------------------------
# Lightweight, reusable assertions for:
#  - leakage control (forbid current-race obs__* predictors),
#  - probability sanity (per-race probs sum to 1),
#  - NA/Inf hygiene on numeric features,
#  - composite-key uniqueness checks.
#
# These helpers are intentionally small and side-effect free so they can be used
# inside feature builders, model evaluators, or tests without pulling in the
# rest of the pipeline.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterable, Sequence
import numpy as np
import pandas as pd

_OBS_PREFIX = "obs__"


def forbid_current_obs_columns(
    df: pd.DataFrame,
    *,
    action: str = "raise",
    additional_prefixes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Guard against leakage: forbid any *current-race* observation columns.

    Any column whose name starts with "obs__" (or provided extra prefixes) is
    considered a leak for *current-race* prediction and must not be present in
    the predictor set.  This function:
      - returns the DataFrame unchanged if none are found,
      - drops them if action="drop",
      - raises if action="raise" (default).
    """
    prefixes: list[str] = [_OBS_PREFIX]
    if additional_prefixes:
        prefixes.extend(additional_prefixes)

    forbidden = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not forbidden:
        return df

    if action == "drop":
        return df.drop(columns=forbidden).copy()

    if action == "raise":
        cols_preview = ", ".join(sorted(forbidden))
        raise ValueError(
            f"Leak guard triggered: forbidden current-race columns detected: [{cols_preview}]. "
            "Remove these predictors (no current-race obs__* allowed)."
        )

    raise ValueError(f"Unknown action '{action}'. Use 'raise' or 'drop'.")


def assert_prob_sum(
    df_probs: pd.DataFrame,
    *,
    race_col: str = "race_id",
    prob_col: str = "p",
    atol: float = 1e-9,
) -> None:
    """
    Assert that probabilities are race-normalized: sum_{runners} p = 1 ± atol.

    Parameters
    ----------
    df_probs : DataFrame with at least {race_col, prob_col}.
    race_col : race key column (default 'race_id').
    prob_col : probability column (default 'p').
    atol     : absolute tolerance for floating-point comparisons.
    """
    if race_col not in df_probs or prob_col not in df_probs:
        missing = [c for c in (race_col, prob_col) if c not in df_probs]
        raise KeyError(f"assert_prob_sum: missing columns: {missing}")

    grp = df_probs.groupby(race_col, sort=False, observed=True, dropna=False)[prob_col].sum()
    sums = grp.to_numpy(dtype=float, copy=False)
    close_mask = np.isclose(sums, 1.0, rtol=0.0, atol=atol)
    bad = grp[~close_mask]
    if not bad.empty:
        sample = bad.head(10)
        details = "; ".join(f"{k} -> {float(v):.12f}" for k, v in sample.items())
        extra = "" if len(bad) <= 10 else f" (+{len(bad)-10} more)"
        raise AssertionError(
            f"Per-race probability sums must equal 1±{atol}. Violations: {details}{extra}"
        )


def assert_no_na_inf(
    df: pd.DataFrame,
    *,
    subset: Iterable[str] | None = None,
) -> None:
    """
    Ensure there are no NA/Inf values in the DataFrame (or in a specified subset).

    - Checks NA for all requested columns (any dtype).
    - Checks finiteness (not Inf/NaN) only for numeric columns.
    """
    to_check = list(subset) if subset is not None else list(df.columns)
    missing = [c for c in to_check if c not in df]
    if missing:
        raise KeyError(f"assert_no_na_inf: subset columns not in df: {missing}")

    # NA check for all columns
    na_any = df[to_check].isna()
    if na_any.any().any():
        bad_cols = sorted(na_any.any()[na_any.any()].index.tolist())
        raise ValueError(f"NA detected in columns: {bad_cols}")

    # Finite check for numeric columns only
    numeric_cols = [c for c in to_check if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        vals = df[numeric_cols].to_numpy(dtype=float, copy=False)
        if not np.isfinite(vals).all():
            bad_cols_set: set[str] = set()
            arr = df[numeric_cols]
            for c in numeric_cols:
                col = arr[c].to_numpy(dtype=float, copy=False)
                if not np.isfinite(col).all():
                    bad_cols_set.add(str(c))
            bad_cols = sorted(bad_cols_set)
            raise ValueError(f"Non-finite values detected in numeric columns: {bad_cols}")


def assert_unique_keys(df: pd.DataFrame, keys: Sequence[str]) -> None:
    """
    Enforce uniqueness of a composite key (e.g., ['race_id', 'horse_id']).

    Raises with examples if duplicate key rows are detected.
    """
    missing = [k for k in keys if k not in df]
    if missing:
        raise KeyError(f"assert_unique_keys: missing key columns: {missing}")

    dup_mask = df.duplicated(subset=list(keys), keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, list(keys)].drop_duplicates().head(10)
        examples = "; ".join(
            ", ".join(f"{k}={row[k]!r}" for k in keys) for _, row in dups.iterrows()
        )
        extra = "" if len(dups) <= 10 else f" (+{len(dups)-10} more)"
        raise ValueError(
            f"Duplicate key rows detected for keys {list(keys)}. Examples: {examples}{extra}"
        )
