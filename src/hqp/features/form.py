# src/hqp/features/form.py
# -----------------------------------------------------------------------------
# Form features (leak-free)
#
# Purpose
#   Derive **past-only** per-horse form signals using both fixed time windows
#   (rolling counts/rates) and exponentially decayed histories:
#     • Rolling over horizons (e.g., 90/180/365 days): prior runs, win/place sums,
#       win/place rates with a minimum history requirement.
#     • Decayed sums with specified half-lives (e.g., 90/180 days).
#
# Contracts
#   • Requires race ordering by time; rolling windows are closed='left' to
#     exclude the current event. Current-race `obs__*` are forbidden as inputs.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FormConfig:
    """Configuration for rolling form features.

    Attributes
    ----------
    horizons_days : Sequence[int]
        Window sizes in days for time-based rolling features.
    half_lives_days : Sequence[int]
        Half-life windows in days for exponentially decayed stats.
        If empty, decay features are skipped.
    min_history: int
        Minimum prior runs required before computing rates; otherwise NaN.
    """

    horizons_days: Sequence[int] = (90, 180, 365)
    half_lives_days: Sequence[int] = (90, 180)
    min_history: int = 1


def _ensure_sorted(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Stable sort by keys and reset index for predictable alignment."""
    return df.sort_values(list(by)).reset_index(drop=True)


def _guard_no_current_obs_columns(df: pd.DataFrame) -> None:
    """Fail if any current-race observed fields are present as inputs."""
    leak_cols = [c for c in df.columns if c.startswith("obs__")]
    if leak_cols:
        raise ValueError(
            "Current-race observed variables (obs__*) are not allowed in features. "
            f"Found: {leak_cols}"
        )


# Past-only rolling sum over a time window, aligned to g.index.
def _rolling_time_sum(g: pd.DataFrame, value_col: str, days: int) -> pd.Series:
    """Sum of `value_col` over the trailing `days`, excluding current row."""
    parts: list[pd.Series] = []
    for _, grp in g.groupby("horse_id", sort=False):
        grp = grp.sort_values(["race_dt", "race_id"], kind="mergesort")
        idx = grp.index  # preserve original row indices
        s = grp.set_index("race_dt")[value_col]
        # closed='left' → excludes current row (past-only)
        rolled = s.rolling(f"{days}D", closed="left").sum()
        rolled.index = idx  # realign to original row index
        parts.append(rolled)
    out: pd.Series = pd.concat(parts).sort_index()
    return out


# Past-only rolling count over a time window, aligned to g.index.
def _rolling_time_count(g: pd.DataFrame, days: int) -> pd.Series:
    """Count of prior runs over the trailing `days` (closed='left')."""
    parts: list[pd.Series] = []
    for _, grp in g.groupby("horse_id", sort=False):
        grp = grp.sort_values(["race_dt", "race_id"], kind="mergesort")
        idx = grp.index
        s = grp.set_index("race_dt")["race_id"]
        # count prior rows in window; closed='left' excludes current
        rolled = s.rolling(f"{days}D", closed="left").count()
        rolled.index = idx
        parts.append(rolled.astype(float))
    out: pd.Series = pd.concat(parts).sort_index()
    return out


def _decay_stats_per_group(
    df: pd.DataFrame,
    half_life_days: int,
    flag_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute exponentially decayed sums per horse using time deltas.

    Uses weights w = 0.5 ** (delta_days / half_life_days) for *prior* rows only.
    Returns a DataFrame aligned to df.index with columns:
    {f"form_decayed_{col}_hl{half_life_days}d" for col in flag_cols}.
    """
    out_parts: list[pd.DataFrame] = []
    for _, grp in df.groupby("horse_id", sort=False):
        grp = grp.sort_values("race_dt").copy()

        # Convert datetime64[ns] → int64 nanoseconds safely (pandas 2.x friendly)
        times_ns = grp["race_dt"].astype("int64").to_numpy()  # nanoseconds since epoch
        times_days = (times_ns - times_ns.min()) / 86_400_000_000_000.0

        n = len(grp)
        # Lower-triangular day deltas: td[i, j] = days between row i and j
        td = times_days.reshape(-1, 1) - times_days.reshape(1, -1)
        prior_mask = np.tri(n, n, k=-1, dtype=bool)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(prior_mask, np.power(0.5, td / float(half_life_days)), 0.0)

        part = pd.DataFrame(index=grp.index)
        for col in flag_cols:
            v = grp[col].to_numpy(dtype=float)
            part[f"form_decayed_{col}_hl{half_life_days}d"] = (w * v.reshape(1, -1)).sum(axis=1)

        out_parts.append(part)

    return pd.concat(out_parts).sort_index()


def compute_form_features(
    runners: pd.DataFrame,
    config: FormConfig | None = None,
) -> pd.DataFrame:
    """Compute past-only rolling form features per horse and race.

    Parameters
    ----------
    runners : DataFrame
        Must include: ['race_id','race_dt','horse_id','pos','places_paid' (optional)].
        Optional: 'is_win','is_place' precomputed flags; if absent, derived from pos.
    config : FormConfig, optional

    Returns
    -------
    DataFrame with original index and added form_* columns (no leakage).
    """
    if config is None:
        config = FormConfig()

    required = {"race_id", "race_dt", "horse_id", "pos"}
    missing = required - set(runners.columns)
    if missing:
        raise KeyError(f"runners missing columns: {sorted(missing)}")

    _guard_no_current_obs_columns(runners)

    g = _ensure_sorted(runners.copy(), ["horse_id", "race_dt"])

    # Flags
    if "is_win" not in g.columns:
        g["is_win"] = (g["pos"] == 1).astype(int)
    if "is_place" not in g.columns:
        if "places_paid" not in g.columns:
            # default: top-3 as place if field >=8 else top-2
            field_sizes = g.groupby("race_id")["horse_id"].transform("size")
            default_places = np.where(field_sizes >= 8, 3, 2)
            g["is_place"] = (g["pos"] <= default_places).astype(int)
        else:
            g["is_place"] = (g["pos"] <= g["places_paid"]).astype(int)

    out = pd.DataFrame(index=g.index)

    # Time-based rolling features for each horizon
    for h in config.horizons_days:
        prev_runs = _rolling_time_count(g, h)
        win_sum = _rolling_time_sum(g, "is_win", h)
        place_sum = _rolling_time_sum(g, "is_place", h)

        out[f"form_prev_runs_{h}d"] = prev_runs.to_numpy()
        out[f"form_win_sum_{h}d"] = win_sum.to_numpy()
        out[f"form_place_sum_{h}d"] = place_sum.to_numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"form_win_rate_{h}d"] = np.where(
                prev_runs >= config.min_history, win_sum / prev_runs, np.nan
            )
            out[f"form_place_rate_{h}d"] = np.where(
                prev_runs >= config.min_history, place_sum / prev_runs, np.nan
            )

    # Exponential decay features by half-life
    for hl in config.half_lives_days:
        decayed = _decay_stats_per_group(
            g[["race_dt", "horse_id", "is_win", "is_place"]], hl, ["is_win", "is_place"]
        )
        out[f"form_win_sum_hl{hl}d"] = decayed[f"form_decayed_is_win_hl{hl}d"]
        out[f"form_place_sum_hl{hl}d"] = decayed[f"form_decayed_is_place_hl{hl}d"]

    # Final index alignment preserved
    return out
