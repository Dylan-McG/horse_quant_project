# src/hqp/features/context.py
# -----------------------------------------------------------------------------
# Context encodings (leak-free)
#
# Purpose
#   Provide static race-context covariates and optional **past-only** horse
#   affinities without using any current-race observed variables. Concretely:
#     • Bucketise distance into configured meter ranges.
#     • One-hot encode going and course with level caps (rare → OTHER).
#     • (Optional) Past-only “affinity” rates per horse × {going, distance-bin}
#       computed via cumulative prior outcomes (win/place), excluding the
#       current row by construction.
#
# Contracts
#   • Inputs must *not* include `obs__*` as features (guarded explicitly).
#   • Past-only rates rely on sorting by time and using cumulative sums that
#     subtract the current row, ensuring no look-ahead.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextConfig:
    """Config for course/going/distance encodings and affinities.

    Attributes
    ----------
    distance_bins : Sequence[int]
        Distance (meters) bin edges for bucket encoding (left-closed, right-open).
    use_affinities : bool
        If True, compute past-only performance affinities per horse × context.
    max_going_levels : int
        Cap the number of one-hot going levels; rest collapsed to OTHER.
    max_course_levels : int
        Cap the number of one-hot course levels; rest collapsed to OTHER.
    """

    distance_bins: Sequence[int] = (0, 1200, 1600, 2000, 2600, 3600, 99999)
    use_affinities: bool = True
    max_going_levels: int = 12
    max_course_levels: int = 20


def _guard_no_current_obs_columns(df: pd.DataFrame) -> None:
    """Fail if any current-race observed fields are present as inputs."""
    leak_cols = [c for c in df.columns if c.startswith("obs__")]
    if leak_cols:
        raise ValueError(
            "Current-race observed variables (obs__*) are not allowed in features. "
            f"Found: {leak_cols}"
        )


def _bucket_distance(distance_m: pd.Series, bins: Sequence[int]) -> pd.Categorical:
    """Left-closed, right-open distance binning with stable categorical dtype."""
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
    cat = pd.cut(
        distance_m.astype(float),
        bins=list(bins),
        right=False,
        include_lowest=True,
        labels=labels,
    )
    return cast(pd.Categorical, cat.astype("category"))


def _one_hot_topn(s: pd.Series, prefix: str, topn: int) -> pd.DataFrame:
    """One-hot encode with level cap: rare levels collapsed to OTHER."""
    s = s.astype("string")
    if topn > 0 and s.nunique(dropna=False) > topn:
        keep = s.value_counts(dropna=False).nlargest(topn).index
        s = s.where(s.isin(keep), other="OTHER")
    # Ensure categorical for stable column order
    dummies = pd.get_dummies(s.astype("category"), prefix=prefix)
    return dummies


def _ensure_flags(runners: pd.DataFrame) -> pd.DataFrame:
    """Create is_win / is_place flags if missing (derived from pos / places_paid)."""
    g = runners.copy()
    if "is_win" not in g.columns:
        g["is_win"] = (g["pos"] == 1).astype(int)
    if "is_place" not in g.columns:
        if "places_paid" in g.columns:
            g["is_place"] = (g["pos"] <= g["places_paid"]).astype(int)
        else:
            # default: top-3 place if field >=8 else top-2
            field_sizes = g.groupby("race_id")["horse_id"].transform("size")
            default_places = np.where(field_sizes >= 8, 3, 2)
            g["is_place"] = (g["pos"] <= default_places).astype(int)
    return g


def _past_only_rate(df: pd.DataFrame, key_col: str, flag_col: str, out_name: str) -> pd.Series:
    """Compute past-only rate per (horse_id × key_col) using cumulative sums.

    Assumes the DataFrame is pre-sorted by time externally. For each group:
        prior_cnt_t = number of prior rows
        prior_sum_t = cumulative sum up to t minus current flag
        rate_t      = prior_sum_t / prior_cnt_t  (NaN if no history)
    """
    grp = df.groupby(["horse_id", key_col], sort=False)
    # Number of prior rows in the group (0 for the first occurrence)
    prior_cnt = grp.cumcount()
    # Sum of flags up to current row, then subtract current to keep *prior* sum
    prior_sum = grp[flag_col].cumsum() - df[flag_col]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(prior_cnt > 0, prior_sum / prior_cnt, np.nan)
    return pd.Series(rate, index=df.index, name=out_name)


def encode_context(
    runners: pd.DataFrame,
    config: ContextConfig | None = None,
) -> pd.DataFrame:
    """Static encodings + optional past-only affinities (leak-free).

    Parameters
    ----------
    runners : DataFrame
        Requires ['race_id','race_dt','horse_id','course','going','distance_m'].
        Optional: 'is_win','is_place','pos','places_paid' (for affinity fallback).
    config : ContextConfig, optional

    Returns
    -------
    DataFrame aligned to runners index with context_* columns.
    """
    if config is None:
        config = ContextConfig()

    required = {"race_id", "race_dt", "horse_id", "course", "going", "distance_m"}
    missing = required - set(runners.columns)
    if missing:
        raise KeyError(f"runners missing columns: {sorted(missing)}")

    _guard_no_current_obs_columns(runners)

    g = runners.copy()
    # Distance buckets (categorical label retained for potential downstream use)
    dist_cat = _bucket_distance(g["distance_m"], config.distance_bins)
    out = pd.DataFrame(index=g.index)
    out["context_distance_bucket"] = dist_cat

    # Going and course one-hots (capped levels for stability)
    going_oh = _one_hot_topn(g["going"], "going", config.max_going_levels)
    course_oh = _one_hot_topn(g["course"], "course", config.max_course_levels)
    out = pd.concat([out, going_oh, course_oh], axis=1)

    if not config.use_affinities:
        return out

    # ---- Past-only affinities (exclude current row by construction)
    gg = g.sort_values(["horse_id", "race_dt"]).copy()
    gg = _ensure_flags(gg)

    # Going affinities (win / place)
    aff_g_win = _past_only_rate(
        gg[["horse_id", "going", "is_win"]].assign(dummy=1), "going", "is_win", "aff_win_rate_going"
    )
    aff_g_plc = _past_only_rate(
        gg[["horse_id", "going", "is_place"]].assign(dummy=1),
        "going",
        "is_place",
        "aff_place_rate_going",
    )

    # Distance-bucket affinities
    dist_lbl = _bucket_distance(gg["distance_m"], config.distance_bins)
    gg = gg.assign(dist_bucket_lbl=pd.Series(dist_lbl).astype(str))
    aff_d_win = _past_only_rate(
        gg[["horse_id", "dist_bucket_lbl", "is_win"]],
        "dist_bucket_lbl",
        "is_win",
        "aff_win_rate_dist",
    )
    aff_d_plc = _past_only_rate(
        gg[["horse_id", "dist_bucket_lbl", "is_place"]],
        "dist_bucket_lbl",
        "is_place",
        "aff_place_rate_dist",
    )

    out = out.join(
        pd.concat([aff_g_win, aff_g_plc, aff_d_win, aff_d_plc], axis=1).reindex(index=g.index)
    )

    # Final leakage guard (paranoia)
    _guard_no_current_obs_columns(out)
    return out
