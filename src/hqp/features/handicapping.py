# src/hqp/features/handicapping.py
# -----------------------------------------------------------------------------
# Handicapping features (leak-free)
#
# Purpose
#   Construct simple, robust handicapping covariates that do not rely on any
#   current-race observations:
#     • Official Rating (OR) delta vs race median.
#     • Carried weight delta vs race median (lbs).
#     • Draw signals: normalized stall position in [0,1] and binned indicators.
#
# Contracts
#   • Inputs must not include `obs__*` as features (explicit guard).
#   • Group-wise medians and normalisations are done per-race to avoid leakage.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HandicappingConfig:
    """Config for handicapping feature construction.

    Attributes
    ----------
    draw_bins : Sequence[float]
        Bin edges applied to normalized draw in [0, 1]. Defaults produce
        inner/middle/outer terciles: [0, 0.33, 0.66, 1.01)
    """

    draw_bins: Sequence[float] = (0.0, 0.33, 0.66, 1.01)


def _guard_no_current_obs_columns(df: pd.DataFrame) -> None:
    """Fail if any current-race observed fields are present as inputs."""
    leak_cols = [c for c in df.columns if c.startswith("obs__")]
    if leak_cols:
        raise ValueError(
            "Current-race observed variables (obs__*) are not allowed in features. "
            f"Found: {leak_cols}"
        )


def _first_present(colnames: Sequence[str], df: pd.DataFrame) -> str | None:
    """Return the first column name present in `df` from `colnames`, else None."""
    for c in colnames:
        if c in df.columns:
            return c
    return None


def compute_handicapping_features(
    runners: pd.DataFrame,
    config: HandicappingConfig | None = None,
) -> pd.DataFrame:
    """Compute OR deltas, weight deltas vs race median, and draw encodings.

    Parameters
    ----------
    runners : DataFrame
        Must include ['race_id','horse_id'] and preferably:
        'official_rating','weight_lbs' or 'carried_weight_lbs','draw'.
    config : HandicappingConfig, optional

    Returns
    -------
    DataFrame aligned to runners index with columns:
      - hc_or_delta
      - hc_weight_delta
      - hc_draw_norm
      - draw_bin_* (if draw present)
    """
    if config is None:
        config = HandicappingConfig()

    required = {"race_id", "horse_id"}
    missing = required - set(runners.columns)
    if missing:
        raise KeyError(f"runners missing columns: {sorted(missing)}")

    _guard_no_current_obs_columns(runners)

    g = runners.copy()
    out = pd.DataFrame(index=g.index)

    # OR delta vs race median
    if "official_rating" in g.columns:
        med_or = g.groupby("race_id", sort=False)["official_rating"].transform("median")
        out["hc_or_delta"] = g["official_rating"] - med_or
    else:
        out["hc_or_delta"] = np.nan

    # Weight delta vs race median (lbs)
    weight_col = _first_present(["weight_lbs", "carried_weight_lbs"], g)
    if weight_col is not None:
        med_w = g.groupby("race_id", sort=False)[weight_col].transform("median")
        out["hc_weight_delta"] = g[weight_col] - med_w
    else:
        out["hc_weight_delta"] = np.nan

    # Draw encodings: normalized draw in [0,1] within stalls, plus bins
    if "draw" in g.columns:
        # Field size per race (avoid div-by-zero if single runner)
        field_size = g.groupby("race_id", sort=False)["horse_id"].transform("size").astype(float)
        # Many stalls start at 1; normalize to [0,1]
        denom = np.maximum(field_size - 1.0, 1.0)
        out["hc_draw_norm"] = (g["draw"].astype(float) - 1.0) / denom

        # Bin normalized draw by configured edges
        bins = list(config.draw_bins)
        labels = [f"draw_bin_{i}" for i in range(len(bins) - 1)]
        draw_bin = pd.cut(
            out["hc_draw_norm"],
            bins=bins,
            right=False,
            include_lowest=True,
            labels=labels,
        )
        out = pd.concat([out, pd.get_dummies(draw_bin)], axis=1)
    else:
        out["hc_draw_norm"] = np.nan

    # Final guard against leakage in constructed features
    _guard_no_current_obs_columns(out)
    return out
