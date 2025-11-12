# src/hqp/features/build.py
# -----------------------------------------------------------------------------
# Feature orchestration (leak-free)
#
# Purpose
#   Assemble a **past-only** feature matrix for each runner by:
#     1) Normalising time/position fields required by submodules.
#     2) Stripping any `obs__*` current-race columns (explicit leak guard).
#     3) Building feature blocks:
#          • Form features  (history windows, decayed stats)
#          • Context        (course/going/distance encodings, affinities)
#          • Handicapping   (draw/weight/OR style signals)
#     4) Optionally merging **ratings** strictly as-of (backward-looking).
#     5) Prepending passthrough identifiers, timestamps, and labels for training.
#
# Contracts
#   • No `obs__*` fields are allowed in inputs to model features.
#   • Time columns are coerced so that every feature uses data available at or
#     before `race_dt`. Ratings are joined via backward `merge_asof`.
#   • The function is schema-tolerant: it derives `race_dt` and `pos` if needed.
# -----------------------------------------------------------------------------
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from .context import ContextConfig, encode_context
from .form import FormConfig, compute_form_features
from .handicapping import HandicappingConfig, compute_handicapping_features


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BuildConfig:
    """Top-level feature build configuration.

    Fields
    ------
    features_out
        Destination path for the assembled feature table (parquet).
    form_horizons_days
        Rolling lookback windows (in days) for form features.
    form_half_lives_days
        Exponential forgetting half-lives (in days) for decayed form stats.
    min_form_history
        Minimum number of historical observations required to emit form stats.
    use_affinities
        If True, `encode_context` will generate horse x (course/going/distance)
        interaction features using history prior to the race.
    distance_bins_m
        Distance (meters) bin edges for context encoding.
    max_going_levels / max_course_levels
        Caps for one-hot target cardinality (rare levels grouped).
    draw_bins
        Quantile-like bins for normalised draw position.
    """

    features_out: Path = Path("data/features/features.parquet")
    # form
    form_horizons_days: Sequence[int] = (90, 180, 365)
    form_half_lives_days: Sequence[int] = (90, 180)
    min_form_history: int = 1
    # context
    use_affinities: bool = True
    distance_bins_m: Sequence[int] = (0, 1200, 1600, 2000, 2600, 3600, 99999)
    max_going_levels: int = 12
    max_course_levels: int = 20
    # handicapping
    draw_bins: Sequence[float] = (0.0, 0.33, 0.66, 1.01)


# =============================================================================
# Leak guards and schema normalisation helpers
# =============================================================================


def _guard_no_current_obs_columns(df: pd.DataFrame) -> None:
    """Fail fast if any current-race observed columns (obs__*) are present.

    Rationale
    ---------
    The assignment forbids using `obs__*` (observed *after* the stalls open)
    as features. These can be used for *evaluation* only.
    """
    leak_cols = [c for c in df.columns if c.startswith("obs__")]
    if leak_cols:
        raise ValueError(
            "Current-race observed variables (obs__*) are not allowed in features. "
            f"Found: {leak_cols}"
        )


def _with_required_form_columns(runners: pd.DataFrame) -> pd.DataFrame:
    """Ensure minimal inputs expected by form features exist.

    Required
    --------
    race_dt : naive UTC timestamp of the race off time.
        If missing, it is derived from one of:
        - race_datetime (tz-aware or naive)
        - off_time
        - date + race_time

    pos : finishing position within race (1 = winner).
        If missing, derive by ranking `obs__completion_time` ascending.
        If completion time is unavailable, fallback:
            pos = 1 if obs__is_winner else 2

    Notes
    -----
    * We derive these here so downstream feature builders can assume the columns
      exist and focus on their own logic.
    * We deliberately keep tz-naive UTC for `race_dt` to enable fast groupby/sort.
    """
    df = runners.copy()

    # race_dt from race_datetime/off_time/date+race_time
    if "race_dt" not in df.columns:
        if "race_datetime" in df.columns:
            dt = pd.to_datetime(df["race_datetime"], errors="coerce", utc=True)
            df["race_dt"] = dt.dt.tz_localize(None)
        elif "off_time" in df.columns:
            dt = pd.to_datetime(df["off_time"], errors="coerce", utc=True)
            df["race_dt"] = dt.dt.tz_localize(None)
        elif {"date", "race_time"}.issubset(df.columns):
            combo = (
                df["date"].astype(str).str.strip() + " " + df["race_time"].astype(str).str.strip()
            )
            dt = pd.to_datetime(combo, errors="coerce", utc=True)
            df["race_dt"] = dt.dt.tz_localize(None)
        else:
            raise KeyError(
                "runners missing time column; need 'race_dt' or 'race_datetime' "
                "(or date+race_time)"
            )

    # pos from completion time (ascending rank) → lower time = better position
    if "pos" not in df.columns:
        if "obs__completion_time" in df.columns:
            ct = pd.to_numeric(df["obs__completion_time"], errors="coerce")
            df["_ct_rank_key"] = ct.where(ct.notna(), float("inf"))
            df["pos"] = (
                df.groupby("race_id", sort=False)["_ct_rank_key"]
                .rank(method="dense", ascending=True)
                .astype("int64")
            )
            df.drop(columns=["_ct_rank_key"], inplace=True, errors="ignore")
        elif "obs__is_winner" in df.columns:
            df["pos"] = (
                df["obs__is_winner"].astype(bool).map(lambda w: 1 if w else 2).astype("int64")
            )
        else:
            raise KeyError(
                "runners missing position; need 'pos' or one of "
                "'obs__completion_time'/'obs__is_winner'"
            )

    return df


def _strip_obs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any `obs__*` columns to enforce leak-free feature inputs."""
    keep = [c for c in df.columns if not c.startswith("obs__")]
    return df[keep].copy()


# =============================================================================
# Context standardisation (distance parsing, aliases)
# =============================================================================

_FURLONG_M = 201.168
_MILE_M = 1609.344
_YARD_M = 0.9144
_DIST_RX = re.compile(r"^\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*f)?\s*(?:(\d+)\s*y)?\s*$", re.IGNORECASE)


def _parse_distance_to_m(x: object) -> float | None:
    """Coerce a distance field to meters, if possible.

    Accepted inputs
    ---------------
    * Numeric (int/float): assumed to already be meters (ignoring NaN/inf).
    * String: compact forms like "1m2f110y", with/without spaces.
    * Anything else: returns None.

    Returns
    -------
    float | None
        Distance in meters or None if unparsable.
    """
    # Numeric → meters (guard NaN/inf)
    if isinstance(x, int | float):
        xf = float(x)
        if not np.isfinite(xf):
            return None
        return xf

    # None / obvious empties
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s in {"nan", "none"}:
        return None

    # Parse '1m 2f 110y' style strings
    m = _DIST_RX.match(s.replace(" ", ""))
    if not m:
        m = _DIST_RX.match(s)
    if m:
        miles = float(m.group(1) or 0)
        furlongs = float(m.group(2) or 0)
        yards = float(m.group(3) or 0)
        return miles * _MILE_M + furlongs * _FURLONG_M + yards * _YARD_M
    return None


# Concrete callable type for Series.map to appease type checkers
PARSE_DISTANCE_FN: Callable[[object], float | None] = _parse_distance_to_m


def _with_required_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure minimal context columns exist for encoding.

    Creates
    -------
    course      : copied from racecourse_name if absent.
    going       : copied from going_clean if absent.
    distance_m  : meters parsed from race_distance or distance (string).

    Notes
    -----
    `encode_context` will decide which of these are actually required,
    but we prepare them consistently here so downstream logic is simple.
    """
    g = df.copy()

    if "course" not in g.columns and "racecourse_name" in g.columns:
        g["course"] = g["racecourse_name"]

    if "going" not in g.columns and "going_clean" in g.columns:
        g["going"] = g["going_clean"]

    if "distance_m" not in g.columns:
        if "race_distance" in g.columns:
            # try numeric first; if some strings remain, parse them
            dist = pd.to_numeric(g["race_distance"], errors="coerce")
            need_parse = dist.isna()
            if need_parse.any():
                parsed = g.loc[need_parse, "race_distance"].map(PARSE_DISTANCE_FN)
                dist.loc[need_parse] = parsed
            g["distance_m"] = dist
        elif "distance" in g.columns:
            g["distance_m"] = g["distance"].map(PARSE_DISTANCE_FN)
        # else: leave missing; encode_context will error with a clear message if required

    return g


# =============================================================================
# Ratings join (as-of)
# =============================================================================


def merge_ratings_asof(
    runners: pd.DataFrame,
    ratings: pd.DataFrame,
    on_keys: Sequence[str] = ("horse_id",),
    left_time: str = "race_dt",
    right_time: str = "rating_dt",
    suffix: str = "_rt",
) -> pd.DataFrame:
    """Attach most recent *past* rating per entity to each runner row (no leakage).

    Behaviour
    ---------
    * If `ratings[right_time]` exists, perform a keyed `merge_asof` with
      `direction='backward'`, guaranteeing we only use historical information.
    * Otherwise, if both sides share `race_id`, fall back to a strict left join
      on `['race_id', *on_keys]` (useful for pre-snapshotted per-race ratings).

    Parameters
    ----------
    runners
        Runner-level table; must include `race_id`, `left_time` and `on_keys`.
    ratings
        Ratings table; must include `right_time` and `on_keys` for as-of,
        or `race_id` and `on_keys` for the fallback.
    on_keys
        Entity key(s) used to match ratings (e.g., horse_id).
    left_time / right_time
        Timestamp columns for as-of alignment on left/right tables.
    suffix
        Suffix appended to rating columns to avoid name collisions.

    Returns
    -------
    pd.DataFrame
        `runners` with rating columns appended (suffixed); row count preserved.
    """
    required_left = {"race_id", left_time, *on_keys}
    missing_l = required_left - set(runners.columns)
    if missing_l:
        raise KeyError(f"runners missing columns: {sorted(missing_l)}")

    _guard_no_current_obs_columns(runners)
    _guard_no_current_obs_columns(ratings)

    # If ratings has the time column → do proper asof (past-only).
    if right_time in ratings.columns:
        required_right = {right_time, *on_keys}
        missing_r = required_right - set(ratings.columns)
        if missing_r:
            raise KeyError(f"ratings missing columns: {sorted(missing_r)}")

        left = runners.sort_values([*on_keys, left_time]).copy()
        right = ratings.sort_values([*on_keys, right_time]).copy()

        bring_cols = [c for c in right.columns if c not in set(on_keys) | {right_time}]
        merged = pd.merge_asof(
            left,
            right[[*on_keys, right_time, *bring_cols]],
            left_on=left_time,
            right_on=right_time,
            by=list(on_keys),
            direction="backward",  # strictly <= left_time
            allow_exact_matches=True,
        )
        rename = {c: f"{c}{suffix}" for c in bring_cols}
        merged = merged.rename(columns=rename).drop(columns=[right_time])
        return merged

    # Fallback: join on race_id + on_keys if possible (ratings without timestamps).
    if "race_id" in ratings.columns and "race_id" in runners.columns:
        join_keys = ["race_id", *on_keys] if "race_id" not in on_keys else list(on_keys)
        bring_cols = [c for c in ratings.columns if c not in set(join_keys)]
        tmp = ratings[join_keys + bring_cols].copy()
        # suffix rating columns (avoid double suffix)
        rename = {c: f"{c}{suffix}" for c in bring_cols if not c.endswith(suffix)}
        tmp = tmp.rename(columns=rename)
        merged = runners.merge(tmp, on=join_keys, how="left", validate="many_to_one")
        return merged

    raise KeyError(
        f"ratings needs '{right_time}' for as-of or 'race_id' to join; "
        f"have columns: {sorted(ratings.columns)}"
    )


# =============================================================================
# Orchestration: build features
# =============================================================================


def build_features(
    runners: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    cfg: BuildConfig | None = None,
) -> pd.DataFrame:
    """Assemble a leak-free, model-ready feature table for each runner row.

    Safety & Contracts
    ------------------
    * All `obs__*` columns are stripped before any computation.
    * Time alignment: form and context features rely only on information
      available at or before `race_dt` for the given runner.
    * Ratings (if provided) are joined strictly as-of (see `merge_ratings_asof`).

    Notes
    -----
    This function is intentionally tolerant to upstream schema variations and
    will derive `race_dt` and `pos` when missing, so the feature builders can
    assume a consistent minimal schema.
    """
    if cfg is None:
        cfg = BuildConfig()

    typer.echo("[features] start (normalizing)")

    # Normalize/augment required columns and strip observed current-race columns
    runners_norm = _with_required_form_columns(runners)
    runners_norm = _with_required_context_columns(runners_norm)
    runners_clean = _strip_obs_columns(runners_norm)

    # (safe micro-optimisation) Use categoricals for high-cardinality key-ish columns.
    # This does not change values; it can reduce memory and speed up groupby/joins.
    for col in ("race_id", "horse_id", "jockey_id", "trainer_id", "course", "going"):
        if col in runners_clean.columns:
            try:
                runners_clean[col] = runners_clean[col].astype("category")
            except Exception:
                # Non-fatal: keep original dtype if casting fails
                pass

    typer.echo("[features] form …")
    form_feats = compute_form_features(
        runners_clean,
        FormConfig(
            horizons_days=tuple(cfg.form_horizons_days),
            half_lives_days=tuple(cfg.form_half_lives_days),
            min_history=cfg.min_form_history,
        ),
    )

    typer.echo("[features] context …")
    ctx_feats = encode_context(
        runners_clean,
        ContextConfig(
            distance_bins=tuple(cfg.distance_bins_m),
            use_affinities=cfg.use_affinities,
            max_going_levels=cfg.max_going_levels,
            max_course_levels=cfg.max_course_levels,
        ),
    )

    typer.echo("[features] handicapping …")
    hc_feats = compute_handicapping_features(
        runners_clean, HandicappingConfig(draw_bins=tuple(cfg.draw_bins))
    )

    typer.echo("[features] concat + passthrough …")
    feats = pd.concat([form_feats, ctx_feats, hc_feats], axis=1)

    # Ratings as-of join (optional)
    if ratings is not None and not ratings.empty:
        typer.echo("[features] ratings (as-of) …")
        merged = merge_ratings_asof(runners_clean, ratings)
        rating_cols = [c for c in merged.columns if c.endswith("_rt")]
        if rating_cols:
            feats = pd.concat([feats, merged[rating_cols]], axis=1)

    # ---- Add training passthrough columns expected by the modelling stack ----
    passthru = pd.DataFrame(index=runners_clean.index)
    if "race_id" in runners_clean.columns:
        passthru["race_id"] = runners_clean["race_id"].values
    if "horse_id" in runners_clean.columns:
        passthru["horse_id"] = runners_clean["horse_id"].values

    # Time columns (model configs often use 'off_time'; fallback to race_dt)
    if "off_time" in runners_clean.columns:
        passthru["off_time"] = runners_clean["off_time"].values
    else:
        passthru["off_time"] = runners_clean["race_dt"].values
    passthru["race_dt"] = runners_clean["race_dt"].values

    # Target (label) — derived only from finishing position (safe; not obs__).
    passthru["is_win"] = (runners_clean["pos"] == 1).astype("int8").values

    # Optional: places flag if available (derived from pos & places_paid if present)
    if "places_paid" in runners_clean.columns:
        passthru["is_place"] = (
            (runners_clean["pos"] <= runners_clean["places_paid"]).astype("int8").values
        )

    # Prepend passthrough columns for convenience
    feats = pd.concat([passthru, feats], axis=1)

    # Final leakage guard on output
    _guard_no_current_obs_columns(feats)

    typer.echo(f"[features] done → rows={len(feats)} cols={feats.shape[1]}")
    return feats


def build_and_write(
    runners: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    cfg: BuildConfig | None = None,
) -> pd.DataFrame:
    """Build features and write to parquet path specified in cfg."""
    typer.echo("[features] build_and_write …")
    feats = build_features(runners, ratings, cfg)
    out_path = cfg.features_out if cfg is not None else BuildConfig().features_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    typer.echo(f"[features] wrote → {out_path}")
    return feats


# =============================================================================
# Minimal CLI (convenience for local runs)
# =============================================================================

if __name__ == "__main__":
    # Lightweight CLI (expects pre-prepared Parquets/CSVs in conventional paths)
    import argparse

    parser = argparse.ArgumentParser("hqp.features.build")
    parser.add_argument("--runners", type=Path, required=True, help="Path to runners parquet/csv")
    parser.add_argument("--ratings", type=Path, required=False, help="Optional ratings parquet/csv")
    parser.add_argument("--out", type=Path, required=False, default=BuildConfig().features_out)
    args = parser.parse_args()

    def _read_any(path: Path, time_cols: Sequence[str]) -> pd.DataFrame:
        """Read parquet or CSV. For CSV, parse only date-like columns that exist.

        This avoids Pandas' 'Unknown string format' errors when a requested
        parse_dates column is absent.
        """
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        # For CSVs, only parse date columns that actually exist to avoid pandas errors.
        if path.exists():
            # Peek columns
            head = pd.read_csv(path, nrows=0)
            parse_cols = [c for c in time_cols if c in head.columns]
            return pd.read_csv(path, parse_dates=parse_cols)
        return pd.read_csv(path)

    runners_df = _read_any(
        args.runners, time_cols=["race_dt", "race_datetime", "off_time", "date", "race_time"]
    )
    ratings_df = _read_any(args.rettings, time_cols=["rating_dt"]) if args.ratings else None

    cfg = BuildConfig(features_out=args.out)
    build_and_write(runners_df, ratings_df, cfg)
