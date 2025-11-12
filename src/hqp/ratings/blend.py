# src/hqp/ratings/blend.py
# -----------------------------------------------------------------------------
# Blended ratings = weighted combination of:
#   - per-entity Elo-like priors (horse, jockey, trainer)   -> skill proxy
#   - per-horse Empirical-Bayes prior on win-rate            -> reliability proxy
#
# The blend uses an *online decayed z-scoring* of each component to (a) remove
# units, (b) handle drift, and (c) avoid look-ahead:
#   For each component X_t observed at timestamp t, we:
#     1) decay running {mean, variance} to time t,
#     2) compute z_t = (X_t - mean_{t^-}) / std_{t^-}  using PRIOR stats,
#     3) update stats with X_t.
#
# This ensures each z-score is “as of t” without peeking at the current event.
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast, Optional

import numpy as np
import pandas as pd
import yaml

from .eb import EBConfig, EmpiricalBayesRater
from .elo import EloConfig, EloRater


# ---- YAML sections (typed) ----------------------------------------------------


class BlendSection(TypedDict, total=False):
    w_horse_elo: float
    w_horse_eb: float
    w_jockey_elo: float
    w_trainer_elo: float
    z_half_life_days: float
    ts_col: str
    race_id_col: str
    horse_col: str
    jockey_col: str
    trainer_col: str
    field_size_col: str
    rank_col: Optional[str]
    win_col: str


class EloSection(TypedDict, total=False):
    half_life_days: float
    k_base: float
    k_field_ref: float
    mean_rating: float


class EBSection(TypedDict, total=False):
    half_life_days: float
    prior_strength: float
    baseline_rate: Optional[float]
    mean_center: bool


@dataclass
class BlendConfig:
    # Component weights in the final blend (sum need not be 1; linear combo).
    w_horse_elo: float = 0.5
    w_horse_eb: float = 0.2
    w_jockey_elo: float = 0.15
    w_trainer_elo: float = 0.15

    # Half-life (days) used by the online z-scoring statistics.
    z_half_life_days: float = 180.0

    # Column names
    ts_col: str = "ts"  # set to "race_dt" in your config if preferred
    race_id_col: str = "race_id"
    horse_col: str = "horse_id"
    jockey_col: str = "jockey_id"
    trainer_col: str = "trainer_id"
    field_size_col: str = "field_size"
    rank_col: Optional[str] = "rank"  # if None, EloRater must get a score_col
    win_col: str = "win"


# ---- Online decayed z-scoring (leak-free) ------------------------------------


class _DecayedZ:
    """
    Welford-style running mean/variance with exponential forgetting.

    State at time t^-:
      mean_{t^-}, m2_{t^-}, w_{t^-}

    Given new value x_t at timestamp t:
      - decay the state from last_ts -> t by λ = 2^{-(Δdays / half_life)}
      - compute z_t using PRIOR (decayed) mean/std
      - update state with x_t

    Notes:
      * We maintain an "effective sample size" w to stabilise early steps.
      * Variance uses m2 / w, bounded away from 0 for numerical safety.
    """

    def __init__(self, half_life_days: float) -> None:
        self.half_life_days = float(max(half_life_days, 1e-9))
        self._last_ts: Optional[pd.Timestamp] = None
        self._mean: float = 0.0
        self._m2: float = 1.0
        self._w: float = 1e-6  # epsilon to avoid division by zero

    def _decay_factor(self, delta_days: float) -> float:
        return float(2.0 ** (-max(delta_days, 0.0) / self.half_life_days))

    def z(self, ts: pd.Timestamp, x: float) -> float:
        # Decay state to current timestamp
        if self._last_ts is None:
            self._last_ts = ts
        else:
            delta_days = float((ts - self._last_ts).total_seconds()) / 86400.0
            lam = self._decay_factor(delta_days)
            self._w *= lam
            self._m2 *= lam
            self._mean *= lam
            self._last_ts = ts

        # PRIOR variance/std (guard against zero)
        var_prior = max(1e-9, self._m2 / max(1e-6, self._w))
        std_prior = float(np.sqrt(var_prior))
        zval = (x - self._mean) / std_prior

        # Welford update (post-observation)
        w_new = self._w + 1.0
        delta = x - self._mean
        mean_new = self._mean + delta / w_new
        self._m2 = max(1e-12, self._m2 + delta * (x - mean_new))
        self._mean = mean_new
        self._w = w_new
        return float(zval)


# ---- YAML load & casting helpers ---------------------------------------------


def _load_yaml(path: str | Path | None) -> dict[str, object]:
    """Load YAML to a shallow dict[str, object]. Missing/empty → {}."""
    if not path:
        return {}
    p = Path(str(path))
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data_obj: Any = yaml.safe_load(f)

    if isinstance(data_obj, dict):
        # Constrain the dynamic YAML mapping to a well-typed dict so Pylance
        # doesn't propagate Unknown into our dict[str, object].
        data_map: dict[Any, Any] = cast(dict[Any, Any], data_obj)
        return {str(cast(str, k)): cast(object, v) for k, v in data_map.items()}

    return {}


def _as_blend_section(raw: object) -> BlendSection:
    out: BlendSection = {}
    if not isinstance(raw, dict):
        return out
    for k in ("w_horse_elo", "w_horse_eb", "w_jockey_elo", "w_trainer_elo", "z_half_life_days"):
        v = raw.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    for k in (
        "ts_col",
        "race_id_col",
        "horse_col",
        "jockey_col",
        "trainer_col",
        "field_size_col",
        "win_col",
    ):
        v = raw.get(k)
        if isinstance(v, str):
            out[k] = v
    v_rank = raw.get("rank_col")
    if isinstance(v_rank, str) or v_rank is None:
        out["rank_col"] = v_rank
    return out


def _as_elo_section(raw: object) -> EloSection:
    out: EloSection = {}
    if not isinstance(raw, dict):
        return out
    for k in ("half_life_days", "k_base", "k_field_ref", "mean_rating"):
        v = raw.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _as_eb_section(raw: object) -> EBSection:
    out: EBSection = {}
    if not isinstance(raw, dict):
        return out
    v = raw.get("half_life_days")
    if isinstance(v, (int, float)):
        out["half_life_days"] = float(v)
    v = raw.get("prior_strength")
    if isinstance(v, (int, float)):
        out["prior_strength"] = float(v)
    vb = raw.get("baseline_rate")
    if isinstance(vb, (int, float)):
        out["baseline_rate"] = float(vb)
    elif vb is None:
        out["baseline_rate"] = None
    vm = raw.get("mean_center")
    if isinstance(vm, bool):
        out["mean_center"] = vm
    return out


# ---- Orchestrator: build blended ratings -------------------------------------


def build_blended_ratings(
    events: pd.DataFrame,
    *,
    config_yaml: str | None = None,
    out_dir: str = "data/ratings",
) -> pd.DataFrame:
    """
    Compute leak-free, as-of blended ratings for each runner:
      rating_blend = w1 * z(horse_elo) + w2 * z(horse_eb) + w3 * z(jockey_elo) + w4 * z(trainer_elo)

    Inputs
    ------
    events : pd.DataFrame
        Must contain columns configured in BlendConfig (ids, timestamps, rank/field, win).
        'win' expected in {0,1}. Timestamps are coerced to UTC and sorted.

    Outputs
    -------
    pd.DataFrame with columns:
        [race_id, ts, horse_id, jockey_id, trainer_id,
         horse_elo, horse_eb, jockey_elo, trainer_elo,
         horse_elo_z, horse_eb_z, jockey_elo_z, trainer_elo_z, rating_blend]
    Also persisted to <out_dir>/ratings.parquet.
    """
    # --- parse config
    cfg_dict = _load_yaml(config_yaml)
    blend_cfg = BlendConfig(**_as_blend_section(cfg_dict.get("blend")))
    elo_cfg = EloConfig(**_as_elo_section(cfg_dict.get("elo")))
    eb_cfg = EBConfig(**_as_eb_section(cfg_dict.get("eb")))

    # --- require columns
    need = {
        blend_cfg.ts_col,
        blend_cfg.race_id_col,
        blend_cfg.horse_col,
        blend_cfg.jockey_col,
        blend_cfg.trainer_col,
        blend_cfg.field_size_col,
        blend_cfg.win_col,
    }
    missing = need - set(map(str, events.columns))
    if missing:
        raise KeyError(f"Missing columns for ratings: {missing}")

    # --- canonical ordering by (time, race) for stable merges
    df = events.copy()
    df[blend_cfg.ts_col] = pd.to_datetime(df[blend_cfg.ts_col], utc=True, errors="coerce")
    df.sort_values([blend_cfg.ts_col, blend_cfg.race_id_col], inplace=True, kind="mergesort")

    # base keys retained in final output
    left_keys = [
        blend_cfg.race_id_col,
        blend_cfg.ts_col,
        blend_cfg.horse_col,
        blend_cfg.jockey_col,
        blend_cfg.trainer_col,
    ]
    base = df.loc[:, left_keys].copy()

    # --- component priors (snapshots BEFORE the current event update)
    # Horse Elo
    horse_elo = (
        EloRater(elo_cfg)
        .rate(
            df,
            entity_col=blend_cfg.horse_col,
            ts_col=blend_cfg.ts_col,
            race_id_col=blend_cfg.race_id_col,
            field_size_col=blend_cfg.field_size_col,
            rank_col=blend_cfg.rank_col,  # if None, a score_col would be required
            score_col=None,
        )
        .rename(columns={"elo_prior": "horse_elo"})
    )

    # Jockey Elo
    jockey_elo = (
        EloRater(elo_cfg)
        .rate(
            df,
            entity_col=blend_cfg.jockey_col,
            ts_col=blend_cfg.ts_col,
            race_id_col=blend_cfg.race_id_col,
            field_size_col=blend_cfg.field_size_col,
            rank_col=blend_cfg.rank_col,
            score_col=None,
        )
        .rename(columns={"elo_prior": "jockey_elo"})
    )

    # Trainer Elo
    trainer_elo = (
        EloRater(elo_cfg)
        .rate(
            df,
            entity_col=blend_cfg.trainer_col,
            ts_col=blend_cfg.ts_col,
            race_id_col=blend_cfg.race_id_col,
            field_size_col=blend_cfg.field_size_col,
            rank_col=blend_cfg.rank_col,
            score_col=None,
        )
        .rename(columns={"elo_prior": "trainer_elo"})
    )

    # Horse EB win-rate prior
    horse_eb = (
        EmpiricalBayesRater(eb_cfg)
        .rate(
            df,
            entity_col=blend_cfg.horse_col,
            ts_col=blend_cfg.ts_col,
            race_id_col=blend_cfg.race_id_col,
            success_col=blend_cfg.win_col,
        )
        .rename(columns={"eb_prior": "horse_eb"})
    )

    # --- join priors back to the base key set (one-to-one per entity at (race,ts))
    keys_rt = [blend_cfg.race_id_col, blend_cfg.ts_col]
    m1 = base.merge(
        horse_elo, on=keys_rt + [blend_cfg.horse_col], how="left", validate="one_to_one"
    )
    m2 = m1.merge(horse_eb, on=keys_rt + [blend_cfg.horse_col], how="left", validate="one_to_one")
    m3 = m2.merge(
        jockey_elo, on=keys_rt + [blend_cfg.jockey_col], how="left", validate="one_to_one"
    )
    merged = m3.merge(
        trainer_elo, on=keys_rt + [blend_cfg.trainer_col], how="left", validate="one_to_one"
    )

    # --- decayed z-scoring per component (no look-ahead: prior stats only)
    z_cols = ["horse_elo", "horse_eb", "jockey_elo", "trainer_elo"]
    zers = {c: _DecayedZ(blend_cfg.z_half_life_days) for c in z_cols}
    merged.sort_values([blend_cfg.ts_col, blend_cfg.race_id_col], inplace=True, kind="mergesort")

    ts_list = [cast(pd.Timestamp, t) for t in merged[blend_cfg.ts_col].tolist()]
    for c in z_cols:
        vals = merged[c].astype(float).fillna(0.0).to_numpy()
        merged[f"{c}_z"] = [zers[c].z(ts, float(x)) for ts, x in zip(ts_list, vals, strict=False)]

    # --- linear blend in z-space
    merged["rating_blend"] = (
        blend_cfg.w_horse_elo * merged["horse_elo_z"]
        + blend_cfg.w_horse_eb * merged["horse_eb_z"]
        + blend_cfg.w_jockey_elo * merged["jockey_elo_z"]
        + blend_cfg.w_trainer_elo * merged["trainer_elo_z"]
    )

    # --- persist
    out_cols = [
        blend_cfg.race_id_col,
        blend_cfg.ts_col,
        blend_cfg.horse_col,
        blend_cfg.jockey_col,
        blend_cfg.trainer_col,
        "horse_elo",
        "horse_eb",
        "jockey_elo",
        "trainer_elo",
        "horse_elo_z",
        "horse_eb_z",
        "jockey_elo_z",
        "trainer_elo_z",
        "rating_blend",
    ]
    out = merged.loc[:, out_cols].copy()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path / "ratings.parquet", index=False)

    return out
