# src/hqp/ratings/elo.py
# -----------------------------------------------------------------------------
# Elo-like per-entity rating with:
#   - exponential decay toward a mean rating (time half-life),
#   - K scaling with field size:  K = k_base * sqrt(field_size / k_field_ref),
#   - snapshot PRIOR strictly before using the current event outcome/score.
#
# Score model:
#   If rank is provided among fs runners, we map rank -> score in [0,1] linearly:
#     score = 1 - (rank - 1)/(fs - 1)   (winner=1, last=0).
#   Alternatively, a precomputed score in [0,1] can be supplied via score_col.
#
# Update:
#   rating_t = rating_{t^-} + K * (score - expected),  with expected=0.5 baseline.
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional
import math

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    half_life_days: float = 90.0  # decay toward mean_rating over time
    k_base: float = 16.0  # base learning rate
    k_field_ref: float = 8.0  # reference field size for K scaling
    mean_rating: float = 0.0  # long-run mean


@dataclass
class _EloState:
    rating: float
    last_ts: pd.Timestamp


class EloRater:
    """Online Elo-like rater with leak-free prior snapshots."""

    def __init__(self, cfg: EloConfig | None = None) -> None:
        self.cfg: EloConfig = cfg or EloConfig()
        self._state: dict[int, _EloState] = {}

    @staticmethod
    def _days_between(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
        return float((t2 - t1).total_seconds()) / 86400.0

    def _decay_factor(self, delta_days: float) -> float:
        if self.cfg.half_life_days <= 0:
            return 1.0
        return float(2.0 ** (-max(delta_days, 0.0) / self.cfg.half_life_days))

    def _ensure_entity(self, ent: int, ts: pd.Timestamp) -> None:
        if ent not in self._state:
            self._state[ent] = _EloState(rating=float(self.cfg.mean_rating), last_ts=ts)

    @staticmethod
    def _score_from_rank(rank: int, field_size: int) -> float:
        fs = max(1, int(field_size))
        if fs == 1:
            return 1.0
        r = max(1, int(rank))
        # winner=1.0, last=0.0 linearly in finishing position
        return float(1.0 - (r - 1) / (fs - 1)) if fs > 1 else 1.0

    def rate(
        self,
        events: pd.DataFrame,
        *,
        entity_col: str,
        ts_col: str,
        race_id_col: str,
        field_size_col: str,
        rank_col: Optional[str] = "rank",
        score_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns [race_id_col, ts_col, entity_col, elo_prior].
        """
        req = {entity_col, ts_col, race_id_col, field_size_col}
        if rank_col is None and score_col is None:
            raise ValueError("Provide rank_col or score_col.")
        if rank_col is not None:
            req.add(rank_col)
        if score_col is not None:
            req.add(score_col)

        missing = req - set(map(str, events.columns))
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        df = events.copy()
        # Coerce timestamps to UTC, deterministic ordering
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if df[ts_col].isna().any():
            raise ValueError("Elo: some timestamps could not be parsed to datetime.")
        df.sort_values([ts_col, race_id_col], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        # Prepare score series
        if score_col is not None:
            score_vals: pd.Series = (
                pd.to_numeric(df[score_col], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
                .astype("float64")
            )
        else:
            ranks_np: np.ndarray = pd.to_numeric(df[rank_col or "rank"], errors="coerce").to_numpy(
                dtype="float64"
            )
            fs_np: np.ndarray = (
                pd.to_numeric(df[field_size_col], errors="coerce").fillna(0).to_numpy(dtype="int64")
            )
            n = len(df)
            scores_arr: np.ndarray = np.zeros(n, dtype="float64")
            for pos in range(n):
                r_val = ranks_np[pos]
                f_val = int(fs_np[pos])
                if not np.isfinite(r_val) or f_val <= 0:
                    scores_arr[pos] = 0.0
                else:
                    scores_arr[pos] = self._score_from_rank(int(r_val), f_val)
            score_vals = pd.Series(data=scores_arr, index=df.index, dtype="float64")

        # Pre-extract typed arrays (avoid Pylance 'Unknown' by using NumPy)
        ts_ser: pd.Series = df[ts_col]
        ent_np: np.ndarray = (
            pd.to_numeric(df[entity_col], errors="coerce").fillna(0).to_numpy(dtype="int64")
        )
        race_id_ser: pd.Series = df[race_id_col]
        field_np: np.ndarray = (
            pd.to_numeric(df[field_size_col], errors="coerce").fillna(0).to_numpy(dtype="int64")
        )
        score_arr: np.ndarray = score_vals.to_numpy(dtype="float64", copy=False)

        out_rows: list[dict[str, object]] = []
        nrows = len(df)

        for pos in range(nrows):
            ts_val: pd.Timestamp = cast(pd.Timestamp, ts_ser.iat[pos])
            ent_val: int = int(ent_np[pos])
            race_id_val: object = race_id_ser.iat[pos]
            field_size_val: int = int(field_np[pos])
            # Pull from typed NumPy array instead of Series.iat to satisfy Pylance
            score_val: float = float(score_arr[pos])

            # init/decay
            self._ensure_entity(ent_val, ts_val)
            st = self._state[ent_val]

            lam: float = self._decay_factor(self._days_between(st.last_ts, ts_val))
            st.rating = float(self.cfg.mean_rating + (st.rating - self.cfg.mean_rating) * lam)
            st.last_ts = ts_val

            # snapshot
            out_rows.append(
                {
                    race_id_col: race_id_val,
                    ts_col: ts_val,
                    entity_col: ent_val,
                    "elo_prior": float(st.rating),
                }
            )

            # update
            k_base: float = float(self.cfg.k_base)
            field_ref: float = float(self.cfg.k_field_ref) if self.cfg.k_field_ref > 0 else 1.0
            ratio: float = field_size_val / field_ref
            if ratio < 1.0:
                ratio = 1.0
            k: float = k_base * math.sqrt(ratio)

            expected: float = 0.5
            st.rating = float(st.rating + k * (score_val - expected))

        return pd.DataFrame(out_rows)
