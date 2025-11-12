# src/hqp/ratings/eb.py
# -----------------------------------------------------------------------------
# Empirical-Bayes decayed success-rate rater (Beta-Binomial).
#
# Model intuition:
#   y_i ~ Bernoulli(p),   p ~ Beta(a0, b0),  with time-decayed sufficient stats.
#   Posterior mean:  E[p | data] = (a0 + S) / (a0 + b0 + T),
#   where S = decayed successes, T = decayed trials.
#
# We snapshot the PRIOR for the current event (i.e., using stats up to t^-),
# then update the state with the current outcome—ensuring no look-ahead.
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass
class EBConfig:
    half_life_days: float = 120.0  # forgetting in days (S,T decay toward 0)
    prior_strength: float = 20.0  # a0 + b0 (equivalent sample size)
    baseline_rate: float | None = None  # if None, inferred as empirical mean
    mean_center: bool = False  # if True, return posterior - baseline


@dataclass
class _EBState:
    succ: float  # decayed successes
    trial: float  # decayed trials
    last_ts: pd.Timestamp  # timestamp of last update (UTC)


class EmpiricalBayesRater:
    """
    Online Beta–Binomial EB rater with exponential forgetting and as-of snapshots.

    Leak-safety: for each row, we (1) decay previous sufficient statistics to t^-,
    (2) compute the prior/posterior mean using only those stats, then (3) update
    with the current outcome y_t. This guarantees no look-ahead.
    """

    def __init__(self, cfg: EBConfig | None = None) -> None:
        self.cfg: EBConfig = cfg or EBConfig()
        self._state: dict[int, _EBState] = {}
        self._baseline_rate_final: float | None = None

    @staticmethod
    def _days_between(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
        return float((t2 - t1).total_seconds()) / 86400.0

    def _decay_factor(self, delta_days: float) -> float:
        if self.cfg.half_life_days <= 0:
            return 1.0
        return float(2.0 ** (-max(delta_days, 0.0) / self.cfg.half_life_days))

    def _ensure_entity(self, ent: int, ts: pd.Timestamp) -> None:
        if ent not in self._state:
            self._state[ent] = _EBState(succ=0.0, trial=0.0, last_ts=ts)

    def _prior(self) -> tuple[float, float]:
        """
        Construct (a0, b0) given baseline rate r and prior strength n0:
          a0 = n0 * r,   b0 = n0 * (1 - r).
        """
        if self._baseline_rate_final is None:
            r = self.cfg.baseline_rate if self.cfg.baseline_rate is not None else 0.1
        else:
            r = self._baseline_rate_final
        # numeric safety
        r = float(np.clip(r, 1e-9, 1 - 1e-9))
        n0 = float(max(self.cfg.prior_strength, 1e-9))
        a0 = n0 * r
        b0 = n0 * (1.0 - r)
        return a0, b0

    def rate(
        self,
        events: pd.DataFrame,
        *,
        entity_col: str,
        ts_col: str,
        race_id_col: str,
        success_col: str = "win",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        events : pd.DataFrame
            Must contain columns: [entity_col, ts_col, race_id_col, success_col].
            success_col must be in {0,1}.
        Returns
        -------
        Snapshot table with columns [race_id_col, ts_col, entity_col, eb_prior].
        """
        req = {entity_col, ts_col, race_id_col, success_col}
        missing = req - set(map(str, events.columns))
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        df = events.copy()
        # Coerce to UTC (tz-aware) for consistent diffs; stable ordering
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if df[ts_col].isna().any():
            raise ValueError("EB: some timestamps could not be parsed to datetime.")
        df.sort_values([ts_col, race_id_col], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        # Guard success values; coerce to {0,1}
        s = pd.to_numeric(df[success_col], errors="coerce").fillna(0.0)
        s = s.clip(0.0, 1.0)
        df[success_col] = s

        # If not provided, infer baseline as empirical mean win-rate (global).
        if self.cfg.baseline_rate is None:
            trials = float(len(df))
            successes = float(s.sum())
            self._baseline_rate_final = successes / trials if trials > 0.0 else 0.1
        else:
            self._baseline_rate_final = float(self.cfg.baseline_rate)

        out_rows: list[dict[str, object]] = []

        for _, row in df.iterrows():
            ts = cast(pd.Timestamp, row[ts_col])
            ent = int(row[entity_col])
            race_id = row[race_id_col]
            y = float(row[success_col])  # expects 0/1

            self._ensure_entity(ent, ts)
            st = self._state[ent]

            # Decay sufficient stats since last update
            lam = self._decay_factor(self._days_between(st.last_ts, ts))
            st.succ *= lam
            st.trial *= lam
            st.last_ts = ts

            # Snapshot PRIOR (posterior mean using stats up to t^-)
            a0, b0 = self._prior()
            # Use beta parameters as: alpha = a0 + successes, beta = b0 + failures
            alpha = a0 + st.succ
            beta = b0 + max(0.0, st.trial - st.succ)
            den = alpha + beta
            mean = float(alpha / den) if den > 1e-12 else float(a0 / (a0 + b0))
            val = mean - float(self._baseline_rate_final) if self.cfg.mean_center else mean

            out_rows.append(
                {race_id_col: race_id, ts_col: ts, entity_col: ent, "eb_prior": float(val)}
            )

            # Update AFTER snapshot
            st.succ += 1.0 if y >= 0.5 else 0.0
            st.trial += 1.0

        return pd.DataFrame(out_rows)
