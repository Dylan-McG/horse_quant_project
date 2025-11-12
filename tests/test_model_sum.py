# tests/test_model_sum.py
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hqp.models.calibrate import isotonic_apply, isotonic_fit
from hqp.models.dataset import grouped_time_cv, make_xy
from hqp.models.logistic import train_logistic
from hqp.models.metrics import check_probs_sum_to_1, race_normalize

Fold = tuple[NDArray[np.int64], NDArray[np.int64]]


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for rid, dt, k in [("R1", "2021-01-01", 3), ("R2", "2021-01-02", 5), ("R3", "2021-01-03", 2)]:
        winner_idx = int(rng.integers(0, k))
        for i in range(k):
            rows.append(
                {
                    "race_id": rid,
                    "race_dt": pd.Timestamp(dt),
                    "horse_id": f"{rid}_H{i}",
                    "is_win": int(i == winner_idx),
                    "f_speed": float(rng.normal()),
                    "f_draw": int(rng.integers(1, 12)),
                }
            )
    return pd.DataFrame(rows).sort_values(["race_dt", "race_id"]).reset_index(drop=True)


def _materialize_cv(df: pd.DataFrame) -> list[Fold]:
    cv: list[Fold] = []
    # IMPORTANT: these are keyword-only params (because of "*," in the function)
    for tr, va in grouped_time_cv(df, race_id_col="race_id", ts_col="race_dt", n_splits=3):
        cv.append((tr, va))  # already NDArray[np.int64] from the function
    return cv


def test_sum_1_after_training_and_calibration() -> None:
    df = _toy_df()
    X, y = make_xy(df, target_col="is_win")
    groups = df["race_id"]

    cv = _materialize_cv(df)

    out = train_logistic(X, y, groups=groups, cv=cv)
    probs = np.asarray(out["probs"], dtype=float)
    race_ids = df["race_id"].to_numpy()

    check_probs_sum_to_1(probs, race_ids)

    iso = isotonic_fit(probs, y.to_numpy(dtype=int))
    probs_cal = isotonic_apply(iso, probs)
    probs_cal = race_normalize(probs_cal, race_ids)
    check_probs_sum_to_1(probs_cal, race_ids)


def test_grouped_time_cv_chronology() -> None:
    df = _toy_df()
    cv = _materialize_cv(df)
    dt = df["race_dt"].to_numpy()
    for tr, va in cv:
        assert dt[tr].max() < dt[va].min()
