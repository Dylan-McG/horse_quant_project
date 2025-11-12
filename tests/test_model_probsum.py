# tests/test_model_probsum.py
from __future__ import annotations

import numpy as np
import pandas as pd

from hqp.models.calibrate import isotonic_apply, isotonic_fit
from hqp.models.dataset import grouped_time_cv, make_xy
from hqp.models.logistic import train_logistic
from hqp.models.metrics import check_probs_sum_to_1, race_normalize


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    for rid, dt, k in [("A", "2022-01-01", 4), ("B", "2022-01-02", 6), ("C", "2022-01-03", 3)]:
        win = int(rng.integers(0, k))
        for i in range(k):
            rows.append(
                {
                    "race_id": rid,
                    "race_dt": pd.Timestamp(dt),
                    "horse_id": f"{rid}_H{i}",
                    "is_win": int(i == win),
                    "f1": float(rng.normal()),
                    "f2": float(rng.uniform()),
                }
            )
    return pd.DataFrame(rows).sort_values(["race_dt", "race_id"]).reset_index(drop=True)


def test_probs_sum_one_before_after_calibration() -> None:
    df = _toy_df()
    X, y = make_xy(df)
    cv = list(grouped_time_cv(df, race_id_col="race_id", ts_col="race_dt", n_splits=3))
    out = train_logistic(X, y, groups=df["race_id"], cv=cv)
    probs = np.asarray(out["probs"], dtype=float)
    rids = df["race_id"].to_numpy()

    check_probs_sum_to_1(probs, rids)

    iso = isotonic_fit(probs, y.to_numpy(int))
    probs_cal = isotonic_apply(iso, probs)
    probs_cal = race_normalize(probs_cal, rids)
    check_probs_sum_to_1(probs_cal, rids)
