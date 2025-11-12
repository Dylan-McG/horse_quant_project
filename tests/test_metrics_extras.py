# tests/test_metrics_extras.py
from __future__ import annotations

import numpy as np
import pandas as pd

from hqp.models.dataset import grouped_time_cv, make_xy
from hqp.models.logistic import train_logistic
from hqp.models.metrics import ece, pit_basic_checks, pit_values


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for rid, dt, k in [("R1", "2021-01-01", 3), ("R2", "2021-01-02", 5), ("R3", "2021-01-03", 4)]:
        w = int(rng.integers(0, k))
        for i in range(k):
            rows.append(
                {
                    "race_id": rid,
                    "race_dt": pd.Timestamp(dt),
                    "horse_id": f"{rid}_H{i}",
                    "is_win": int(i == w),
                    "f1": float(rng.normal()),
                    "f2": float(rng.uniform()),
                }
            )
    return pd.DataFrame(rows).sort_values(["race_dt", "race_id"]).reset_index(drop=True)


def test_metrics_ece_pit() -> None:
    df = _toy_df()
    X, y = make_xy(df)
    cv = list(grouped_time_cv(df, race_id_col="race_id", ts_col="race_dt", n_splits=3))
    out = train_logistic(X, y, groups=df["race_id"], cv=cv)
    p = np.asarray(out["probs"], float)
    rids = df["race_id"].to_numpy()

    e = ece(y.to_numpy(int), p, n_bins=10, strategy="quantile")
    assert np.isfinite(e) and 0.0 <= e <= 1.0

    u = pit_values(y.to_numpy(int), p, rids)
    assert u.size > 0 and np.all((u >= 0.0) & (u <= 1.0))
    m, v = pit_basic_checks(u)
    assert np.isfinite(m) and np.isfinite(v)
