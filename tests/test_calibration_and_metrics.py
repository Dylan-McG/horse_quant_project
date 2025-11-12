# tests/test_calibration_and_metrics.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path for direct test execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from hqp.models.calibrate import isotonic_apply, isotonic_fit
from hqp.models.dataset import grouped_time_cv
from hqp.models.logistic import train_logistic
from hqp.models.metrics import (
    accuracy_at_threshold,
    brier_score,
    check_probs_sum_to_1,
    log_loss_by_race,
    race_normalize,
)


def _toy(n_races: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    t0 = pd.Timestamp("2021-01-01")
    for r in range(n_races):
        n = int(rng.integers(4, 8))
        winner = int(rng.integers(0, n))
        dt = t0 + pd.Timedelta(days=r)
        s = rng.normal(0, 1, size=n)
        s[winner] += 0.8
        for i in range(n):
            rows.append(
                {
                    "race_id": f"R{r:03d}",
                    "race_dt": dt,
                    "is_win": 1 if i == winner else 0,
                    "x1": float(s[i] + rng.normal(0, 0.2)),
                    "x2": float(rng.normal(0, 1)),
                }
            )
    return pd.DataFrame(rows)


def test_isotonic_and_metrics_pipeline() -> None:
    df = _toy()
    X = df[["x1", "x2"]]
    y = df["is_win"]
    g = df["race_id"]
    cv = list(grouped_time_cv(df, race_id_col="race_id", ts_col="race_dt", n_splits=4))

    res = train_logistic(X, y, g, cv=cv)
    oof = np.asarray(res["oof_proba"], dtype=float)
    mask = ~np.isnan(oof)

    y_mask = np.asarray(y.values[mask], dtype=float)
    g_mask = np.asarray(g.values[mask])
    oof_mask = oof[mask]

    iso = isotonic_fit(oof_mask, y_mask)
    oof_cal = isotonic_apply(iso, oof_mask)
    oof_cal = race_normalize(oof_cal, g_mask)
    check_probs_sum_to_1(oof_cal, g_mask)

    ll = log_loss_by_race(y_mask, oof_cal, g_mask)
    br = brier_score(y_mask, oof_cal)
    acc = accuracy_at_threshold(y_mask, oof_cal, thr=0.5)

    assert np.isfinite(ll)
    assert 0.0 <= br <= 1.0
    assert 0.0 <= acc <= 1.0
