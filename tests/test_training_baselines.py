# tests/test_training_baselines.py
# tests/test_training_baselines.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path for direct test execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from hqp.models.dataset import grouped_time_cv
from hqp.models.logistic import train_logistic

# Mark as used for Pylance when test collection is partial
assert train_logistic is not None


@pytest.mark.skipif(
    pytest.importorskip("lightgbm", reason="lightgbm not installed") is None, reason="no lgbm"
)
def test_lgbm_training_oof_and_prob_sum() -> None:
    from hqp.models.train_lgbm import train_lgbm

    # toy data
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []
    t0 = pd.Timestamp("2022-01-01")
    for r in range(10):
        n = int(rng.integers(5, 9))
        winner = int(rng.integers(0, n))
        dt = t0 + pd.Timedelta(days=r)
        base = rng.normal(0, 1, size=n)
        base[winner] += 1.0
        for i in range(n):
            rows.append(
                {
                    "race_id": f"R{r:03d}",
                    "horse_id": f"H{r:03d}_{i:02d}",
                    "race_dt": dt,
                    "is_win": 1 if i == winner else 0,
                    "f1": float(base[i] + rng.normal(0, 0.3)),
                    "f2": float(rng.normal(0, 1)),
                }
            )
    df = pd.DataFrame(rows)

    X = df[["f1", "f2"]]
    y = df["is_win"]
    groups = df["race_id"]
    cv = list(grouped_time_cv(df, race_id_col="race_id", ts_col="race_dt", n_splits=5))

    params = {
        # keep deterministic for test
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": 1,
        # optional: try early stopping; should still be deterministic
        "early_stopping_rounds": 20,
    }

    res = train_lgbm(X, y, groups, params=params, cv=cv)
    oof = np.asarray(res["oof_proba"], dtype=float)
    assert isinstance(oof, np.ndarray)
    assert oof.shape == (len(df),)

    mask_scored = ~np.isnan(oof)
    for rid in df.loc[mask_scored, "race_id"].unique():
        idx = (df["race_id"] == rid) & mask_scored
        s = float(np.nansum(oof[idx]))
        assert abs(s - 1.0) < 1e-9

    # determinism check (mean OOF close)
    res2 = train_lgbm(X, y, groups, params=params, cv=cv)
    oof2 = np.asarray(res2["oof_proba"], dtype=float)
    assert np.isclose(float(np.nanmean(oof)), float(np.nanmean(oof2)), atol=1e-9)
