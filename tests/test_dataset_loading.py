# tests/test_dataset_loading.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path for direct test execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from hqp.models.dataset import grouped_time_cv, load_features, make_xy


def _toy_features(n_races: int = 6, rng_seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    t0 = pd.Timestamp("2022-01-01")
    for r in range(n_races):
        n = int(rng.integers(4, 8))
        winner = int(rng.integers(0, n))
        race_time = t0 + pd.Timedelta(days=int(r))
        for i in range(n):
            rows.append(
                {
                    "race_id": f"R{r:03d}",
                    "horse_id": f"H{r:03d}_{i:02d}",
                    "race_dt": race_time,
                    "pos": 1 if i == winner else int(rng.integers(2, 12)),
                    "form_score": float(rng.normal(0, 1)),
                    "ctx_speed": float(rng.normal(0, 1)),
                    "obs__leak": float(rng.normal(0, 1)),
                }
            )
    return pd.DataFrame(rows)


def test_load_features_derives_target_and_blocks_forbidden(tmp_path: Path) -> None:
    df = _toy_features()
    fpath = tmp_path / "features.parquet"
    df.to_parquet(fpath)

    # Should raise due to forbidden column 'obs__leak'
    raised = False
    try:
        _ = load_features(fpath)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for forbidden columns"

    # Remove forbidden and try again
    df2 = df.drop(columns=["obs__leak"])
    df2.to_parquet(fpath)
    loaded = load_features(fpath)
    assert "is_win" in loaded.columns
    assert loaded["is_win"].isin([0, 1]).all()
    assert loaded["race_dt"].dtype.kind == "M"  # datetime64


def test_make_xy_numeric_only(tmp_path: Path) -> None:
    df = _toy_features().drop(columns=["obs__leak"])
    fpath = tmp_path / "features2.parquet"
    df.to_parquet(fpath)

    loaded = load_features(fpath)
    X, y = make_xy(loaded, target_col="is_win", drop_cols=("race_id", "horse_id"))

    assert len(X) == len(loaded)
    assert len(y) == len(loaded)
    assert y.name == "is_win"
    assert "race_id" not in X.columns
    assert "horse_id" not in X.columns
    assert "is_win" not in X.columns
    for c in X.columns:
        assert pd.api.types.is_numeric_dtype(X[c])


def test_grouped_time_cv_is_chronological(tmp_path: Path) -> None:
    df = _toy_features(n_races=8).drop(columns=["obs__leak"])
    fpath = tmp_path / "features3.parquet"
    df.to_parquet(fpath)
    loaded = load_features(fpath)

    folds = list(grouped_time_cv(loaded, race_id_col="race_id", ts_col="race_dt", n_splits=4))
    assert len(folds) == 4
    for tr_idx, va_idx in folds:
        if len(tr_idx) and len(va_idx):
            t_train_max = loaded.loc[tr_idx, "race_dt"].max()
            t_valid_min = loaded.loc[va_idx, "race_dt"].min()
            assert t_train_max <= t_valid_min
