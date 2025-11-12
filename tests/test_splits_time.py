# tests/horse_quant/test_splits_time.py
from __future__ import annotations

import numpy as np
import pandas as pd

from hqp.split.time import SplitConfig, create_time_splits


def _toy_df(n_races: int = 10, runners_per: int = 3) -> pd.DataFrame:
    rows = []
    for ridx in range(n_races):
        race_id = f"R{ridx:02d}"
        ts = pd.Timestamp("2021-01-01") + pd.Timedelta(days=ridx)
        for h in range(runners_per):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": f"H{ridx:02d}_{h}",
                    "race_datetime": ts,
                    "course": "X",
                    "horse_name": f"N{h}",
                }
            )
    return pd.DataFrame(rows)


def test_single_cutoff_covers_all_once_and_monotonic() -> None:
    df = _toy_df(n_races=12, runners_per=2)
    cfg: SplitConfig = {
        "strategy": "single_cutoff",
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "output_dir": "data/splits/test_single",
    }
    masks = create_time_splits(df, cfg)

    # coverage exactly once
    total = masks["train"].astype(int) + masks["valid"].astype(int) + masks["test"].astype(int)
    assert int(total.max()) == 1
    assert int(total.sum()) == len(df)

    # monotonicity: min(valid_ts) >= max(train_ts) if both non-empty
    if masks["train"].any() and masks["valid"].any():
        gmin = df.loc[masks["valid"], "race_datetime"].min()
        gmax = df.loc[masks["train"], "race_datetime"].max()
        assert gmin >= gmax
    if masks["valid"].any() and masks["test"].any():
        gmin = df.loc[masks["test"], "race_datetime"].min()
        gmax = df.loc[masks["valid"], "race_datetime"].max()
        assert gmin >= gmax

    # group integrity
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ra = set(df.loc[masks[a], "race_id"])
        rb = set(df.loc[masks[b], "race_id"])
        assert ra.isdisjoint(rb)


def test_kfold_group_integrity_and_coverage() -> None:
    df = _toy_df(n_races=9, runners_per=4)
    cfg: SplitConfig = {"strategy": "kfold", "k_folds": 3, "output_dir": "data/splits/test_kfold"}
    masks = create_time_splits(df, cfg)

    # validation folds cover all rows once, no overlap
    valid_sum = pd.Series(0, index=df.index, dtype="int64")
    for i in range(3):
        vt = masks[f"fold{i}_valid"].astype(int)
        valid_sum = valid_sum + vt
        # group integrity within fold
        ra = set(df.loc[masks[f"fold{i}_train"], "race_id"])
        rb = set(df.loc[masks[f"fold{i}_valid"], "race_id"])
        assert ra.isdisjoint(rb)
        # monotonicity
        if masks[f"fold{i}_train"].any():
            gmin = df.loc[masks[f"fold{i}_valid"], "race_datetime"].min()
            gmax = df.loc[masks[f"fold{i}_train"], "race_datetime"].max()
            assert gmin >= gmax
    assert int(valid_sum.max()) == 1
    assert int(valid_sum.sum()) == len(df)


def test_reproducibility_same_masks() -> None:
    df = _toy_df(n_races=8, runners_per=2)
    cfg: SplitConfig = {"strategy": "kfold", "k_folds": 4, "output_dir": "data/splits/test_repro"}
    m1 = create_time_splits(df, cfg)
    m2 = create_time_splits(df, cfg)
    # identical booleans
    assert set(m1.keys()) == set(m2.keys())
    for k in m1:
        assert np.array_equal(m1[k].to_numpy(), m2[k].to_numpy())
