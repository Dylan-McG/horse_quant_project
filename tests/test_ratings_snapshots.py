# tests/test_ratings_snapshots.py
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

# Ensure src/ is importable (src-layout project)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from hqp.ratings.blend import build_blended_ratings


def _toy_events() -> pd.DataFrame:
    rows = [
        # day 1
        {
            "race_id": 1,
            "ts": "2024-01-01 14:00:00Z",
            "horse_id": 101,
            "jockey_id": 201,
            "trainer_id": 301,
            "field_size": 3,
            "rank": 1,
            "win": 1,
        },
        {
            "race_id": 1,
            "ts": "2024-01-01 14:00:00Z",
            "horse_id": 102,
            "jockey_id": 202,
            "trainer_id": 302,
            "field_size": 3,
            "rank": 2,
            "win": 0,
        },
        {
            "race_id": 1,
            "ts": "2024-01-01 14:00:00Z",
            "horse_id": 103,
            "jockey_id": 203,
            "trainer_id": 303,
            "field_size": 3,
            "rank": 3,
            "win": 0,
        },
        # day 2
        {
            "race_id": 2,
            "ts": "2024-01-02 14:00:00Z",
            "horse_id": 101,
            "jockey_id": 201,
            "trainer_id": 301,
            "field_size": 3,
            "rank": 2,
            "win": 0,
        },
        {
            "race_id": 2,
            "ts": "2024-01-02 14:00:00Z",
            "horse_id": 102,
            "jockey_id": 202,
            "trainer_id": 302,
            "field_size": 3,
            "rank": 1,
            "win": 1,
        },
        {
            "race_id": 2,
            "ts": "2024-01-02 14:00:00Z",
            "horse_id": 103,
            "jockey_id": 203,
            "trainer_id": 303,
            "field_size": 3,
            "rank": 3,
            "win": 0,
        },
        # day 3
        {
            "race_id": 3,
            "ts": "2024-01-03 14:00:00Z",
            "horse_id": 101,
            "jockey_id": 201,
            "trainer_id": 301,
            "field_size": 3,
            "rank": 3,
            "win": 0,
        },
        {
            "race_id": 3,
            "ts": "2024-01-03 14:00:00Z",
            "horse_id": 102,
            "jockey_id": 202,
            "trainer_id": 302,
            "field_size": 3,
            "rank": 2,
            "win": 0,
        },
        {
            "race_id": 3,
            "ts": "2024-01-03 14:00:00Z",
            "horse_id": 103,
            "jockey_id": 203,
            "trainer_id": 303,
            "field_size": 3,
            "rank": 1,
            "win": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_snapshots_no_leakage_and_time_order(tmp_path: Path) -> None:
    events = _toy_events()

    out = build_blended_ratings(events, config_yaml=None, out_dir=tmp_path.as_posix())

    # First-ever horse appearances => elo prior at mean 0.0
    r1 = out[out["race_id"] == 1].sort_values("horse_id")
    assert np.allclose(r1["horse_elo"].to_numpy(dtype=float), np.array([0.0, 0.0, 0.0]), atol=1e-12)

    # After day 1 winner=101, its day-2 prior should be > 0 BEFORE day-2 update
    h101_d2_prior = float(
        out.loc[(out["race_id"] == 2) & (out["horse_id"] == 101), "horse_elo"].to_numpy(
            dtype=float
        )[0]
    )
    assert h101_d2_prior > 0.0

    # Last place day 1 => negative prior on day 2
    h103_d2_prior = float(
        out.loc[(out["race_id"] == 2) & (out["horse_id"] == 103), "horse_elo"].to_numpy(
            dtype=float
        )[0]
    )
    assert h103_d2_prior < 0.0

    # The day-2 winner (102) shouldn't be boosted yet in its PRIOR
    h102_d2_prior = float(
        out.loc[(out["race_id"] == 2) & (out["horse_id"] == 102), "horse_elo"].to_numpy(
            dtype=float
        )[0]
    )
    assert h102_d2_prior <= 0.0 or np.isclose(h102_d2_prior, 0.0, atol=1e-6)

    # Deterministic: same input twice => identical output
    out2 = build_blended_ratings(events, config_yaml=None, out_dir=tmp_path.as_posix())
    pd.testing.assert_frame_equal(
        out.sort_values(["race_id", "horse_id"]).reset_index(drop=True),
        out2.sort_values(["race_id", "horse_id"]).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )

    # Column presence
    expected_cols = {
        "race_id",
        "ts",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "horse_elo",
        "horse_eb",
        "jockey_elo",
        "trainer_elo",
        "horse_elo_z",
        "horse_eb_z",
        "jockey_elo_z",
        "trainer_elo_z",
        "rating_blend",
    }
    assert expected_cols.issubset(set(out.columns))

    # Parquet written
    assert (tmp_path / "ratings.parquet").exists()
