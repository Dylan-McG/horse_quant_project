# tests/horse_quant/test_checks_guards.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hqp.checks.guards import (
    ensure_chrono_no_leak,
    ensure_no_current_obs,
    ensure_prob_sums,
    list_current_obs,
)


def test_list_and_ensure_no_current_obs() -> None:
    cols = ["hist__speed", "rat__rpr", "obs__draw", "obs__going", "context__field"]
    offenders = list_current_obs(cols)
    assert set(offenders) == {"obs__draw", "obs__going"}
    with pytest.raises(ValueError):
        ensure_no_current_obs(cols)

    # Safe when no offenders
    ensure_no_current_obs(["hist__speed", "rat__rpr"])


def test_ensure_prob_sums_pass_and_fail() -> None:
    df_ok = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 2, 2],
            "p_win": [0.5, 0.3, 0.2, 0.7, 0.3],
        }
    )
    ensure_prob_sums(df_ok, race_col="race_id", prob_col="p_win", atol=1e-9)

    df_bad = pd.DataFrame(
        {
            "race_id": [10, 10, 10, 11, 11],
            "p_win": [0.5, 0.5, 0.1, 0.4, 0.4],  # sums: 1.1 and 0.8
        }
    )
    with pytest.raises(ValueError):
        ensure_prob_sums(df_bad, race_col="race_id", prob_col="p_win", atol=1e-9)


def test_ensure_chrono_no_leak_ok_and_gap() -> None:
    ts = pd.date_range("2023-01-01 12:00:00Z", periods=6, freq="30min")
    df = pd.DataFrame({"off_time": ts, "race_id": [1, 1, 2, 2, 3, 3]})

    train_mask = np.array([True, True, True, True, False, False], dtype=bool)
    test_mask = np.array([False, False, False, False, True, True], dtype=bool)

    ensure_chrono_no_leak(df, "off_time", train_mask, test_mask)
    ensure_chrono_no_leak(df, "off_time", train_mask, test_mask, gap=pd.Timedelta(minutes=0))
    ensure_chrono_no_leak(df, "off_time", train_mask, test_mask, gap=pd.Timedelta(minutes=30))

    with pytest.raises(AssertionError):
        ensure_chrono_no_leak(df, "off_time", train_mask, test_mask, gap=pd.Timedelta(hours=1))


def test_ensure_chrono_no_leak_violates_forward_order() -> None:
    ts = pd.date_range("2023-02-01 10:00:00Z", periods=4, freq="15min")
    df = pd.DataFrame({"off_time": ts})
    train_mask = np.array([True, False, True, False], dtype=bool)  # t0, t2
    test_mask = np.array([False, True, False, True], dtype=bool)  # t1, t3
    with pytest.raises(AssertionError):
        ensure_chrono_no_leak(df, "off_time", train_mask, test_mask)
