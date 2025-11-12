# tests/test_market_parsing.py

import numpy as np
import pandas as pd

from hqp.market.parse import (
    coalesce_decimal_fractional,
    parse_decimal_series,
    parse_fractional_series,
)


def test_parse_decimal_series_basic() -> None:
    s = pd.Series(["2.5", "EVS", "evens", "EVEN", "SP", "1.0", "0.9", "3"])
    out = parse_decimal_series(s)

    # valid
    assert np.isclose(out.iloc[0], 2.5)
    assert np.isclose(out.iloc[1], 2.0)
    assert np.isclose(out.iloc[2], 2.0)
    assert np.isclose(out.iloc[3], 2.0)
    assert np.isclose(out.iloc[7], 3.0)

    # invalid/NaN
    assert np.isnan(out.iloc[4])  # "SP"
    assert np.isnan(out.iloc[5])  # 1.0
    assert np.isnan(out.iloc[6])  # 0.9

    assert out.dtype == "float64"


def test_parse_fractional_series_basic() -> None:
    s = pd.Series(["7/2", "100/30", "evs", "EVEN", "bad", "2/0"])
    out = parse_fractional_series(s)

    # 7/2 -> 1 + 7/2 = 4.5
    assert np.isclose(out.iloc[0], 4.5)
    # 100/30 -> 1 + (100/30) ~= 4.33333...
    assert np.isclose(out.iloc[1], 1.0 + 100 / 30, rtol=1e-12, atol=1e-12)
    # evens variants -> 2.0
    assert np.isclose(out.iloc[2], 2.0)
    assert np.isclose(out.iloc[3], 2.0)
    # invalid -> NaN
    assert np.isnan(out.iloc[4])
    assert np.isnan(out.iloc[5])

    assert out.dtype == "float64"


def test_coalesce_decimal_fractional_prefers_decimal() -> None:
    dec = pd.Series(["SP", "2.2", None, "EVS"])
    frac = pd.Series(["7/2", "3/1", "100/30", "even"])

    out = coalesce_decimal_fractional(dec, frac)

    # row0: decimal NaN -> use fractional 7/2 = 4.5
    assert np.isclose(out.iloc[0], 4.5)
    # row1: decimal present 2.2 -> prefer this
    assert np.isclose(out.iloc[1], 2.2)
    # row2: decimal NaN -> fractional 100/30
    assert np.isclose(out.iloc[2], 1.0 + 100 / 30, rtol=1e-12, atol=1e-12)
    # row3: decimal EVS -> 2.0 (already parsed inside helper)
    assert np.isclose(out.iloc[3], 2.0)

    assert out.dtype == "float64"
