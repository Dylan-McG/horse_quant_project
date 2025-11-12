# test_market_normalize.py

import numpy as np
import pandas as pd

from hqp.market.normalise import implied_raw, normalize_per_race


def _sum_by_group(vals: pd.Series, groups: pd.Series, gid: object) -> float:
    m = groups == gid
    x = np.asarray(vals[m], float)
    x = x[np.isfinite(x)]
    return float(x.sum()) if x.size else float("nan")


def test_implied_raw_and_normalize_simple_cases() -> None:
    # two races
    df = pd.DataFrame(
        {
            "race_id": ["A", "A", "A", "B", "B", "B"],
            # already probabilities (we’ll directly place into prob col below)
            # but also check implied_raw from decimal odds quickly
            "decimal_odds": [2.0, 2.5, 10.0, 1.9230769, 2.0833333, np.nan],
            # raw probs we want to normalize within race (simulate overround already ~1)
            "raw": [0.5, 0.4, 0.1, 0.52, 0.48, np.nan],
        }
    )

    # quick implied_raw check
    imp = implied_raw(df["decimal_odds"])
    assert np.isclose(imp.iloc[0], 0.5)  # 1/2
    assert np.isfinite(imp.iloc[1]) and imp.iloc[1] > 0.0
    assert np.isnan(imp.iloc[5])

    # now normalize the 'raw' column per race
    df2 = df.rename(columns={"raw": "mkt_implied_raw"})
    df2 = normalize_per_race(
        df2, prob_col="mkt_implied_raw", group_col="race_id", out_col="mkt_implied"
    )

    # Race A: sums to 1 (0.5+0.4+0.1)
    sA = _sum_by_group(df2["mkt_implied"], df2["race_id"], "A")
    assert np.isfinite(sA) and abs(sA - 1.0) <= 1e-9

    # Race B: sums to 1 over non-NaNs (0.52 + 0.48)
    sB = _sum_by_group(df2["mkt_implied"], df2["race_id"], "B")
    assert np.isfinite(sB) and abs(sB - 1.0) <= 1e-9

    # NaN remains NaN
    mask_nan = (df2["race_id"] == "B") & df2["mkt_implied_raw"].isna()
    assert df2.loc[mask_nan, "mkt_implied"].isna().all()
