# src/hqp/analysis/ages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ._io import ensure_run_dir, safe_write_parquet

PathLike = Union[str, Path]
T = TypeVar("T")  # DataFrame or Series


def _coerce_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'age' is numeric Int64 and drop rows with missing or invalid age.
    """
    if "age" not in df.columns:
        raise ValueError("Missing column: 'age'")
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["age"])
    return out


def _ensure_won(df: pd.DataFrame, win_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure a binary 'won' column exists.

    Priority:
      1) If win_col is provided, coerce it to {0,1}
      2) Existing boolean-ish winners: ['won','is_winner','winner','won_flag','obs__is_winner']
      3) Position-based numeric winners: ['finish_position','position','pos','finpos','placing','obs__uposition']
         -> won = (value == 1)
      4) String-based positions/results (e.g., '1st','Winner','1') mapped to won=1
    """
    out = df.copy()

    def coerce_boolish(col: str) -> Optional[pd.Series]:
        if col not in out.columns:
            return None
        s: pd.Series = out[col]
        s_num: pd.Series = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            s_num = s_num.fillna(0)
            return (s_num != 0).astype(int)
        s_str: pd.Series = s.astype(str).str.strip().str.lower()
        truthy = {"true", "yes", "y", "winner", "win", "won", "1"}
        falsy = {"false", "no", "n", "loser", "lose", "lost", "0"}
        known_mask: pd.Series = s_str.isin(truthy.union(falsy))
        if known_mask.any():
            return s_str.isin(truthy).astype(int)
        return None

    def from_position_numeric(col: str) -> Optional[pd.Series]:
        if col not in out.columns:
            return None
        s: pd.Series = out[col]
        s_num: pd.Series = pd.to_numeric(s, errors="coerce")
        if not s_num.notna().any():
            return None
        return (s_num == 1).astype("Int64").fillna(0).astype(int)

    def from_position_string(col: str) -> Optional[pd.Series]:
        if col not in out.columns:
            return None
        s: pd.Series = out[col].astype(str).str.strip().str.lower()
        won_tokens = {"1", "1st", "first", "winner", "won"}
        digits: pd.Series = s.str.extract(r"^\s*(\d+)", expand=False)
        is_one: pd.Series = digits == "1"
        is_token: pd.Series = s.isin({t.lower() for t in won_tokens})
        won_series: pd.Series = (is_one | is_token).astype(int)
        if (won_series == 0).all():
            return None
        return won_series

    # 1) explicit
    if win_col:
        if win_col not in out.columns:
            raise ValueError(
                f"Specified win_col='{win_col}' not found. Available: {list(out.columns)}"
            )
        cand = coerce_boolish(win_col)
        if cand is None:
            cand = from_position_numeric(win_col)
        if cand is None:
            cand = from_position_string(win_col)
        if cand is not None:
            out["won"] = cand
            return out
        raise ValueError(
            f"Could not coerce win_col '{win_col}' into binary 0/1. "
            "Provide a different column or precompute 'won'."
        )

    # 2) boolean-ish direct
    for col in ["won", "is_winner", "winner", "won_flag", "obs__is_winner"]:
        cand = coerce_boolish(col)
        if cand is not None:
            out["won"] = cand
            return out

    # 3) numeric positions
    for col in ["finish_position", "position", "pos", "finpos", "placing", "obs__uposition"]:
        cand = from_position_numeric(col)
        if cand is not None:
            out["won"] = cand
            return out

    # 4) string positions/results
    for col in [
        "finish_position",
        "position",
        "pos",
        "result",
        "outcome",
        "placing",
        "finish",
        "obs__uposition",
    ]:
        cand = from_position_string(col)
        if cand is not None:
            out["won"] = cand
            return out

    raise ValueError(
        "Could not infer a binary 'won' column. "
        "Please specify 'win_col=' (e.g., obs__is_winner / obs__uposition) or add a numeric 0/1 'won'. "
        f"Available columns: {list(out.columns)}"
    )


# ---- Tiny helper to bypass strict pandas-stubs on sort_values ----
def _sort(df: T, *, by: str, ascending: bool) -> T:
    """
    Wrapper around .sort_values that preserves the input type (DataFrame or Series)
    while bypassing pyright/pylance call-overload issues with pandas-stubs.
    """
    df_any = cast(Any, df)
    return cast(T, df_any.sort_values(by=by, ascending=ascending))


# ---- Stats helpers ---------------------------------------------------------
def _wilson_interval(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """
    Wilson score interval for binomial proportion (vectorized).
    k = wins, n = trials
    """
    k = pd.to_numeric(k, errors="coerce")
    n = pd.to_numeric(n, errors="coerce").replace(0, np.nan)
    p = (k / n).astype(float)
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    term = (p * (1.0 - p) + (z * z) / (4.0 * n)) / n
    margin = z * np.sqrt(term) / denom
    lo = (centre - margin).clip(0.0, 1.0).fillna(0.0)
    hi = (centre + margin).clip(0.0, 1.0).fillna(0.0)
    return lo, hi


def _beta_shrink_rate(k: pd.Series, n: pd.Series, a: float = 0.5, b: float = 0.5) -> pd.Series:
    """
    Empirical-Bayes shrinkage with Beta(a,b) prior (Jeffreys default: 0.5,0.5).
    Returns posterior mean (k+a)/(n+a+b).
    """
    k = pd.to_numeric(k, errors="coerce").fillna(0.0)
    n = pd.to_numeric(n, errors="coerce").replace(0, np.nan)
    out = (k + a) / (n + a + b)
    return out.fillna(0.0)


def performance_by_age(
    df: pd.DataFrame,
    *,
    min_support: int = 1,
    age_range: Optional[Tuple[int, int]] = (2, 20),
    round_dp: int = 4,
    win_col: Optional[str] = None,
    use_shrunk_for_peak: bool = True,
) -> pd.DataFrame:
    """
    Compute win rate by age within each race_type_simple.
    Returns: ['race_type_simple','age','win_rate','win_rate_lo','win_rate_hi',
              'win_rate_shrunk','n','is_peak']
    """
    if "race_type_simple" not in df.columns:
        raise ValueError("Missing column: 'race_type_simple'")

    df = _coerce_age(df)
    df = _ensure_won(df, win_col=win_col)

    if age_range is not None:
        lo, hi = age_range
        df = df[(df["age"] >= lo) & (df["age"] <= hi)]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "race_type_simple",
                "age",
                "win_rate",
                "win_rate_lo",
                "win_rate_hi",
                "win_rate_shrunk",
                "n",
                "is_peak",
            ]
        )

    # Aggregate (SeriesGroupBy-friendly)
    g = df.groupby(["race_type_simple", "age"])["won"]
    out = g.agg(["mean", "count"]).reset_index().rename(columns={"mean": "win_rate", "count": "n"})

    if min_support > 1:
        out = out[out["n"] >= min_support]

    if out.empty:
        return pd.DataFrame(
            columns=[
                "race_type_simple",
                "age",
                "win_rate",
                "win_rate_lo",
                "win_rate_hi",
                "win_rate_shrunk",
                "n",
                "is_peak",
            ]
        )

    # Compute wins, Wilson CI, and shrunk rate
    out["wins"] = (out["win_rate"] * out["n"]).round().astype(int)
    lo, hi = _wilson_interval(out["wins"], out["n"])
    out["win_rate_lo"] = lo.round(round_dp)
    out["win_rate_hi"] = hi.round(round_dp)
    out["win_rate_shrunk"] = _beta_shrink_rate(out["wins"], out["n"]).round(round_dp)

    # ---- Peak selection using stable chained sorts ----
    # Priority: race_type_simple ASC, metric DESC, n DESC
    metric = "win_rate_shrunk" if use_shrunk_for_peak else "win_rate"
    sorted_for_peaks = _sort(out, by="n", ascending=False)
    sorted_for_peaks = _sort(sorted_for_peaks, by=metric, ascending=False)
    sorted_for_peaks = _sort(sorted_for_peaks, by="race_type_simple", ascending=True)

    peaks = (
        sorted_for_peaks.groupby("race_type_simple", as_index=False)
        .head(1)[["race_type_simple", "age"]]
        .assign(is_peak=True)
    )

    out = out.merge(peaks, how="left", on=["race_type_simple", "age"])
    # Warning-safe boolean fill
    out["is_peak"] = out["is_peak"].astype("boolean").fillna(False).astype(bool)

    # Round for presentation (raw win_rate kept as rounded)
    out["win_rate"] = out["win_rate"].round(round_dp)

    # Final readable sort
    out = _sort(out, by="age", ascending=True)
    out = _sort(out, by="race_type_simple", ascending=True)
    out = out.reset_index(drop=True)

    # Drop helper
    out = out.drop(columns=["wins"], errors="ignore")
    return out


def plot_age_curves(tbl: pd.DataFrame, out_dir: PathLike) -> Path:
    """
    Plot win-rate vs age for each race_type_simple. Saves PNG+PDF.
    Includes Wilson CI ribbons and peak annotations if available.
    Returns the PNG path.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    p_png = out_path / "age_winrate_curves.png"
    p_pdf = out_path / "age_winrate_curves.pdf"

    fig, ax = plt.subplots()

    if not tbl.empty:
        have_ci = {"win_rate_lo", "win_rate_hi"}.issubset(tbl.columns)
        have_peaks = "is_peak" in tbl.columns

        for rtype, sub in tbl.groupby("race_type_simple", sort=False):
            sub = _sort(sub, by="age", ascending=True)

            # CI ribbon
            if have_ci:
                ax.fill_between(
                    sub["age"].astype(float),
                    sub["win_rate_lo"].astype(float),
                    sub["win_rate_hi"].astype(float),
                    alpha=0.15,
                    label=None,
                )

            # Main curve
            ax.plot(sub["age"], sub["win_rate"], marker="o", label=str(rtype))

            # Peak annotations
            if have_peaks:
                peak_rows = sub[sub["is_peak"]]
                if not peak_rows.empty:
                    ax.scatter(peak_rows["age"], peak_rows["win_rate"], s=80, zorder=3)
                    for _, r in peak_rows.iterrows():
                        ax.annotate(
                            f"peak {int(r['age'])}",
                            (float(r["age"]), float(r["win_rate"])),
                            textcoords="offset points",
                            xytext=(6, 6),
                        )

    ax.set_title("Win rate by Age (per race_type_simple)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Win rate")
    if not tbl.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(p_png, dpi=150)
    fig.savefig(p_pdf)
    plt.close(fig)
    return p_png


def _write_copy_paste_peaks(tbl: pd.DataFrame, out_dir: PathLike) -> None:
    """
    Save compact peak table (CSV + TSV + TXT) for easy report pasting.
    """
    out_path = Path(out_dir)
    cols = ["race_type_simple", "age", "win_rate", "win_rate_shrunk", "n"]
    cols = [c for c in cols if c in tbl.columns]
    peaks = (
        tbl.loc[tbl["is_peak"], cols]
        .sort_values(by="race_type_simple", ascending=True)
        .reset_index(drop=True)
    )

    peaks.to_csv(out_path / "peaks.csv", index=False)
    peaks.to_csv(out_path / "peaks.tsv", index=False, sep="\t")

    txt_lines = [
        "[q1-ages] peaks:",
        (peaks.to_string(index=False) if not peaks.empty else "(no peaks — table empty)"),
        "",
        "Copy-paste (TSV) below:",
        (peaks.to_csv(sep="\t", index=False).strip() if not peaks.empty else ""),
        "",
    ]
    (out_path / "peaks.txt").write_text("\n".join(txt_lines), encoding="utf-8")


def run_q1(
    data_path: str,
    out_base: str,
    *,
    min_support: int = 1,
    age_range: Optional[Tuple[int, int]] = (2, 20),
    round_dp: int = 4,
    win_col: Optional[str] = None,
    use_shrunk_for_peak: bool = True,
) -> str:
    """
    Acceptance entry point.
    """
    p = Path(data_path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    run = ensure_run_dir(out_base, "q1")

    tbl = performance_by_age(
        df,
        min_support=min_support,
        age_range=age_range,
        round_dp=round_dp,
        win_col=win_col,
        use_shrunk_for_peak=use_shrunk_for_peak,
    )

    safe_write_parquet(tbl, run / "table_winrate_by_age.parquet")
    tbl.to_csv(run / "table_winrate_by_age.csv", index=False)

    plot_age_curves(tbl, run)

    _write_copy_paste_peaks(tbl, run)

    display_cols = ["race_type_simple", "age", "win_rate", "win_rate_shrunk", "n"]
    display_cols = [c for c in display_cols if c in tbl.columns]
    display_peaks = (
        tbl.loc[tbl["is_peak"], display_cols]
        .sort_values(by="race_type_simple", ascending=True)
        .reset_index(drop=True)
    )
    print(
        "[q1-ages] peaks:\n",
        (display_peaks.to_string(index=False) if not display_peaks.empty else "(none)"),
    )
    if not display_peaks.empty:
        print(
            "\n[q1-ages] TSV (copy/paste):\n", display_peaks.to_csv(sep="\t", index=False).strip()
        )
    print("\n[q1-ages] run dir:", run)

    return str(run)
