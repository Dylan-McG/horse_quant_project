# src/hqp/analysis/ratings_inspect.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


def _latest_series(
    df: pd.DataFrame, id_col: str, value_col: str, topn: int, time_col: str
) -> pd.Series:
    """
    Latest snapshot per id, sorted descending by the value.
    """
    if time_col not in df.columns:
        raise KeyError(f"Missing time column '{time_col}' in dataframe.")
    return (
        df.sort_values(time_col)
        .groupby(id_col, sort=False)[value_col]
        .tail(1)
        .sort_values(ascending=False)
        .head(topn)
    )


def _latest_series_by_category(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    cat_col: str,
    topn: int,
    time_col: str,
) -> Dict[str, pd.Series]:
    """
    For each category level in cat_col, compute latest-per-id and return topn.
    """
    if cat_col not in df.columns:
        return {}
    out: Dict[str, pd.Series] = {}
    for lvl, sub in df.groupby(cat_col, dropna=True, sort=False):
        if sub.empty:
            continue
        out[str(lvl)] = _latest_series(sub, id_col, value_col, topn, time_col)
    return out


def _series_lines(label: str, s: pd.Series) -> List[str]:
    """
    Pretty-print a Series[id -> value] block as lines of text.
    """
    lines: List[str] = [label]
    if len(s) == 0:
        lines.append("  (no data)")
        lines.append("")
        return lines
    maxw = max((len(str(i)) for i in s.index), default=4)
    for i, v in s.items():
        lines.append(f"  {str(i):<{maxw}}  {float(v):.4f}")
    lines.append("")  # blank line after block
    return lines


def _plot_distributions(ratings: pd.DataFrame, out_dir: Path) -> None:
    """
    Save histograms (PNG) for rating components.
    """
    cols = [
        "horse_elo",
        "horse_eb",
        "jockey_elo",
        "trainer_elo",
        "horse_elo_z",
        "horse_eb_z",
        "jockey_elo_z",
        "trainer_elo_z",
        "rating_blend",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cols:
        if c not in ratings.columns:
            continue
        data = pd.to_numeric(ratings[c], errors="coerce").dropna()
        if data.empty:
            continue
        plt.figure()
        data.hist(bins=50)
        plt.title(f"Distribution of {c}")
        plt.xlabel(c)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{c}.png")
        plt.close()


def _correlation_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix for z-scored components + blend.
    """
    cols = ["horse_elo_z", "horse_eb_z", "jockey_elo_z", "trainer_elo_z", "rating_blend"]
    have: List[str] = [c for c in cols if c in ratings.columns]
    if not have:
        return pd.DataFrame()
    df_num = pd.DataFrame({c: pd.to_numeric(ratings[c], errors="coerce") for c in have})
    return df_num.corr()


def run_inspect_extended(
    parquet_path: str,
    out_dir: str = "reports/analysis/q2/inspect",
    topn: int = 10,
    race_type_col: str = "race_type_simple",
) -> str:
    """
    Extended inspector for Q2 ratings:
      - Coverage & time span
      - Top-N (latest) entities:
          * Horses by rating_blend
          * Horses by horse_elo (overall)
          * Horses by horse_elo per race_type (if available)
          * Jockeys by jockey_elo
          * Trainers by trainer_elo
      - Histograms for distributions (PNG files)
      - Correlation matrix of z-scored components + blend (CSV)

    Returns the directory where artifacts were written.
    """
    # Load
    df = pd.read_parquet(parquet_path).copy()

    # choose time column
    time_col = "race_dt" if "race_dt" in df.columns else ("ts" if "ts" in df.columns else None)
    if time_col is None:
        raise KeyError("Expected 'race_dt' or 'ts' in the ratings parquet.")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # Coverage
    races = int(df["race_id"].nunique()) if "race_id" in df else 0
    horses = int(df["horse_id"].nunique()) if "horse_id" in df else 0
    jockeys = int(df["jockey_id"].nunique()) if "jockey_id" in df else 0
    trainers = int(df["trainer_id"].nunique()) if "trainer_id" in df else 0
    tmin: Optional[pd.Timestamp] = pd.to_datetime(df[time_col]).min()
    tmax: Optional[pd.Timestamp] = pd.to_datetime(df[time_col]).max()

    # Top-N
    top_horses_blend = (
        _latest_series(df, "horse_id", "rating_blend", topn, time_col)
        if "rating_blend" in df.columns
        else pd.Series(dtype=float)
    )
    top_horses_elo = (
        _latest_series(df, "horse_id", "horse_elo", topn, time_col)
        if "horse_elo" in df.columns
        else pd.Series(dtype=float)
    )

    top_jockeys = (
        _latest_series(df, "jockey_id", "jockey_elo", topn, time_col)
        if "jockey_elo" in df.columns
        else pd.Series(dtype=float)
    )
    top_trainers = (
        _latest_series(df, "trainer_id", "trainer_elo", topn, time_col)
        if "trainer_elo" in df.columns
        else pd.Series(dtype=float)
    )

    # Per-discipline horse Elo (if race_type_col present)
    by_disc = (
        _latest_series_by_category(
            df,
            id_col="horse_id",
            value_col="horse_elo",
            cat_col=race_type_col,
            topn=topn,
            time_col=time_col,
        )
        if "horse_elo" in df.columns and race_type_col in df.columns
        else {}
    )

    # Prepare output dir
    ts_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"q2_inspect_{ts_label}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Write a text summary
    lines: List[str] = []
    lines.append("Q2 — Ratings Inspector (extended)")
    lines.append("")
    lines.append("Coverage:")
    lines.append(f"  Races={races}  Horses={horses}  Jockeys={jockeys}  Trainers={trainers}")
    lines.append(f"  Time span: {tmin} → {tmax}")
    lines.append("")

    lines += _series_lines("Top Horses (rating_blend):", top_horses_blend)
    lines += _series_lines("Top Horses (horse_elo overall):", top_horses_elo)
    for k, s in by_disc.items():
        lines += _series_lines(f"Top Horses (horse_elo by {race_type_col} = {k}):", s)
    lines += _series_lines("Top Jockeys (jockey_elo):", top_jockeys)
    lines += _series_lines("Top Trainers (trainer_elo):", top_trainers)

    txt_file = out_path / "q2_ratings_inspect.txt"
    txt_file.write_text("\n".join(lines), encoding="utf-8")

    # Save histograms
    _plot_distributions(df, out_path)

    # Save correlations
    corr = _correlation_matrix(df)
    if not corr.empty:
        corr.to_csv(out_path / "correlations.csv", index=True)

    print(f"[Q2 Inspector] wrote: {txt_file}")
    return str(out_path)
