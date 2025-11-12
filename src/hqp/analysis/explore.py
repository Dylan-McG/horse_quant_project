# src/hqp/analysis/explore.py
from __future__ import annotations

from pathlib import Path
from typing import Final
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.legend import Legend

# package import (works when run as module); safe absolute fallback for direct run
try:
    from ._io import ensure_run_dir, safe_write_parquet
except ImportError:  # pragma: no cover
    from hqp.analysis._io import ensure_run_dir, safe_write_parquet  # type: ignore

# ===========================
# Aesthetics (dark, CB-safe)
# ===========================

DARK_BG: Final[str] = "#0b0f14"  # deep slate
DARK_AX: Final[str] = "#11161c"
FG_TEXT: Final[str] = "#E6E6E6"
GRID: Final[str] = "#22303b"
# Okabe–Ito color-blind friendly palette
PALETTE: Final[list[str]] = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]


def _set_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_AX,
            "axes.edgecolor": FG_TEXT,
            "axes.labelcolor": FG_TEXT,
            "axes.titlecolor": FG_TEXT,
            "xtick.color": FG_TEXT,
            "ytick.color": FG_TEXT,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "font.size": 12,
            "figure.dpi": 120,
            "savefig.facecolor": DARK_BG,
            "savefig.edgecolor": DARK_BG,
            "legend.facecolor": DARK_AX,
            "legend.edgecolor": GRID,
            "legend.fontsize": 10,
        }
    )


def _legend_to_fg(leg: Legend | None) -> None:
    if leg is None:
        return
    for text in leg.get_texts():
        text.set_color(FG_TEXT)


def _add_caption(fig: Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, color=FG_TEXT, ha="left", va="bottom", fontsize=10)


# ===========================
# Columns / schema
# ===========================

# Numeric columns to try plotting (only if present)
BASIC_NUMERIC_COLS: Final[tuple[str, ...]] = (
    "age",
    "obs__bsp",  # exchange SP (in your data)
    "ltp_5min",  # pre-off LTP
    "official_rating",
    "carried_weight",
    "draw",
    "n_runners",
)

# Candidate odds columns (first present will be used for overround/odds plots)
ODDS_CANDIDATES: Final[tuple[str, ...]] = (
    "sp_decimal",
    "bsp_decimal",
    "obs__bsp",
    "ltp_5min",
)

# Default categoricals in your dataset
DEFAULT_CAT_COLS: Final[tuple[str, ...]] = (
    "race_type_simple",
    "racecourse_name",
    "going_clean",
)

# ---------------------------
# Small IO helpers
# ---------------------------


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_dual(df: pd.DataFrame, base: Path) -> None:
    """Write both parquet and csv with the same base name."""
    safe_write_parquet(df, base.with_suffix(".parquet"))
    _safe_write_csv(df, base.with_suffix(".csv"))


# ---------------------------
# Table builders (EDA basics)
# ---------------------------


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    counts = {
        "rows": len(df),
        "races": df["race_id"].nunique() if "race_id" in df else None,
        "horses": df["horse_id"].nunique() if "horse_id" in df else None,
        "jockeys": df["jockey_id"].nunique() if "jockey_id" in df else None,
        "trainers": df["trainer_id"].nunique() if "trainer_id" in df else None,
        "race_types": df["race_type_simple"].nunique() if "race_type_simple" in df else None,
        "date_range_start": str(df["date"].min()) if "date" in df else None,
        "date_range_end": str(df["date"].max()) if "date" in df else None,
        "avg_runners_per_race": (
            float(df.groupby("race_id")["horse_id"].nunique().mean())
            if {"race_id", "horse_id"}.issubset(df.columns)
            else (
                float(df.groupby("race_id")["n_runners"].first().mean())
                if {"race_id", "n_runners"}.issubset(df.columns)
                else None
            )
        ),
        "dup_key_pairs": (
            int(df.duplicated(subset=["race_id", "horse_id"]).sum())
            if {"race_id", "horse_id"}.issubset(df.columns)
            else None
        ),
    }
    return pd.DataFrame([counts])


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    non_null = df.notna().sum()
    nulls = df.isna().sum()
    if len(df):
        pct_null = (nulls.astype(float) / float(len(df))) * 100.0
    else:
        pct_null = pd.Series(0.0, index=df.columns, dtype=float)
    out = pd.DataFrame(
        {
            "column": cols,
            "non_null": [int(non_null[c]) for c in cols],
            "nulls": [int(nulls[c]) for c in cols],
            "pct_null": [float(pct_null[c]) for c in cols],
        }
    )
    return out.sort_values("pct_null", ascending=False).reset_index(drop=True)


def dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}).reset_index(
        drop=True
    )


def categorical_breakdown(
    df: pd.DataFrame, cat_cols: tuple[str, ...] = DEFAULT_CAT_COLS
) -> pd.DataFrame:
    records: list[tuple[str, str, int, float]] = []
    n = len(df)
    for c in cat_cols:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).head(20)
            for val, cnt in vc.items():
                freq = float(cnt / n) if n else 0.0
                records.append((c, str(val), int(cnt), freq))
    return pd.DataFrame(records, columns=["column", "value", "count", "freq"]).reset_index(
        drop=True
    )


def runners_per_race(df: pd.DataFrame) -> pd.DataFrame:
    if {"race_id", "horse_id"}.issubset(df.columns):
        g = df.groupby("race_id")["horse_id"].nunique().reset_index(name="n_runners")
        return g
    if {"race_id", "n_runners"}.issubset(df.columns):
        g = df.groupby("race_id")["n_runners"].first().reset_index(name="n_runners")
        return g
    return pd.DataFrame(columns=["race_id", "n_runners"])


def _pick_odds_col(df: pd.DataFrame) -> str | None:
    for c in ODDS_CANDIDATES:
        if c in df.columns:
            return c
    return None


def overround_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overround per race: sum_i (1/odds_i). Uses first available odds column.
    Filters out invalid odds (<= 1.0).
    """
    if "race_id" not in df.columns:
        return pd.DataFrame(columns=["race_id", "overround", "n_runners_with_odds", "odds_source"])
    odds_col = _pick_odds_col(df)
    if odds_col is None:
        return pd.DataFrame(columns=["race_id", "overround", "n_runners_with_odds", "odds_source"])

    work = df[["race_id", odds_col]].copy()
    work = work[work[odds_col].notna()]
    work[odds_col] = work[odds_col].astype(float)
    work = work[work[odds_col] > 1.0]  # guard against bad odds
    if work.empty:
        return pd.DataFrame(columns=["race_id", "overround", "n_runners_with_odds", "odds_source"])

    work["inv_odds"] = 1.0 / work[odds_col]
    agg = work.groupby("race_id").agg(
        overround=("inv_odds", "sum"),
        n_runners_with_odds=(odds_col, "count"),
    )
    agg["odds_source"] = odds_col
    return agg.reset_index()


# ---------------------------
# Plots (matplotlib only)
# ---------------------------


def plot_basic_distributions(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """(Optional) generic histograms for basic numeric cols."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    cols = [c for c in BASIC_NUMERIC_COLS if c in df.columns]
    for c in cols:
        fig, ax = plt.subplots()
        data = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        ax.hist(data, bins=30, color=PALETTE[6])
        ax.set_title(f"Distribution of {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("Count (rows)")
        _add_caption(fig, f"Histogram of {c} across all rows (NaNs dropped).")
        fig.tight_layout()
        p_png = out_path / f"dist_{c}.png"
        p_pdf = out_path / f"dist_{c}.pdf"
        fig.savefig(p_png, dpi=150)
        fig.savefig(p_pdf)
        plt.close(fig)
        outputs += [p_png, p_pdf]
    return outputs


def plot_runners_hist(runners: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Field size distribution, coloured per bin, clamped to 1–30 runners."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    if "n_runners" in runners.columns and len(runners):
        # build counts by integer field size
        r = pd.to_numeric(runners["n_runners"], errors="coerce").dropna().astype(int)
        counts = r.value_counts().sort_index()
        counts = counts[(counts.index >= 1) & (counts.index <= 30)]
        if counts.empty:
            return figs

        x_vals = counts.index.to_list()
        heights = counts.to_list()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(x_vals))]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x_vals, heights, color=colors, width=0.85, edgecolor=DARK_BG, linewidth=0.5)
        ax.set_title("Field Size Distribution")
        ax.set_xlabel("Runners in race")
        ax.set_ylabel("Number of races")
        ax.set_xlim(0.5, 30.5)
        # mean/median lines
        mean_v = float(np.mean(r))
        median_v = float(np.median(r))
        ax.axvline(
            mean_v, color=PALETTE[4], linestyle="--", linewidth=1.5, label=f"Mean {mean_v:.1f}"
        )
        ax.axvline(
            median_v, color=PALETTE[5], linestyle=":", linewidth=1.5, label=f"Median {median_v:.1f}"
        )
        leg = ax.legend()
        _legend_to_fg(leg)
        _add_caption(
            fig,
            "Each bar is a distinct field size (1–30). Mean/median shown as vertical lines. "
            "Y-axis = number of races at that field size.",
        )
        fig.tight_layout()
        p_png = out_path / "dist_n_runners.png"
        p_pdf = out_path / "dist_n_runners.pdf"
        fig.savefig(p_png, dpi=150)
        fig.savefig(p_pdf)
        plt.close(fig)
        figs += [p_png, p_pdf]
    return figs


def plot_overround_hist(ovr: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Overround distribution with fair-book and mean markers."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    if "overround" in ovr.columns and len(ovr):
        vals = pd.to_numeric(ovr["overround"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return figs
        # sensible window; most books live near 1.0–1.3
        xmin = max(0.85, float(np.nanpercentile(vals, 0.5)))
        xmax = min(1.50, float(np.nanpercentile(vals, 99.5)))
        bins_list: list[float] = list(np.arange(0.90, 1.501, 0.01, dtype=float))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(vals, bins=bins_list, color=PALETTE[2], edgecolor=DARK_BG, linewidth=0.3)
        ax.set_xlim(xmin, xmax)
        src = (
            str(ovr["odds_source"].iloc[0]) if "odds_source" in ovr.columns and len(ovr) else "odds"
        )
        ax.set_title(f"Per-race Overround using {src}")
        ax.set_xlabel("Overround (sum of implied probabilities)")
        ax.set_ylabel("Number of races")
        # markers
        mean_v = float(np.nanmean(vals))
        ax.axvline(1.0, color=PALETTE[4], linestyle="--", linewidth=1.5, label="Fair book (1.00)")
        ax.axvline(
            mean_v, color=PALETTE[5], linestyle=":", linewidth=1.5, label=f"Mean {mean_v:.3f}"
        )
        leg = ax.legend()
        _legend_to_fg(leg)
        _add_caption(
            fig,
            "Overround = sum of implied win probabilities in a race. >1 implies margin/overround; "
            "1.00 is a fair book with no margin.",
        )
        fig.tight_layout()
        p_png = out_path / "dist_overround.png"
        p_pdf = out_path / "dist_overround.pdf"
        fig.savefig(p_png, dpi=150)
        fig.savefig(p_pdf)
        plt.close(fig)
        figs += [p_png, p_pdf]
    return figs


def _plot_monthly_volume(df: pd.DataFrame, out_dir: Path) -> None:
    """Monthly race counts with YYYY-Mon labels and month-of-year colour."""
    if "date" not in df.columns or "race_id" not in df.columns:
        return
    work = df[["date", "race_id"]].dropna().copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    monthly = (
        work.drop_duplicates(["date", "race_id"])
        .assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["race_id"]
        .nunique()
        .reset_index(name="races")
    )
    if monthly.empty:
        return

    # colour by month-of-year so adjacent months are visually distinct
    months = monthly["month"].dt.month.to_list()
    colors = [PALETTE[(m - 1) % len(PALETTE)] for m in months]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = monthly["month"].dt.to_pydatetime()
    y = monthly["races"].to_numpy(dtype=float)
    ax.bar(x, y, color=colors, edgecolor=DARK_BG, linewidth=0.4, width=20)  # ~20-day wide bars
    ax.set_title("Races per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of races")
    ax.set_xlim(min(x), max(x))

    # Ticks: no more than ~12 labels; format as YYYY-Mon
    step = max(1, len(x) // 12)
    tick_positions = x[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.Timestamp(tp).strftime("%Y-%b") for tp in tick_positions])

    fig.autofmt_xdate()
    _add_caption(
        fig, "Monthly race volume. Bar hue encodes month-of-year for quick seasonal parsing."
    )
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_races.png")
    fig.savefig(out_dir / "monthly_races.pdf")
    plt.close(fig)


def _plot_odds_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Linear odds histogram, capped at robust high end for clarity."""
    col = _pick_odds_col(df)
    if col is None:
        return
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[(s > 1.0) & np.isfinite(s)]
    if s.empty:
        return
    # Robust cap: min(600, 99.5th pct)
    cap = float(min(600.0, np.nanpercentile(s.to_numpy(dtype=float), 99.5)))
    data = s.clip(upper=cap).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(data, bins=60, color=PALETTE[0], edgecolor=DARK_BG, linewidth=0.3)
    ax.set_title(f"Distribution of {col} (capped at {cap:.0f})")
    ax.set_xlabel("Odds (decimal)")
    ax.set_ylabel("Number of runners")
    ax.set_xlim(1.0, cap)
    _add_caption(
        fig,
        f"Histogram of {col}. Values above {cap:.0f} are capped for readability; "
        "y-axis counts individual runners.",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"dist_{col}.png")
    fig.savefig(out_dir / f"dist_{col}.pdf")
    plt.close(fig)


def _plot_age_profile_by_code(df: pd.DataFrame, out_dir: Path) -> None:
    """Age share curves by race code with clear legend + caption."""
    if "age" not in df.columns or "race_type_simple" not in df.columns:
        return

    work = df[["age", "race_type_simple"]].dropna().copy()
    work = work[(work["age"] >= 2) & (work["age"] <= 15)]
    if work.empty:
        return

    g = work.groupby(["race_type_simple", "age"]).size().rename("count").reset_index()
    totals = g.groupby("race_type_simple")["count"].transform("sum")
    g["share"] = g["count"] / totals.replace(0, np.nan)

    # annotate legend entries with race counts for that code
    code_counts = work.groupby("race_type_simple").size().rename("rows").reset_index()
    code_to_rows = {str(r["race_type_simple"]): int(r["rows"]) for _, r in code_counts.iterrows()}

    codes: list[str] = [str(c) for c in g["race_type_simple"].unique().tolist()]
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for i, code in enumerate(codes):
        sub = g[g["race_type_simple"] == code]
        x: list[float] = list(map(float, sub["age"].tolist()))
        y: list[float] = list(map(float, sub["share"].tolist()))
        color = PALETTE[i % len(PALETTE)]
        label = f"{code} (n={code_to_rows.get(code, 0)})"
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=label, color=color)

    ax.set_title("Age Profile by Race Type (share within code)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Share of runners")
    leg = ax.legend(ncol=min(2, len(codes)))
    _legend_to_fg(leg)
    _add_caption(
        fig,
        "Within each race type, line shows the share of all runners at each age "
        "(areas under curves sum to 1 per code).",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "age_profile_by_code.png")
    fig.savefig(out_dir / "age_profile_by_code.pdf")
    plt.close(fig)


# ----- Info card figure
def _plot_info_card(summary: dict[str, object], out_dir: Path) -> None:
    keys = [
        ("Rows", "rows"),
        ("Races", "races"),
        ("Horses", "horses"),
        ("Date range", None),
        ("Avg runners/race", "avg_runners_per_race"),
        ("Dup (race_id, horse_id)", "dup_key_pairs"),
    ]
    start = str(summary.get("date_range_start", "NA"))
    end = str(summary.get("date_range_end", "NA"))
    dr = f"{start} → {end}"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_axis_off()
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)

    lines: list[tuple[str, str]] = []
    for label, field in keys:
        if field is None:
            val = dr
        else:
            v = summary.get(field, "NA")
            if isinstance(v, float):
                val = f"{v:.3f}" if np.isfinite(v) else "NA"
            else:
                val = str(v)
        lines.append((label, val))

    y = 0.9
    ax.text(
        0.05,
        0.95,
        "Dataset Summary",
        color=FG_TEXT,
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    for i, (label, val) in enumerate(lines):
        ax.text(
            0.08, y - i * 0.12, f"{label}:", color=PALETTE[4], fontsize=12, transform=ax.transAxes
        )
        ax.text(0.40, y - i * 0.12, val, color=FG_TEXT, fontsize=12, transform=ax.transAxes)
    _add_caption(fig, "Quick-glance card for pasting into the report.")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_info_card.png")
    fig.savefig(out_dir / "dataset_info_card.pdf")
    plt.close(fig)


# ---------------------------
# Notes.md generator
# ---------------------------


def _fmt(v: object) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.3f}"
    return str(v)


def _write_notes_md(
    run_dir: Path,
    summary: pd.DataFrame,
    missing: pd.DataFrame,
    runners: pd.DataFrame,
    ovr_summary: pd.DataFrame,
) -> None:
    s = summary.iloc[0].to_dict()
    lines: list[str] = []
    lines.append("# Q0 — Dataset Exploration Notes\n")
    lines.append("## Snapshot\n")
    lines.append(f"- Rows: **{_fmt(s.get('rows'))}**")
    lines.append(f"- Races: **{_fmt(s.get('races'))}**, Horses: **{_fmt(s.get('horses'))}**")
    lines.append(
        f"- Date range: **{_fmt(s.get('date_range_start'))} → {_fmt(s.get('date_range_end'))}**"
    )
    lines.append(f"- Avg runners/race: **{_fmt(s.get('avg_runners_per_race'))}**")
    lines.append(f"- Duplicate (race_id, horse_id) pairs: **{_fmt(s.get('dup_key_pairs'))}**\n")

    lines.append("## Missingness (top 10 by % null)\n")
    head = missing.head(10).copy()
    md = head.to_markdown(index=False)  # requires 'tabulate'
    lines.append(md)
    lines.append("")

    if len(runners):
        lines.append("## Field Size\n")
        lines.append("- See `plots/dist_n_runners.png` and `runners_per_race.{parquet,csv}`.\n")

    if len(ovr_summary):
        os = ovr_summary.iloc[0].to_dict()
        lines.append("## Overround (Bookmaker Margin Proxy)\n")
        lines.append(
            f"- Mean: **{_fmt(os.get('overround_mean'))}**, "
            f"Median: **{_fmt(os.get('overround_median'))}**, "
            f"P10: **{_fmt(os.get('overround_p10'))}**, "
            f"P90: **{_fmt(os.get('overround_p90'))}**."
        )
        lines.append("- See `plots/dist_overround.png` and `overround_by_race.{parquet,csv}`.\n")

    notes_path = run_dir / "notes.md"
    notes_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# Report-helper artifacts
# ---------------------------


def _build_report_context(
    run_dir: Path,
    summary: pd.DataFrame,
    missing: pd.DataFrame,
    dtypes: pd.DataFrame,
    cats: pd.DataFrame,
    runners: pd.DataFrame,
    ovr: pd.DataFrame,
    ovr_summary: pd.DataFrame,
) -> dict[str, object]:
    """Assemble a compact JSON-able dict with just what we need for drafting the report."""
    plots_dir = run_dir / "plots"

    # small helpers (avoid numpy/NaN leaking to json)
    def df_head_records(df: pd.DataFrame | None, n: int = 10) -> list[dict[str, object]]:
        if df is None or df.empty:
            return []
        out: list[dict[str, object]] = []
        for row in df.head(n).to_dict(orient="records"):
            rec: dict[str, object] = {}
            for k, v in row.items():
                key = str(k)
                # normalise numpy scalars/NaNs to python types
                if isinstance(v, (np.generic,)):
                    v = v.item()
                if isinstance(v, float) and np.isnan(v):
                    v = None
                rec[key] = v
            out.append(rec)
        return out

    # precise nested type for plots
    plots: dict[str, str | None] = {
        "info_card": str(plots_dir / "dataset_info_card.png"),
        "monthly_races": str(plots_dir / "monthly_races.png"),
        "field_size": str(plots_dir / "dist_n_runners.png"),
        "overround": str(plots_dir / "dist_overround.png"),
        "odds_hist": None,  # maybe absent if no odds col
        "age_profile": str(plots_dir / "age_profile_by_code.png"),
    }

    ctx: dict[str, object] = {
        "run_dir": str(run_dir),
        "plots": plots,
        "summary": summary.iloc[0].to_dict() if len(summary) else {},
        "missing_top10": df_head_records(missing, 10),
        "dtypes": df_head_records(dtypes, 100),
        "categorical_top": df_head_records(cats, 50),
        "runners_examples": df_head_records(runners, 20),
        "overround_examples": df_head_records(ovr, 20),
        "overround_summary": df_head_records(ovr_summary, 1),
        "odds_source": (
            ovr["odds_source"].iloc[0] if "odds_source" in ovr.columns and len(ovr) else None
        ),
    }

    # detect which odds plot we wrote (depends on which column existed)
    for cand in ODDS_CANDIDATES:
        p = plots_dir / f"dist_{cand}.png"
        if p.exists():
            plots["odds_hist"] = str(p)
            break

    return ctx


def _write_report_context(run_dir: Path, context: dict[str, object]) -> None:
    """Write report_context.json only (no prompt text)."""
    (run_dir / "report_context.json").write_text(
        json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------
# Orchestrator
# ---------------------------


def run_q0(data_path: str, out_base: str) -> str:
    """
    Q0: Explore dataset (raw CSV/Parquet)
    - Writes BOTH parquet and csv for all tables
    - Emits plots (PNG+PDF with captions) and a concise notes.md for report copy/paste
    - Emits report_context.json (structured dump ready to paste into ChatGPT)
    """
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    run = ensure_run_dir(out_base, "q0")
    _set_dark_style()

    # Tables
    summary = summarize_dataset(df)
    missing = missingness_table(df)
    dtypes = dtypes_table(df)
    cats = categorical_breakdown(df)
    runners = runners_per_race(df)
    ovr = overround_by_race(df)

    if len(ovr):
        ovr_vals = ovr["overround"].to_numpy(dtype=float)
        ovr_summary = pd.DataFrame(
            [
                {
                    "overround_mean": float(np.nanmean(ovr_vals)) if ovr_vals.size else np.nan,
                    "overround_median": float(np.nanmedian(ovr_vals)) if ovr_vals.size else np.nan,
                    "overround_p10": (
                        float(np.nanpercentile(ovr_vals, 10)) if ovr_vals.size else np.nan
                    ),
                    "overround_p90": (
                        float(np.nanpercentile(ovr_vals, 90)) if ovr_vals.size else np.nan
                    ),
                    "races_with_odds": int(len(ovr)),
                }
            ]
        )
    else:
        ovr_summary = pd.DataFrame(
            [
                {
                    "overround_mean": np.nan,
                    "overround_median": np.nan,
                    "overround_p10": np.nan,
                    "overround_p90": np.nan,
                    "races_with_odds": 0,
                }
            ]
        )

    # Dual-write (parquet + csv)
    run_path = Path(run)
    _write_dual(summary, run_path / "summary")
    _write_dual(missing, run_path / "missingness")
    _write_dual(dtypes, run_path / "dtypes")
    _write_dual(cats, run_path / "categoricals")
    _write_dual(runners, run_path / "runners_per_race")
    _write_dual(ovr, run_path / "overround_by_race")
    _write_dual(ovr_summary, run_path / "overround_summary")

    # Plots (the “Top 5” + info card)
    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_monthly_volume(df, plots_dir)  # 1) time coverage
    plot_runners_hist(runners, plots_dir)  # 2) field size (improved)
    _plot_odds_distribution(df, plots_dir)  # 3) odds (linear, capped)
    plot_overround_hist(ovr, plots_dir)  # 4) overround with markers
    _plot_age_profile_by_code(df, plots_dir)  # 5) age profile
    _plot_info_card(summary.iloc[0].to_dict(), plots_dir)

    # (Optional) keep simple numeric histograms for completeness
    plot_basic_distributions(df, plots_dir)

    # Notes for the report
    _write_notes_md(run_path, summary, missing, runners, ovr_summary)

    # Structured context (JSON only; no prompt text)
    context = _build_report_context(
        run_path, summary, missing, dtypes, cats, runners, ovr, ovr_summary
    )
    _write_report_context(run_path, context)

    return str(run)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Q0 — Explore dataset (raw CSV/Parquet).")
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet dataset")
    parser.add_argument("--out", default="reports/analysis", help="Base output directory")
    args = parser.parse_args()
    print(run_q0(args.data, args.out))
