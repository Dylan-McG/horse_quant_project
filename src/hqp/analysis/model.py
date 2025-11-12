# src/hqp/analysis/model.py
# -----------------------------------------------------------------------------
# Q3 — Predictive Model (report-from-artifacts)
#
# This module does NOT train. It consumes predictions produced by the
# config-driven pipeline in src/hqp/models/* (e.g., LightGBM trainer),
# verifies per-race normalization, computes metrics/plots, and emits a
# Q3-ready report under reports/model/q3/<timestamp>/.
#
# Typical usage (PowerShell):
#   poetry run python -c "from hqp.analysis.model import run_q3; out = run_q3(run_dir=r'reports/model/20250921_233551'); print(out['report_md'])"
#   # or point directly at a predictions file:
#   poetry run python -c "from hqp.analysis.model import run_q3; out = run_q3(predictions_path=r'reports/model/20250921_233551/predictions_cal.parquet'); print(out['report_md'])"
#   # if labels are stored elsewhere:
#   poetry run python -c "from hqp.analysis.model import run_q3; out = run_q3(predictions_path=r'.../predictions.parquet', labels_path=r'data/labels/winners.parquet'); print(out['report_md'])"
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timezone

from hqp.models.metrics import (
    check_probs_sum_to_1,
    race_normalize,
    log_loss_by_race,
    brier_score,
    ece,
    pit_values,
    pit_basic_checks,
)

PROB_TOL: float = 1e-9
EPS: float = 1e-12


@dataclass(frozen=True)
class Q3ReportConfig:
    # Column names we expect to find in predictions artifacts
    race_id_col: str = "race_id"
    runner_id_col: str = "horse_id"
    target_col: str = "won"
    # Candidate probability columns (in priority order)
    prob_cols_priority: tuple[str, ...] = (
        "model_prob_cal",
        "p_lgbm_cal",
        "p_cal",
        "prob_cal",
        "model_prob",
        "p_lgbm",
        "p",
        "prob",
        "prob_win",
        "p_win",
        "proba",
    )
    # Odds candidates for ROI deciles (first match is used)
    odds_cols_priority: tuple[str, ...] = (
        "mkt_odds_decimal",
        "sp_odds_decimal",
        "bsp_decimal",
        "ltp_5min",
    )
    n_calib_bins: int = 15
    n_roi_deciles: int = 10


# ------------------------------ filesystem utils ------------------------------


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------ plotting -------------------------------------


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _calibration_plot(
    df: pd.DataFrame, proba_col: str, y_col: str, out_png: Path, n_bins: int
) -> None:
    p = df[proba_col].clip(0, 1).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)

    xs: list[float] = []
    ys: list[float] = []
    ns: list[int] = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not np.any(m):
            continue
        xs.append(float(np.mean(p[m])))
        ys.append(float(np.mean(y[m])))
        ns.append(int(np.sum(m)))

    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    sizes = np.clip(np.asarray(ns, dtype=float) * 0.5, 10.0, 120.0)

    plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Perfect")
    plt.scatter(xs_a, ys_a, s=sizes)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    _save_plot(out_png)


def _prob_hist(df: pd.DataFrame, proba_col: str, out_png: Path) -> None:
    plt.figure()
    plt.hist(df[proba_col].to_numpy(), bins=30)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Histogram")
    _save_plot(out_png)


def _pit_plot(u: np.ndarray, out_png: Path) -> None:
    plt.figure()
    plt.hist(u, bins=20)
    plt.xlabel("PIT value")
    plt.ylabel("Count")
    mu, var = pit_basic_checks(u)
    plt.title(f"PIT (mean={mu:.3f}, var={var:.3f})")
    _save_plot(out_png)


# ------------------------------ core helpers ---------------------------------


def _guess_prob_col(df: pd.DataFrame, cfg: Q3ReportConfig) -> str:
    # First: named priorities
    for c in cfg.prob_cols_priority:
        if c in df.columns:
            return c
    # Next: any float-like column in [0,1] not an id/target/odds
    deny = {cfg.race_id_col, cfg.runner_id_col, cfg.target_col}
    deny |= set(cfg.odds_cols_priority)
    cand = []
    for c in df.columns:
        if c in deny:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            frac_01 = float(((vals >= -1e-9) & (vals <= 1 + 1e-9)).mean())
            if frac_01 > 0.98:
                cand.append(c)
    if cand:
        cand.sort(key=len)  # prefer simple names like 'p'
        return cand[0]
    raise ValueError("Could not identify a probability column in predictions.")


def _load_predictions_any(run_dir_or_file: Path) -> pd.DataFrame:
    """
    Accept either a directory (containing predictions_cal / predictions)
    or a direct parquet/csv file path.
    """
    p = run_dir_or_file
    if p.is_file():
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)

    # Directory case: prefer calibrated parquet
    cal = p / "predictions_cal.parquet"
    raw = p / "predictions.parquet"
    cal_csv = p / "predictions_cal.csv"
    raw_csv = p / "predictions.csv"
    for cand in (cal, raw, cal_csv, raw_csv):
        if cand.exists():
            if cand.suffix.lower() == ".parquet":
                return pd.read_parquet(cand)
            return pd.read_csv(cand)
    raise FileNotFoundError(
        f"No predictions file found in {p.as_posix()} "
        "(looked for predictions_cal.parquet, predictions.parquet, .csv variants)."
    )


# -------- winner inference / labels join (for missing `won`) ------------------


def _derive_won_from_columns(df: pd.DataFrame, *, out_col: str = "won") -> pd.DataFrame:
    """Try to construct a binary 'won' from common post-race columns."""
    g = df.copy()
    if out_col in g.columns:
        g[out_col] = pd.to_numeric(g[out_col], errors="coerce").fillna(0).astype(int)
        return g

    # Direct flags first
    direct_flags = ["obs__is_winner", "is_winner", "label", "target", "obs__won"]
    for c in direct_flags:
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce").fillna(0)
            g[out_col] = (s == 1).astype(int)
            return g

    # Position-like columns -> winner if position == 1
    posish = [
        "obs__finish_position",
        "finish_position",
        "finish_pos",
        "final_pos",
        "pos",
        "position",
        "rank",
        "fin_pos",
        "placing",
        "obs__uposition",
        "obs__Place",
    ]
    # Also include any column name containing 'finish' or 'position'
    dynamic = [c for c in g.columns if ("finish" in c.lower() or "position" in c.lower())]
    for c in list(dict.fromkeys(posish + dynamic)):  # preserve order, de-dup
        if c in g.columns:
            vals = pd.to_numeric(g[c], errors="coerce")
            if vals.notna().any():
                g[out_col] = (vals == 1).astype(int)
                return g

    # Could not infer -> leave as is (no 'won' added)
    return g


def _try_attach_labels(
    preds: pd.DataFrame,
    *,
    cfg: Q3ReportConfig,
    predictions_source: Path,
    labels_path: str | Path | None,
) -> pd.DataFrame:
    """
    Ensure a 'won' column exists. If missing:
      1) Try to derive from columns in preds.
      2) If still missing, try to merge labels from labels_path (if provided)
         or from a few likely relative locations.
    """
    out_col = cfg.target_col
    if out_col in preds.columns:
        return preds

    # (1) Try to derive from present columns
    g = _derive_won_from_columns(preds, out_col=out_col)
    if out_col in g.columns:
        return g

    # (2) Try to locate an external labels file and merge
    candidates: list[Path] = []
    if labels_path is not None:
        candidates.append(Path(labels_path))

    # sensible relatives to predictions file/directory
    base = predictions_source if predictions_source.is_dir() else predictions_source.parent
    candidates.extend(
        [
            base / "labels.parquet",
            base / "labels.csv",
            base / "winners.parquet",
            base / "winners.csv",
            base.parent / "labels" / "winners.parquet",
            base.parent / "labels" / "labels.parquet",
            Path("data") / "labels" / "winners.parquet",
            Path("data") / "labels" / "labels.parquet",
        ]
    )

    # deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(c)

    labels_df: Optional[pd.DataFrame] = None
    for cand in uniq:
        try:
            if cand.exists():
                labels_df = (
                    pd.read_parquet(cand)
                    if cand.suffix.lower() == ".parquet"
                    else pd.read_csv(cand)
                )
                # must contain minimal keys
                if all(k in labels_df.columns for k in (cfg.race_id_col, cfg.runner_id_col)):
                    if cfg.target_col not in labels_df.columns:
                        # Try to derive inside the labels file, too
                        labels_df = _derive_won_from_columns(labels_df, out_col=cfg.target_col)
                    if cfg.target_col in labels_df.columns:
                        break
                labels_df = None
        except Exception:
            labels_df = None
    if labels_df is not None:
        keep = [cfg.race_id_col, cfg.runner_id_col, cfg.target_col]
        labels_df = labels_df[keep].copy()
        # Harmonize dtypes for safe merge (string is safest/common from artifacts)
        preds_h = preds.copy()
        preds_h[cfg.race_id_col] = preds_h[cfg.race_id_col].astype("string")
        preds_h[cfg.runner_id_col] = preds_h[cfg.runner_id_col].astype("string")
        labels_df[cfg.race_id_col] = labels_df[cfg.race_id_col].astype("string")
        labels_df[cfg.runner_id_col] = labels_df[cfg.runner_id_col].astype("string")

        merged = preds_h.merge(labels_df, on=[cfg.race_id_col, cfg.runner_id_col], how="left")
        merged[cfg.target_col] = merged[cfg.target_col].fillna(0).astype(int)
        return merged

    raise ValueError(
        "No 'won' column found or derivable in predictions, and no labels file was found. "
        "Provide labels_path=... (a CSV/Parquet with race_id, horse_id, won)."
    )


def _maybe_roi_deciles(
    df: pd.DataFrame,
    proba_col: str,
    cfg: Q3ReportConfig,
) -> Optional[pd.DataFrame]:
    odds_col: Optional[str] = None
    for c in cfg.odds_cols_priority:
        if c in df.columns:
            odds_col = c
            break
    if odds_col is None:
        return None

    y_col = cfg.target_col
    d = df[[proba_col, y_col, odds_col]].copy()
    # robust ranking to avoid tied quantiles
    d["__rank__"] = d[proba_col].rank(method="first")
    d["decile"] = pd.qcut(d["__rank__"], cfg.n_roi_deciles, labels=False, duplicates="drop")
    d.drop(columns="__rank__", inplace=True)

    # unit-stake P&L: + (odds - 1) on win, else -1
    d["pnl"] = np.where(d[y_col].astype(int) == 1, d[odds_col] - 1.0, -1.0)

    out = (
        d.groupby("decile", dropna=True)
        .agg(
            n=("pnl", "size"),
            roi=("pnl", "mean"),
            avg_prob=(proba_col, "mean"),
            hit_rate=(y_col, "mean"),
            avg_odds=(odds_col, "mean"),
        )
        .reset_index()
        .sort_values("decile")
    )
    return out


# ------------------------------ public API -----------------------------------


def summarize_run(
    *,
    predictions_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    out_root: str | Path = "reports/model/q3",
    config: Q3ReportConfig | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """
    Summarize an existing model run by reading its predictions file(s),
    computing metrics/plots, and writing a Q3 report.

    If 'won' is missing in predictions, we try to derive it from common
    columns; failing that, we try to merge from labels_path or auto-locate
    a labels file nearby.

    Returns: dict with keys {run_dir, metrics, report_md, report_tex, predictions_used}
    """
    cfg = config or Q3ReportConfig()

    if predictions_path is None and run_dir is None:
        raise ValueError("Provide either predictions_path or run_dir.")

    src = Path(predictions_path) if predictions_path else Path(run_dir)  # type: ignore[arg-type]
    preds_raw = _load_predictions_any(src)

    # Validate required id columns
    for col in (cfg.race_id_col, cfg.runner_id_col):
        if col not in preds_raw.columns:
            raise ValueError(f"'{col}' column not found in predictions.")

    # Attach labels (if missing) or derive
    preds = _try_attach_labels(preds_raw, cfg=cfg, predictions_source=src, labels_path=labels_path)

    # Pick probability column
    prob_col = _guess_prob_col(preds, cfg)

    # Enforce/verify per-race normalization
    rids = preds[cfg.race_id_col].to_numpy()
    probs = pd.to_numeric(preds[prob_col], errors="coerce").to_numpy(dtype=float)
    probs = race_normalize(probs, rids)  # idempotent if already normalized
    check_probs_sum_to_1(probs, rids, tol=PROB_TOL)

    # Targets
    y = pd.to_numeric(preds[cfg.target_col], errors="coerce").fillna(0).astype(int).to_numpy()

    # Output directory
    out_root = Path(out_root)
    run_out = out_root / _now_stamp()
    _ensure_dir(run_out)

    # Metrics
    has_winner = int(y.sum()) > 0

    def _metrics_for(p: np.ndarray) -> dict[str, float]:
        if not has_winner:
            return {
                "logloss": float("nan"),
                "brier": float(brier_score(y, p)),
                "ece": float("nan"),
            }
        return {
            "logloss": float(log_loss_by_race(y, p, rids)),
            "brier": float(brier_score(y, p)),
            "ece": float(ece(y, p, n_bins=cfg.n_calib_bins, strategy="equal_width")),
        }

    m = _metrics_for(probs)

    # PIT (winner rows only)
    if has_winner:
        pit_u = pit_values(y, probs, rids)
        pit_mu, pit_var = pit_basic_checks(pit_u)
    else:
        pit_u = np.array([], dtype=float)
        pit_mu, pit_var = float("nan"), float("nan")

    # ROI deciles (optional if odds present)
    preds_norm = preds.copy()
    preds_norm[prob_col] = probs
    roi_dec = _maybe_roi_deciles(preds_norm, prob_col, cfg)

    # Persist metrics & (optionally) plots
    metrics: dict[str, Any] = {
        "model_run_source": str(src),
        "n_rows": int(len(preds_norm)),
        "n_races": int(preds_norm[cfg.race_id_col].nunique()),
        "prob_col_used": prob_col,
        "logloss": m["logloss"],
        "brier": m["brier"],
        "ece": m["ece"],
        "pit_mean": float(pit_mu),
        "pit_var": float(pit_var),
        "notes": ("no_winners_in_validation" if not has_winner else "ok"),
    }

    if roi_dec is not None and not roi_dec.empty:
        roi_path = run_out / "roi_deciles.csv"
        roi_dec.to_csv(roi_path, index=False)
        metrics["roi_deciles_csv"] = roi_path.as_posix()

    (run_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plots (skip on giant runs if desired)
    if make_plots:
        _calibration_plot(
            preds_norm, prob_col, cfg.target_col, run_out / "calibration.png", cfg.n_calib_bins
        )
        _prob_hist(preds_norm, prob_col, run_out / "prob_hist.png")
        if pit_u.size > 0:
            _pit_plot(pit_u, run_out / "pit.png")

    # Report (MD + TeX)
    md_lines: list[str] = []
    md_lines.append("# Q3 — Predictive Model (from artifacts)")
    md_lines.append("")
    md_lines.append(f"- Source predictions: `{src.as_posix()}`")
    md_lines.append(f"- Rows: **{metrics['n_rows']}**, Races: **{metrics['n_races']}**")
    if metrics["notes"] == "no_winners_in_validation":
        md_lines.append(
            "> Note: validation set has **no winners**; Logloss/ECE/PIT are undefined. Brier is runner-level."
        )
    md_lines.append("")
    md_lines.append("## Metrics")
    md_lines.append(
        f"- logloss (race): {metrics['logloss']:.6f} | Brier: {metrics['brier']:.6f} | ECE({cfg.n_calib_bins}): {metrics['ece']:.6f}"
    )
    md_lines.append(
        f"- PIT mean: {metrics['pit_mean']:.3f} | var: {metrics['pit_var']:.3f} (≈ 0.083 ideal)"
    )
    if "roi_deciles_csv" in metrics:
        md_lines.append("")
        md_lines.append(f"ROI by probability decile written to `{metrics['roi_deciles_csv']}`.")
    report_md = "\n".join(md_lines)

    tex = r"""
\section*{Q3: Predictive Model (Report from Artifacts)}
We load calibrated predictions produced by the config-driven pipeline (see \texttt{src/hqp/models/*}), verify that per-race probabilities sum to $1$, and compute race-level metrics. We report log loss (winner log-prob per race), Brier score, and calibration (ECE). When available, we include randomized PIT diagnostics and ROI by predicted-probability deciles.

\paragraph{Top-line.}
\begin{itemize}
  \item Logloss (race): %(LLOG)s \quad Brier: %(BRI)s \quad ECE(%(NBIN)s): %(ECE)s.
\end{itemize}

Predictions were sourced from: \texttt{%(SRC)s}. All outputs for this summary are written under \texttt{reports/model/q3/}.
""".strip() % {
        "LLOG": f"{m['logloss']:.6f}",
        "BRI": f"{m['brier']:.6f}",
        "NBIN": f"{cfg.n_calib_bins}",
        "ECE": f"{m['ece']:.6f}",
        "SRC": src.as_posix(),
    }

    (run_out / "report.md").write_text(report_md, encoding="utf-8")
    (run_out / "report.tex").write_text(tex, encoding="utf-8")

    return {
        "run_dir": run_out,
        "metrics": metrics,
        "report_md": report_md,
        "report_tex": tex,
        "predictions_used": src.as_posix(),
    }


# Backwards-friendly alias (the project previously used run_q3 as the entry)
def run_q3(
    *,
    predictions_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    out_root: str | Path = "reports/model/q3",
    config: Q3ReportConfig | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """
    Alias to summarize_run() for convenience.
    - If you pass run_dir, it will look for predictions_cal/parquet (preferred).
    - If you pass predictions_path, it will use that file directly.
    - If 'won' is missing, you may pass labels_path to merge labels.
    """
    return summarize_run(
        predictions_path=predictions_path,
        run_dir=run_dir,
        labels_path=labels_path,
        out_root=out_root,
        config=config,
        make_plots=make_plots,
    )
