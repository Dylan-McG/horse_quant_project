# src/hqp/eval/calibration.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Calibration Diagnostics
#
# Purpose
# -------
# Given a predictions parquet with:
#   - model_prob : float in [0,1]
#   - won        : 0/1 outcome
# produce:
#   - deciles.parquet    (expected vs observed by probability decile)
#   - calibration.png    (curve of observed vs expected)
#   - summary.json       (rows, Brier score, ECE over deciles, paths)
#
# Design Notes
# ------------
# - We use equal-width deciles on [0,1]. ECE is sum over bins of
#     w_b * |observed_b - expected_b|
#   where w_b is the bin frequency / N.
# - We clip model_prob to [0,1] and drop NaNs before aggregation.
# - Plotting is minimalist and stable for CI.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CalibrationSummary(TypedDict):
    rows: int
    brier: float
    ece_deciles: float
    deciles_path: str
    plot_path: str
    predictions_path: str


def _decile_edges(n: int = 10) -> List[float]:
    """Return n equal-width bin edges over [0, 1] inclusive (as a list[float] for pandas-stubs)."""
    # np.linspace returns ndarray[float]; convert to list[float] for pd.cut type expectations
    return np.linspace(0.0, 1.0, n + 1, dtype=float).tolist()


def calibrate_from_predictions(pred_path: str, outdir: str) -> CalibrationSummary:
    """
    Read predictions parquet with columns:
      - 'model_prob' (float in [0,1])
      - 'won' (0/1)
    Produce:
      - deciles.parquet: expected vs observed per probability decile
      - calibration.png: calibration curve
      - summary.json: Brier score, ECE, counts

    Returns
    -------
    CalibrationSummary : Typed dict with key paths and summary metrics.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pred_path)
    if "model_prob" not in df.columns or "won" not in df.columns:
        raise ValueError("Expected columns 'model_prob' and 'won' in predictions parquet.")

    # Clean & clip
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_prob", "won"]).copy()
    df["model_prob"] = df["model_prob"].astype(float).clip(0.0, 1.0)
    df["won"] = df["won"].astype(int)

    # Equal-width deciles on [0,1]; bins as list[float] to satisfy pandas-stubs
    bins: List[float] = _decile_edges(10)
    df["decile"] = pd.cut(
        df["model_prob"],
        bins=bins,
        include_lowest=True,
        right=True,
        labels=False,
    )

    agg = (
        df.groupby("decile", dropna=False)
        .agg(
            count=("won", "size"),
            expected=("model_prob", "mean"),
            observed=("won", "mean"),
        )
        .reset_index()
    )

    # Brier score
    brier = float(np.mean((df["model_prob"] - df["won"]) ** 2))

    # Expected Calibration Error (equal-width bins)
    ece = float(np.sum(np.abs(agg["observed"] - agg["expected"]) * (agg["count"] / len(df))))

    # Save deciles table
    deciles_path = out / "deciles.parquet"
    agg.to_parquet(deciles_path, index=False)

    # Plot calibration curve
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(agg["expected"], agg["observed"], marker="o")
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Observed win rate")
    plt.title("Calibration curve")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out / "calibration.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary: CalibrationSummary = {
        "rows": int(len(df)),
        "brier": brier,
        "ece_deciles": ece,
        "deciles_path": str(deciles_path),
        "plot_path": str(plot_path),
        "predictions_path": pred_path,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console preview for quick inspection
    preview = (
        agg.assign(decile_label=lambda x: x["decile"].astype(int) + 1)[
            ["decile_label", "count", "expected", "observed"]
        ]
        .rename(columns={"decile_label": "decile(1-10)"})
        .to_string(index=False, float_format=lambda v: f"{v:0.3f}")
    )
    print("\n[calibration] Deciles:")
    print(preview)
    print(f"\n[calibration] Brier={brier:0.5f}, ECE(deciles)={ece:0.5f}")
    print(f"[calibration] Wrote: {out}")

    return summary
