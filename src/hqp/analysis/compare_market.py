"""
Q4 — Compare Model vs Betfair Starting Price (BSP)
=================================================
Glue orchestrator that **reuses existing pipeline outputs** to answer Q4
without reimplementing backtests/strategies. It consumes a finished
`reports/backtest/<ts>/` folder and (optionally) predictions + market to
compute LL/Brier vs BSP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal, Callable, Any

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# Optional progress bar
try:
    from tqdm import trange  # type: ignore
except Exception:  # pragma: no cover
    trange = None  # fallback: no progress bar

_EPS: float = 1e-6

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _now_ts() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Profitability (reuse existing backtest outputs)
# -----------------------------------------------------------------------------


@dataclass
class BacktestArtifacts:
    summary: dict[str, Any]
    bets: pd.DataFrame


def load_backtest(backtest_dir: str | Path) -> BacktestArtifacts:
    backtest_dir = Path(backtest_dir)
    summary_path = backtest_dir / "summary.json"
    bets_path = backtest_dir / "bets.parquet"
    if not summary_path.exists() or not bets_path.exists():
        raise FileNotFoundError(
            f"Expected backtest artifacts not found under {backtest_dir}. "
            f"Have: summary.json? {summary_path.exists()} bets.parquet? {bets_path.exists()}"
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary: dict[str, Any] = json.load(f)
    bets = _read_table(str(bets_path))

    # normalize columns
    bets.columns = [c.lower() for c in bets.columns]

    # outcome column -> won (0/1)
    won_candidates = ["won", "obs__is_winner", "is_win", "position", "obs__place"]
    won: Optional[pd.Series] = None
    for c in won_candidates:
        if c in bets.columns:
            if c == "position":
                won = (bets[c] == 1).astype(int)
            elif c == "obs__place":
                won = (bets[c].astype(str).str.lower() == "win").astype(int)
            else:
                won = bets[c].astype(int)
            break
    if won is None:
        raise ValueError("bets.parquet lacks any recognizable outcome column")
    bets["won"] = won

    # odds & stakes standardisation
    if "mkt_odds_decimal" in bets.columns:
        odds = bets["mkt_odds_decimal"].astype(float)
    elif "odds" in bets.columns:
        odds = bets["odds"].astype(float)
    else:
        raise ValueError("bets.parquet must contain mkt_odds_decimal or odds column")
    bets["odds"] = odds
    bets["stake"] = bets["stake"].astype(float) if "stake" in bets.columns else 1.0

    # zero-commission per-bet return
    bets["ret"] = bets["won"].astype(float) * (bets["odds"].astype(float) - 1.0) * bets[
        "stake"
    ].astype(float) - (1.0 - bets["won"].astype(float)) * bets["stake"].astype(float)

    return BacktestArtifacts(summary=summary, bets=bets)


# -----------------------------------------------------------------------------
# Probabilistic comparison (optional) — join predictions with market/raw
# -----------------------------------------------------------------------------


def _normalize_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q_raw"] = 1.0 / df["obs__bsp"].astype(float)
    s = df.groupby("race_id")["q_raw"].transform("sum").replace(0, np.nan)
    df["q_mkt"] = (df["q_raw"] / s).clip(_EPS, 1 - _EPS)
    return df


def load_for_metrics(
    preds_path: Optional[str], market_or_raw_path: Optional[str]
) -> Optional[pd.DataFrame]:
    if not preds_path or not market_or_raw_path:
        return None

    dfp = _read_table(preds_path)
    dfr = _read_table(market_or_raw_path)

    # lower-case columns
    dfp.columns = [c.lower() for c in dfp.columns]
    dfr.columns = [c.lower() for c in dfr.columns]

    # accept model_prob or p_model
    if "model_prob" in dfp.columns and "p_model" not in dfp.columns:
        dfp = dfp.rename(columns={"model_prob": "p_model"})

    required_p = {"race_id", "horse_id", "p_model"}
    required_r = {"race_id", "horse_id", "obs__bsp", "obs__is_winner"}
    if not required_p.issubset(dfp.columns):
        miss = required_p - set(dfp.columns)
        raise ValueError(f"Predictions missing columns: {miss}")
    if not required_r.issubset(dfr.columns):
        miss = required_r - set(dfr.columns)
        raise ValueError(f"Market/raw missing columns: {miss}")

    # Coerce merge keys to a common dtype (string)
    for k in ("race_id", "horse_id"):
        dfp[k] = dfp[k].astype(str).str.strip()
        dfr[k] = dfr[k].astype(str).str.strip()

    # merge (allow many-to-many)
    df = dfp.merge(dfr, on=["race_id", "horse_id"], how="inner")

    # clean, clip, and compute market probs
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["p_model", "obs__bsp", "obs__is_winner"])
        .copy()
    )
    df["p_model"] = df["p_model"].astype(float).clip(_EPS, 1 - _EPS)
    df["obs__bsp"] = df["obs__bsp"].astype(float)
    df["y"] = df["obs__is_winner"].astype(int)

    df = _normalize_market_probs(df)

    return df


def compute_overall_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = np.asarray(df["y"].to_numpy(dtype=float))
    pm = np.asarray(df["p_model"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))
    qm = np.asarray(df["q_mkt"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))

    def _brier(pred: np.ndarray, tgt: np.ndarray) -> float:
        return float(np.mean((pred - tgt) ** 2))

    out: dict[str, float] = {
        "logloss_model": float(log_loss(y, pm)),
        "logloss_market": float(log_loss(y, qm)),
        "brier_model": _brier(pm, y),
        "brier_market": _brier(qm, y),
    }
    out["ll_diff_model_minus_market"] = out["logloss_model"] - out["logloss_market"]
    out["brier_diff_model_minus_market"] = out["brier_model"] - out["brier_market"]
    return out


def plot_calibration(df: pd.DataFrame, out_png: Path) -> None:
    # Equal-count bins on model and market separately
    def _tab(by: str) -> pd.DataFrame:
        bins = pd.qcut(df[by], q=10, duplicates="drop")
        return df.groupby(bins, observed=False).agg(p_bar=(by, "mean"), y_bar=("y", "mean"))

    tab_m = _tab("p_model").reset_index(drop=True)
    tab_q = _tab("q_mkt").reset_index(drop=True)

    plt.figure(figsize=(6, 5))
    plt.plot(tab_m["p_bar"], tab_m["y_bar"], marker="o", label="Model")
    plt.plot(tab_q["p_bar"], tab_q["y_bar"], marker="s", label="Market")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration: Model vs Market")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def block_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    block_key: Literal["date", "race_id"] = "race_id",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 42,
    show_progress: bool = True,
) -> Tuple[float, float]:
    # Ensure valid key and contiguous iloc index
    if block_key not in df.columns:
        block_key = "race_id"
    df = df.reset_index(drop=True)

    # Map each block to its row indices (typed so Pylance is happy)
    blk_codes, blk_uniques = pd.factorize(df[block_key], sort=False)
    n_blocks: int = int(blk_uniques.size)
    block_rows: list[np.ndarray] = [
        np.asarray(np.where(blk_codes == b)[0], dtype=np.intp) for b in range(n_blocks)
    ]

    rng = np.random.default_rng(random_state)
    stats: list[float] = []

    iterator = range(n_boot)
    if show_progress and trange is not None:
        iterator = trange(n_boot, desc="Bootstrapping", leave=False)  # type: ignore

    try:
        for _ in iterator:
            sampled_blocks = rng.integers(0, n_blocks, size=n_blocks, endpoint=False)
            # Use tuple(...) so arrays is a tuple[np.ndarray, ...] (no list[Unknown])
            selected_idx = np.concatenate(tuple(block_rows[int(b)] for b in sampled_blocks))
            try:
                stats.append(float(stat_fn(df.iloc[selected_idx])))
            except Exception:
                continue
    except KeyboardInterrupt:
        pass

    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [alpha / 2.0, 1 - alpha / 2.0]).tolist()
    return float(lo), float(hi)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def run_q4(
    backtest_dir: str,
    preds_path: Optional[str] = None,
    market_or_raw_path: Optional[str] = None,
    out_base: str = "reports/analysis/q4",
    n_boot: int = 1000,
) -> str:
    """Assemble Q4 answers using existing artifacts."""
    bt = load_backtest(backtest_dir)

    ts_dir = _ensure_dir(Path(out_base) / _now_ts())

    # 1) Profitability (ROI) + CI from bets
    total_staked = float(bt.bets["stake"].sum()) if len(bt.bets) else 0.0
    total_pnl = float(bt.bets["ret"].sum()) if len(bt.bets) else 0.0
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    def _roi_stat(d: pd.DataFrame) -> float:
        staked = float(d["stake"].sum())
        pnl = float(d["ret"].sum())
        return pnl / staked if staked > 0 else 0.0

    roi_lo, roi_hi = block_bootstrap(
        bt.bets, _roi_stat, block_key="race_id", n_boot=n_boot, show_progress=True
    )

    # 2) Probabilistic comparison (optional)
    metrics: Optional[dict[str, float]] = None
    ll_ci: Tuple[float, float] = (float("nan"), float("nan"))
    brier_ci: Tuple[float, float] = (float("nan"), float("nan"))
    if preds_path and market_or_raw_path:
        dfm = load_for_metrics(preds_path, market_or_raw_path)
        if dfm is not None and not dfm.empty:
            metrics = compute_overall_metrics(dfm)

            def _ll_diff(d: pd.DataFrame) -> float:
                y = np.asarray(d["y"].to_numpy(dtype=float))
                pm = np.asarray(d["p_model"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))
                qm = np.asarray(d["q_mkt"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))
                return float(log_loss(y, pm) - log_loss(y, qm))

            def _brier_diff(d: pd.DataFrame) -> float:
                y = np.asarray(d["y"].to_numpy(dtype=float))
                pm = np.asarray(d["p_model"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))
                qm = np.asarray(d["q_mkt"].clip(_EPS, 1 - _EPS).to_numpy(dtype=float))
                return float(np.mean((pm - y) ** 2) - np.mean((qm - y) ** 2))

            ll_ci = block_bootstrap(
                dfm, _ll_diff, block_key="race_id", n_boot=n_boot, show_progress=True
            )
            brier_ci = block_bootstrap(
                dfm, _brier_diff, block_key="race_id", n_boot=n_boot, show_progress=True
            )

            # Calibration & densities
            try:
                plot_calibration(dfm, ts_dir / "calibration_model_vs_market.png")
                plt.figure(figsize=(6, 4))
                plt.hist(dfm["p_model"], bins=40, alpha=0.6, label="Model p", density=True)
                plt.hist(dfm["q_mkt"], bins=40, alpha=0.6, label="Market q", density=True)
                plt.xlabel("Probability")
                plt.ylabel("Density")
                plt.title("Model vs Market Distributions")
                plt.legend()
                plt.tight_layout()
                plt.savefig(ts_dir / "density_bsp_vs_model.png")
                plt.close()
            except Exception:
                pass

    # 3) Write outputs
    met_rows: dict[str, float | int] = {
        "roi": roi,
        "roi_lo": roi_lo,
        "roi_hi": roi_hi,
        "bets": int(bt.summary.get("bets", len(bt.bets))),
        "pnl": float(bt.summary.get("pnl", total_pnl)),
        "hit_rate": float(
            bt.summary.get("hit_rate", float(bt.bets["won"].mean()) if len(bt.bets) else 0.0)
        ),
        "avg_odds": float(
            bt.summary.get("avg_odds", float(bt.bets["odds"].mean()) if len(bt.bets) else 0.0)
        ),
    }
    if metrics is not None:
        met_rows.update({k: float(v) for k, v in metrics.items()})
        met_rows.update(
            {
                "ll_diff_ci_lo": float(ll_ci[0]),
                "ll_diff_ci_hi": float(ll_ci[1]),
                "brier_diff_ci_lo": float(brier_ci[0]),
                "brier_diff_ci_hi": float(brier_ci[1]),
            }
        )

    pd.DataFrame([met_rows]).to_csv(ts_dir / "metrics_overall.csv", index=False)

    per_race = (
        bt.bets.groupby("race_id")
        .agg(
            bets=("race_id", "size"),
            staked=("stake", "sum"),
            pnl=("ret", "sum"),
        )
        .reset_index()
    )
    per_race.to_csv(ts_dir / "per_race_strategy.csv", index=False)

    ci_rows: list[dict[str, float | str]] = [
        {"metric": "ROI", "lo": float(roi_lo), "hi": float(roi_hi)}
    ]
    if metrics is not None:
        ci_rows.append(
            {"metric": "LL_diff(model-market)", "lo": float(ll_ci[0]), "hi": float(ll_ci[1])}
        )
        ci_rows.append(
            {
                "metric": "Brier_diff(model-market)",
                "lo": float(brier_ci[0]),
                "hi": float(brier_ci[1]),
            }
        )
    pd.DataFrame(ci_rows).to_csv(ts_dir / "bootstrap_ci.csv", index=False)

    # diagnostics
    (ts_dir / "diagnostics.md").write_text(
        json.dumps(
            {
                "backtest_dir": str(backtest_dir),
                "preds_path": preds_path,
                "market_or_raw_path": market_or_raw_path,
                "n_bets": int(len(bt.bets)),
                "bootstrap": {"n_boot": n_boot, "block": "race_id"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # summary.txt — human-readable answer
    lines: list[str] = []
    lines.append("Does the model beat BSP overall?\n")
    if metrics is None:
        lines.append("- Probabilistic comparison not computed (no preds + market join provided).\n")
    else:
        lines.append(
            "- LogLoss: model={:.4f} vs market={:.4f} \u2192 \u0394={:.4f} (95% CI [{:.4f}, {:.4f}]).\n".format(
                float(met_rows.get("logloss_model", float("nan"))),
                float(met_rows.get("logloss_market", float("nan"))),
                float(met_rows.get("ll_diff_model_minus_market", float("nan"))),
                float(met_rows.get("ll_diff_ci_lo", float("nan"))),
                float(met_rows.get("ll_diff_ci_hi", float("nan"))),
            )
        )
        lines.append(
            "- Brier:   model={:.4f} vs market={:.4f} \u2192 \u0394={:.4f} (95% CI [{:.4f}, {:.4f}]).\n".format(
                float(met_rows.get("brier_model", float("nan"))),
                float(met_rows.get("brier_market", float("nan"))),
                float(met_rows.get("brier_diff_model_minus_market", float("nan"))),
                float(met_rows.get("brier_diff_ci_lo", float("nan"))),
                float(met_rows.get("brier_diff_ci_hi", float("nan"))),
            )
        )
    lines.append("\nProfitable strategy (zero commission)?\n")
    lines.append(
        "- Backtest ROI={:.2f}% (95% CI [{:.2f}%, {:.2f}%]), bets={}, hit_rate={:.3f}, avg_odds={:.2f}.\n".format(
            roi * 100.0,
            roi_lo * 100.0,
            roi_hi * 100.0,
            int(met_rows["bets"]),
            float(met_rows["hit_rate"]),
            float(met_rows["avg_odds"]),
        )
    )
    if roi_lo > 0:
        lines.append("- Conclusion: Profitable and statistically above 0 at 95% level.\n")
    elif roi_hi < 0:
        lines.append("- Conclusion: Unprofitable (ROI significantly below 0).\n")
    else:
        lines.append("- Conclusion: Not statistically different from 0 given CI.\n")

    (ts_dir / "summary.txt").write_text("".join(lines), encoding="utf-8")

    return str(ts_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Q4: Compare model vs BSP (glue-only)")
    p.add_argument("--backtest_dir", required=True, help="Path to reports/backtest/<ts> directory")
    p.add_argument(
        "--preds_path",
        default=None,
        help="Optional predictions parquet/csv with p_model/model_prob",
    )
    p.add_argument(
        "--market_or_raw_path",
        default=None,
        help="Optional market/raw file with obs__bsp and obs__is_winner columns",
    )
    p.add_argument("--out_base", default="reports/analysis/q4", help="Output base directory")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations for CIs")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args: argparse.Namespace = _build_argparser().parse_args(argv)
    # Note: we don't thread no_progress into run_q4 signature; instead, flip trange to None if disabled.
    global trange  # simple toggle
    if getattr(args, "no_progress", False):
        trange = None  # type: ignore

    out = run_q4(
        backtest_dir=str(args.backtest_dir),
        preds_path=(str(args.preds_path) if args.preds_path is not None else None),
        market_or_raw_path=(
            str(args.market_or_raw_path) if args.market_or_raw_path is not None else None
        ),
        out_base=str(args.out_base),
        n_boot=int(args.n_boot),
    )
    print(out)


if __name__ == "__main__":
    main()
