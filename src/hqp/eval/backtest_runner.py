# src/hqp/eval/backtest_runner.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Backtest Runner (Config Compare)
#
# Purpose
# -------
# Drive the CLI backtest for multiple YAML configs, parse the printed summary
# lines from stdout, and return a compact comparison table. Also writes a
# timestamped parquet with all results for auditability.
#
# Design notes
# ------------
# - We invoke the Typer CLI (`hqp backtest ...`), but fall back to
#   `python -m hqp.cli backtest ...` if the shim isn't on PATH.
# - The parser is tolerant to minor format changes: it looks for key tokens
#   using regex groups (bets / pnl / roi / hit_rate / avg_odds / report_dir).
# - We *do not* fail the whole comparison if an individual run prints slightly
#   different lines; missing fields simply remain None.
# - Output parquet path: reports/backtest/compare_<YYYYmmdd_HHMMSS>.parquet
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict

import pandas as pd

# Regexes:
# - Be permissive with floats (allow integer-like too): -12, 3, -0.12, 10.88, etc.
_FLOAT = r"-?\d+(?:\.\d+)?"

_RUN_LINE = re.compile(
    rf"roi=(?P<roi>{_FLOAT}).*?hit_rate=(?P<hit_rate>{_FLOAT}).*?avg_odds=(?P<avg_odds>{_FLOAT})",
    re.IGNORECASE,
)
_REPORT_LINE = re.compile(r"->\s*(?P<path>reports[\\/].+)", re.IGNORECASE)
_HEADER_LINE = re.compile(
    rf"bets=(?P<bets>\d+).*?roi=(?P<roi>{_FLOAT}).*?pnl=(?P<pnl>{_FLOAT})",
    re.IGNORECASE,
)


@dataclass
class BacktestResult:
    """
    Parsed summary for a single backtest config.
    All numeric fields are Optional: if parsing fails for a line, the field remains None.
    """

    label: str
    config_path: str
    report_dir: Optional[str]
    bets: Optional[int]
    roi: Optional[float]
    hit_rate: Optional[float]
    avg_odds: Optional[float]
    pnl: Optional[float]


class _ParsedStdout(TypedDict, total=False):
    bets: Optional[int]
    pnl: Optional[float]
    roi: Optional[float]
    hit_rate: Optional[float]
    avg_odds: Optional[float]
    report_dir: Optional[str]


def _call_cli_backtest(
    edges: str, config_path: str, extra_args: Optional[Iterable[str]] = None
) -> str:
    """
    Call the Typer CLI:
        hqp backtest --edges <...> --config <...> [extra_args...]
    Fallback to:
        python -m hqp.cli backtest ...
    Returns combined stdout+stderr for robust parsing.
    """
    cmd = ["hqp", "backtest", "--edges", edges, "--config", config_path]
    if extra_args:
        cmd.extend(map(str, extra_args))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (proc.stdout or "") + ("\n" + (proc.stderr or ""))
    except (FileNotFoundError, subprocess.CalledProcessError):
        alt = [
            sys.executable,
            "-m",
            "hqp.cli",
            "backtest",
            "--edges",
            edges,
            "--config",
            config_path,
        ]
        if extra_args:
            alt.extend(map(str, extra_args))
        proc = subprocess.run(alt, check=True, capture_output=True, text=True)
        return (proc.stdout or "") + ("\n" + (proc.stderr or ""))


def _parse_stdout(stdout: str) -> _ParsedStdout:
    """
    Parse known backtest log lines like:
      "[backtest] rows=... bets=56316 races=... pnl=-190.71 roi=-0.126 hit_rate=0.107 avg_odds=10.88 -> reports\\backtest\\YYYYmmdd_HHMMSS"
    We scan all lines, updating fields as we find matches. The last occurrence wins.
    """
    bets: Optional[int] = None
    pnl: Optional[float] = None
    roi: Optional[float] = None
    hit_rate: Optional[float] = None
    avg_odds: Optional[float] = None
    report_dir: Optional[str] = None

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue

        m0 = _HEADER_LINE.search(line)
        if m0:
            try:
                bets = int(m0.group("bets"))
            except Exception:
                pass
            try:
                pnl = float(m0.group("pnl"))
            except Exception:
                pass
            try:
                roi = float(m0.group("roi"))
            except Exception:
                pass

        m1 = _RUN_LINE.search(line)
        if m1:
            try:
                roi = float(m1.group("roi"))
            except Exception:
                pass
            try:
                hit_rate = float(m1.group("hit_rate"))
            except Exception:
                pass
            try:
                avg_odds = float(m1.group("avg_odds"))
            except Exception:
                pass

        m2 = _REPORT_LINE.search(line)
        if m2:
            report_dir = m2.group("path").replace("\\", "/")

    return {
        "bets": bets,
        "pnl": pnl,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "report_dir": report_dir,
    }


def _validate_config_paths(configs: Dict[str, str]) -> None:
    """
    Ensure every declared config exists; raise with a helpful message otherwise.
    We validate up front so that typos or missing files don't lead to partial runs.
    """
    missing = [(label, path) for label, path in configs.items() if not Path(path).exists()]
    if missing:
        lines = "\n".join(f"  - {lbl}: {pth}" for lbl, pth in missing)
        raise FileNotFoundError(
            "Missing backtest config file(s):\n"
            f"{lines}\n\nCreate them or pass an explicit --configs label=path set."
        )


def compare_configs(
    edges_path: str,
    configs: Dict[str, str],
    outdir_root: str = "reports/backtest",
    extra_args: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Run backtests for each config and return a summary DataFrame with:
      [label, bets, roi, hit_rate, avg_odds, pnl, config_path, report_dir].
    Also writes: reports/backtest/compare_<ts>.parquet

    This is intentionally side-effect-light: the actual per-config backtest
    still writes its own report directory; here we only collect a summary table.
    """
    _validate_config_paths(configs)

    results: List[BacktestResult] = []

    for label, cfg in configs.items():
        stdout = _call_cli_backtest(edges=edges_path, config_path=cfg, extra_args=extra_args)
        parsed = _parse_stdout(stdout)
        results.append(
            BacktestResult(
                label=label,
                config_path=cfg,
                report_dir=parsed.get("report_dir"),
                bets=parsed.get("bets"),
                roi=parsed.get("roi"),
                hit_rate=parsed.get("hit_rate"),
                avg_odds=parsed.get("avg_odds"),
                pnl=parsed.get("pnl"),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = Path(outdir_root) / f"compare_{ts}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    # Compact console table (robust to missing values)
    display_cols = ["label", "bets", "roi", "hit_rate", "avg_odds", "pnl"]
    present_cols = [c for c in display_cols if c in df.columns]
    display = df[present_cols].sort_values("roi", ascending=False, na_position="last")
    print("\n[backtest-compare] Summary (best ROI first):")
    print(display.to_string(index=False))

    return df


def default_config_set() -> Dict[str, str]:
    """
    Defines a sensible trio: default / strict / relaxed.
    Make sure these paths exist in your repo.
    """
    return {
        "default": "configs/backtest.yaml",
        "strict": "configs/backtest_baseline.yaml",
        "relaxed": "configs/backtest_relaxed.yaml",  # optional; create if you want looser thresholds
    }


if __name__ == "__main__":
    # CLI-lite entry for ad-hoc use (python -m hqp.eval.backtest_runner --edges ... --configs ...):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="data/market/edges.parquet")
    ap.add_argument(
        "--configs", nargs="*", help="Pairs label=path. If omitted, uses default_config_set()."
    )
    args, extra = ap.parse_known_args()

    if args.configs:
        cfgs = dict(pair.split("=", 1) for pair in args.configs)
    else:
        cfgs = default_config_set()

    compare_configs(edges_path=args.edges, configs=cfgs, extra_args=extra)
