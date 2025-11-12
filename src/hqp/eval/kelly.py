# src/hqp/eval/kelly.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, Union, TypeAlias

import pandas as pd
import yaml


@dataclass(frozen=True)
class FamilySpec:
    name: str
    glob: str  # e.g. "configs/backtest_calibrated_precision_kelly_*.yaml"


@dataclass(frozen=True)
class CalibrationSpec:
    name: str  # "platt" | "iso"
    edges_path: Path  # data/market/edges_cal.parquet | data/market/edges_cal_iso.parquet


StrPath: TypeAlias = Union[str, os.PathLike[str], Path]


def _ensure_utf8(paths: List[Path]) -> None:
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="utf-16")
        p.write_text(txt, encoding="utf-8")


def _yaml_load(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data or {})


def _sanity_check_edges(edges: Path) -> None:
    if not edges.exists():
        raise RuntimeError(f"[kelly-sweep] edges file not found: {edges}")
    try:
        df = pd.read_parquet(edges)
    except Exception as e:
        raise RuntimeError(f"[kelly-sweep] failed to read edges {edges}: {e}")
    missing = [c for c in ("race_id", "horse_id", "edge") if c not in df.columns]
    if missing:
        raise RuntimeError(f"[kelly-sweep] edges missing columns {missing} in {edges}")
    if len(df) == 0:
        raise RuntimeError(f"[kelly-sweep] edges file is empty: {edges}")


def _as_path(x: Path | os.PathLike[str] | str | bytes | bytearray) -> Path:
    if isinstance(x, Path):
        return x
    if isinstance(x, os.PathLike):
        fs: str | bytes = os.fspath(x)
        if isinstance(fs, bytes):
            fs = fs.decode("utf-8", "ignore")
        return Path(fs)
    if isinstance(x, (bytes, bytearray)):
        return Path(bytes(x).decode("utf-8", "ignore"))
    return Path(x)


def _attach_labels_to_edges(edges_in: Path, out_dir: Path) -> Path:
    df = pd.read_parquet(edges_in)
    if "won" in df.columns:
        return edges_in

    label_sources: List[Path] = [
        Path("data/interim/base.parquet"),
        Path("data/features/features.parquet"),
    ]
    src: Optional[Path] = next((p for p in label_sources if p.exists()), None)
    if src is None:
        return edges_in

    df_lbl = pd.read_parquet(src)
    cand: List[str] = [
        c for c in ["won", "obs__is_winner", "is_winner", "label", "target"] if c in df_lbl.columns
    ]
    if not cand:
        return edges_in
    ycol = cand[0]

    for k in ("race_id", "horse_id"):
        if k in df.columns:
            df[k] = df[k].astype("string")
        if k in df_lbl.columns:
            df_lbl[k] = df_lbl[k].astype("string")

    merged = df.merge(df_lbl[["race_id", "horse_id", ycol]], on=["race_id", "horse_id"], how="left")
    if merged[ycol].isna().all():
        return edges_in

    won_series = pd.to_numeric(merged[ycol], errors="coerce").fillna(0.0)
    won_series = won_series.astype("float64").round().clip(lower=0, upper=1).astype("int64")
    merged["won"] = won_series
    merged = merged.drop(columns=[ycol], errors="ignore")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "edges_with_labels.parquet"
    merged.to_parquet(out_p, index=False)
    return out_p


def _to_expected_key_from_int_code(code: int) -> Optional[str]:
    if code in (0, 5, 10, 20, 30):
        return f"k{code:03d}"
    if code in (50, 100, 200, 300):
        mapped = {50: 5, 100: 10, 200: 20, 300: 30}[code]
        return f"k{mapped:03d}"
    choices: tuple[int, ...] = (0, 5, 10, 20, 30)
    diffs = [abs(c - code) for c in choices]
    nearest = choices[diffs.index(min(diffs))]
    if abs(nearest - code) <= 2:
        return f"k{nearest:03d}"
    return None


def _discover_kelly_configs_by_content(pattern: str) -> Dict[str, Path]:
    files = sorted(Path(".").glob(pattern))
    out: Dict[str, Path] = {}
    debug_lines: List[str] = []

    for p in files:
        key: Optional[str] = None
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            data: Dict[str, Any] = cast(Dict[str, Any], loaded)
        except Exception:
            data = {}

        if data:
            bt_raw = cast(Dict[str, Any], (data.get("backtest") or {}))
            if "kelly_fraction" in bt_raw:
                val_any: Any = bt_raw.get("kelly_fraction", 0.0)
                if isinstance(val_any, (int, float, str)):
                    try:
                        frac_val = float(val_any)
                        code = int(round(frac_val * 100))
                        key = _to_expected_key_from_int_code(code)
                    except Exception:
                        key = None
            else:
                staking_raw = cast(Dict[str, Any], (data.get("staking") or {}))
                stype = staking_raw.get("type")
                if stype == "fixed":
                    key = "k000"
                elif stype == "kelly":
                    frac_any: Any = staking_raw.get("fraction")
                    if isinstance(frac_any, (int, float, str)):
                        try:
                            frac_val = float(frac_any)
                            code = int(round(frac_val * 100))
                            key = _to_expected_key_from_int_code(code)
                        except Exception:
                            key = None

        if key is None:
            stem = p.stem
            suffix = stem.split("_")[-1]
            if suffix.isdigit():
                code = int(suffix)
                key = _to_expected_key_from_int_code(code)

        if key is not None:
            out.setdefault(key, p)
            debug_lines.append(f"  -> {p.name}  ==>  {key}")
        else:
            debug_lines.append(f"  !! {p.name}  ==>  could not map to k000/005/010/020/030")

    expected = ["k000", "k005", "k010", "k020", "k030"]
    mapped = {k: out[k] for k in expected if k in out}

    if debug_lines:
        print("[kelly-sweep] Discovery report for pattern:", pattern)
        for line in debug_lines:
            print(line)
        missing = [k for k in expected if k not in mapped]
        if missing:
            print("[kelly-sweep] Missing expected keys:", ", ".join(missing))

    return mapped


def _run_backtest_compare(
    edges: Path, keyed_configs: Dict[str, Path], tag: str, cwd: Optional[Path] = None
) -> Path:
    _sanity_check_edges(edges)
    tmp_dir = Path("reports/backtest/_kelly_tmp") / tag
    edges_for_bt = _attach_labels_to_edges(edges, tmp_dir)

    market_default = Path("data/market/market_join.parquet")
    market_exists = market_default.exists()

    try:
        from hqp.eval.backtest import run as _bt_run  # type: ignore
    except Exception as e:
        raise RuntimeError("[kelly-sweep] could not import hqp.eval.backtest.run") from e

    rows_for_csv: List[Dict[str, Any]] = []

    for key, cfg_path in keyed_configs.items():
        cfg_data: Dict[str, Any] = _yaml_load(cfg_path)

        if "backtest" in cfg_data and isinstance(cfg_data["backtest"], dict):
            bt_raw = cast(Dict[str, Any], cfg_data["backtest"])
            bt_cfg: Dict[str, Any] = {str(k): v for k, v in bt_raw.items()}
            for k in ("staking", "filters", "selection", "pricing", "notes"):
                if k in cfg_data:
                    bt_cfg[k] = cfg_data[k]
        else:
            bt_cfg = dict(cfg_data)

        bt_cfg.setdefault("try_join_market", True)
        if market_exists:
            bt_cfg.setdefault("market_path", str(market_default))

        # --- translate staking -> TOP-LEVEL engine fields + enable bankroll mode ---
        staking = cast(Dict[str, Any], bt_cfg.get("staking", {}) or {})
        stype = staking.get("type")
        if stype == "kelly":
            frac = staking.get("fraction", 0.0)
            try:
                bt_cfg["kelly_fraction"] = float(frac)
            except Exception:
                bt_cfg["kelly_fraction"] = 0.0

            # Enable bankroll mode so ROI changes with Kelly fraction
            # (defaults; can be overridden by adding keys under staking)
            bt_cfg.setdefault("bankroll_init", float(staking.get("bankroll", 1000.0)))
            if "stake_min" in staking:
                bt_cfg["stake_min"] = float(staking["stake_min"])
            if "stake_max" in staking:
                bt_cfg["stake_max"] = float(staking["stake_max"])
        elif stype == "fixed":
            stake = staking.get("stake", 1.0)
            try:
                bt_cfg["stake"] = float(stake)
            except Exception:
                bt_cfg["stake"] = 1.0
            bt_cfg["kelly_fraction"] = 0.0
            # Do not set bankroll_init for fixed

        print(
            f"[debug] key={key} -> kelly_fraction={bt_cfg.get('kelly_fraction')} stake={bt_cfg.get('stake', 'n/a')}"
        )

        result_any: Any = _bt_run(edges_for_bt, bt_cfg)
        out_dir: Path = _as_path(cast(StrPath, result_any))

        summary_p = out_dir / "summary.json"
        try:
            s_raw = json.loads(summary_p.read_text(encoding="utf-8"))
            s: Dict[str, Any] = cast(Dict[str, Any], s_raw)
            row: Dict[str, Any] = {
                "key": key,
                "config_path": str(cfg_path),
                "rows": s.get("rows"),
                "bets": s.get("bets"),
                "races_bet": s.get("races_bet", s.get("races")),
                "pnl": s.get("pnl"),
                "roi": s.get("roi"),
                "hit_rate": s.get("hit_rate"),
                "avg_odds": s.get("avg_odds"),
                "result_dir": str(out_dir),
            }
        except Exception:
            row = {
                "key": key,
                "config_path": str(cfg_path),
                "rows": None,
                "bets": None,
                "races_bet": None,
                "pnl": None,
                "roi": None,
                "hit_rate": None,
                "avg_odds": None,
                "result_dir": str(out_dir),
            }
        rows_for_csv.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/backtest") / f"compare_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_cols = ["key", "bets", "roi", "hit_rate", "avg_odds", "pnl", "result_dir", "config_path"]
    df = pd.DataFrame(rows_for_csv)
    df = df[[c for c in tidy_cols if c in df.columns]]
    df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")

    prov: Dict[str, Any] = {
        "edges": str(edges_for_bt),
        "configs": {k: str(v) for k, v in keyed_configs.items()},
        "tag": tag,
        "ts": ts,
        "market_path": str(market_default) if market_exists else None,
        "labels_attached": edges_for_bt != edges,
    }
    (out_dir / "_source.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return out_dir


def _read_summary_csv(path: Path) -> List[Dict[str, str]]:
    fp = path / "summary.csv"
    if not fp.exists():
        raise FileNotFoundError(f"summary.csv not found in {path}")
    rows: List[Dict[str, str]] = []
    with fp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # type: ignore[type-arg]
        for r in reader:
            rows.append(cast(Dict[str, str], r))
    return rows


def _write_master_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)  # type: ignore[type-arg]
        w.writeheader()
        w.writerows(rows)


def _write_master_md(rows: List[Dict[str, str]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        key = (r.get("family", ""), r.get("calibration", ""))
        grouped.setdefault(key, []).append(r)

    lines: List[str] = ["# Kelly Sweep — Summary\n"]
    for (fam, cal), rs in grouped.items():
        lines.append(f"## {fam} — {cal}")
        if not rs:
            lines.append("_No rows._\n")
            continue
        rs_sorted = sorted(rs, key=lambda x: x.get("key", ""))
        cols = [
            c
            for c in ["key", "bets", "roi", "pnl", "hit_rate", "avg_odds", "result_dir"]
            if c in rs_sorted[0]
        ]
        if not cols:
            cols = list(rs_sorted[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for r in rs_sorted:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def kelly_sweep(
    families: Optional[List[FamilySpec]] = None,
    calibrations: Optional[List[CalibrationSpec]] = None,
    project_root: Path = Path("."),
    output_root: Path = Path("reports/backtest"),
) -> Dict[str, object]:
    if families is None:
        families = [
            FamilySpec("precision", "configs/backtest_calibrated_precision_kelly_*.yaml"),
            FamilySpec("balanced", "configs/backtest_calibrated_balanced_kelly_*.yaml"),
            FamilySpec("high_volume", "configs/backtest_calibrated_high_volume_kelly_*.yaml"),
        ]
    if calibrations is None:
        calibrations = [
            CalibrationSpec("platt", project_root / "data/market/edges_cal.parquet"),
            CalibrationSpec("iso", project_root / "data/market/edges_cal_iso.parquet"),
        ]

    all_rows: List[Dict[str, str]] = []
    created_dirs: List[Path] = []

    for fam in families:
        keyed = _discover_kelly_configs_by_content(fam.glob)
        if not keyed or set(keyed.keys()) != {"k000", "k005", "k010", "k020", "k030"}:
            raise RuntimeError(
                f"[kelly-sweep] Expected 5 Kelly configs for family '{fam.name}' via '{fam.glob}', got: {list(keyed.keys())}"
            )
        _ensure_utf8(list(keyed.values()))

        for cal in calibrations:
            tag = f"{fam.name}_{cal.name}"
            out_dir = _run_backtest_compare(
                edges=cal.edges_path, keyed_configs=keyed, tag=tag, cwd=project_root
            )
            created_dirs.append(out_dir)

            rows = _read_summary_csv(out_dir)

            if rows and "key" not in rows[0] and "label" in rows[0]:
                for r in rows:
                    r["key"] = r.get("label", "")

            for r in rows:
                r["family"] = fam.name
                r["calibration"] = cal.name
                r["compare_dir"] = str(out_dir)
            all_rows.extend(rows)

    master_dir = output_root / "kelly_master"
    master_csv = master_dir / "summary.csv"
    master_md = master_dir / "summary.md"
    _write_master_csv(all_rows, master_csv)
    _write_master_md(all_rows, master_md)

    return {
        "created_compare_dirs": [str(p) for p in created_dirs],
        "master_csv": str(master_csv),
        "master_md": str(master_md),
        "rows": all_rows,
    }
