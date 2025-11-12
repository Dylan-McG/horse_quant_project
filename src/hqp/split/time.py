# src/hqp/split/time.py
# -----------------------------------------------------------------------------
# Forward-chronological data splits with *race-level grouping*.
#
# Why grouping?
#   All runners from a race must be in the same partition to avoid label/market
#   leakage across rows of the same event.
#
# Strategies:
#   1) "single_cutoff":
#        - Train: races with min(timestamp) <= cutoff (if provided), else the
#          earliest (1 - val_fraction - test_fraction) of races by time.
#        - Valid/Test: the remaining races in temporal order by fraction.
#        - Enforces: exact coverage, no overlap, and chronology (train <= valid <= test).
#
#   2) "kfold":
#        - Time-ordered “expanding window” GroupKFold over races:
#          Fold i: Train = all earlier blocks; Valid = block i.
#        - Enforces: no overlap; valid never precedes train end.
#
# Outputs:
#   - A dict of boolean masks aligned to df.index.
#   - Side effect: writes masks as row-index parquet files under output_dir and a manifest.json.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd
import yaml

Strategy = Literal["single_cutoff", "kfold"]


@dataclass(frozen=True)
class Defaults:
    seed: int = 42
    output_dir: str = "data/splits"
    date_column: str = "race_datetime"
    group_column: str = "race_id"
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    k_folds: int = 5


class SplitConfig(TypedDict, total=False):
    # Common
    strategy: Strategy
    date_column: str
    group_column: str
    seed: int
    output_dir: str
    # single_cutoff
    cutoff: str | None
    val_fraction: float
    test_fraction: float
    # kfold
    k_folds: int


# YAML surface schema (used by forward_splits)
class YamlStrategy(TypedDict, total=False):
    type: str
    seed: int
    group_key: str
    time_key: str
    n_splits: int
    shuffle: bool  # ignored (we never shuffle time)


class YamlConfig(TypedDict, total=False):
    version: int
    strategy: YamlStrategy
    discipline: dict[str, Any]  # ignored here; preserved for forwards-compat
    parquet_in: str
    output_dir: str


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for splitting: {missing}")


def _order_groups(df: pd.DataFrame, date_col: str, group_col: str) -> pd.DataFrame:
    """
    Compute per-group (race) time order by each group's minimum timestamp.

    Returns
    -------
    DataFrame with columns [group_col, "__min_ts__"] sorted ascending by time.
    """
    grp = (
        df[[group_col, date_col]]
        .dropna(subset=[group_col, date_col])
        .groupby(group_col, as_index=False)
        .agg({date_col: "min"})
        .rename(columns={date_col: "__min_ts__"})
    )
    grp = grp.sort_values(by=["__min_ts__", group_col], kind="mergesort").reset_index(drop=True)
    return grp


def _mask_from_groups(
    df: pd.DataFrame, group_col: str, groups: pd.Index | Iterable[object]
) -> pd.Series:
    """Boolean mask selecting rows whose group is in `groups` (aligned to df.index)."""
    groups_idx = pd.Index(groups)
    return df[group_col].isin(groups_idx.to_numpy()).astype(bool)


def _persist_masks(
    df: pd.DataFrame,
    masks: Mapping[str, pd.Series],
    output_dir: Path,
    meta: Mapping[str, object],
) -> None:
    """
    Persist each mask as a parquet of row indices and write a small JSON manifest.
    This keeps downstream consumers decoupled from schema changes in `df`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        mask_bool = mask.astype(bool, copy=False)
        true_idx = df.index[mask_bool]
        pd.DataFrame({"row_index": true_idx.astype("int64")}).to_parquet(
            output_dir / f"{name}.parquet", index=False
        )
    manifest = {
        "strategy": meta.get("strategy"),
        "seed": meta.get("seed"),
        "date_column": meta.get("date_column"),
        "group_column": meta.get("group_column"),
        "k_folds": meta.get("k_folds"),
        "cutoff": meta.get("cutoff"),
        "val_fraction": meta.get("val_fraction"),
        "test_fraction": meta.get("test_fraction"),
        "n_rows": meta.get("n_rows"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _single_cutoff_masks(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    cutoff: str | None,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, pd.Series]:
    """
    Build train/valid/test masks under a single temporal ordering.

    If `cutoff` is provided:
      Train = groups with min(timestamp) <= cutoff.
      Remaining groups are split (by time) into valid/test via fractions.
      Any rounding remainder is assigned to validation to maintain coverage.

    If `cutoff` is None:
      The entire ordered set is partitioned by fractions (train, then valid, then test).
    """
    grp = _order_groups(df, date_col, group_col)

    if cutoff is not None:
        cutoff_ts = pd.to_datetime(cutoff)
        train_groups_s = grp.loc[grp["__min_ts__"] <= cutoff_ts, group_col]
        remain_groups_s = grp.loc[~grp[group_col].isin(train_groups_s), group_col]

        n_remain = int(len(remain_groups_s))
        n_val = int(math.floor(n_remain * float(val_fraction)))
        n_test = int(math.floor(n_remain * float(test_fraction)))

        # clamp & ensure non-negative; remainder pushed into valid
        n_val = min(max(n_val, 0), n_remain)
        n_test = min(max(n_test, 0), max(0, n_remain - n_val))

        val_s = remain_groups_s.iloc[:n_val].reset_index(drop=True)
        test_s = remain_groups_s.iloc[n_val : n_val + n_test].reset_index(drop=True)

        if (n_val + n_test) < n_remain:
            extra = remain_groups_s.iloc[n_val + n_test :].reset_index(drop=True)
            val_list = val_s.tolist() + extra.tolist()
            val_groups = pd.Index(val_list)
        else:
            val_groups = pd.Index(val_s)

        test_groups = pd.Index(test_s)
        train_groups = pd.Index(train_groups_s)
    else:
        n = int(len(grp))
        n_test = int(math.floor(n * float(test_fraction)))
        n_val = int(math.floor((n - n_test) * float(val_fraction)))
        n_train = max(n - n_val - n_test, 0)
        n_val = max(n_val, 0)
        n_test = max(n_test, 0)

        train_groups = pd.Index(grp.iloc[:n_train][group_col])
        val_groups = pd.Index(grp.iloc[n_train : n_train + n_val][group_col])
        test_groups = pd.Index(grp.iloc[n_train + n_val :][group_col])

    masks: dict[str, pd.Series] = {
        "train": _mask_from_groups(df, group_col, train_groups),
        "valid": _mask_from_groups(df, group_col, val_groups),
        "test": _mask_from_groups(df, group_col, test_groups),
    }

    # Integrity checks: coverage + no overlap + chronology
    total_true = masks["train"].astype(int) + masks["valid"].astype(int) + masks["test"].astype(int)
    if int(total_true.max()) > 1:
        raise AssertionError("Overlap detected between split masks.")
    if int(total_true.sum()) != int(len(df)):
        raise AssertionError("Masks do not cover all rows exactly once.")

    # valid must not precede end of train; test must not precede end of valid
    if masks["train"].any() and masks["valid"].any():
        gmin = df.loc[masks["valid"], date_col].min()
        gmax = df.loc[masks["train"], date_col].max()
        if pd.notna(gmin) and pd.notna(gmax) and gmin < gmax:
            raise AssertionError("Temporal leakage: valid before end of train in single_cutoff.")
    if masks["valid"].any() and masks["test"].any():
        gmin = df.loc[masks["test"], date_col].min()
        gmax = df.loc[masks["valid"], date_col].max()
        if pd.notna(gmin) and pd.notna(gmax) and gmin < gmax:
            raise AssertionError("Temporal leakage: test before end of valid in single_cutoff.")

    return masks


def _kfold_masks(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    k_folds: int,
) -> dict[str, pd.Series]:
    """
    Time-ordered “expanding window” GroupKFold:
      - Split the ordered groups into k blocks.
      - For fold i: Train = union of blocks [0, ..., i-1]; Valid = block i.
    """
    if int(k_folds) < 2:
        raise ValueError("k_folds must be >= 2")
    grp = _order_groups(df, date_col, group_col)
    n = int(len(grp))

    # Balanced block sizes, first blocks get the remainder
    fold_sizes: list[int] = [n // int(k_folds)] * int(k_folds)
    for i in range(n % int(k_folds)):
        fold_sizes[i] += 1

    folds: list[pd.Index] = []
    start = 0
    for sz in fold_sizes:
        part = grp.iloc[start : start + int(sz)][group_col]
        folds.append(pd.Index(part.reset_index(drop=True)))
        start += int(sz)

    masks: dict[str, pd.Series] = {}
    for i, val_groups in enumerate(folds):
        if i == 0:
            train_groups = pd.Index([])
        else:
            train_list: list[object] = []
            for f in folds[:i]:
                train_list.extend(f.tolist())
            train_groups = pd.Index(train_list)

        masks[f"fold{i}_train"] = _mask_from_groups(df, group_col, train_groups)
        masks[f"fold{i}_valid"] = _mask_from_groups(df, group_col, val_groups)

        if bool((masks[f"fold{i}_train"] & masks[f"fold{i}_valid"]).any()):
            raise AssertionError(f"Group leakage between train/valid in fold {i}")

        if bool(masks[f"fold{i}_train"].any()):
            gmin = df.loc[masks[f"fold{i}_valid"], date_col].min()
            gmax = df.loc[masks[f"fold{i}_train"], date_col].max()
            if pd.isna(gmin) or pd.isna(gmax) or (gmin < gmax):
                raise AssertionError(f"Temporal leakage in fold {i}")

    # Validation coverage: each row must appear in exactly one validation fold
    valid_sum = pd.Series(0, index=df.index, dtype="int64")
    for i in range(int(k_folds)):
        valid_sum = valid_sum + masks[f"fold{i}_valid"].astype(int)
    assert int(valid_sum.max()) <= 1, "Validation folds overlap"
    assert int(valid_sum.sum()) == int(len(df)), "Validation folds must cover all rows exactly once"

    return masks


def _require_float(name: str, value: object) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(f"Config '{name}' must be a float (got {type(value).__name__}).")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, numbers.Integral):
        return int(value)
    raise TypeError(f"Config '{name}' must be an int (got {type(value).__name__}).")


def _require_opt_str(name: str, value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Config '{name}' must be a string or None (got {type(value).__name__}).")


def create_time_splits(df: pd.DataFrame, split_cfg: SplitConfig) -> dict[str, pd.Series]:
    """
    Create forward-chronological splits with race-level grouping.

    Returns
    -------
    dict[str, pd.Series]: boolean masks aligned to df.index.
    Side effect: writes `train.parquet`, `valid.parquet`, `test.parquet`
    (or fold masks) of row indices under output_dir, plus manifest.json.
    """
    cfg = dict(split_cfg)
    date_col = cast(str, cfg.get("date_column", Defaults.date_column))
    group_col = cast(str, cfg.get("group_column", Defaults.group_column))
    out_dir = Path(cast(str, cfg.get("output_dir", Defaults.output_dir)))
    strategy: Strategy = cast(Strategy, cfg.get("strategy", "single_cutoff"))

    _ensure_columns(df, [date_col, group_col])

    # Coerce time column if needed (ingest usually provides proper dtype)
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)

    if strategy == "single_cutoff":
        cutoff_str = _require_opt_str("cutoff", cfg.get("cutoff", None))
        val_fraction = _require_float(
            "val_fraction", cfg.get("val_fraction", Defaults.val_fraction)
        )
        test_fraction = _require_float(
            "test_fraction", cfg.get("test_fraction", Defaults.test_fraction)
        )

        masks = _single_cutoff_masks(
            df=df,
            date_col=date_col,
            group_col=group_col,
            cutoff=cutoff_str,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        _persist_masks(
            df,
            masks,
            out_dir,
            {
                "strategy": strategy,
                "seed": _require_int("seed", cfg.get("seed", Defaults.seed)),
                "date_column": date_col,
                "group_column": group_col,
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "n_rows": int(len(df)),
            },
        )
        return masks

    if strategy == "kfold":
        k_folds = _require_int("k_folds", cfg.get("k_folds", Defaults.k_folds))
        masks = _kfold_masks(df=df, date_col=date_col, group_col=group_col, k_folds=k_folds)
        _persist_masks(
            df,
            masks,
            out_dir,
            {
                "strategy": strategy,
                "seed": _require_int("seed", cfg.get("seed", Defaults.seed)),
                "date_column": date_col,
                "group_column": group_col,
                "k_folds": k_folds,
                "n_rows": int(len(df)),
            },
        )
        return masks

    raise ValueError(f"Unknown strategy: {strategy!r}")


def forward_splits(config_path: Path) -> dict[str, pd.Series]:
    """
    YAML-driven convenience wrapper for forward, race-grouped splits.

    Supported config shape:
      strategy:
        type: "time_series" | "kfold" | "groupkfold"  # others fallback to single_cutoff
        seed: 42
        group_key: "race_id"
        time_key: "race_datetime"
        n_splits: 5
        shuffle: false           # ignored (we never shuffle time)
      discipline: {...}          # ignored here
      parquet_in: "data/interim/base.parquet"  # optional (default)
      output_dir: "data/splits"               # optional (default)
    """
    cfg_any: Any = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = cast(YamlConfig, cfg_any)

    strat: YamlStrategy = cfg.get("strategy") or {}
    parquet_in = Path(cfg.get("parquet_in") or "data/interim/base.parquet")
    out_dir = Path(cfg.get("output_dir") or Defaults.output_dir)

    type_str = str(strat.get("type") or "time_series").lower()
    seed = int(strat.get("seed") or Defaults.seed)
    group_col = strat.get("group_key") or Defaults.group_column
    date_col = strat.get("time_key") or Defaults.date_column
    n_splits = int(strat.get("n_splits") or Defaults.k_folds)

    # Read the minimal columns we need; fall back to default time column if necessary
    try:
        df: pd.DataFrame = pd.read_parquet(parquet_in, columns=[group_col, date_col])
    except Exception:
        df_all = pd.read_parquet(parquet_in)
        if Defaults.date_column in df_all.columns:
            date_col = Defaults.date_column
            df = df_all[[group_col, date_col]].copy()
        else:
            raise

    # Map external config → internal SplitConfig
    if type_str in {"time_series", "kfold", "groupkfold"} and n_splits >= 2:
        internal_cfg: SplitConfig = {
            "strategy": "kfold",
            "seed": seed,
            "date_column": date_col,
            "group_column": group_col,
            "output_dir": str(out_dir),
            "k_folds": n_splits,
        }
    else:
        internal_cfg = {
            "strategy": "single_cutoff",
            "seed": seed,
            "date_column": date_col,
            "group_column": group_col,
            "output_dir": str(out_dir),
            "val_fraction": Defaults.val_fraction,
            "test_fraction": Defaults.test_fraction,
            "cutoff": None,
        }

    return create_time_splits(df, internal_cfg)
