# src/hqp/ingest/reader.py
# -----------------------------------------------------------------------------
# Ingestion of raw CSV(s) into a canonical, runner-level Parquet ("base").
#
# Responsibilities
# - load CSVs (single file or glob),
# - apply renames, dtype coercions, basic category cleaning,
# - ensure required columns exist,
# - normalize keys (race_id, horse_id) and derive canonical time columns,
# - deduplicate rows by (race_id, horse_id),
# - deterministic sorting, write Parquet, optional audit + ledger append.
#
# Design notes
# - Be permissive at ingest time (coercions best-effort) and defer strict
#   dataset checks to `validate` (schema expectations).
# - Time columns: we standardize on `race_dt` (naive) with `off_time` as alias,
#   derived from race_datetime/off_time or (date + race_time).
# - Logging/ledger is best-effort: ingestion should succeed even if audit/ledger
#   sinks are unavailable (e.g., CI filesystem).
# -----------------------------------------------------------------------------
from __future__ import annotations

import glob
import inspect
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd
import yaml

from hqp.common.logging import RunLedger, RunRecord, log_stdout, sha256_file
from hqp.common.schema import DEFAULT_SCHEMA, coerce_frame, ensure_required_columns
from hqp.common.schema import Schema as SchemaType
from hqp.ingest.keys import check_unique_keys, normalize_keys


class IngestConfig(TypedDict, total=False):
    """
    YAML-backed config shape for ingestion.

    One of:
      raw_csv_path: str        # single CSV path
      raw_glob: str            # glob pattern for multiple CSVs

    Optional parsing/coercion:
      rename_map: dict[str, str]
      date_columns: list[str]             # will be parsed to tz-naive datetime64[ns]
      dtypes: dict[str, str]              # runtime-tolerant coercion (float/int/string/category/...)
      categorical_columns: list[str]      # additional columns to coerce to Categorical

    Validation:
      required_columns: list[str]         # must exist, else ValueError
      drop_na_on: list[str]               # rows with NA in any are dropped

    Output:
      output_parquet: str                 # path for the canonical base parquet
      audit: dict[str, Any]               # audit sink {enabled: bool, path: str}
    """

    raw_csv_path: str
    raw_glob: str
    rename_map: dict[str, str]
    date_columns: list[str]
    dtypes: dict[str, str]
    categorical_columns: list[str]
    required_columns: list[str]
    drop_na_on: list[str]
    output_parquet: str
    audit: dict[str, Any]


def _load_config(path: Path) -> dict[str, Any]:
    """Load YAML from `path` (empty docs return {})."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return cast(dict[str, Any], data)


def _augment_schema(
    base_schema: SchemaType,
    date_cols: Iterable[str],
    cat_cols: Iterable[str],
) -> SchemaType:
    """
    Return a Schema object like base_schema but with dtypes overridden for
    the listed date/categorical columns. Used by the ingest_csv helper.
    """
    base_dtypes: dict[str, str] = dict(base_schema.dtypes)
    base_orders_raw = getattr(base_schema, "category_orders", {})
    base_orders: Mapping[str, list[str]] = cast(Mapping[str, list[str]], base_orders_raw)
    for c in date_cols:
        base_dtypes[c] = "datetime64[ns]"
    for c in cat_cols:
        base_dtypes[c] = "category"
    SchemaCls = type(base_schema)
    try:
        return SchemaCls(dtypes=base_dtypes, category_orders=base_orders)
    except TypeError:
        # older schema constructor signature
        return SchemaCls(base_dtypes, base_orders)


def _build_runrecord_kwargs(
    started: datetime,
    finished: datetime,
    input_path: str,
    output_path: Path,
    input_rows: int,
    output_rows: int,
    removed: int,
) -> dict[str, Any]:
    """Build kwargs that match either RunRecord.new(...) or RunRecord(...) signatures."""
    return {
        "started_at": started,
        "finished_at": finished,
        "input_path": input_path,
        "output_path": output_path,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "status": "ok",
        "message": f"ingest completed; dedup_removed={removed}",
    }


def _maybe_make_runrecord(kwargs: dict[str, Any]) -> Any:
    """
    Instantiate a RunRecord using the most compatible available constructor.
    If anything fails, we log and return None (ingest should remain successful).
    """
    try:
        maker = getattr(RunRecord, "new", None)
        if callable(maker):
            params = set(inspect.signature(maker).parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in params}
            return maker(**filtered)
        params = set(inspect.signature(RunRecord).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return RunRecord(**filtered)
    except Exception as e:
        log_stdout(f"[ingest] ledger append skipped ({type(e).__name__}: {e})")
        return None


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical time columns exist:

      race_dt  : naive UTC datetime of race start (primary, used for sorting)
      off_time : alias of race_dt (kept for compatibility/plotting)

    Derivation priority:
      1) 'race_datetime' (tz-aware or naive, coerced to naive UTC)
      2) 'off_time'
      3) 'date' + 'race_time' (both strings, parsed to datetime)

    Raises
    ------
    KeyError if none of the above sources can be found.
    """
    g = df.copy()

    def _to_naive_utc(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        return dt.dt.tz_localize(None)

    if "race_dt" in g.columns and pd.api.types.is_datetime64_ns_dtype(g["race_dt"]):
        # Already normalized
        pass
    elif "race_datetime" in g.columns:
        g["race_dt"] = _to_naive_utc(g["race_datetime"])
    elif "off_time" in g.columns:
        g["race_dt"] = _to_naive_utc(g["off_time"])
    elif {"date", "race_time"}.issubset(g.columns):
        combo = g["date"].astype(str).str.strip() + " " + g["race_time"].astype(str).str.strip()
        g["race_dt"] = _to_naive_utc(combo)
    else:
        raise KeyError(
            "Could not derive 'race_dt'. Provide race_datetime/off_time or date+race_time."
        )

    if "off_time" not in g.columns:
        g["off_time"] = g["race_dt"]

    return g


def ingest_base(
    config_path: Path,
    output_parquet: Path | None = None,
    ledger_path: Path | None = None,  # (kept for compatibility; not used directly)
) -> Path:
    """
    Legacy entry-point used in older tests/scripts.
    Loads YAML from `config_path` and delegates to `run`.
    """
    raw_cfg = _load_config(Path(config_path))
    out_path = (
        Path(output_parquet).expanduser()
        if output_parquet is not None
        else Path(raw_cfg.get("output_parquet", "data/canonical/base.parquet")).expanduser()
    )
    run(raw_cfg, out_path)  # Mapping[str, Any] accepted
    return out_path


def _coerce_series_dtype(s: pd.Series, dt: str) -> pd.Series:
    """
    Runtime-tolerant dtype coercion used by `run`:

    Supported shorthands:
      - "float", "float64", "double" → numeric (NaNs allowed)
      - "int", "int64" → pandas nullable Int64
      - "string"/"str"/"object" → pandas StringDtype
      - "category"/"categorical" → pandas Categorical
      - otherwise: pass-through to `.astype(dt)` (e.g., 'datetime64[ns]')

    We intentionally `errors="coerce"` for numerics to avoid hard failures at ingest
    and defer strictness to validation.
    """
    dt_l = dt.lower()
    if dt_l in ("float", "float64", "double"):
        return pd.to_numeric(s, errors="coerce")
    if dt_l in ("int", "int64"):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if dt_l in ("string", "str", "object"):
        return s.astype("string")  # pandas StringDtype
    if dt_l in ("category", "categorical"):
        return s.astype("category")  # type: ignore[arg-type]
    # Fallback: allow pandas to resolve (e.g., 'datetime64[ns]')
    return s.astype(dt)  # type: ignore[arg-type]


def run(cfg: Mapping[str, Any], out_path: Path) -> None:
    """
    Main ingestion entry-point (used by CLI).

    Steps
    -----
    1) Load raw CSV(s) per cfg (raw_glob or raw_csv_path).
    2) Apply rename_map, parse date_columns, coerce dtypes, and category cleanup.
    3) Enforce required_columns and optionally drop rows with NA in drop_na_on.
    4) Normalize keys (race_id, horse_id) and derive canonical time columns.
    5) Deduplicate by (race_id, horse_id) and sort deterministically by race_dt, race_id, horse_id.
    6) Write Parquet to `out_path`; optionally write an audit CSV; append ledger record.

    Raises
    ------
    ValueError
        If neither raw_glob nor raw_csv_path is provided, or if required columns are missing.
    """
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load raw ----
    paths: list[Path] = []
    raw_glob = cfg.get("raw_glob")
    raw_csv_path = cfg.get("raw_csv_path")
    if isinstance(raw_glob, str) and raw_glob:
        paths = [Path(p) for p in glob.glob(raw_glob, recursive=True)]
    if not paths and isinstance(raw_csv_path, str) and raw_csv_path:
        paths = [Path(raw_csv_path).expanduser()]
    if not paths:
        raise ValueError("Provide one of 'raw_glob' or 'raw_csv_path' in configs/ingest.yaml")

    frames: list[pd.DataFrame] = []
    for p in paths:
        log_stdout(f"[ingest] reading CSV: {p}")
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    # ---- renames / dates / dtypes ----
    rename_map = cast(dict[str, str], cfg.get("rename_map", {}) or {})
    if rename_map:
        df = df.rename(columns=rename_map)

    # Parse listed date columns to tz-naive datetime64[ns], preserving NA
    for c in cast(list[str], cfg.get("date_columns", []) or []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)

    # Coerce dtypes with tolerant helpers; strictness belongs in validation
    for col, dt in cast(dict[str, str], cfg.get("dtypes", {}) or {}).items():
        if col in df.columns:
            try:
                df[col] = _coerce_series_dtype(df[col], dt)
            except Exception:
                # be lenient at ingest; deeper validation happens later
                pass

    # --- optional categorical coercion (config-driven) ---
    cat_cols = cast(list[str], cfg.get("categorical_columns", []) or [])
    for col in cat_cols:
        if col in df.columns:
            s = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            df[col] = s.astype("category")

    # --- hardening for common fields regardless of config ---
    # going → categorical (light normalization)
    if "going" in df.columns and df["going"].dtype.name != "category":
        s = df["going"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df["going"] = s.astype("category")

    # draw → nullable Int32 (to handle "", NaN gracefully)
    if "draw" in df.columns:
        df["draw"] = pd.to_numeric(df["draw"], errors="coerce").astype("Int32")

    # ---- validation ----
    required = cast(list[str], cfg.get("required_columns", []) or [])
    if required:
        ensure_required_columns(df, required)
    for c in cast(list[str], cfg.get("drop_na_on", []) or []):
        if c in df.columns:
            df = df[df[c].notna()]

    # ---- normalize keys & time ----
    df = normalize_keys(df)
    df = _add_time_columns(df)

    # ---- dedupe & sort ----
    rows_in = len(df)
    if {"race_id", "horse_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["race_id", "horse_id"], keep="first").reset_index(drop=True)
    total, uniq = check_unique_keys(df)
    if total != uniq:
        raise ValueError(f"Duplicate (race_id, horse_id) keys remain after dedupe: {total-uniq}")
    removed = rows_in - len(df)

    sort_cols: list[str] = []
    if "race_dt" in df.columns:
        sort_cols.append("race_dt")
    sort_cols.extend([c for c in ("race_id", "horse_id") if c in df.columns])
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # ---- write & audit ----
    df.to_parquet(out_path, index=False)

    audit = cfg.get("audit")
    if isinstance(audit, Mapping) and audit.get("enabled") and audit.get("path"):
        ap = Path(cast(str, audit["path"])).expanduser()
        ap.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "rows_in": [rows_in],
                "rows_out": [len(df)],
                "dedup_removed": [removed],
                "distinct_races": [df["race_id"].nunique() if "race_id" in df else None],
                "off_time_min": [df["off_time"].min() if "off_time" in df else None],
                "off_time_max": [df["off_time"].max() if "off_time" in df else None],
            }
        ).to_csv(ap, index=False)

    # ---- ledger/log summary ----
    si = sha256_file(paths[0])[:12] if paths else "NA"
    so = sha256_file(out_path)[:12]
    msg = (
        f"[ingest] done: rows_in={rows_in} rows_out={len(df)} removed={removed} "
        f"sha_in={si} sha_out={so} -> {out_path}"
    )
    log_stdout(msg)

    # optional ledger (best-effort, never fail the run)
    try:
        led_path = Path("data/ledger/ingest_runs.csv")
        ledger = RunLedger(led_path)
        rr_kwargs = _build_runrecord_kwargs(
            started=datetime.now(),
            finished=datetime.now(),
            input_path=str(paths[0]),
            output_path=out_path,
            input_rows=rows_in,
            output_rows=len(df),
            removed=removed,
        )
        rec = _maybe_make_runrecord(rr_kwargs)
        if rec is not None:
            ledger.append(rec)
    except Exception as e:
        log_stdout(f"[ingest] ledger append failed ({type(e).__name__}: {e})")


def ingest_csv(csv_path: Path, out_path: Path) -> Path:
    """
    Minimal CSV → canonical Parquet helper for ad-hoc runs (kept for compatibility).

    Requirements
    ------------
    - Either an existing 'race_datetime' column or both 'date' and 'race_time' columns
      (which will be combined into 'race_datetime').

    Output
    ------
    Parquet at `out_path` with:
      - normalized keys (race_id, horse_id),
      - canonical time columns (race_dt, off_time),
      - deduplicated and deterministically sorted rows.
    """
    raw_csv = Path(csv_path).expanduser()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv, low_memory=False)

    if "race_datetime" not in df.columns:
        if {"date", "race_time"}.issubset(df.columns):
            dt = pd.to_datetime(
                df["date"].astype(str) + " " + df["race_time"].astype(str),
                errors="coerce",
                utc=True,
            )
            df["race_datetime"] = dt.dt.tz_localize(None)
        else:
            raise ValueError(
                "race_datetime missing and cannot be built (need columns: 'date' and 'race_time')."
            )

    required_columns = ["race_id", "horse_id", "race_datetime"]
    ensure_required_columns(df, required_columns)

    schema_obj: SchemaType = _augment_schema(DEFAULT_SCHEMA, ["race_datetime"], [])
    df = coerce_frame(df, schema_obj)

    # Harden common fields here too (mirror main run)
    if "going" in df.columns and df["going"].dtype.name != "category":
        s = df["going"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df["going"] = s.astype("category")
    if "draw" in df.columns:
        df["draw"] = pd.to_numeric(df["draw"], errors="coerce").astype("Int32")

    df = normalize_keys(df)
    rows_in = len(df)
    if {"race_id", "horse_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["race_id", "horse_id"], keep="first").reset_index(drop=True)
    total, uniq = check_unique_keys(df)
    if total != uniq:
        raise ValueError(f"Duplicate (race_id, horse_id) keys detected after dedupe: {total-uniq}")

    df = _add_time_columns(df)
    sort_cols: list[str] = ["race_dt"]
    sort_cols.extend([c for c in ("race_id", "horse_id") if c in df.columns])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    log_stdout(f"[ingest_csv] done: rows_in={rows_in} rows_out={len(df)} -> {out_path}")
    return out_path
