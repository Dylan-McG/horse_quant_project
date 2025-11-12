# src/hqp/common/schema.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# -----------------------------------------------------------------------------
# Canonical dtype targets + best-effort coercion to keep raw feeds consistent.
# Used by ingest to:
#  - standardize common columns (nullable Int32, Float32, StringDtype, category),
#  - apply fixed category orders (e.g., going),
#  - assert required columns exist.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from pandas import Float32Dtype, Int32Dtype, StringDtype
from pandas.api.types import CategoricalDtype

# Canonical dtype targets (can be extended/overridden by configs)
CANONICAL_DTYPES: dict[str, str] = {
    "race_id": "string",
    "horse_id": "string",
    "race_datetime": "datetime64[ns]",
    "course": "string",
    "horse_name": "string",
    "draw": "Int32",
    "age": "Int32",
    "weight_lbs": "Int32",
    "official_rating": "Int32",
    "n_runners": "Int32",
    "distance_yards": "Int32",
    "sp": "Float32",
    "position": "Int32",
    "going": "category",
    "sex": "category",
    "trainer": "string",
    "jockey": "string",
}

# Optional fixed category orders (used during coercion)
CATEGORY_ORDERS: dict[str, list[str]] = {
    "going": [
        "FIRM",
        "GOOD TO FIRM",
        "GOOD",
        "GOOD TO SOFT",
        "SOFT",
        "HEAVY",
        "STANDARD",
        "STANDARD TO SLOW",
        "SLOW",
    ],
    "sex": ["F", "M", "G", "C", "H"],
}


@dataclass(frozen=True)
class Schema:
    dtypes: Mapping[str, str]
    category_orders: Mapping[str, Sequence[str]]

    def categorical_cols(self) -> list[str]:
        return [c for c, t in self.dtypes.items() if t == "category"]


DEFAULT_SCHEMA = Schema(dtypes=CANONICAL_DTYPES, category_orders=CATEGORY_ORDERS)


def _to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(Int32Dtype())


def _to_float32(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(Float32Dtype())


def _to_string(series: pd.Series) -> pd.Series:
    return series.astype(StringDtype()).str.strip()


def _to_category(series: pd.Series, order: Sequence[str] | None) -> pd.Series:
    s = series.astype(StringDtype()).str.strip().str.upper()
    if order:
        dtype = CategoricalDtype(categories=list(order), ordered=True)
        return s.astype(dtype)
    return s.astype(CategoricalDtype())


Coercer = Callable[[pd.Series], pd.Series]
_COERCERS: dict[str, Coercer] = {
    "Int32": _to_nullable_int,
    "Float32": _to_float32,
    "string": _to_string,
}


def _astype_any(s: pd.Series, dtype: Any) -> pd.Series:
    """Helper to keep type-checkers happy for dynamic astype fallbacks."""
    return cast(pd.Series, s.astype(dtype))


def coerce_frame(df: pd.DataFrame, schema: Schema = DEFAULT_SCHEMA) -> pd.DataFrame:
    """
    Best-effort coercion to canonical dtypes + category orders.

    - Datetime columns are parsed with errors='coerce' (preserve NA).
    - Scalar types use nullable pandas dtypes where applicable.
    - Categorical columns are uppercased and optionally ordered.
    """
    out: pd.DataFrame = df.copy()

    # Datetimes
    if "race_datetime" in out.columns:
        out["race_datetime"] = pd.to_datetime(out["race_datetime"], errors="coerce", utc=False)

    # Scalar type coercion (skip datetime; handled above)
    for col, target in schema.dtypes.items():
        if col not in out.columns:
            continue
        if target == "datetime64[ns]":
            continue
        if target == "category":
            order = schema.category_orders.get(col)
            out[col] = _to_category(out[col], order)
            continue
        coercer = _COERCERS.get(target)
        if coercer is not None:
            out[col] = coercer(out[col])
        else:
            # Fallback: best-effort cast; ignore failures
            try:
                out[col] = _astype_any(out[col], target)
            except Exception:
                pass

    return out


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise if any required columns are missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------- light aliases / derivations used across the pipeline ----------


def alias_race_time(df: pd.DataFrame, prefer: str = "race_dt") -> pd.DataFrame:
    """
    Create a canonical race time column (default: 'race_dt') from existing options.
    Reuses your already-coerced 'race_datetime' if present.
    """
    out = df.copy()
    if prefer in out.columns:
        out[prefer] = pd.to_datetime(out[prefer], errors="coerce")
        return out

    # prefer your canonical 'race_datetime' if available
    time_cands = ["race_datetime", "off_time", "date", "timestamp", "time", "ts"]
    for c in time_cands:
        if c in out.columns:
            out = out.rename(columns={c: prefer})
            break
    if prefer not in out.columns:
        raise KeyError(
            "Could not find a race time column; expected one of "
            "['race_dt','race_datetime','off_time','date','timestamp','time','ts']."
        )
    out[prefer] = pd.to_datetime(out[prefer], errors="coerce")
    return out


def ensure_field_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide a 'field_size' column from your canonical 'n_runners' if present,
    else derive as per-race row count.
    """
    out = df.copy()
    if "field_size" in out.columns:
        return out
    if "n_runners" in out.columns:
        return out.rename(columns={"n_runners": "field_size"})
    if "race_id" in out.columns:
        out["field_size"] = out.groupby("race_id")["race_id"].transform("count")
        return out
    return out


def ensure_rank_and_win(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise 'rank' (numeric) and 'win' (0/1) with best-effort inference.
    Never uses obs__* as model features; only for label/diagnostics.
    """
    out = df.copy()

    # Win flag
    if "win" not in out.columns:
        win_cands = ["win", "won", "winner", "is_winner", "obs__is_winner", "result", "outcome"]
        for c in win_cands:
            if c in out.columns:
                s = out[c]
                if s.dtype == bool:
                    out["win"] = s.astype("int64")
                else:
                    sval = s.astype("string").str.strip().str.lower()
                    mapped = sval.map(
                        {
                            "y": 1,
                            "yes": 1,
                            "true": 1,
                            "t": 1,
                            "1": 1,
                            "win": 1,
                            "won": 1,
                            "winner": 1,
                            "n": 0,
                            "no": 0,
                            "false": 0,
                            "f": 0,
                            "0": 0,
                            "lose": 0,
                            "lost": 0,
                            "non-winner": 0,
                        }
                    )
                    mapped = mapped.where(~sval.str.match(r"^1(st)?$", na=False), 1)
                    out["win"] = mapped.astype("Int64").fillna(0).astype("int64")
                break

    # Rank → numeric
    if "rank" not in out.columns:
        rank_cands = [
            "rank",
            "position",
            "pos",
            "finish_position",
            "official_position",
            "placing",
            "place",
            "obs__uposition",
        ]
        for c in rank_cands:
            if c in out.columns:
                s = out[c].astype("string")
                digits = s.str.extract(r"^\s*(\d+)", expand=False)
                out["rank"] = pd.to_numeric(digits, errors="coerce")
                break

    if "win" not in out.columns and "rank" in out.columns:
        out["win"] = (
            (pd.to_numeric(out["rank"], errors="coerce") == 1).fillna(False).astype("int64")
        )

    if "rank" in out.columns:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce")

    return out
