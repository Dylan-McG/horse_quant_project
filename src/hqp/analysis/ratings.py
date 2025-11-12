# src/hqp/analysis/ratings.py

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import yaml

from ..ratings.blend import build_blended_ratings
from ..common.schema import (
    alias_race_time,  # canonicalise/alias race time to 'race_dt'
    ensure_rank_and_win,  # robust win/rank inference
    ensure_field_size,  # derive/standardise field_size
)
from ._io import ensure_run_dir, safe_write_parquet


# ----------------------- small helpers for the text report --------------------
def _load_config_summary(config_yaml: str | None) -> dict[str, Any]:
    if not config_yaml:
        return {"note": "No config provided; defaults from src/hqp/ratings/*.py used."}
    p = Path(config_yaml)
    if not p.exists():
        return {"note": f"Config not found at {config_yaml}; defaults used."}
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            return {"note": "Config YAML did not parse to a mapping; defaults used."}
        out: dict[str, Any] = {}
        for k in ("blend", "elo", "eb"):
            v = raw.get(k)
            if isinstance(v, dict):
                out[k] = {kk: v[kk] for kk in v.keys()}
        return out
    except Exception as e:  # pragma: no cover
        return {"note": f"Config read error: {e!s}"}


def _format_overview(ratings: pd.DataFrame) -> str:
    races = ratings["race_id"].nunique() if "race_id" in ratings else 0
    horses = ratings["horse_id"].nunique() if "horse_id" in ratings else 0
    jockeys = ratings["jockey_id"].nunique() if "jockey_id" in ratings else 0
    trainers = ratings["trainer_id"].nunique() if "trainer_id" in ratings else 0
    tmin = pd.to_datetime(ratings["race_dt"]).min() if "race_dt" in ratings else None
    tmax = pd.to_datetime(ratings["race_dt"]).max() if "race_dt" in ratings else None
    return "\n".join(
        [
            "Q2 Ratings — coverage overview:",
            f"  Races:    {races}",
            f"  Horses:   {horses}",
            f"  Jockeys:  {jockeys}",
            f"  Trainers: {trainers}",
            f"  Time span: {tmin} → {tmax}",
        ]
    )


# ----------------------- name/context lookups ---------------------------------
def _make_name_lookups(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _mk(df: pd.DataFrame, idc: str, namec: str) -> pd.DataFrame:
        keep = [c for c in (idc, namec) if c in df.columns]
        if len(keep) < 2:
            return pd.DataFrame(columns=[idc, namec])
        out = df[keep].dropna(subset=[idc]).drop_duplicates(subset=[idc])
        return out

    h_lu = _mk(raw, "horse_id", "horse_name")
    j_lu = _mk(raw, "jockey_id", "jockey_name")
    t_lu = _mk(raw, "trainer_id", "trainer_name")
    return h_lu, j_lu, t_lu


def _attach_names(ratings: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    h_lu, j_lu, t_lu = _make_name_lookups(raw)
    out = ratings.merge(h_lu, on="horse_id", how="left", validate="m:1")
    out = out.merge(j_lu, on="jockey_id", how="left", validate="m:1")
    out = out.merge(t_lu, on="trainer_id", how="left", validate="m:1")
    return out


def _attach_discipline(
    ratings: pd.DataFrame, raw: pd.DataFrame, col: str = "race_type_simple"
) -> pd.DataFrame:
    """
    Attach discipline/context column(s) from the raw dataset to the ratings frame.
    """
    if col not in raw.columns:
        return ratings

    ctx = raw.loc[:, ["race_id", "race_dt", col]].drop_duplicates()

    # --- normalise datetimes to naive (no tz) for stable merge
    for df in (ratings, ctx):
        if "race_dt" in df.columns:
            df["race_dt"] = pd.to_datetime(df["race_dt"], errors="coerce").dt.tz_localize(None)

    keys = ["race_id", "race_dt"]
    return ratings.merge(ctx, on=keys, how="left", validate="m:1")


# ----------------------- report helpers with names ----------------------------
def _latest_topn_with_names(
    df: pd.DataFrame, id_col: str, name_col: str, value_col: str, n: int = 10
) -> pd.DataFrame:
    if name_col not in df.columns:
        df = df.copy()
        df[name_col] = pd.NA
    latest = (
        df.sort_values("race_dt")
        .groupby(id_col, sort=False)
        .tail(1)
        .sort_values(value_col, ascending=False)
        .loc[:, [id_col, name_col, value_col]]
        .head(n)
        .copy()
    )
    latest.columns = ["id", "name", "value"]
    return latest


def _latest_topn_with_names_by_cat(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    value_col: str,
    cat_col: str,
    n: int = 10,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    For each category level, compute latest-per-id and return topn rows with names.
    """
    if cat_col not in df.columns:
        return []
    out: List[Tuple[str, pd.DataFrame]] = []
    for lvl, sub in df.groupby(cat_col, dropna=True, sort=False):
        sub_top = _latest_topn_with_names(sub, id_col, name_col, value_col, n)
        out.append((str(lvl), sub_top))
    # Order categories by number of rows descending for nicer display
    out.sort(key=lambda kv: len(kv[1]), reverse=True)
    return out


def _format_top_block(title: str, top_df: pd.DataFrame) -> str:
    lines: List[str] = [title]
    if top_df.empty:
        lines.append("  (no data)")
        return "\n".join(lines)
    max_idw = max(len(str(x)) for x in top_df["id"])
    max_nmw = max(len(str(x)) for x in top_df["name"].fillna(""))
    for _, row in top_df.iterrows():
        ident = f"{row['id']}".ljust(max_idw)
        name = ("" if pd.isna(row["name"]) else f"{row['name']}").ljust(max_nmw)
        val = float(row["value"])
        lines.append(f"  {ident}  {name}  {val: .4f}" if max_nmw else f"  {ident}  {val: .4f}")
    return "\n".join(lines)


def _write_text_report(ratings: pd.DataFrame, run_dir: Path | str, config_yaml: str | None) -> Path:
    cfg_summary = _load_config_summary(config_yaml)

    # latest snapshots (with names if available)
    top_horses_blend = _latest_topn_with_names(
        ratings, id_col="horse_id", name_col="horse_name", value_col="rating_blend", n=10
    )
    top_horses_elo = _latest_topn_with_names(
        ratings, id_col="horse_id", name_col="horse_name", value_col="horse_elo", n=10
    )
    top_jockeys = _latest_topn_with_names(
        ratings, id_col="jockey_id", name_col="jockey_name", value_col="jockey_elo", n=10
    )
    top_trainers = _latest_topn_with_names(
        ratings, id_col="trainer_id", name_col="trainer_name", value_col="trainer_elo", n=10
    )

    # per-discipline horse Elo (if available)
    by_disc = _latest_topn_with_names_by_cat(
        ratings, "horse_id", "horse_name", "horse_elo", cat_col="race_type_simple", n=10
    )

    parts: list[str] = []
    parts.append("Q2 — Ratings: leak-free skill and reliability priors\n")
    parts.append(_format_overview(ratings))

    parts.append("\n\nHyperparameters (from YAML if provided):")
    parts.append(
        yaml.safe_dump(cfg_summary, sort_keys=False)
        if not isinstance(cfg_summary, str)
        else str(cfg_summary)
    )

    parts.append("\nOutputs (per runner, as-of each race timestamp):")
    parts.append(
        "  Columns: [race_id, race_dt, horse_id, jockey_id, trainer_id,\n"
        "            horse_elo, horse_eb, jockey_elo, trainer_elo,\n"
        "            horse_elo_z, horse_eb_z, jockey_elo_z, trainer_elo_z, rating_blend,\n"
        "            horse_name, jockey_name, trainer_name, race_type_simple?]"
    )

    parts.append("\nTop-10 snapshots (latest as-of):")
    parts.append(_format_top_block("Horses — rating_blend (overall):", top_horses_blend))
    parts.append("")
    parts.append(_format_top_block("Horses — horse_elo (overall):", top_horses_elo))
    parts.append("")
    if by_disc:
        for lvl, df_top in by_disc:
            parts.append(
                _format_top_block(f"Horses — horse_elo by race_type_simple = {lvl}:", df_top)
            )
            parts.append("")

    parts.append(_format_top_block("Jockeys — jockey_elo (overall):", top_jockeys))
    parts.append("")
    parts.append(_format_top_block("Trainers — trainer_elo (overall):", top_trainers))

    text = "\n".join(parts) + "\n"
    out_path = Path(run_dir) / "q2_ratings_report.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


# ----------------------------- orchestrator -----------------------------------
def run_q2(data_path: str, out_base: str, config_yaml: str | None = None) -> str:
    """
    Q2 ratings analysis, aligned to pipeline conventions:
      - canonical time column: race_dt
      - shared coercion for rank/win/field_size via hqp.common.schema
      - uses build_blended_ratings() (Elo + EB; online, leak-free)
      - adds Top Horses by Elo (overall) and per-discipline if available
    """
    # 1) Load
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)

    # 2) Harmonise → canonical names
    df = alias_race_time(df, prefer="race_dt")
    df = ensure_field_size(df)
    df = ensure_rank_and_win(df)

    # 3) De-dupe runner rows to guarantee one-to-one merges
    required_keys = ["race_id", "race_dt", "horse_id", "jockey_id", "trainer_id"]
    missing = [k for k in required_keys if k not in df.columns]
    if missing:
        raise KeyError(f"Missing required ID keys for ratings: {missing}")

    pre_n = len(df)
    dupe_mask = df.duplicated(subset=required_keys, keep="first")
    n_dupes = int(dupe_mask.sum())
    if n_dupes > 0:
        print(f"[Q2] Dropping {n_dupes} duplicate runner rows (out of {pre_n}).")
        df = df.loc[~dupe_mask].copy()

    # Extra safety per entity
    for ent_col in ["horse_id", "jockey_id", "trainer_id"]:
        dup_count = df.duplicated(subset=["race_id", "race_dt", ent_col], keep=False).sum()
        if dup_count:
            print(f"[Q2] {dup_count} duplicates on ({ent_col}, race_id, race_dt) → keeping first.")
            df = df.drop_duplicates(subset=["race_id", "race_dt", ent_col], keep="first")

    # 4) Compute ratings (this writes its own artefacts to data/ratings if configured)
    ratings = build_blended_ratings(df, config_yaml=config_yaml, out_dir="data/ratings")

    # 5) Attach names and discipline (optional, for prettier text report & stratified Elo)
    ratings_named = _attach_names(ratings, df)
    ratings_named = _attach_discipline(ratings_named, df, col="race_type_simple")

    # 6) Persist under timestamped analysis directory
    run_dir = ensure_run_dir(out_base, "q2")
    safe_write_parquet(ratings_named, Path(run_dir) / "ratings.parquet")

    # 7) Write pasteable text report
    report_path = _write_text_report(ratings_named, run_dir, config_yaml)
    print(f"\nWrote text report to: {report_path}")

    # 8) Console top-10s (now includes horses by Elo as well)
    top_horses_blend = _latest_topn_with_names(
        ratings_named, "horse_id", "horse_name", "rating_blend", n=10
    )
    top_horses_elo = _latest_topn_with_names(
        ratings_named, "horse_id", "horse_name", "horse_elo", n=10
    )
    top_jockeys = _latest_topn_with_names(
        ratings_named, "jockey_id", "jockey_name", "jockey_elo", n=10
    )
    top_trainers = _latest_topn_with_names(
        ratings_named, "trainer_id", "trainer_name", "trainer_elo", n=10
    )

    pd.set_option("display.max_rows", 20)
    print("\nTop 10 Horses by blended rating (latest snapshot):")
    print(top_horses_blend.set_index(["id", "name"])["value"])
    print("\nTop 10 Horses by Elo prior (latest snapshot):")
    print(top_horses_elo.set_index(["id", "name"])["value"])
    print("\nTop 10 Jockeys by Elo prior (latest snapshot):")
    print(top_jockeys.set_index(["id", "name"])["value"])
    print("\nTop 10 Trainers by Elo prior (latest snapshot):")
    print(top_trainers.set_index(["id", "name"])["value"])

    return str(run_dir)
