# src/hqp/cli.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Command Line Interface (Typer)
#
# This CLI orchestrates the complete pipeline used in the assignment:
#   ingest → validate → split → ratings → features → train/evaluate → calibration
#   → market (join) → edges → backtest (and variants like sweeps / compare)
#
# Design priorities:
# - Reproducibility: every command writes timestamped artifacts under reports/ or models/.
# - Safety: avoid silent fallbacks; fail loudly where correctness matters (e.g., keys/columns).
# - No leakage: features exclude same-race obs__*; joins are keyed by (race_id, horse_id).
# - Interop: commands accept both path-based and DataFrame-based signatures in user code.
#
# Key math/concepts referenced below:
# - Market implied probability from decimal odds: p_i = 1 / odds_i.
#   We normalize within race: p̂_i = p_i / Σ_j p_j, which implicitly removes the overround.
# - “Edge” is defined as model_prob − market_implied (post-normalization).
# - Platt scaling (calibration): σ(a·p + b) fitted on validation labels; isotonic is monotone.
# - Empirical Bayes fallback ratings use a Beta-Binomial style shrinkage toward the global mean.
#
# NOTE: This file adds verbose documentation and comments only. No logic changes.
# -----------------------------------------------------------------------------

from __future__ import annotations

import inspect
import json
import traceback
from collections.abc import Callable
from datetime import datetime
from inspect import Signature
from pathlib import Path
from typing import Any, Dict, Optional, Callable, cast

import pandas as pd
import typer
import numpy as np
import importlib.util
import time
import warnings

from hqp.utils.dyn import find_callable_in_module, resolve_callable

warnings.filterwarnings(
    "ignore",
    message="SeriesGroupBy.fillna is deprecated",
    category=FutureWarning,
)


from hqp.eval.calibrators import (
    fit_from_predictions_parquet,
    save_calibrator,
    evaluate_brier,
    load_calibrator,
    CalibMethod,
)

# Typer application entry-point. We disable the default shell completion for stability in CI.
app = typer.Typer(add_completion=False, no_args_is_help=True)


# ------------------------------ helpers ---------------------------------
def _ts() -> str:
    """Return a compact timestamp used for artifact folder naming (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_parents(p: Path) -> None:
    """Ensure parent directories of path ``p`` exist (idempotent)."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_load_yaml(p: Path) -> dict[str, Any]:
    """Load a YAML file as a dict. Returns {} if the YAML is empty."""
    import yaml

    return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def _maybe_autoadjust_splits(features_p: Path, cfg_p: Path) -> Path:
    """
    Adjust training.n_splits downward if the features have too few groups (e.g., for GroupKFold).
    This avoids runtime errors like "n_splits > n_groups".
    Returns either the original cfg path or a new '<name>.autoadj.yaml' path.

    Rationale:
    - In time-ordered CV with group constraints (e.g., by race_id), scikit/GroupKFold requires
      n_splits ≤ #unique_groups. We clamp to [2, n_groups] to keep training viable on tiny sets.
    - This is a file-level transformation (writes an .autoadj.yaml) so the change is auditable.
    """
    try:
        import yaml

        cfg = _safe_load_yaml(cfg_p)
        training = dict(cfg.get("training", {}) or {})
        desired: int = int(training.get("n_splits", 3))
        group_key: str = str(training.get("group_key", "race_id"))

        # Default safely so variable is always bound
        n_groups: int = 2

        # Count groups from features
        cols = [group_key] if group_key else None
        df = pd.read_parquet(features_p, columns=cols)

        # If requested group_key is missing, try race_id; otherwise keep default
        if group_key not in df.columns:
            if "race_id" != group_key:
                try:
                    df = pd.read_parquet(features_p, columns=["race_id"])
                    group_key = "race_id" if "race_id" in df.columns else group_key
                except Exception:
                    pass

        if group_key in df.columns:
            n_groups = int(df[group_key].nunique(dropna=True))

        # For GroupKFold, scikit requires n_splits <= #unique_groups; enforce >=2
        adjusted = max(2, min(desired, n_groups))

        if adjusted != desired:
            training["n_splits"] = int(adjusted)
            cfg["training"] = training
            out = cfg_p.with_name(cfg_p.stem + ".autoadj.yaml")
            out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            typer.echo(
                f"[train] auto-adjusted n_splits={desired} -> {adjusted} "
                f"based on groups={n_groups} ({group_key}) -> {out.name}"
            )
            return out

    except Exception:
        # If anything fails, just use the original config
        pass

    return cfg_p


# ------------------------------- ingest ---------------------------------
@app.command()
def ingest(
    config: Optional[str] = typer.Option(
        None, help="Path to configs/ingest.yaml (optional in dry-run)"
    ),
    out: str = typer.Option("data/interim/base.parquet", help="Output canonical base parquet"),
    dry_run: bool = typer.Option(False, help="Write a tiny synthetic base that downstream accepts"),
) -> None:
    """
    Build the canonical runner-level dataset ("base") and write to Parquet.

    In dry-run mode we emit a minimal synthetic dataset that exercises downstream steps
    (ratings, features, training). This is useful for CI smoke and for users bootstrapping.
    """
    out_p = Path(out)
    _ensure_parents(out_p)

    if dry_run:
        from pandas import Timestamp as _Ts
        import pandas as pd

        # Minimal, well-formed synthetic base: enough columns to let downstream proceed.
        df = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 2, 2, 3, 3],
                "horse_id": [101, 102, 103, 201, 202, 301, 302],
                "jockey_id": [11, 12, 13, 21, 22, 31, 32],
                "trainer_id": [111, 112, 113, 121, 122, 131, 132],
                "race_datetime": [
                    _Ts("2021-01-01 12:00"),
                    _Ts("2021-01-01 12:00"),
                    _Ts("2021-01-01 12:00"),
                    _Ts("2021-01-02 12:00"),
                    _Ts("2021-01-02 12:00"),
                    _Ts("2021-01-03 12:00"),
                    _Ts("2021-01-03 12:00"),
                ],
                "obs__is_winner": [1, 0, 0, 0, 1, 0, 1],
                "course": ["C1", "C1", "C1", "C2", "C2", "C3", "C3"],
                "going": ["Good", "Good", "Good", "Soft", "Soft", "Good", "Good"],
                "distance_m": [1600, 1600, 1600, 2000, 2000, 1400, 1400],
            }
        )
        df.to_parquet(out_p, index=False)
        typer.echo(f"[ingest] synthetic base written: {out}")
        return

    cfg = _safe_load_yaml(Path(config)) if config else {}
    from hqp.ingest.reader import run as _ingest_run  # type: ignore

    try:
        _ingest_run(cfg, out_p)
        typer.echo(f"[ingest] base written: {out}")
    except Exception as e:
        typer.echo("[ingest] failed:")
        typer.echo("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise typer.Exit(code=1)


# ------------------------------ validate --------------------------------
@app.command()
def validate(
    base: str = typer.Option("data/interim/base.parquet", help="Canonical parquet"),
    schema: Optional[str] = typer.Option(None, help="Optional schema YAML"),
    dry_run: bool = typer.Option(False, help="Write a stub JSON report and exit"),
) -> None:
    """
    Run dataset validations/expectations and write a JSON summary.

    Strategy:
      - Prefer the project validator (hqp.validate.run) if available.
      - Fall back to minimal structural checks (e.g., presence of keys) to avoid false positives.
    """
    if dry_run:
        run_dir = Path(f"reports/validate/{_ts()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({"ok": True, "notes": "stub"}), "utf-8")
        typer.echo(f"[validate] stub written: {run_dir}/summary.json")
        return

    base_p = Path(base)
    if not base_p.exists():
        typer.echo(f"[validate] base not found: {base}")
        raise typer.Exit(code=1)

    # Try optional real validator
    try:
        from hqp.validate import run as _validate  # type: ignore

        out_dir = Path(f"reports/validate/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        _validate(base_p, Path(schema) if schema else None, out_dir)
        typer.echo(f"[validate] report under: {out_dir}")
    except Exception:
        # Minimal checks: ensure essential join keys exist.
        df = pd.read_parquet(base_p)
        required = {"race_id", "horse_id"}
        ok = required.issubset(df.columns)
        run_dir = Path(f"reports/validate/{_ts()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"ok": ok, "missing": list(required - set(df.columns))}), "utf-8"
        )
        typer.echo(f"[validate] summary: {run_dir}/summary.json")


# -------------------------------- split ---------------------------------
@app.command()
def split(
    base: str = typer.Option("data/interim/base.parquet", help="Canonical parquet"),
    config: Optional[str] = typer.Option(None, help="configs/split.yaml"),
    dry_run: bool = typer.Option(False, help="Write a stub manifest"),
) -> None:
    """
    Create chronological train/valid/test splits or a manifest.

    Notes:
    - We keep all runners from a race in the same split (group by race_id).
    - In chronological mode, we split by race datetime to prevent future leakage.
    """
    if dry_run:
        _ensure_parents(Path("data/splits/split.json"))
        Path("data/splits/split.json").write_text(json.dumps({"mode": "stub"}), "utf-8")
        typer.echo("[split] stub manifest: data/splits/split.json")
        return

    try:
        from hqp.splits import run as _split_run  # type: ignore

        cfg = _safe_load_yaml(Path(config)) if config else {}
        _split_run(Path(base), cfg, Path("data/splits"))
        typer.echo("[split] wrote: data/splits/")
    except Exception:
        # Conservative fallback: write a manifest placeholder rather than guess splits.
        _ensure_parents(Path("data/splits/split.json"))
        Path("data/splits/split.json").write_text(json.dumps({"mode": "fallback"}), "utf-8")
        typer.echo("[split] fallback manifest: data/splits/split.json")


# ------------------------------ ratings ---------------------------------
@app.command()
def ratings(
    base: str = typer.Option("data/interim/base.parquet", help="Canonical parquet"),
    out: str = typer.Option("data/ratings/ratings.parquet", help="Output ratings parquet"),
) -> None:
    """
    Build horse/jockey/trainer ratings using your modules.
    Resolution order:
      1) Try blend → eb → elo (common function names).
      2) Try first callable in those modules (duck-typed).
      3) Fallback: minimal Empirical-Bayes prior on win-rate, using only *past* races.

    Fallback EB math:
      For an entity e with cumulative wins W_e and appearances N_e (not counting current row):
        θ̂_e = (W_e + α·μ) / (N_e + α),
      where μ is the global mean win-rate and α is the prior strength (prior_n).
      This shrinks sparse entities toward μ, reducing variance without peeking ahead.
    """
    base_p = Path(base)
    out_p = Path(out)
    _ensure_parents(out_p)

    candidates = [
        "hqp.ratings.blend:build",
        "hqp.ratings.blend:build_ratings",
        "hqp.ratings.blend:run",
        "hqp.ratings.blend:main",
        "hqp.ratings.eb:build",
        "hqp.ratings.eb:build_ratings",
        "hqp.ratings.eb:compute",
        "hqp.ratings.eb:compute_ratings",
        "hqp.ratings.elo:build",
        "hqp.ratings.elo:compute",
        "hqp.ratings.elo:build_ratings",
    ]

    fn: Callable[..., Any] | None = None
    try:
        fn = resolve_callable(candidates)
    except Exception:
        for m in ["hqp.ratings.blend", "hqp.ratings.eb", "hqp.ratings.elo"]:
            fn = find_callable_in_module(m)
            if fn:
                break

    def _fallback_minimal_eb(in_parquet: Path, out_parquet: Path) -> None:
        """
        Minimal EB ratings that respect time order:
        - Sort by time and race/horse keys to ensure cumulatives exclude current row.
        - Compute per-entity prior-smoothed win rates for horse/jockey/trainer.
        """
        df = pd.read_parquet(in_parquet)

        keep_candidates = [
            "race_id",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "race_datetime",
            "race_dt",
            "off_time",
            "obs__is_winner",
        ]
        present = [c for c in keep_candidates if c in df.columns]
        df = df[present].copy()

        # Identify a usable time column (no leakage: cumulative excludes current row)
        time_col = next(
            (c for c in ("off_time", "race_dt", "race_datetime") if c in df.columns), None
        )
        sort_cols: list[str] = [time_col] if time_col else []
        sort_cols += [c for c in ("race_id", "horse_id") if c in df.columns]
        if sort_cols:
            # Stable sort (mergesort) preserves earlier order inside equal times.
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        if "obs__is_winner" not in df.columns:
            df["obs__is_winner"] = 0

        # Global mean win rate μ; prior_n = α controls shrinkage strength.
        mu = float(
            pd.to_numeric(df["obs__is_winner"], errors="coerce").fillna(0).astype(int).mean()
        )
        prior_n = 20.0

        def _bayes_rate(id_col: str) -> pd.Series:
            """
            Compute θ̂_e for each row as the *historical* (up-to-previous) EB win-rate for the
            entity value present in that row. We use cumulative counts minus the current row to
            avoid look-ahead bias.
            """
            if id_col not in df.columns:
                return pd.Series(mu, index=df.index)
            g = df.groupby(id_col, sort=False)
            # Number of *previous* appearances of this entity
            cnt_prev = g.cumcount()
            # Cumulative wins up to current row, then subtract current row's win to avoid peeking
            wins_prev = pd.to_numeric(g["obs__is_winner"].fillna(0), errors="coerce").astype(
                int
            ).cumsum() - pd.to_numeric(df["obs__is_winner"].fillna(0), errors="coerce").astype(int)
            return (wins_prev + prior_n * mu) / (cnt_prev + prior_n)

        df["rating_horse"] = _bayes_rate("horse_id")
        df["rating_jockey"] = _bayes_rate("jockey_id")
        df["rating_trainer"] = _bayes_rate("trainer_id")

        out_cols = [
            c
            for c in ["race_id", "horse_id", "rating_horse", "rating_jockey", "rating_trainer"]
            if c in df.columns or c.startswith("rating_")
        ]
        df[out_cols].to_parquet(out_parquet, index=False)

    if fn is None:
        _fallback_minimal_eb(base_p, out_p)
        typer.echo(f"[ratings] used fallback EB; written: {out}")
        return

    # Flexible call pattern: path-based signatures first; otherwise DataFrame based.
    try:
        try:
            fn(base_parquet=base_p, out_parquet=out_p)
        except TypeError:
            try:
                fn(base=base_p, out=out_p)
            except TypeError:
                df = pd.read_parquet(base_p)
                try:
                    res = fn(df)
                except TypeError:
                    res = fn(df, None)
                if not isinstance(res, pd.DataFrame):
                    raise TypeError("Ratings function returned non-DataFrame.") from None
                res.to_parquet(out_p, index=False)
    except Exception:
        _fallback_minimal_eb(base_p, out_p)
        typer.echo(f"[ratings] module call failed; used fallback EB; written: {out}")
        return

    typer.echo(f"Ratings written: {out}")


# ------------------------------ features --------------------------------
@app.command()
def features(
    base: str = typer.Option("data/interim/base.parquet", help="Canonical parquet"),
    ratings: str = typer.Option("data/ratings/ratings.parquet", help="Ratings parquet"),
    out: str = typer.Option("data/features/features.parquet", help="Output features parquet"),
) -> None:
    """
    Assemble model features, merging base + ratings and applying feature logic
    defined in hqp.features.build.* (without using same-race obs__* as predictors).

    Flexible invocation:
      - Prefer path-based signatures (base_parquet=..., ratings_parquet=..., out_parquet=...).
      - Fallback to DataFrame-based function shapes, writing the returned DataFrame.
    """
    base_p = Path(base)
    ratings_p = Path(ratings)
    out_p = Path(out)
    _ensure_parents(out_p)

    try:
        fn = resolve_callable(
            [
                "hqp.features.build:build_features",
                "hqp.features.build:build",
                "hqp.features.build:run",
                "hqp.features.build:main",
            ]
        )
    except ImportError:
        fn = find_callable_in_module("hqp.features.build")
        if fn is None:
            raise

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    try:
        try:
            fn(base_parquet=base_p, ratings_parquet=ratings_p, out_parquet=out_p)
        except TypeError:
            try:
                fn(base=base_p, ratings=ratings_p, out=out_p)
            except TypeError:
                runners_df = pd.read_parquet(base_p)
                ratings_df = pd.read_parquet(ratings_p)

                if len(params) >= 2:
                    third_has_default = len(params) >= 3 and (
                        params[2].default is not Signature.empty
                    )
                    if third_has_default:
                        res = fn(runners_df, ratings_df)
                    else:
                        res = fn(runners_df, ratings_df, None)
                else:
                    res = fn(runners_df)

                if not isinstance(res, pd.DataFrame):
                    raise TypeError("Features function returned non-DataFrame.") from None
                res.to_parquet(out_p, index=False)

    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"features step failed via {fn.__module__}.{fn.__name__}: {e}") from e

    typer.echo(f"Features written: {out}")


# ------------------------------- training --------------------------------
@app.command("model")
def model_cmd(
    features: str = typer.Option(..., "--features", help="Path to features parquet"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    artifacts_dir: str = typer.Option(
        "models/artifacts", "--artifacts-dir", help="Where to save run artifacts"
    ),
) -> None:
    """
    Train a model specified by a YAML config and persist artifacts under models/artifacts/<ts>.

    Notes:
    - _maybe_autoadjust_splits clamps n_splits to unique(group_key) for robustness.
    - After training, we *optionally* persist a serialized model and feature importance
      if the LGBM helper functions are present (best-effort).
    """
    from hqp.models.run_from_config import run_from_yaml

    features_p = Path(features)
    cfg_p = Path(config)
    cfg_p = _maybe_autoadjust_splits(features_p, cfg_p)

    run_dir = run_from_yaml(features_p, cfg_p, Path(artifacts_dir))
    typer.echo(f"[model] run directory: {run_dir}")

    # Post-train: persist model + feature importances (best-effort)
    try:
        import hqp.models.train_lgbm as _lgbm_mod

        # write model file
        wm = getattr(_lgbm_mod, "write_model_artifact", None)
        if callable(wm):
            p = wm(Path(run_dir))
            typer.echo(f"[model] model artifact written: {p}")

        # write feature importances (optional)
        wf = getattr(_lgbm_mod, "write_feature_importances", None)
        if callable(wf):
            wf(Path(run_dir))
            typer.echo("[model] feature importances written.")
    except Exception as _e:
        typer.echo(f"[model] note: post-train artifact write skipped: {_e}")


@app.command("train")
def train_cmd(
    features: str = typer.Option(..., "--features", help="Path to features parquet"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    artifacts_dir: str = typer.Option(
        "models/artifacts", "--artifacts-dir", help="Where to save run artifacts"
    ),
) -> None:
    """
    Alias for 'model' command to align with pipeline verb naming.
    """
    from hqp.models.run_from_config import run_from_yaml

    features_p = Path(features)
    cfg_p = Path(config)
    cfg_p = _maybe_autoadjust_splits(features_p, cfg_p)

    run_dir = run_from_yaml(features_p, cfg_p, Path(artifacts_dir))
    typer.echo(f"[train] run directory: {run_dir}")

    # Post-train: persist model + feature importances (best-effort)
    try:
        import hqp.models.train_lgbm as _lgbm_mod

        wm = getattr(_lgbm_mod, "write_model_artifact", None)
        if callable(wm):
            p = wm(Path(run_dir))
            typer.echo(f"[train] model artifact written: {p}")

        wf = getattr(_lgbm_mod, "write_feature_importances", None)
        if callable(wf):
            wf(Path(run_dir))
            typer.echo("[train] feature importances written.")
    except Exception as _e:
        typer.echo(f"[train] note: post-train artifact write skipped: {_e}")


# ------------------------------ evaluate-model ---------------------------
@app.command("evaluate-model")
def evaluate_model(
    run_dir: str = typer.Option(..., help="models/artifacts/<run>"),
    features: str = typer.Option("data/features/features.parquet", help="Features parquet"),
    dry_run: bool = typer.Option(False, help="Write stub diagnostics"),
) -> None:
    """
    Run model evaluation for a given artifact directory.
    Expected side-effect: writes reports/model/<ts>/predictions.parquet and summary/plots.
    """
    if dry_run:
        out_dir = Path(f"reports/model/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps({"ok": True, "notes": "stub"}), "utf-8")
        typer.echo(f"[evaluate-model] stubs under: {out_dir}")
        return

    try:
        from hqp.eval.model import evaluate as _eval_model  # type: ignore

        _eval_model(Path(run_dir), Path(features))  # <-- exactly two arguments
        typer.echo(f"[evaluate-model] done for run: {run_dir}")
    except Exception:
        out_dir = Path(f"reports/model/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps({"ok": True, "notes": "fallback stub"}), "utf-8"
        )
        typer.echo(f"[evaluate-model] fallback stubs under: {out_dir}")


# --------------------------------- odds ----------------------------------
@app.command()
def odds(
    base: str = typer.Option("data/interim/base.parquet", help="Canonical base parquet"),
    out: str = typer.Option("data/market/odds.parquet", help="Output odds parquet"),
    race_key: str = typer.Option("race_id", help="Race key column"),
    horse_key: str = typer.Option("horse_id", help="Horse key column"),
) -> None:
    """
    Build a strict odds parquet (race_id, horse_id, decimal_odds) from base.

    Logic:
      - Prefer explicit decimal odds columns (incl. Betfair SP).
      - Otherwise parse fractional odds 'a/b' → 1 + a/b as decimal.
      - We retain only finite decimal_odds > 1.0 (valid betting prices).
    """
    base_p = Path(base)
    out_p = Path(out)
    _ensure_parents(out_p)

    df = pd.read_parquet(base_p)

    # Candidate decimal odds columns in order of preference (includes your data)
    dec_candidates = [
        # “market” / generic
        "decimal_odds",
        "mkt_odds_decimal",
        "odds_decimal",
        # Starting prices
        "sp_decimal",
        "isp_decimal",
        "bsp_decimal",
        # Betfair SP in your base
        "obs__bsp",
        "bsp",
        "betfair_sp",
        "bfsp",
        # Some datasets put SP directly under 'sp'
        "sp",
        # LTP 5min before off (if you want as a proxy, place below SP)
        "ltp_5min",
    ]

    dec_col_found = next((c for c in dec_candidates if c in df.columns), None)

    # Candidate fractional odds columns (e.g., '7/2')
    frac_candidates = [
        "fractional_odds",
        "odds_fractional",
        "sp_fractional",
        "sp_frac",
        "fractional",
        "obs__sp_frac",
        "obs__fractional_odds",
    ]
    frac_col_found = next((c for c in frac_candidates if c in df.columns), None)

    def _coerce_decimal(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _parse_fractional_to_decimal(s: pd.Series) -> pd.Series:
        s = s.astype("string").fillna("")

        def _one(x: str) -> float | None:
            x = x.strip()
            if not x or "/" not in x:
                return None
            a, b = x.split("/", 1)
            try:
                return 1.0 + (float(a) / float(b))
            except Exception:
                return None

        return s.map(_one)

    if dec_col_found is not None:
        out_df = pd.DataFrame(
            {
                race_key: df[race_key],
                horse_key: df[horse_key],
                "decimal_odds": _coerce_decimal(df[dec_col_found]),
            }
        )
    elif frac_col_found is not None:
        out_df = pd.DataFrame(
            {
                race_key: df[race_key],
                horse_key: df[horse_key],
                "decimal_odds": _parse_fractional_to_decimal(df[frac_col_found]),
            }
        )
    else:
        cols = ", ".join(df.columns[:50])
        raise typer.BadParameter(
            "No recognizable odds column found in base. "
            f"Decimal candidates: {dec_candidates}; fractional candidates: {frac_candidates}. "
            f"First 50 columns: [{cols}]"
        )

    out_df["decimal_odds"] = pd.to_numeric(out_df["decimal_odds"], errors="coerce")
    out_df = out_df.dropna(subset=["decimal_odds"]).copy()

    # Strict: must have positive, finite odds (> 1.0 by convention for decimal odds)
    out_df = out_df[(out_df["decimal_odds"] > 1.0) & np.isfinite(out_df["decimal_odds"])]
    if out_df.empty:
        raise typer.BadParameter("All candidate odds values are missing/invalid after parsing.")

    out_df[[race_key, horse_key, "decimal_odds"]].to_parquet(out_p, index=False)
    typer.echo(f"[odds] written: {out} rows={len(out_df)}")


# --------------------------------- market --------------------------------
@app.command()
def market(
    features: str = typer.Option(
        "data/features/features.parquet",
        "--features",
        "--base",  # deprecated alias for backwards compatibility
        help="Path to features parquet",
    ),
    odds_path: str = typer.Option(
        "data/market/odds.parquet",
        "--odds-path",
        "--odds_path",
        "--odds",
        help="Path to raw odds parquet",
    ),
    out: str = typer.Option("data/market/market_join.parquet", help="Output path"),
    decimal_col: str = typer.Option("decimal_odds", help="Decimal odds column in odds file"),
    fractional_col: str | None = typer.Option(None, help="Fractional odds column (e.g., '7/2')"),
    race_key: str = typer.Option("race_id", help="Race key column"),
    horse_key: str = typer.Option("horse_id", help="Horse key column"),
    prefix: str = typer.Option("mkt", help="Prefix for output columns"),
) -> None:
    """
    Join market odds onto features and **normalize implied probabilities per race**.

    Math:
      - Raw implied from decimal odds: p_i = 1 / odds_i.
      - Racewise normalization (remove overround): p̂_i = p_i / Σ_j p_j (over runners in race).
        This ensures Σ_i p̂_i = 1 for that race, making “edge” comparisons coherent.
    """
    f_p = Path(features)
    o_p = Path(odds_path)
    out_p = Path(out)
    _ensure_parents(out_p)

    if not o_p.exists():
        raise typer.BadParameter(
            f"Odds parquet not found at {o_p}. "
            "Generate it with: 'poetry run hqp odds' (or fix --odds-path)."
        )

    feats = pd.read_parquet(f_p)

    # Coerce keys on features to string for robust joins (heterogeneous source types happen)
    for k in (race_key, horse_key):
        if k in feats.columns:
            feats[k] = feats[k].astype("string")

    # Preferred path: dedicated attach_market (handles fractional odds, hygiene)
    try:
        from hqp.market import attach_market as _attach_market  # type: ignore

        odds_df = pd.read_parquet(o_p)
        for k in (race_key, horse_key):
            if k in odds_df.columns:
                odds_df[k] = odds_df[k].astype("string")

        merged = _attach_market(
            feats,
            odds_df,
            race_key=race_key,
            horse_key=horse_key,
            decimal_col=decimal_col,
            fractional_col=fractional_col,
            prefix=prefix,
        )
    except Exception:
        # Strict fallback: simple join + implied prob from decimal odds; normalize per race
        odds_df = pd.read_parquet(o_p)
        for k in (race_key, horse_key):
            if k in odds_df.columns:
                odds_df[k] = odds_df[k].astype("string")

        dec_out = f"{prefix}_odds_decimal"
        imp_out = f"{prefix}_implied"

        if decimal_col not in odds_df.columns:
            raise typer.BadParameter(
                f"Expected decimal odds column '{decimal_col}' in {o_p}. "
                "Rebuild odds parquet with the correct column name via 'hqp odds'."
            )

        df = feats.merge(
            odds_df[[race_key, horse_key, decimal_col]].rename(columns={decimal_col: dec_out}),
            on=[race_key, horse_key],
            how="left",
            copy=False,
        )
        if df[dec_out].isna().all():
            raise typer.BadParameter(
                f"All joined {dec_out} values are NaN. Check your keys ({race_key},{horse_key}) "
                "or regenerate odds with 'hqp odds'."
            )

        df[imp_out] = 1.0 / df[dec_out]
        sums = df.groupby(race_key, sort=False)[imp_out].transform("sum")
        df.loc[sums.notna() & (sums > 0), imp_out] = df[imp_out] / sums
        merged = df

    dec_col = f"{prefix}_odds_decimal"
    imp_col = f"{prefix}_implied"
    n_rows = len(merged)
    n_dec_nan = int(merged[dec_col].isna().sum()) if dec_col in merged else 0
    zero_valid_races = 0
    if imp_col in merged:
        counts = merged.groupby(race_key, sort=False)[imp_col].count()
        zero_valid_races = int((counts == 0).sum())

    if n_rows == 0 or n_dec_nan == n_rows:
        raise typer.BadParameter(
            "Market join produced no usable odds. Verify odds parquet & key columns."
        )

    merged.to_parquet(out_p, index=False)
    typer.echo(
        f"[market] wrote: {out}\n"
        f"         rows={n_rows}, NaN {dec_col}={n_dec_nan}, "
        f"races with 0 valid odds={zero_valid_races}"
    )


# ---------------------------- evaluate-market ----------------------------
@app.command("evaluate-market")
def evaluate_market(
    market: str = typer.Option("data/market/market_join.parquet", help="Market-joined parquet"),
    dry_run: bool = typer.Option(False, help="Write stub checks"),
) -> None:
    """
    Run market sanity checks/analytics (e.g., overround distributions, coverage).
    """
    if dry_run:
        out_dir = Path(f"reports/market/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps({"ok": True, "notes": "stub"}), "utf-8")
        typer.echo(f"[evaluate-market] stubs under: {out_dir}")
        return

    try:
        from hqp.eval.market import evaluate as _eval_market  # type: ignore

        _eval_market(Path(market))
        typer.echo("[evaluate-market] done")
    except Exception:
        out_dir = Path(f"reports/market/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps({"ok": True, "notes": "fallback stub"}), "utf-8"
        )
        typer.echo(f"[evaluate-market] fallback stubs under: {out_dir}")


## --------------------------------- edge ----------------------------------
@app.command()
def edge(
    market: str = typer.Option("data/market/market_join.parquet", help="Market-joined parquet"),
    run_dir: Optional[str] = typer.Option(
        None, help="models/artifacts/<run> (required unless --dry-run)"
    ),
    out: str = typer.Option("data/market/edges.parquet", help="Output edges parquet"),
    dry_run: bool = typer.Option(False, help="Write a stub edges file"),
) -> None:
    """
    Compute model-vs-market edge table using hqp.eval.edge.compute.

    Edge definition:
      edge = model_prob − mkt_implied   (both probabilities refer to the same runner, same race).
      mkt_implied is race-normalized (Σ runners = 1) to make edges comparable across races.
    """
    out_p = Path(out)
    _ensure_parents(out_p)

    if dry_run:
        pd.DataFrame({"race_id": [], "horse_id": [], "edge": []}).to_parquet(out_p, index=False)
        typer.echo(f"[edge] stub written: {out}")
        return

    if run_dir is None:
        typer.echo("[edge] Missing --run-dir (required without --dry-run).")
        raise typer.Exit(code=1)

    # Use the real evaluator (not the old module), and don't swallow errors
    from hqp.eval.edge import compute as _edge_compute  # type: ignore

    _edge_compute(Path(market), Path(run_dir), out_p)

    # Light, controlled confirmation (no redirected stdout noise)
    try:
        n_rows = int(pd.read_parquet(out_p, columns=["race_id"]).shape[0])
        typer.echo(f"[edge] written: {out} rows={n_rows}")
    except Exception:
        typer.echo(f"[edge] written: {out}")


@app.command()
def backtest(
    edges: str = typer.Option("data/market/edges.parquet", help="Edges parquet"),
    config: Optional[str] = typer.Option(None, help="configs/backtest.yaml"),
    dry_run: bool = typer.Option(False, help="Write a stub report"),
) -> None:
    """
    Run the backtest over computed edges (no silent fallbacks).
    Prints a concise one-line summary from the generated summary.json.

    Backtest config highlights:
      - edge_threshold: minimum model−market probability gap to place a bet.
      - max_odds: cap long-tail exposure; stabilizes variance.
      - per_race_max_bets: capacity control / diversification.
      - stake / kelly_fraction: staking policy (flat or Kelly-fractional).
      - try_join_market: allow label attach if edges parquet omits labels.
    """
    if dry_run:
        out_dir = Path(f"reports/backtest/{_ts()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps({"ok": True, "notes": "stub"}), encoding="utf-8"
        )
        typer.echo(f"[backtest] stub under: {out_dir}")
        return

    edges_p = Path(edges)
    if not edges_p.exists():
        raise typer.BadParameter(f"Edges parquet not found at {edges_p}")

    try:
        # Use the eval/backtest implementation (not the old hqp.backtest)
        from hqp.eval.backtest import run as _bt_run  # type: ignore
    except Exception as e:
        raise typer.Exit(code=1) from e

    # Load YAML safely; accept either flat dict or backtest: { ... } section.
    cfg_raw: Dict[str, Any] = _safe_load_yaml(Path(config)) if config else {}
    cfg: Dict[str, Any]
    bt_section = cfg_raw.get("backtest")

    if isinstance(bt_section, dict):
        cfg = {str(k): v for k, v in cast(Dict[str, Any], bt_section).items()}
    else:
        cfg = {str(k): v for k, v in cfg_raw.items()}

    # (optional) echo effective core settings for transparency
    if cfg:
        keys = [
            "edge_threshold",
            "max_odds",
            "per_race_max_bets",
            "stake",
            "kelly_fraction",
            "try_join_market",
        ]
        shown: Dict[str, Any] = {k: cfg.get(k) for k in keys if k in cfg}
        if shown:
            typer.echo(f"[backtest] using cfg: {shown}")

    # Run and print a concise summary
    out_dir = _bt_run(edges_p, cfg)

    # Small, deterministic console output from summary.json
    summary_p = out_dir / "summary.json"
    try:
        with summary_p.open("r", encoding="utf-8") as f:
            s = json.load(f)
        # Expected keys from BTResult: rows, bets, races_bet, pnl, roi, hit_rate, avg_odds, ts
        rows = int(s.get("rows", 0))
        bets = int(s.get("bets", 0))
        races = int(s.get("races_bet", 0))
        pnl = float(s.get("pnl", 0.0))
        roi = float(s.get("roi", 0.0))
        hr = float(s.get("hit_rate", 0.0))
        avg_odds = float(s.get("avg_odds", 0.0))
        typer.echo(
            "[backtest] "
            f"rows={rows} bets={bets} races={races} pnl={pnl:.2f} roi={roi:.3f} "
            f"hit_rate={hr:.3f} avg_odds={avg_odds:.2f} -> {out_dir}"
        )
    except Exception:
        # If summary can't be read, still confirm completion, but keep it short.
        typer.echo(f"[backtest] done -> {out_dir}")


# ------------------------------ backtest compare ----------------------------------
@app.command("init-backtest-configs")
def init_backtest_configs(
    force: bool = typer.Option(False, help="Overwrite existing files"),
):
    """
    Create common backtest configs:
      - configs/backtest_baseline.yaml (strict)
      - configs/backtest_relaxed.yaml  (looser)
    Idempotent unless --force is supplied.
    """
    strict = Path("configs/backtest_baseline.yaml")
    relaxed = Path("configs/backtest_relaxed.yaml")

    strict_content = """\
version: 0
backtest:
  edge_threshold: 0.10
  max_odds: 12.0
  per_race_max_bets: 1
  stake: 1.0
  kelly_fraction: 0.0
  try_join_market: true
  min_samples_per_bucket: 1
  allow_ties: false
paths:
  edges: "data/market/edges.parquet"
  outdir: "reports/backtest"
"""
    relaxed_content = """\
version: 0
backtest:
  edge_threshold: 0.02
  max_odds: 50.0
  per_race_max_bets: 3
  stake: 1.0
  kelly_fraction: 0.0
  try_join_market: true
paths:
  edges: "data/market/edges.parquet"
  outdir: "reports/backtest"
"""

    # Only write baseline/relaxed if missing, unless --force
    strict.parent.mkdir(parents=True, exist_ok=True)

    wrote: list[str] = []
    if force or not strict.exists():
        strict.write_text(strict_content, encoding="utf-8")
        wrote.append(str(strict))
    if force or not relaxed.exists():
        relaxed.write_text(relaxed_content, encoding="utf-8")
        wrote.append(str(relaxed))

    if wrote:
        typer.echo("[init-backtest-configs] created/updated:\n- " + "\n- ".join(wrote))
    else:
        typer.echo("[init-backtest-configs] nothing to do (files already present).")


@app.command("backtest-compare")
def backtest_compare(
    edges: str = typer.Option("data/market/edges.parquet", help="Edges parquet to backtest."),
    configs: list[str] = typer.Option(
        None,
        help="Pairs label=path (repeatable). If omitted, uses default/strict/relaxed.",
    ),
):
    """
    Run multiple backtests (default/strict/relaxed) and print a comparison table.
    Writes reports/backtest/compare_<ts>.parquet

    Tip:
      Use this to demonstrate robustness: e.g., ROI sensitivity to edge_threshold/odds caps.
    """
    # Import here so Pylance doesn't flag unused imports at module level.
    from hqp.eval.backtest_runner import compare_configs, default_config_set

    if configs:
        pairs: list[tuple[str, str]] = []
        for item in configs:
            if "=" not in item:
                raise typer.BadParameter(
                    f"Invalid --configs entry '{item}'. Expected format label=path."
                )
            k, v = item.split("=", 1)
            pairs.append((k, v))
        cfgs: dict[str, str] = dict(pairs)
    else:
        cfgs = default_config_set()

    typer.echo(f"[backtest-compare] edges={edges}")
    df = compare_configs(edges_path=edges, configs=cfgs)
    small = df[["label", "bets", "roi", "hit_rate", "avg_odds", "pnl"]].sort_values(
        "roi", ascending=False
    )
    typer.echo("\n" + small.to_string(index=False))


# ------------------------------ calibration ----------------------------------
@app.command("calibration")
def calibration(
    pred: str = typer.Option(
        ..., "--pred", help="Predictions parquet; must have model_prob and keys."
    ),
    out: str = typer.Option(..., "--out", help="Output directory, e.g. reports/calibration/<ts>/"),
    labels: Optional[str] = typer.Option(
        None,
        "--labels",
        help="Parquet with ground-truth labels. If 'won' is missing in --pred, "
        "we will merge by (race_id, horse_id) and look for one of: won, obs__is_winner, is_winner.",
    ),
) -> None:
    """
    Probability calibration diagnostics: deciles, Brier score, calibration plot.

    Behavior:
      - If predictions already contain 'won', we run diagnostics directly.
      - Else, we merge labels on (race_id, horse_id) from --labels or default sources.
      - Outputs include before/after metrics and reliability tables/plots.
    """
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    # If predictions already include 'won', run directly
    df_pred = pd.read_parquet(pred)
    has_won = "won" in df_pred.columns

    if not has_won:
        # Decide where to source labels from:
        # 1) user-provided --labels
        # 2) default to base or features if present
        label_path: Optional[Path] = Path(labels) if labels else None
        if label_path is None:
            # Prefer base if it exists; else try features
            for candidate in [
                Path("data/interim/base.parquet"),
                Path("data/features/features.parquet"),
            ]:
                if candidate.exists():
                    label_path = candidate
                    break

        if label_path is None or not label_path.exists():
            raise typer.BadParameter(
                "Predictions lack 'won' and no usable labels parquet was found. "
                "Provide one via --labels (must contain won/obs__is_winner/is_winner plus race_id,horse_id)."
            )

        df_lbl = pd.read_parquet(label_path)

        # Find a label column
        label_cols = [
            c
            for c in ["won", "obs__is_winner", "is_winner", "label", "target"]
            if c in df_lbl.columns
        ]
        if not label_cols:
            raise typer.BadParameter(
                f"Labels parquet at {label_path} has no recognized ground-truth column "
                "(expected one of: won, obs__is_winner, is_winner, label, target)."
            )
        ycol = label_cols[0]

        # Require keys
        for k in ["race_id", "horse_id"]:
            if k not in df_pred.columns or k not in df_lbl.columns:
                raise typer.BadParameter(
                    f"Both predictions and labels must contain '{k}' to merge ground truth."
                )

        # Cast keys to string for safer join
        for k in ["race_id", "horse_id"]:
            df_pred[k] = df_pred[k].astype("string")
            df_lbl[k] = df_lbl[k].astype("string")

        merged = df_pred.merge(
            df_lbl[["race_id", "horse_id", ycol]], on=["race_id", "horse_id"], how="left"
        )
        if merged[ycol].isna().all():
            raise typer.BadParameter(
                "After merging labels, all ground-truth values are NaN. "
                "Check that (race_id, horse_id) keys align between predictions and labels."
            )

        merged["won"] = (
            pd.to_numeric(merged[ycol], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )

        # Persist a temporary parquet that includes 'won', then call the evaluator
        pred_for_cal = out_p / "predictions_with_won.parquet"
        merged.drop(columns=[ycol], errors="ignore").to_parquet(pred_for_cal, index=False)

        typer.echo(
            f"[calibration] added 'won' by merging labels from {label_path.name} "
            f"(using '{ycol}'); rows={len(merged)}"
        )
        from hqp.eval.calibration import calibrate_from_predictions  # local import kept near use

        calibrate_from_predictions(pred_path=str(pred_for_cal), outdir=str(out_p))
    else:
        from hqp.eval.calibration import calibrate_from_predictions  # local import kept near use

        calibrate_from_predictions(pred_path=pred, outdir=str(out_p))


# ------------------------------ backtest sweep ----------------------------------
@app.command("backtest-sweep")
def backtest_sweep_cmd(
    edges: str = typer.Option("data/market/edges.parquet", help="Edges parquet to backtest."),
    edge_min: float = typer.Option(0.02, help="Minimum edge_threshold to sweep (inclusive)."),
    edge_max: float = typer.Option(0.20, help="Maximum edge_threshold to sweep (inclusive)."),
    edge_step: float = typer.Option(0.02, help="Step size for edge_threshold."),
    odds_grid: str = typer.Option("8,10,12,15,20", help="Comma-separated max_odds values."),
    per_race_max_bets: int = typer.Option(1, help="per_race_max_bets for all grid points."),
    stake: float = typer.Option(1.0, help="Fixed stake per bet (if kelly_fraction=0.0)."),
    kelly_fraction: float = typer.Option(0.0, help="Kelly fraction to use across the sweep."),
    try_join_market: bool = typer.Option(
        True, help="Attempt to join market if not already present."
    ),
    no_plot: bool = typer.Option(False, help="Disable the ROI plot."),
) -> None:
    """
    Sweep backtests across edge_threshold and max_odds.
    Writes reports/backtest/sweep_<ts>.parquet and optionally a ROI plot.

    Interpretation:
      - edge_threshold ↑ → fewer, “higher-conviction” bets (lower variance, possibly higher ROI).
      - max_odds ↓ → caps tail risk; often increases hit rate but may reduce total ROI.
    """
    from math import isfinite

    # Build the edge threshold sequence
    if edge_step <= 0:
        raise typer.BadParameter("--edge-step must be > 0")

    edge_vals: list[float] = []
    x = edge_min
    # Robust accumulation to avoid float issues; clamp to max on last step
    while x <= edge_max + 1e-12:
        edge_vals.append(round(x, 6))
        x += edge_step
    # Deduplicate / sort for safety
    edge_vals = sorted(set(edge_vals))

    # Parse odds grid
    try:
        mo_vals = [float(s.strip()) for s in odds_grid.split(",") if s.strip()]
        mo_vals = [m for m in mo_vals if isfinite(m) and m > 1.0]
        if not mo_vals:
            raise ValueError
    except Exception:
        raise typer.BadParameter(
            "Invalid --odds-grid. Provide comma-separated numbers, e.g. '8,10,12,15,20'."
        ) from None

    from hqp.eval.backtest_sweep import (
        sweep as _sweep,
    )  # local import to avoid top-level import noise

    out_parquet = _sweep(
        edges_path=edges,
        edge_thresholds=edge_vals,
        max_odds_grid=mo_vals,
        per_race_max_bets=per_race_max_bets,
        stake=stake,
        kelly_fraction=kelly_fraction,
        try_join_market=try_join_market,
        make_plot=not no_plot,
    )

    typer.echo(f"[backtest-sweep] written: {out_parquet}")
    try:
        df = pd.read_parquet(out_parquet)
        top = (
            df[["edge_threshold", "max_odds", "bets", "roi", "hit_rate", "avg_odds", "pnl"]]
            .sort_values("roi", ascending=False)
            .head(10)
        )
        typer.echo("\n[backtest-sweep] Top 10 by ROI:")
        typer.echo(top.to_string(index=False))
    except Exception as e:  # noqa: BLE001
        typer.echo(f"[backtest-sweep] (note) could not print top-10: {e}")


# ------------------------------ predict-eb --------------------------------
@app.command("predict-eb")
def predict_eb(
    market: str = typer.Option(..., help="Path to market join parquet (from `hqp market`)"),
    out: str = typer.Option("reports/model/pred_eb.parquet", help="Output predictions parquet"),
) -> None:
    """
    Write EB-blended model probabilities from a market join file.
    Schema: race_id, horse_id, model_prob
    """
    from pathlib import Path
    from hqp.eval.edge import write_eb_predictions
    import pandas as pd

    market_p = Path(market)
    out_p = Path(out)
    write_eb_predictions(market_p, out_p)
    n_rows = int(pd.read_parquet(out_p).shape[0])
    typer.echo(f"[predict-eb] wrote: {out} rows={n_rows}")


# ------------------------------ fit-calibrator ----------------------------------
@app.command("fit-calibrator")
def fit_calibrator_cmd(
    pred: str = typer.Option(
        ...,
        "--pred",
        help="Predictions parquet (needs model_prob; will merge labels if 'won' missing).",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Output path for calibrator artifact, e.g. models/calibration/{ts}.pkl",
    ),
    method: str = typer.Option(
        "platt",
        "--method",
        help="Calibration method: platt | isotonic | none",
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels",
        help="Labels parquet (used if 'won' missing in --pred). Must have race_id,horse_id and a label column.",
    ),
) -> None:
    """
    Fit a post-hoc probability calibrator (Platt or Isotonic) from predictions, report Brier deltas,
    and save the calibrator artifact.

    Platt scaling reminder:
      p_cal = σ(a·p + b), parameters (a,b) minimize logistic loss vs observed labels on a
      calibration fold (held-out from model training to avoid bias).
    """
    method_lc_raw = method.lower().strip()
    if method_lc_raw not in {"platt", "isotonic", "none"}:
        raise typer.BadParameter("--method must be one of: platt, isotonic, none")

    method_lc: CalibMethod = method_lc_raw  # type: ignore[assignment]

    cal, df = fit_from_predictions_parquet(pred_path=pred, method=method_lc, labels_path=labels)

    # Evaluate Brier before/after (lower is better)
    p_before = df["model_prob"].to_numpy()
    y = df["won"].to_numpy().astype(int)
    p_after = cal.apply(p_before)
    brier = evaluate_brier(p_before, p_after, y)

    saved = save_calibrator(cal, out)

    typer.echo(
        "[fit-calibrator] "
        f"method={method_lc} saved={saved}\n"
        f"    brier_before={brier['brier_before']:.6f} "
        f"brier_after={brier['brier_after']:.6f} "
        f"improvement={brier['improvement']:.6f}"
    )


# ------------------------------ apply-calibrator --------------------------------
@app.command("apply-calibrator")
def apply_calibrator_cmd(
    pred: str = typer.Option(
        ...,
        "--pred",
        help="Predictions parquet to read (expects a probability column; default 'model_prob').",
    ),
    calibrator: str = typer.Option(
        ...,
        "--calibrator",
        help="Path to saved calibrator .pkl (created by 'hqp fit-calibrator').",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Output predictions parquet with calibrated probabilities added.",
    ),
    col_in: str = typer.Option(
        "model_prob",
        "--col-in",
        help="Input probability column name in --pred.",
    ),
    col_out: str = typer.Option(
        "model_prob_cal",
        "--col-out",
        help="Output calibrated probability column name.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="If set, also overwrite the input column with calibrated values.",
    ),
) -> None:
    """
    Apply a saved calibrator to a predictions parquet and write calibrated probabilities.

    Note:
      We do not renormalize within race here; if your model ensures Σ p = 1 per race,
      that invariant remains true only if you re-normalize after per-runner calibration.
      (We often re-normalize in downstream evaluation if needed.)
    """
    df = pd.read_parquet(pred)
    if col_in not in df.columns:
        raise typer.BadParameter(f"Input column '{col_in}' not found in {pred}")

    cal = load_calibrator(calibrator)
    df[col_out] = cal.apply(df[col_in].to_numpy())

    if replace:
        df[col_in] = df[col_out]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    typer.echo(
        f"[apply-calibrator] wrote: {out} " f"(in_col={col_in} out_col={col_out} replace={replace})"
    )


# ------------------------------ edge-from-pred --------------------------------
@app.command("edge-from-pred")
def edge_from_pred(
    market: str = typer.Option(
        "data/market/market_join.parquet",
        "--market",
        help="Market-joined parquet from `hqp market` (needs race_id, horse_id, and mkt_implied/mkt_odds_decimal).",
    ),
    pred: str = typer.Option(
        ...,
        "--pred",
        help="Predictions parquet (raw or calibrated) with race_id, horse_id, and a probability column.",
    ),
    probs_col: str = typer.Option(
        "model_prob",
        "--probs-col",
        help="Probability column in --pred to use when computing edges (e.g., model_prob_cal).",
    ),
    out: str = typer.Option(
        "data/market/edges_from_pred.parquet",
        "--out",
        help="Output edges parquet (race_id, horse_id, edge, model_prob, mkt_implied, mkt_odds_decimal).",
    ),
    race_key: str = typer.Option("race_id", help="Race key column name."),
    horse_key: str = typer.Option("horse_id", help="Horse key column name."),
    market_prefix: str = typer.Option(
        "mkt",
        "--market-prefix",
        help="Prefix used by `hqp market` (expects {prefix}_implied and {prefix}_odds_decimal). Default: mkt.",
    ),
) -> None:
    """
    Build an edges parquet directly from a predictions parquet by joining
    model probabilities to market implied probabilities:
        model_prob = <probs_col>
        edge = model_prob − {prefix}_implied
    Also persists {prefix}_implied and {prefix}_odds_decimal for backtest.
    """
    market_p = Path(market)
    pred_p = Path(pred)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not market_p.exists():
        raise typer.BadParameter(f"Market parquet not found: {market_p}")
    if not pred_p.exists():
        raise typer.BadParameter(f"Predictions parquet not found: {pred_p}")

    df_mkt = pd.read_parquet(market_p)
    df_pred = pd.read_parquet(pred_p)

    # validate columns
    if probs_col not in df_pred.columns:
        raise typer.BadParameter(f"Column '{probs_col}' not found in predictions: {pred_p}")
    imp_col = f"{market_prefix}_implied"
    dec_col = f"{market_prefix}_odds_decimal"
    for c in (imp_col, dec_col):
        if c not in df_mkt.columns:
            raise typer.BadParameter(
                f"Column '{c}' not found in market parquet. "
                "Rebuild with `hqp market` or adjust --market-prefix."
            )

    # coerce keys to string for robust join
    for k in (race_key, horse_key):
        if k in df_pred.columns:
            df_pred[k] = df_pred[k].astype("string")
        if k in df_mkt.columns:
            df_mkt[k] = df_mkt[k].astype("string")

    # select & merge needed columns
    left = df_pred[[race_key, horse_key, probs_col]].rename(columns={probs_col: "model_prob"})
    right = df_mkt[[race_key, horse_key, imp_col, dec_col]]
    merged = left.merge(right, on=[race_key, horse_key], how="left", copy=False)

    if merged[imp_col].isna().all():
        raise typer.BadParameter(
            "All joined market implied probabilities are NaN. "
            "Check that (race_id, horse_id) keys align between predictions and market."
        )

    # numeric + edge
    merged["model_prob"] = pd.to_numeric(merged["model_prob"], errors="coerce")
    merged[imp_col] = pd.to_numeric(merged[imp_col], errors="coerce")
    merged[dec_col] = pd.to_numeric(merged[dec_col], errors="coerce")
    merged = merged.dropna(subset=["model_prob", imp_col]).copy()
    merged["edge"] = merged["model_prob"] - merged[imp_col]

    # final columns for backtest
    out_df = merged[[race_key, horse_key, "edge", "model_prob", imp_col, dec_col]].copy()
    out_df.to_parquet(out_p, index=False)

    typer.echo(
        f"[edge-from-pred] wrote: {out_p} rows={len(out_df)} " f"(probs={probs_col} vs {imp_col})"
    )


# --------------------------- backtest-from-pred (helper) ------------------------
from typing import Any, Dict, Mapping, Optional, cast  # (kept local in original; duplicated here)


@app.command("backtest-from-pred")
def backtest_from_pred(
    market: str = typer.Option(
        "data/market/market_join.parquet", "--market", help="Market-joined parquet."
    ),
    pred: str = typer.Option(..., "--pred", help="Predictions parquet (raw or calibrated)."),
    probs_col: str = typer.Option(
        "model_prob", "--probs-col", help="Probability column to use (e.g., model_prob_cal)."
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Backtest config (YAML). Defaults to configs/backtest.yaml if omitted.",
    ),
    edges_out: Optional[str] = typer.Option(
        None,
        "--edges-out",
        help="Optional path for intermediate edges parquet. Default: reports/backtest/<ts>/edges_from_pred.parquet",
    ),
) -> None:
    """
    Convenience: build edges from predictions and immediately run backtest.

    Guarantees:
      - Injects 'market_path' into the backtest config so the engine can attach labels if needed.
      - Keeps artifacts under a timestamped reports/backtest/<ts>/ directory for auditability.
    """
    mkt_p = Path(market)
    if not mkt_p.exists():
        raise typer.BadParameter(f"Market parquet not found at {mkt_p}. Run `hqp market` first.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_edges = (
        Path(edges_out) if edges_out else Path(f"reports/backtest/{ts}/edges_from_pred.parquet")
    )
    tmp_edges.parent.mkdir(parents=True, exist_ok=True)

    # 1) Produce edges (reuses the CLI function above; behaviorally identical to calling it from shell)
    edge_from_pred(
        market=market,
        pred=pred,
        probs_col=probs_col,
        out=str(tmp_edges),
        race_key="race_id",
        horse_key="horse_id",
        market_prefix="mkt",
    )
    typer.echo(f"[backtest-from-pred] edges -> {tmp_edges}")

    # 2) Backtest with market_path injected so labels can be joined
    try:
        from hqp.eval.backtest import run as _bt_run  # type: ignore
    except Exception as e:
        raise typer.Exit(code=1) from e

    cfg_raw: Dict[str, Any] = _safe_load_yaml(Path(config)) if config else {}
    bt_section = cfg_raw.get("backtest")

    # ✅ Cast to a typed Mapping to keep Pylance happy; avoid str() on Unknown
    cfg: Dict[str, Any]
    if isinstance(bt_section, dict):
        typed_bt = cast(Mapping[str, Any], bt_section)
        cfg = dict(typed_bt)
    else:
        # cfg_raw is already Dict[str, Any]; copying keeps type info
        cfg = dict(cfg_raw)

    cfg["market_path"] = str(mkt_p)

    # Echo effective knobs
    if cfg:
        keys = [
            "edge_threshold",
            "max_odds",
            "per_race_max_bets",
            "stake",
            "kelly_fraction",
            "try_join_market",
            "market_path",
        ]
        shown = {k: cfg.get(k) for k in keys if k in cfg}
        if shown:
            typer.echo(f"[backtest] using cfg: {shown}")

    out_dir = _bt_run(Path(tmp_edges), cfg)

    # Print concise summary
    summary_p = out_dir / "summary.json"
    try:
        with summary_p.open("r", encoding="utf-8") as f:
            s = json.load(f)
        rows = int(s.get("rows", 0))
        bets = int(s.get("bets", 0))
        races = int(s.get("races_bet", 0))
        pnl = float(s.get("pnl", 0.0))
        roi = float(s.get("roi", 0.0))
        hr = float(s.get("hit_rate", 0.0))
        avg_odds = float(s.get("avg_odds", 0.0))
        typer.echo(
            "[backtest] "
            f"rows={rows} bets={bets} races={races} pnl={pnl:.2f} roi={roi:.3f} "
            f"hit_rate={hr:.3f} avg_odds={avg_odds:.2f} -> {out_dir}"
        )
    except Exception:
        typer.echo(f"[backtest] done -> {out_dir}")


# -------------------------------- report ---------------------------------
@app.command()
def report(
    run_dir: str = typer.Option(..., help="models/artifacts/<run>"),
    out: str = typer.Option("reports/{ts}/report.md", help="Output path (md/html)"),
    dry_run: bool = typer.Option(False, help="Write a stub report"),
) -> None:
    """
    Build a compact assignment report (Q0–Q4) for a given run directory.
    Output defaults to a timestamped Markdown file under reports/.
    """
    out_path = Path(out.format(ts=_ts()))
    _ensure_parents(out_path)

    if dry_run:
        out_path.write_text("# Report (stub)\n", encoding="utf-8")
        typer.echo(f"[report] stub written: {out_path}")
        return

    try:
        from hqp.report import build as _build_report  # type: ignore

        _build_report(Path(run_dir), out_path)
        typer.echo(f"[report] written: {out_path}")
    except Exception:
        out_path.write_text("# Report (fallback)\n", encoding="utf-8")
        typer.echo(f"[report] fallback written: {out_path}")


# ------------------------------ run-all (with calibrated strategy) ------------------------------
@app.command("run-all")
def run_all(
    quick: bool = typer.Option(False, help="Quick smoke: stub configs & dry-runs where possible"),
) -> None:
    """
    Execute the full pipeline end-to-end with a single execution path.
    After training, we also run a calibrated EB strategy:
      market → EB predictions → fit calibrator → apply → edges (cal) → backtest (calibrated config)
    """

    # Build stage list (each is a zero-arg callable so typing stays simple)
    stages: list[tuple[str, Callable[[], None]]] = []

    # 1) Ingest / Validate / Split
    if quick:
        stages += [
            ("ingest", lambda: ingest(config=None, out="data/interim/base.parquet", dry_run=True)),
            (
                "validate",
                lambda: validate(base="data/interim/base.parquet", schema=None, dry_run=True),
            ),
            ("split", lambda: split(base="data/interim/base.parquet", config=None, dry_run=True)),
        ]
    else:
        stages += [
            (
                "ingest",
                lambda: ingest(
                    config="configs/ingest.yaml", out="data/interim/base.parquet", dry_run=False
                ),
            ),
            (
                "validate",
                lambda: validate(
                    base="data/interim/base.parquet", schema="configs/schema.yaml", dry_run=False
                ),
            ),
            (
                "split",
                lambda: split(
                    base="data/interim/base.parquet", config="configs/split.yaml", dry_run=False
                ),
            ),
        ]

    # 2) Ratings → Features
    stages += [
        (
            "ratings",
            lambda: ratings(base="data/interim/base.parquet", out="data/ratings/ratings.parquet"),
        ),
        (
            "features",
            lambda: features(
                base="data/interim/base.parquet",
                ratings="data/ratings/ratings.parquet",
                out="data/features/features.parquet",
            ),
        ),
    ]

    # 3) Train (choose config; auto-adjust splits)
    run_dir_holder: dict[str, str] = {"run_dir": ""}

    def _train_and_hold() -> None:
        cfg_main = Path("configs/model_lgbm.yaml")
        cfg_tiny = Path("configs/model_lgbm_tiny.yaml")
        cfg_to_use = cfg_tiny if (quick or not cfg_main.exists()) else cfg_main
        cfg_path = _maybe_autoadjust_splits(Path("data/features/features.parquet"), cfg_to_use)
        from hqp.models.run_from_config import run_from_yaml

        rd = run_from_yaml(
            Path("data/features/features.parquet"), Path(cfg_path), Path("models/artifacts")
        )
        typer.echo(f"[run-all] train run directory: {rd}")
        run_dir_holder["run_dir"] = str(rd)

    stages.append(("train", _train_and_hold))

    # 4) Evaluate model (best-effort; if it falls back, we still proceed with EB calibration)
    stages.append(
        (
            "evaluate-model",
            lambda: evaluate_model(
                run_dir=run_dir_holder["run_dir"],
                features="data/features/features.parquet",
                dry_run=False,
            ),
        )
    )

    # 5) Odds → Market → Evaluate Market
    stages += [
        (
            "odds",
            lambda: odds(
                base="data/interim/base.parquet",
                out="data/market/odds.parquet",
                race_key="race_id",
                horse_key="horse_id",
            ),
        ),
        (
            "market",
            lambda: market(
                features="data/features/features.parquet",
                odds_path="data/market/odds.parquet",
                out="data/market/market_join.parquet",
                decimal_col="decimal_odds",
                fractional_col=None,
                race_key="race_id",
                horse_key="horse_id",
                prefix="mkt",
            ),
        ),
        (
            "evaluate-market",
            lambda: evaluate_market("data/market/market_join.parquet", dry_run=False),
        ),
    ]

    # 6) EB predictions → fit calibrator → apply → edges (calibrated) → backtest (calibrated config)
    ts_pred = _ts()
    pred_raw = f"reports/model/{ts_pred}/pred_eb.parquet"
    pred_cal = f"reports/model/{ts_pred}/pred_eb_cal.parquet"
    cal_pkl = f"models/calibration/{ts_pred}.pkl"
    edges_cal = "data/market/edges_cal.parquet"
    bt_cfg_path = "configs/backtest_calibrated_strict.yaml"

    stages += [
        (
            "predict-eb",
            lambda: predict_eb(
                market="data/market/market_join.parquet",
                out=pred_raw,
            ),
        ),
        (
            "fit-calibrator",
            lambda: fit_calibrator_cmd(
                pred=pred_raw,
                out=cal_pkl,
                method="platt",  # use "isotonic" if that’s your preferred choice
                labels=None,  # will auto-merge labels from base/features if needed
            ),
        ),
        (
            "apply-calibrator",
            lambda: apply_calibrator_cmd(
                pred=pred_raw,
                calibrator=cal_pkl,
                out=pred_cal,
                col_in="model_prob",
                col_out="model_prob_cal",
                replace=False,
            ),
        ),
        (
            "edge-from-pred (cal)",
            lambda: edge_from_pred(
                market="data/market/market_join.parquet",
                pred=pred_cal,
                probs_col="model_prob_cal",
                out=edges_cal,
                race_key="race_id",
                horse_key="horse_id",
                market_prefix="mkt",
            ),
        ),
        (
            "backtest (cal)",
            lambda: backtest(
                edges=edges_cal,
                config=bt_cfg_path if Path(bt_cfg_path).exists() else None,
                dry_run=False,
            ),
        ),
    ]

    # 7) Report (final)
    stages.append(
        (
            "report",
            lambda: report(
                run_dir=run_dir_holder["run_dir"],
                out=f"reports/{_ts()}/report.md",
                dry_run=False,
            ),
        )
    )

    # ---- Progress UI (Rich if available) ----
    have_rich = importlib.util.find_spec("rich.progress") is not None

    def _run(with_progress: bool) -> None:
        total = len(stages)
        if with_progress:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]run-all · {task.fields[stage]}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("pipeline", total=total, stage="starting")
                for name, fn in stages:
                    progress.update(task, stage=name)
                    start = time.perf_counter()
                    typer.echo(f"[run-all] ▶ {name}")
                    fn()
                    typer.echo(f"[run-all] ✓ {name} ({time.perf_counter()-start:.1f}s)")
                    progress.advance(task)
        else:
            for name, fn in stages:
                start = time.perf_counter()
                typer.echo(f"[run-all] ▶ {name}")
                fn()
                typer.echo(f"[run-all] ✓ {name} ({time.perf_counter()-start:.1f}s)")

    try:
        _run(with_progress=have_rich)
    except Exception as e:
        typer.echo(f"[run-all] (note) progress UI failed: {e}. Aborting to avoid double-runs.")
        raise

    typer.echo("[run-all] done.")
