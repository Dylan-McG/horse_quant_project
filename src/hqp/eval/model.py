# src/hqp/eval/model.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Model Forward Pass (Evaluation)
#
# Purpose
# -------
# Load a trained model artifact from models/artifacts/<run_dir>, build the
# design matrix using the *training* feature order, produce probabilities,
# and write a self-contained report folder:
#   reports/model/<timestamp>/{summary.json, predictions.parquet}
#
# Notes
# -----
# - We infer the training feature list from <run_dir>/features.json first
#   (preferred), then try model attributes (e.g., LightGBM feature_name_).
# - We do *not* force per-race normalization here; probabilities are emitted
#   as given by the model (downstream steps may normalize as needed).
# - Output predictions include model_prob and, if available in features,
#   race_id and horse_id for convenient merging.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Dict, List, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray  # explicit numpy array typing

# Optional joblib
try:
    import joblib  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_model_path(run_dir: Path) -> Path:
    """
    Search for a model file inside run_dir (or run_dir/artifacts) and return
    the most explicit match (model.pkl/joblib/sav) or the newest candidate.
    """
    candidates: List[Path] = []
    patterns: Sequence[str] = [
        "model.pkl",
        "model.joblib",
        "model.sav",
        "*.pkl",
        "*.joblib",
        "*.sav",
    ]
    for pat in patterns:
        candidates.extend(run_dir.glob(pat))
        candidates.extend((run_dir / "artifacts").glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No model artifact found under {run_dir}")
    preferred = [p for p in candidates if p.name in {"model.pkl", "model.joblib", "model.sav"}]
    if preferred:
        return preferred[0]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_model(path: Path) -> Any:
    """Load a model using joblib if available, else pickle."""
    if joblib is not None and path.suffix in {".joblib", ".pkl"}:
        try:
            return joblib.load(path)  # type: ignore[no-any-return]
        except Exception:
            pass
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logloss(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    y = y_true
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _safe_auc(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return None


def _accuracy(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    thr: float = 0.5,
) -> float:
    y_true_bin: NDArray[np.int_] = (y_true >= 0.5).astype(np.int_)
    y_pred_bin: NDArray[np.int_] = (y_pred >= thr).astype(np.int_)
    equal_mask = np.equal(y_pred_bin, y_true_bin)
    return float(np.mean(equal_mask, dtype=np.float64))


def _load_training_feature_list(run_dir: Path, model: Any) -> list[str]:
    """
    Determine the exact training feature order to reconstruct X:
      1) <run_dir>/features.json (preferred; written by training helpers)
      2) model.feature_name_ (e.g., LightGBM)
      3) model.booster.feature_name() (fallback for LGBM)
    """
    # 1) explicit file saved by training
    feats_json = run_dir / "features.json"
    if feats_json.exists():
        try:
            loaded_obj: object = json.loads(feats_json.read_text(encoding="utf-8"))
            if isinstance(loaded_obj, list):
                names_list = cast(List[Any], loaded_obj)
                return [str(c) for c in names_list]
        except Exception:
            pass

    # 2) model-provided names (LightGBM)
    for attr in ("feature_name_",):
        if hasattr(model, attr):
            try:
                raw_names: object = getattr(model, attr)
                if isinstance(raw_names, (list, tuple)):
                    names_seq = cast(Sequence[Any], raw_names)
                    return [str(c) for c in list(names_seq)]
            except Exception:
                pass

    # 3) booster feature names
    try:
        booster: Any = getattr(model, "booster_", None) or getattr(model, "booster", None)
        if booster is not None and hasattr(booster, "feature_name"):
            raw_booster_names: object = booster.feature_name()
            if isinstance(raw_booster_names, (list, tuple)):
                names_seq2 = cast(Sequence[Any], raw_booster_names)
                return [str(c) for c in list(names_seq2)]
    except Exception:
        pass

    # last resort: empty -> caller will error clearly
    return []


def _predict_proba(model: Any, X: pd.DataFrame) -> NDArray[np.float64]:
    """
    Best-effort probability prediction:
      - predict_proba[:, 1] when available
      - decision_function -> sigmoid
      - predict() already in [0,1] (clipped)
    """
    if hasattr(model, "predict_proba"):
        p_any: Any = model.predict_proba(X)  # type: ignore[attr-defined]
        p = np.asarray(p_any, dtype=float)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(np.float64, copy=False)
        if p.ndim == 1:
            return p.astype(np.float64, copy=False)
    if hasattr(model, "decision_function"):
        s_any: Any = model.decision_function(X)  # type: ignore[attr-defined]
        s = np.asarray(s_any, dtype=float).reshape(-1).astype(np.float64, copy=False)
        return _sigmoid(s)
    if hasattr(model, "predict"):
        y_any: Any = model.predict(X)  # type: ignore[attr-defined]
        yhat = np.asarray(y_any, dtype=float).reshape(-1).astype(np.float64, copy=False)
        return np.clip(yhat, 0.0, 1.0)
    raise AttributeError("Model has no predict_proba/decision_function/predict.")


def evaluate(run_dir: Path, features_path: Path) -> Path:
    """
    Evaluate a trained binary classifier on features.
    Writes reports/model/<timestamp>/{summary.json,predictions.parquet}.
    """
    run_dir = Path(run_dir)
    features_path = Path(features_path)
    out_dir = Path("reports") / "model" / _now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)

    # Identify label if present (optional metrics)
    label_col: Optional[str] = None
    for candidate in ("obs__is_winner", "won"):
        if candidate in df.columns:
            label_col = candidate
            break

    # Load model and training feature list
    model_path = _find_model_path(run_dir)
    model = _load_model(model_path)
    train_feats = _load_training_feature_list(run_dir, model)
    if not train_feats:
        raise RuntimeError(
            "Could not determine training feature list. "
            "Expected features.json or model feature names."
        )

    # Build X with exact training columns / order
    # - drop any extras
    # - fill missing with 0.0
    X = df.reindex(columns=train_feats, fill_value=0.0).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64").fillna(0.0)

    proba: NDArray[np.float64] = _predict_proba(model, X)

    # predictions
    out_keys: List[str] = [k for k in ("race_id", "horse_id") if k in df.columns]
    pred_df = pd.DataFrame({"model_prob": pd.Series(proba, index=df.index)})
    for k in out_keys:
        pred_df[k] = df[k].values

    metrics: Dict[str, Optional[float] | int | str] = {
        "logloss": None,
        "auc": None,
        "accuracy@0.5": None,
        "n_rows": int(len(df)),
        "n_features": int(len(train_feats)),
        "label_col": label_col if label_col is not None else "",
        "run_dir": str(run_dir),
        "features_path": str(features_path),
        "model_artifact": str(model_path),
    }

    # Optional metrics if label is in the features
    if label_col is not None:
        y_raw = df[label_col].to_numpy()
        try:
            yf = y_raw.astype(float)
            uniq_vals: np.ndarray = np.unique(yf).astype(np.float64, copy=False)
            uniq_set: set[float] = set(map(float, uniq_vals.tolist()))
            if uniq_set <= {0.0, 1.0}:
                y = yf.astype(np.float64)
                metrics["logloss"] = _safe_logloss(y, proba)
                metrics["auc"] = _safe_auc(y, proba)
                metrics["accuracy@0.5"] = _accuracy(y, proba, 0.5)
                pred_df["y_true"] = y.astype(int)
            else:
                # If labels are not {0,1}, still carry them for inspection
                pred_df["y_true"] = y_raw
        except Exception:
            pred_df["y_true"] = y_raw

    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[evaluate-model] wrote: {out_dir}")
    return out_dir
