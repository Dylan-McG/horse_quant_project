# src/hqp/eval/calibrators.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Probability Calibration Helpers
#
# What this module provides
# -------------------------
# - Thin wrappers around well-tested scikit-learn models for probability
#   calibration: Platt scaling (logistic on p) and Isotonic regression.
# - A unified Calibrator class with .apply(), plus save/load helpers.
# - Convenience function to fit directly from a predictions parquet.
#
# Design choices
# --------------
# - All probability arrays are clipped to (eps, 1-eps) for numerical safety.
# - The public API avoids exotic dependencies and keeps types explicit.
# - We don't enforce race-wise normalization here; calibration operates on
#   *marginal* probabilities (per row) and relies on upstream code to make
#   sure probabilities are sensible for the task at hand.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pickle

# We use scikit-learn for robust, well-tested implementations.
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Public API exported from this module
__all__ = [
    "CalibMethod",
    "PlattParams",
    "IsotonicParams",
    "CalibParams",
    "Calibrator",
    "fit_platt",
    "apply_platt",
    "fit_isotonic",
    "apply_isotonic",
    "save_calibrator",
    "load_calibrator",
    "evaluate_brier",
    "fit_from_predictions_parquet",
]

CalibMethod = Literal["none", "platt", "isotonic"]


@dataclass
class PlattParams:
    """Platt scaling parameters backed by a LogisticRegression on p (prob)."""

    model: LogisticRegression


@dataclass
class IsotonicParams:
    """Isotonic regression model over predicted probabilities."""

    model: IsotonicRegression


CalibParams = Union[PlattParams, IsotonicParams]


def _clip_proba(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically safe clamp to (eps, 1-eps)."""
    return np.clip(p.astype(float), eps, 1 - eps)


@dataclass
class Calibrator:
    """
    Thin wrapper over Platt/Isotonic fit that presents a common .apply() API.
    """

    method: CalibMethod
    params: Optional[CalibParams] = None
    # metadata (optional): versioning, where it was fitted, etc.
    meta: Dict[str, Any] | None = None

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Apply the learned calibration mapping to raw probabilities."""
        probs = _clip_proba(np.asarray(probs))

        # No-op
        if self.method == "none" or self.params is None:
            return probs

        if self.method == "platt":
            # type narrowing for Pylance
            assert isinstance(self.params, PlattParams)
            # LR expects 2D (n, 1)
            out = self.params.model.predict_proba(probs.reshape(-1, 1))[:, 1]
            return _clip_proba(out)

        if self.method == "isotonic":
            assert isinstance(self.params, IsotonicParams)
            out = self.params.model.predict(probs)
            return _clip_proba(out)

        # Defensive default (shouldn't be hit)
        return probs


# ---------------------------- individual fits -------------------------------


def fit_platt(probs: np.ndarray, y: np.ndarray, C: float = 1.0) -> PlattParams:
    """
    Fit Platt scaling via logistic regression mapping p->y.
    Input:
      probs in [0,1], y in {0,1}
    """
    X = _clip_proba(np.asarray(probs)).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    lr = LogisticRegression(C=C, solver="lbfgs", penalty="l2", max_iter=1000)
    lr.fit(X, y)
    return PlattParams(model=lr)


def apply_platt(probs: np.ndarray, params: PlattParams) -> np.ndarray:
    return Calibrator(method="platt", params=params).apply(probs)


def fit_isotonic(
    probs: np.ndarray, y: np.ndarray, y_min: float = 0.0, y_max: float = 1.0
) -> IsotonicParams:
    """
    Fit isotonic regression mapping p->y (monotone, piecewise constant).
    """
    iso = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")
    iso.fit(_clip_proba(np.asarray(probs)), np.asarray(y).astype(float))
    return IsotonicParams(model=iso)


def apply_isotonic(probs: np.ndarray, params: IsotonicParams) -> np.ndarray:
    return Calibrator(method="isotonic", params=params).apply(probs)


# ------------------------------- persistence --------------------------------


def save_calibrator(cal: Calibrator, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_calibrator(path: Union[str, Path]) -> Calibrator:
    with Path(path).open("rb") as f:
        cal: Calibrator = pickle.load(f)
    return cal


# --------------------------------- metrics ----------------------------------


def evaluate_brier(before: np.ndarray, after: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Utility to quantify calibration effect using Brier score.
    Returns a dict with before/after and absolute improvement.
    """
    before = _clip_proba(np.asarray(before))
    after = _clip_proba(np.asarray(after))
    y = np.asarray(y).astype(int)
    b_before = float(brier_score_loss(y, before))
    b_after = float(brier_score_loss(y, after))
    return {
        "brier_before": b_before,
        "brier_after": b_after,
        "improvement": b_before - b_after,
    }


# ----------------------- parquet-driven convenience fit ---------------------


def fit_from_predictions_parquet(
    pred_path: Union[str, Path],
    *,
    method: CalibMethod,
    labels_path: Optional[Union[str, Path]] = None,
) -> Tuple[Calibrator, pd.DataFrame]:
    """
    Load predictions parquet; ensure it has columns: model_prob, race_id, horse_id, and a 'won' label.
    If 'won' is missing, try to merge from labels_path.

    Returns
    -------
    (Calibrator, DataFrame)
        The fitted calibrator and a 2-column DataFrame ['model_prob', 'won'] used in fitting.
    """
    df = pd.read_parquet(pred_path)
    if "model_prob" not in df.columns:
        raise ValueError("predictions parquet must contain 'model_prob' column")

    if "won" not in df.columns:
        if labels_path is None:
            raise ValueError("No 'won' in predictions and no labels_path provided.")
        lbl = pd.read_parquet(labels_path)
        label_col = next(
            (
                c
                for c in ["won", "obs__is_winner", "is_winner", "label", "target"]
                if c in lbl.columns
            ),
            None,
        )
        if label_col is None:
            raise ValueError(
                "labels parquet must contain one of: won, obs__is_winner, is_winner, label, target"
            )
        for k in ["race_id", "horse_id"]:
            if k not in df.columns or k not in lbl.columns:
                raise ValueError(f"Both predictions and labels must have '{k}' to merge")
        # cast ids to string for safer join
        for k in ["race_id", "horse_id"]:
            df[k] = df[k].astype("string")
            lbl[k] = lbl[k].astype("string")
        df = df.merge(
            lbl[["race_id", "horse_id", label_col]], on=["race_id", "horse_id"], how="left"
        )
        df["won"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_prob", "won"]).copy()
    p = _clip_proba(df["model_prob"].to_numpy())
    y = df["won"].to_numpy().astype(int)

    if method == "platt":
        params = fit_platt(p, y)
        cal = Calibrator(method="platt", params=params, meta={"source": str(pred_path)})
    elif method == "isotonic":
        params = fit_isotonic(p, y)
        cal = Calibrator(method="isotonic", params=params, meta={"source": str(pred_path)})
    elif method == "none":
        cal = Calibrator(method="none", params=None, meta={"source": str(pred_path)})
    else:
        raise ValueError(f"Unknown method: {method}")

    return cal, df[["model_prob", "won"]].copy()
