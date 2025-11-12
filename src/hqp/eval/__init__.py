# src/hqp/eval/__init__.py
# -----------------------------------------------------------------------------
# Horse Quant Project – Evaluation Package
#
# Purpose:
# - "Forward pass" utilities for trained models, producing per-runner win
#   probabilities and writing standardized prediction artifacts.
# - This package also hosts backtesting, calibration, and market utilities.
#
# Public surface (re-exported):
# - evaluate_model: run a stored model against a features parquet and write
#   reports/model/<ts>/predictions.parquet (with model_prob).
# - write_predictions_parquet: helper to persist predictions in the standard
#   format (for ad-hoc or custom scoring flows).
# -----------------------------------------------------------------------------
