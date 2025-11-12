# src/hqp/split/__init__.py
# -----------------------------------------------------------------------------
# Public surface for time-based, race-grouped splits.
# - create_time_splits: programmatic API (Single-cutoff / KFold)
# - SplitConfig: typed dict for configuration
# -----------------------------------------------------------------------------
from __future__ import annotations

from .time import SplitConfig, create_time_splits

__all__ = ["create_time_splits", "SplitConfig"]
