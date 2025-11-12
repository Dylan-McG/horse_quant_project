# src/hqp/analysis/_io.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def ensure_run_dir(base_out: str | os.PathLike[str], qtag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(base_out) / qtag / ts
    (run / "plots").mkdir(parents=True, exist_ok=True)
    return run


def safe_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
