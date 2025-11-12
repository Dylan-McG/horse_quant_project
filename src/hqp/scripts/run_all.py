# scripts/run_all.py
from __future__ import annotations

import subprocess
from pathlib import Path

SEQ = [
    ["hqp", "ingest", "--config", "configs/ingest.yaml", "--out", "data/interim/base.parquet"],
    [
        "hqp",
        "validate",
        "--base",
        "data/interim/base.parquet",
        "--schema",
        "configs/schema.yaml",
        "--dry-run",
    ],
    [
        "hqp",
        "split",
        "--base",
        "data/interim/base.parquet",
        "--config",
        "configs/split.yaml",
        "--dry-run",
    ],
    [
        "hqp",
        "ratings",
        "--base",
        "data/interim/base.parquet",
        "--out",
        "data/ratings/ratings.parquet",
    ],
    [
        "hqp",
        "features",
        "--base",
        "data/interim/base.parquet",
        "--ratings",
        "data/ratings/ratings.parquet",
        "--out",
        "data/features/features.parquet",
    ],
    [
        "hqp",
        "train",
        "--features",
        "data/features/features.parquet",
        "--config",
        "configs/model_lgbm_tiny.yaml",
        "--artifacts-dir",
        "models/artifacts",
    ],
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(["poetry", "run", *cmd])


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    for cmd in SEQ:
        run(cmd)
    print("[run_all] sequence complete")


if __name__ == "__main__":
    main()
