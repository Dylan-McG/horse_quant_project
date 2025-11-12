# tests/cli/test_cli_train_smoke.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from hqp.cli import app

runner = CliRunner()

MINI_MODEL_YAML = """\
version: 0
model:
  type: lgbm_classifier
  params:
    n_estimators: 50
    learning_rate: 0.1
    num_leaves: 31
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    random_state: 42
training:
  group_key: race_id
  time_key: off_time
  target: is_win
  n_splits: 3
  per_race_normalize: true
  class_weight: null
"""


def latest_run_dir(root: Path) -> Path:
    arts = sorted((root / "models" / "artifacts").glob("*"))
    assert arts, "no artifacts found"
    return arts[-1]


def test_cli_ratings_features_train_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir(parents=True, exist_ok=True)
    (Path("configs") / "ingest.yaml").write_text("{}", encoding="utf-8")
    (Path("configs") / "schema.yaml").write_text("{}", encoding="utf-8")
    (Path("configs") / "split.yaml").write_text("{}", encoding="utf-8")
    (Path("configs") / "model_lgbm_tiny.yaml").write_text(MINI_MODEL_YAML, encoding="utf-8")

    # ingest (dry-run tiny)
    assert runner.invoke(app, ["ingest", "--dry-run"]).exit_code == 0
    assert Path("data/interim/base.parquet").exists()

    # ratings -> features
    assert runner.invoke(app, ["ratings"]).exit_code == 0
    assert Path("data/ratings/ratings.parquet").exists()
    assert runner.invoke(app, ["features"]).exit_code == 0
    assert Path("data/features/features.parquet").exists()

    # train
    res = runner.invoke(
        app,
        [
            "train",
            "--features",
            "data/features/features.parquet",
            "--config",
            "configs/model_lgbm_tiny.yaml",
            "--artifacts-dir",
            "models/artifacts",
        ],
    )
    assert res.exit_code == 0

    run_dir = latest_run_dir(Path("."))
    metrics = run_dir / "metrics.json"
    assert metrics.exists()
    # sanity keys (allow either known keys or at least non-empty content)
    content: Dict[str, Any] = json.loads(metrics.read_text(encoding="utf-8"))
    assert isinstance(content, dict) and len(content) > 0
