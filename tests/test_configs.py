# tests/test_configs.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Import the module (not the symbol) so Pylance resolves cleanly
import hqp.ingest.reader as ingest_reader


def test_config_missing_keys_raises(tmp_path: Path) -> None:
    bad_cfg_path = tmp_path / "bad.yaml"
    with bad_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"date_columns": ["race_datetime"]}, f)
    with pytest.raises(ValueError):
        ingest_reader.ingest_base(bad_cfg_path)


def test_config_minimal_parse_ok(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "race_id,horse_id,race_datetime,course,horse_name\nR1,H1,2024-01-01 12:00:00,Ascot,A\n",
        encoding="utf-8",
    )
    good_cfg = {
        "raw_csv_path": str(csv_path),
        "date_columns": ["race_datetime"],
        "categorical_columns": ["course"],
        "required_columns": ["race_id", "horse_id", "race_datetime", "course", "horse_name"],
        "output_parquet": str(tmp_path / "out.parquet"),
    }
    cfg_path = tmp_path / "good.yaml"
    cfg_path.write_text(yaml.safe_dump(good_cfg), encoding="utf-8")

    out = ingest_reader.ingest_base(cfg_path)
    assert isinstance(out, Path)
    assert out.exists()
