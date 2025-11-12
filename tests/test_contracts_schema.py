# tests/test_contracts_schema.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

import hqp.ingest.reader as ingest_reader  # module import keeps Pylance happy
from hqp.common.contracts import EntryKey, RaceKey, RaceRow
from hqp.common.schema import ensure_required_columns
from hqp.ingest.keys import check_unique_keys


def _tmp_file(tmp_path: Path, rel: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_contract_key_normalisation() -> None:
    rk = RaceKey(race_id="  abc123 ")
    assert rk.race_id == "ABC123"
    ek = EntryKey(race_id=" r1 ", horse_id=" h7 ")
    assert ek.race_id == "R1" and ek.horse_id == "H7"


def test_contract_row_coercion() -> None:
    row = RaceRow(
        race_id="r1",
        horse_id="h1",
        race_datetime=datetime(2024, 6, 1, 14, 30),
        course=" Ascot ",
        horse_name="  Sea The Stars ",
        draw=7,
        sp=5.5,
        position=1,
    )
    assert row.course == "Ascot"
    assert row.horse_name == "Sea The Stars"
    assert row.draw == 7
    assert isinstance(row.sp, float)


def test_schema_coercion_and_unique_keys(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1", "r2"],
            "horse_id": ["h1", "h1", "h2", "h3"],
            "race_datetime": ["2024-06-01 14:30"] * 3 + ["2024-06-02 15:00"],
            "course": ["Ascot", "Ascot", "Ascot", "York"],
            "horse_name": ["Sea", "Sea", "Gallop", "Runner"],
            "draw": ["7", "7", "", "3"],
            "going": ["good", "good", "Good to Firm", "SOFT"],
            "sp": ["5.5", "5.5", "12.0", ""],
        }
    )
    raw_csv = _tmp_file(tmp_path, "raw/raw.csv")
    data.to_csv(raw_csv, index=False)

    cfg: dict[str, object] = {
        "raw_csv_path": str(raw_csv),
        "date_columns": ["race_datetime"],
        "categorical_columns": ["going", "sex"],
        "output_parquet": str(_tmp_file(tmp_path, "data/canonical/base.parquet")),
        "required_columns": ["race_id", "horse_id", "race_datetime", "course", "horse_name"],
    }
    cfg_path = _tmp_file(tmp_path, "configs/data.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_path = ingest_reader.ingest_base(cfg_path)
    assert isinstance(out_path, Path) and out_path.exists()

    df = pd.read_parquet(str(out_path))

    # Keys normalized and unique
    assert df["race_id"].str.isupper().all()
    assert df["horse_id"].str.isupper().all()
    assert len(df) == 3  # one duplicate (r1,h1) removed

    assert df["going"].dtype.name == "category"
    assert str(df["draw"].dtype) in ("Int32", "int32")

    n_rows, n_unique = check_unique_keys(df)
    assert n_rows == n_unique


def test_schema_helpers_required_columns() -> None:
    df = pd.DataFrame({"race_id": ["R1"], "horse_id": ["H1"]})
    with pytest.raises(ValueError):
        ensure_required_columns(df, ["race_id", "horse_id", "race_datetime"])
