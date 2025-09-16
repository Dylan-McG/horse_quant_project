from __future__ import annotations
from pathlib import Path
from typing import Any
import pytest
import yaml

@pytest.fixture(scope="session")
def project_root() -> Path:
    # conftest.py is now at: <repo>/tests/conftest.py
    # repo root is one parent up
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    return project_root / "configs"

@pytest.fixture(scope="session")
def config_files(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.yaml"))

@pytest.fixture(scope="session")
def loaded_configs(config_files: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in config_files:
        with p.open("r", encoding="utf-8") as fh:
            out[p.name] = yaml.safe_load(fh) or {}
    return out
