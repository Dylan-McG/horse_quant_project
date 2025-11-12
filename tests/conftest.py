# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Ensure src/ is on sys.path for test runtime
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root (one level above tests/)."""
    return ROOT


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Path to configs/ directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def config_files(config_dir: Path) -> list[Path]:
    """All YAML files in configs/."""
    return sorted(config_dir.glob("*.yaml"))


@pytest.fixture(scope="session")
def loaded_configs(config_files: list[Path]) -> dict[str, dict[str, Any]]:
    """Loaded YAML content keyed by filename."""
    out: dict[str, dict[str, Any]] = {}
    for p in config_files:
        with p.open("r", encoding="utf-8") as fh:
            out[p.name] = yaml.safe_load(fh) or {}
    return out
