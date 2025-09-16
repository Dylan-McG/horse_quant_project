from __future__ import annotations

from typing import Any

import pytest


def test_all_configs_parse(loaded_configs: dict[str, dict[str, Any]]) -> None:
    assert loaded_configs, "No configs loaded."
    for name, cfg in loaded_configs.items():
        assert "version" in cfg, f"{name} missing 'version' key"


@pytest.mark.parametrize(
    "required_cfg",
    [
        "data.yaml",
        "split.yaml",
        "features.yaml",
        "ratings.yaml",
        "model_lgbm.yaml",
        "calibrate.yaml",
        "market.yaml",
        "policy.yaml",
    ],
)
def test_required_configs_present(required_cfg: str, loaded_configs: dict[str, dict]) -> None:
    assert required_cfg in loaded_configs, f"Missing {required_cfg}"
