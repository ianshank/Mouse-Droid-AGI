"""Sub-second F-025 sanity: imports, Literal YAML round-trip, load rejects.

Mirrors the PR #104 sanity shape. The YAML round-trip is load-bearing for
``Literal`` fields — a selector silently dropped on ``model_dump`` →
``yaml.safe_dump`` → ``model_validate`` would resurrect the legacy default
on the next config rewrite.
"""

from __future__ import annotations

import importlib

import pytest
import yaml

from mousedroid.config.schema import ESP32Config


@pytest.mark.parametrize(
    "module_name",
    [
        "mousedroid.comms.command_set",
        "mousedroid.comms.base_driver",
        "mousedroid.comms.serial_driver",
        "mousedroid.comms.wifi_driver",
    ],
)
def test_f025_modules_import_clean(module_name: str) -> None:
    """Every F-025-touched comms module imports without side effects."""
    assert importlib.import_module(module_name) is not None


def test_command_set_round_trips_through_yaml() -> None:
    """model_dump → yaml → model_validate preserves the Literal selector."""
    cfg = ESP32Config(command_set="waveshare_stock")
    dumped = yaml.safe_dump(cfg.model_dump(mode="json"))
    restored = ESP32Config.model_validate(yaml.safe_load(dumped))
    assert restored.command_set == "waveshare_stock"
    assert restored.serial_baud == cfg.serial_baud


def test_invalid_command_set_rejected() -> None:
    """A typo'd selector fails loudly at parse time, never at runtime."""
    with pytest.raises(ValueError, match="command_set"):
        ESP32Config.model_validate({"command_set": "waveshare"})


def test_stock_wifi_rejection_survives_yaml_path() -> None:
    """The stock+wifi ValueError fires through the YAML-shaped load too."""
    with pytest.raises(ValueError, match="/cmd"):
        ESP32Config.model_validate(yaml.safe_load("command_set: waveshare_stock\nprotocol: wifi\n"))
