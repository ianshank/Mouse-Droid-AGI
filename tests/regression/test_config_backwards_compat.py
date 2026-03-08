from __future__ import annotations

import yaml

from mousedroid.config.schema import Settings


def test_settings_no_args_works() -> None:
    s = Settings(mock_hardware=True)
    assert s.platform.value == "mouse_droid"


def test_settings_defaults_populated() -> None:
    s = Settings(mock_hardware=True)
    assert s.loop is not None
    assert s.model is not None
    assert s.mcts is not None
    assert s.safety is not None
    assert s.esp32 is not None
    assert s.camera is not None


def test_old_style_minimal_yaml() -> None:
    minimal_yaml = """
    mock_hardware: true
    platform: mouse_droid
    """
    data = yaml.safe_load(minimal_yaml)
    s = Settings(**data)
    assert s.mock_hardware is True
    assert s.platform.value == "mouse_droid"


def test_new_fields_have_defaults() -> None:
    s = Settings(mock_hardware=True)
    assert s.memory is not None
    assert s.learning is not None
    assert s.reward is not None
    assert s.curiosity is not None
    assert s.circuit_breaker is not None
    assert s.metrics is not None
    assert s.health is not None
    assert s.retry is not None


def test_debug_default_false() -> None:
    s = Settings(mock_hardware=True)
    assert s.debug is False
