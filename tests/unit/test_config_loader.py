from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import _deep_merge, _ScopedEnvUnset, load_settings, load_yaml


def test_load_yaml_valid(tmp_path: Path):
    p = tmp_path / "test.yaml"
    p.write_text("foo: bar\nnested:\n  key: value\n")
    data = load_yaml(p)
    assert data["foo"] == "bar"
    assert data["nested"]["key"] == "value"


def test_load_yaml_empty_file(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    data = load_yaml(p)
    assert data == {}


def test_load_yaml_nonexistent_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    overlay = {"b": 3, "c": 4}
    result = _deep_merge(base, overlay)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 10}
    overlay = {"a": {"y": 99, "z": 3}}
    result = _deep_merge(base, overlay)
    assert result["a"] == {"x": 1, "y": 99, "z": 3}
    assert result["b"] == 10


def test_deep_merge_overlay_replaces_non_dict():
    base = {"a": {"x": 1}}
    overlay = {"a": "replaced"}
    result = _deep_merge(base, overlay)
    assert result["a"] == "replaced"


def test_load_settings_with_default_config(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    default = {
        "mock_hardware": True,
        "platform": "mouse_droid",
    }
    (cfg_dir / "default.yaml").write_text(yaml.dump(default))
    settings = load_settings(config_dir=cfg_dir)
    assert settings.mock_hardware is True


def test_load_settings_with_overlay(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    default = {"mock_hardware": True, "debug": False}
    (cfg_dir / "default.yaml").write_text(yaml.dump(default))
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(yaml.dump({"debug": True}))
    settings = load_settings(overlay_path, config_dir=cfg_dir)
    assert settings.debug is True


def test_load_settings_secure_metrics_overlay(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    default = {
        "mock_hardware": True,
        "platform": "mouse_droid",
        "telemetry": {
            "enabled": True,
            "auth": {
                "auth_enabled": True,
                "token_env_var": "MOUSEDROID_TELEMETRY_TOKEN",
                "allowed_origins": [],
                "exempt_paths": ["/health", "/metrics", "/api/v1/health"],
            },
        },
    }
    overlay = {
        "telemetry": {
            "auth": {
                "exempt_paths": ["/health", "/api/v1/health"],
            }
        }
    }

    (cfg_dir / "default.yaml").write_text(yaml.dump(default))
    overlay_path = tmp_path / "jetson_secure_metrics.yaml"
    overlay_path.write_text(yaml.dump(overlay))

    settings = load_settings(overlay_path, config_dir=cfg_dir)

    assert settings.telemetry.auth.exempt_paths == ["/health", "/api/v1/health"]


def test_load_settings_nonexistent_overlay_raises(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text(yaml.dump({"mock_hardware": True}))
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "no_such_file.yaml", config_dir=cfg_dir)


def test_load_settings_no_default_yaml(tmp_path: Path):
    cfg_dir = tmp_path / "empty_config"
    cfg_dir.mkdir()
    settings = load_settings(config_dir=cfg_dir)
    assert settings.mock_hardware is True  # env var from conftest


# ---------------------------------------------------------------------------
# _ScopedEnvUnset tests
# ---------------------------------------------------------------------------


def test_scoped_env_unset_removes_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context manager drops named vars on enter and restores them on exit."""
    monkeypatch.setenv("MOUSEDROID_TEST_TMP_KEY", "some_value")
    assert "MOUSEDROID_TEST_TMP_KEY" in os.environ

    with _ScopedEnvUnset(["MOUSEDROID_TEST_TMP_KEY"]):
        assert "MOUSEDROID_TEST_TMP_KEY" not in os.environ

    assert os.environ.get("MOUSEDROID_TEST_TMP_KEY") == "some_value"


def test_scoped_env_unset_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var is restored even when the body raises."""
    monkeypatch.setenv("MOUSEDROID_TMP_RESTORE_VAR", "original")

    try:
        with _ScopedEnvUnset(["MOUSEDROID_TMP_RESTORE_VAR"]):
            raise RuntimeError("body error")
    except RuntimeError:
        pass

    assert os.environ.get("MOUSEDROID_TMP_RESTORE_VAR") == "original"


def test_scoped_env_unset_skips_absent_names() -> None:
    """No error when a listed var is not present in the environment."""
    with _ScopedEnvUnset(["MOUSEDROID_DOES_NOT_EXIST_ZZ9PLURALZ"]):
        pass  # must not raise


def test_scoped_env_unset_empty_list() -> None:
    """Empty name list is a no-op."""
    with _ScopedEnvUnset([]):
        pass  # must not raise


def test_load_settings_empty_nested_env_var_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty MOUSEDROID_SECTION__FIELD env vars are sanitized before Settings()."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text("mock_hardware: true\n")
    monkeypatch.setenv("MOUSEDROID_GCP__PROJECT_ID", "")

    settings = load_settings(config_dir=cfg_dir)

    assert settings.mock_hardware is True
