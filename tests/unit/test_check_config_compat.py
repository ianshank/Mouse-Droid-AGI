"""Unit tests for ``scripts/check_config_compat.py``.

Avoids the heavyweight integration path (real git worktree + subprocess
schema load) by importing the module and exercising its pure helpers.
The worktree/validate paths are covered by their own integration tests
in CI when the workflow runs against a real PR.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_config_compat.py"
_spec = importlib.util.spec_from_file_location("check_config_compat", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check_config_compat = importlib.util.module_from_spec(_spec)
# dataclass(frozen=True) inside the script reads sys.modules during
# decoration; register the module before exec to avoid AttributeError.
sys.modules["check_config_compat"] = check_config_compat
_spec.loader.exec_module(check_config_compat)


def test_load_deployment_returns_record(tmp_path: Path) -> None:
    """Well-formed deployment JSON parses into a DeployedImage."""
    (tmp_path / "jetson-image.json").write_text(
        json.dumps(
            {
                "sha": "abc123",
                "platform": "jetson",
                "image_tag": "mousedroid:jetson",
            },
        ),
        encoding="utf-8",
    )
    record = check_config_compat.load_deployment(tmp_path, "jetson")
    assert record.sha == "abc123"
    assert record.platform == "jetson"
    assert record.image_tag == "mousedroid:jetson"


def test_load_deployment_missing_file_exits_two(tmp_path: Path) -> None:
    """Missing platform file → exit 2 (invocation error)."""
    with pytest.raises(SystemExit) as exc_info:
        check_config_compat.load_deployment(tmp_path, "nonexistent")
    assert exc_info.value.code == 2


def test_load_deployment_malformed_json_exits_two(tmp_path: Path) -> None:
    """Invalid JSON → exit 2 (invocation error)."""
    (tmp_path / "jetson-image.json").write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        check_config_compat.load_deployment(tmp_path, "jetson")
    assert exc_info.value.code == 2


def test_load_deployment_missing_required_key_exits_two(tmp_path: Path) -> None:
    """Missing required key (``sha``) → exit 2 with a list of missing keys."""
    (tmp_path / "jetson-image.json").write_text(
        json.dumps({"platform": "jetson", "image_tag": "mousedroid:jetson"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        check_config_compat.load_deployment(tmp_path, "jetson")
    assert exc_info.value.code == 2


def test_changed_yaml_files_filters_explicit_list(tmp_path: Path) -> None:
    """Explicit ``paths`` list is filtered to YAML under ``config/``."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_in = config_dir / "default.yaml"
    yaml_in.write_text("platform: mouse_droid\n", encoding="utf-8")
    not_config = tmp_path / "stray.yaml"
    not_config.write_text("foo: bar\n", encoding="utf-8")
    not_yaml = config_dir / "README.md"
    not_yaml.write_text("# notes\n", encoding="utf-8")

    result = check_config_compat.changed_yaml_files(
        base_ref="unused",
        paths=[yaml_in, not_config, not_yaml],
    )
    assert result == [yaml_in]


def test_changed_yaml_files_missing_path_dropped(tmp_path: Path) -> None:
    """Paths that don't exist on disk are silently dropped (renames, deletes)."""
    missing = tmp_path / "config" / "ghost.yaml"
    result = check_config_compat.changed_yaml_files(base_ref="unused", paths=[missing])
    assert result == []


def test_real_deployment_file_loads(tmp_path: Path) -> None:
    """The actual ``deployments/jetson-image.json`` shipped in this PR loads."""
    real = _SCRIPT.parent.parent / "deployments"
    record = check_config_compat.load_deployment(real, "jetson")
    assert len(record.sha) >= 7
    assert record.platform == "jetson"
    assert record.image_tag.startswith("mousedroid:")


def test_deployment_required_keys_constant_includes_sha() -> None:
    """``sha`` is in the required-keys contract — gate breaks if dropped."""
    assert "sha" in check_config_compat.REQUIRED_KEYS
    assert "platform" in check_config_compat.REQUIRED_KEYS
    assert "image_tag" in check_config_compat.REQUIRED_KEYS


# ---------------------------------------------------------------------------
# _validation_env — the subprocess environment for schema validation.
# Regression: replacing the whole env (only PYTHONPATH/PATH/MOCK) breaks the
# interpreter on platforms that need base vars (Windows SYSTEMROOT etc.),
# causing spurious "No module named yaml" failures. It must INHERIT the base
# env, STRIP MOUSEDROID_* (so host overrides don't pollute the file-vs-schema
# check), PIN PYTHONPATH to the deployed worktree's src, and force mock hardware.
# ---------------------------------------------------------------------------
def test_validation_env_inherits_base_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base (non-MOUSEDROID) env vars are inherited so the interpreter works."""
    monkeypatch.setenv("CONFIG_COMPAT_BASE_PROBE", "present")
    env = check_config_compat._validation_env(tmp_path)
    assert env.get("CONFIG_COMPAT_BASE_PROBE") == "present"


def test_validation_env_strips_mousedroid_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host MOUSEDROID_* overrides must not leak into the schema check."""
    monkeypatch.setenv("MOUSEDROID_LLM__ENABLED", "false")
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_TOKEN", "secret")
    env = check_config_compat._validation_env(tmp_path)
    assert not any(k.startswith("MOUSEDROID_") and k != "MOUSEDROID_MOCK_HARDWARE" for k in env)


def test_validation_env_pins_pythonpath_to_worktree_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PYTHONPATH is pinned to the deployed worktree's src, overriding any host value."""
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    env = check_config_compat._validation_env(tmp_path)
    assert env["PYTHONPATH"] == str((tmp_path / "src").resolve())


def test_validation_env_forces_mock_hardware(tmp_path: Path) -> None:
    """Schema load must not init real hardware."""
    env = check_config_compat._validation_env(tmp_path)
    assert env["MOUSEDROID_MOCK_HARDWARE"] == "true"
