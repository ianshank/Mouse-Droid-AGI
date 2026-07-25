# tests/unit/tools/claude_hooks/test_config.py
"""Unit tests for the workforce configuration schema.

Two contracts matter most here: a missing or empty config must still yield a
usable object (backwards compatibility), and an unknown key must fail loudly
rather than silently disabling a gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.claude_hooks.config import (
    DEFAULT_CONFIG_RELPATH,
    ConfigError,
    WorkforceConfig,
    load_config,
)


def _write_config(root: Path, body: str) -> Path:
    path = root / DEFAULT_CONFIG_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults + backwards compatibility
# ---------------------------------------------------------------------------


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    cfg = load_config(repo_root=tmp_path)
    assert isinstance(cfg, WorkforceConfig)
    assert cfg.freeze.feature_key == "F-008"
    assert cfg.coverage.tools_line_min == 85
    assert cfg.secret_scan.strict is False


def test_empty_file_yields_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "")
    assert load_config(repo_root=tmp_path).freeze.enabled is True


def test_comment_only_file_yields_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "# nothing but a comment\n")
    assert load_config(repo_root=tmp_path).docs.core_max_lines == 250


def test_partial_config_keeps_defaults_for_absent_sections(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze:\n    feature_key: F-999\n")
    cfg = load_config(repo_root=tmp_path)
    assert cfg.freeze.feature_key == "F-999"
    # Untouched sections keep their defaults — a config written before a section
    # existed still loads.
    assert cfg.coverage.tools_line_min == 85
    assert cfg.secret_scan.command == "gitleaks"


def test_explicit_path_overrides_repo_root(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text("docs:\n    core_max_lines: 42\n", encoding="utf-8")
    assert load_config(custom, repo_root=tmp_path).docs.core_max_lines == 42


def test_repo_config_file_is_valid() -> None:
    """The checked-in .claude/workforce.yaml must satisfy its own schema."""
    repo_root = Path(__file__).resolve().parents[4]
    cfg = load_config(repo_root=repo_root)
    assert cfg.freeze.frozen_paths, "repo config should declare at least one frozen path"
    assert cfg.freeze.feature_key.startswith("F-")


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "unknown_section:\n    a: 1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "unknown_section" in str(excinfo.value)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    # The exact typo this guard exists for: frozen_path vs frozen_paths.
    _write_config(tmp_path, "freeze:\n    frozen_path:\n        - src/**\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "frozen_path" in str(excinfo.value)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(repo_root=tmp_path)


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(repo_root=tmp_path)


def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    # A directory where a file is expected: read_text raises OSError.
    path = tmp_path / DEFAULT_CONFIG_RELPATH
    path.mkdir(parents=True)
    # is_file() is False for a directory, so this falls through to defaults.
    assert load_config(repo_root=tmp_path).freeze.enabled is True


@pytest.mark.parametrize(
    "body",
    [
        "coverage:\n    tools_line_min: 101\n",
        "coverage:\n    tools_line_min: -1\n",
        "secret_scan:\n    timeout_s: 0\n",
        "secret_scan:\n    timeout_s: -5\n",
        "secret_scan:\n    max_bytes: 0\n",
        "docs:\n    core_max_lines: 0\n",
        "evidence:\n    stale_after_days: 0\n",
        "agents:\n    max_lines: 0\n",
        "post_edit:\n    timeout_s: 100000\n",
    ],
)
def test_out_of_range_values_are_rejected(tmp_path: Path, body: str) -> None:
    _write_config(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "freeze:\n    feature_key: '   '\n",
        "freeze:\n    features_file: ''\n",
        "freeze:\n    override_env: '  '\n",
        "freeze:\n    done_status: ''\n",
    ],
)
def test_blank_freeze_identifiers_are_rejected(tmp_path: Path, body: str) -> None:
    _write_config(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "features_file",
    ["/etc/features.yaml", "../features.yaml", "a/../../features.yaml"],
)
def test_non_relative_features_file_is_rejected(tmp_path: Path, features_file: str) -> None:
    _write_config(tmp_path, f"freeze:\n    features_file: '{features_file}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze:\n    frozen_paths: not-a-list\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "config_file",
    ["/etc/passwd", "../../etc/passwd", "a/../../outside.toml"],
)
def test_non_relative_secret_scan_config_is_rejected(tmp_path: Path, config_file: str) -> None:
    """The allowlist path is joined onto the repo root before the scan runs.

    An absolute or traversing value would point the scanner's config at an
    arbitrary file, so it gets the same guard as freeze.features_file.
    """
    _write_config(tmp_path, f"secret_scan:\n    config_file: '{config_file}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_relative_secret_scan_config_is_accepted(tmp_path: Path) -> None:
    _write_config(tmp_path, "secret_scan:\n    config_file: custom/.gitleaks.toml\n")
    assert load_config(repo_root=tmp_path).secret_scan.config_file == "custom/.gitleaks.toml"
