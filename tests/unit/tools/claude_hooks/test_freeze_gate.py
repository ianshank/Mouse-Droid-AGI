# tests/unit/tools/claude_hooks/test_freeze_gate.py
"""Unit tests for the capability freeze gate.

The split failure posture is the contract worth pinning: a *governance* failure
(missing/malformed catalog, absent feature) denies, because the gate cannot
prove the freeze lifted; an *environment* failure allows, because bricking every
edit in a session is worse than a missed gate.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from tools.claude_hooks import freeze_gate
from tools.claude_hooks.config import WorkforceConfig

_FROZEN_GLOB = "src/mousedroid/arm/**"
_FROZEN_FILE = "src/mousedroid/arm/driver.py"


def _config(**freeze_overrides: Any) -> WorkforceConfig:
    base = {"frozen_paths": [_FROZEN_GLOB]}
    base.update(freeze_overrides)
    return WorkforceConfig.model_validate({"freeze": base})


def _repo(tmp_path: Path, *, status: str | None = "todo", key: str = "F-008") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    if status is not None:
        (tmp_path / "features.yaml").write_text(
            f'features:\n  - id: "{key}"\n    status: "{status}"\n',
            encoding="utf-8",
        )
    return tmp_path


def _payload(path: str) -> dict[str, Any]:
    return {"tool_name": "Edit", "tool_input": {"file_path": path}}


def _evaluate(
    tmp_path: Path,
    payload: dict[str, Any],
    cfg: WorkforceConfig | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    return freeze_gate.evaluate(
        payload,
        cfg or _config(),
        repo_root=tmp_path,
        env=env or {},
    )


# ---------------------------------------------------------------------------
# Core gate behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["todo", "in_progress", "blocked", "deferred"])
def test_frozen_path_denied_while_feature_incomplete(tmp_path: Path, status: str) -> None:
    repo = _repo(tmp_path, status=status)
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "hardware readiness preempts" in reason
    assert _FROZEN_GLOB in reason
    assert status in reason


def test_gate_self_disables_when_feature_done(tmp_path: Path) -> None:
    # The whole point: no code change, no redeploy — just the catalog flipping.
    repo = _repo(tmp_path, status="done")
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is True
    assert reason == ""


def test_unfrozen_path_is_allowed(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    allowed, _ = _evaluate(repo, _payload(str(repo / "src/mousedroid/telemetry/server.py")))
    assert allowed is True


def test_similar_but_distinct_path_is_not_frozen(tmp_path: Path) -> None:
    # 'armour.py' must not be caught by the 'arm/**' glob.
    repo = _repo(tmp_path, status="todo")
    allowed, _ = _evaluate(repo, _payload(str(repo / "src/mousedroid/armour.py")))
    assert allowed is True


def test_disabled_gate_allows_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    cfg = _config(enabled=False)
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), cfg)[0] is True


def test_empty_frozen_paths_allows_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    cfg = WorkforceConfig()
    assert cfg.freeze.frozen_paths == []
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), cfg)[0] is True


def test_payload_without_target_is_allowed(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    assert _evaluate(repo, {"tool_name": "Bash", "tool_input": {}})[0] is True


def test_path_outside_repository_is_allowed(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo", status="todo")
    outside = tmp_path / "elsewhere" / "arm" / "driver.py"
    assert _evaluate(repo, _payload(str(outside)))[0] is True


# ---------------------------------------------------------------------------
# Governance failures fail closed
# ---------------------------------------------------------------------------


def test_missing_catalog_denies(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status=None)
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "not found" in reason


def test_malformed_catalog_denies(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    (repo / "features.yaml").write_text("features: [unclosed\n", encoding="utf-8")
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "not valid YAML" in reason


def test_catalog_without_features_list_denies(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    (repo / "features.yaml").write_text("features: {}\n", encoding="utf-8")
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "no 'features' list" in reason


def test_absent_feature_key_denies(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo", key="F-999")
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "not present in the catalog" in reason


def test_feature_without_status_denies(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status=None)
    (repo / "features.yaml").write_text('features:\n  - id: "F-008"\n', encoding="utf-8")
    allowed, reason = _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))
    assert allowed is False
    assert "no status field" in reason


def test_top_level_list_catalog_is_supported(tmp_path: Path) -> None:
    # Some catalogs are a bare list rather than a 'features:' mapping.
    repo = _repo(tmp_path, status=None)
    (repo / "features.yaml").write_text(
        '- id: "F-008"\n  status: "done"\n',
        encoding="utf-8",
    )
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))[0] is True


def test_malformed_entries_are_skipped(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status=None)
    (repo / "features.yaml").write_text(
        'features:\n  - "just a string"\n  - id: "F-008"\n    status: "done"\n',
        encoding="utf-8",
    )
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)))[0] is True


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


def test_override_env_allows_frozen_edit(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    env = {"MOUSEDROID_WORKFORCE_ALLOW_FROZEN": "1"}
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), env=env)[0] is True


def test_override_env_also_covers_broken_catalog(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status=None)
    env = {"MOUSEDROID_WORKFORCE_ALLOW_FROZEN": "yes"}
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), env=env)[0] is True


def test_blank_override_does_not_activate(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    env = {"MOUSEDROID_WORKFORCE_ALLOW_FROZEN": "   "}
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), env=env)[0] is False


def test_override_env_name_is_configurable(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    cfg = _config(override_env="CUSTOM_OVERRIDE")
    assert _evaluate(repo, _payload(str(repo / _FROZEN_FILE)), cfg, {"CUSTOM_OVERRIDE": "1"})[0]


# ---------------------------------------------------------------------------
# read_feature_status directly
# ---------------------------------------------------------------------------


def test_read_feature_status_returns_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="in_progress")
    status, problem = freeze_gate.read_feature_status(repo / "features.yaml", "F-008")
    assert status == "in_progress"
    assert problem is None


def test_read_feature_status_unreadable_path(tmp_path: Path) -> None:
    target = tmp_path / "catalog"
    target.mkdir()  # a directory is not a file
    status, problem = freeze_gate.read_feature_status(target, "F-008")
    assert status is None
    assert problem is not None


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


def test_main_denies_and_writes_decision_json(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text(
        f"freeze:\n    frozen_paths:\n        - {_FROZEN_GLOB}\n",
        encoding="utf-8",
    )
    stdout = io.StringIO()
    code = freeze_gate.main(
        stdin=io.StringIO(json.dumps(_payload(str(repo / _FROZEN_FILE)))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(repo)},
    )
    assert code == 0
    decision = json.loads(stdout.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"


def test_main_allows_silently(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="done")
    stdout = io.StringIO()
    code = freeze_gate.main(
        stdin=io.StringIO(json.dumps(_payload(str(repo / _FROZEN_FILE)))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(repo)},
    )
    assert code == 0
    assert stdout.getvalue() == ""


def test_main_denies_on_invalid_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="done")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text("bogus_key: 1\n", encoding="utf-8")
    stdout = io.StringIO()
    freeze_gate.main(
        stdin=io.StringIO(json.dumps(_payload(str(repo / _FROZEN_FILE)))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(repo)},
    )
    decision = json.loads(stdout.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "configuration" in decision["permissionDecisionReason"]


def test_main_allows_on_environment_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An environment failure must not brick every edit in the session.
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated environment failure")

    monkeypatch.setattr(freeze_gate, "resolve_repo_root", _boom)
    stdout = io.StringIO()
    code = freeze_gate.main(
        stdin=io.StringIO(json.dumps(_payload("src/mousedroid/arm/x.py"))),
        stdout=stdout,
        env={},
    )
    assert code == 0
    assert stdout.getvalue() == ""


def test_main_tolerates_empty_stdin(tmp_path: Path) -> None:
    repo = _repo(tmp_path, status="todo")
    stdout = io.StringIO()
    assert (
        freeze_gate.main(
            stdin=io.StringIO(""),
            stdout=stdout,
            env={"CLAUDE_PROJECT_DIR": str(repo)},
        )
        == 0
    )
    assert stdout.getvalue() == ""
