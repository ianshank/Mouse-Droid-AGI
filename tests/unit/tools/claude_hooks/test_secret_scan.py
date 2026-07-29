# tests/unit/tools/claude_hooks/test_secret_scan.py
"""Unit tests for the edit-time secret scan.

The scanner binary is not assumed to exist: a stub executable is synthesised per
test so both the "clean" and "leak found" paths are exercised deterministically,
and the genuinely-absent path is exercised by pointing at a missing command.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from tools.claude_hooks import secret_scan
from tools.claude_hooks.config import WorkforceConfig

_WIN32_REASON = "edit-time secret scan hook tests are POSIX-only"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason=_WIN32_REASON)


def _config(**overrides: Any) -> WorkforceConfig:
    return WorkforceConfig.model_validate({"secret_scan": overrides})


def _payload(content: str, path: str = "src/a.py") -> dict[str, Any]:
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


def _stub_scanner(tmp_path: Path, *, exit_code: int, message: str = "") -> Path:
    """Create an executable stub standing in for the scanner binary."""
    script = tmp_path / "bin" / "stub-scanner"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"#!{sys.executable}\nimport sys\nprint({message!r})\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def scanner_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Put a stub-scanner directory on PATH and return its parent."""
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return bindir


# ---------------------------------------------------------------------------
# scan_content
# ---------------------------------------------------------------------------


def test_clean_content_passes(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=0)
    outcome = secret_scan.scan_content("x = 1", _config(command="stub-scanner"), repo_root=tmp_path)
    assert outcome.available is True
    assert outcome.clean is True


def test_leak_finding_is_reported(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=1, message="rule: generic-api-key")
    outcome = secret_scan.scan_content(
        "token = 'x'", _config(command="stub-scanner"), repo_root=tmp_path
    )
    assert outcome.available is True
    assert outcome.clean is False
    assert "generic-api-key" in outcome.detail


def test_missing_scanner_is_unavailable_not_dirty(tmp_path: Path) -> None:
    outcome = secret_scan.scan_content(
        "x = 1", _config(command="definitely-not-installed-xyz"), repo_root=tmp_path
    )
    assert outcome.available is False
    assert outcome.clean is True
    assert "not found on PATH" in outcome.detail


def test_unexpected_exit_code_is_unavailable(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=42)
    outcome = secret_scan.scan_content("x = 1", _config(command="stub-scanner"), repo_root=tmp_path)
    assert outcome.available is False
    assert "exited 42" in outcome.detail


def test_timeout_is_unavailable(tmp_path: Path, scanner_path: Path) -> None:
    slow = tmp_path / "bin" / "slow-scanner"
    slow.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(5)\n",
        encoding="utf-8",
    )
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    outcome = secret_scan.scan_content(
        "x = 1",
        _config(command="slow-scanner", timeout_s=0.25),
        repo_root=tmp_path,
    )
    assert outcome.available is False
    assert "timed out" in outcome.detail


def test_allowlist_config_is_passed_when_present(tmp_path: Path, scanner_path: Path) -> None:
    # The stub echoes its argv so we can assert --config was forwarded.
    script = tmp_path / "bin" / "argv-scanner"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nprint(' '.join(sys.argv))\nsys.exit(1)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    (tmp_path / ".gitleaks.toml").write_text("[extend]\n", encoding="utf-8")
    outcome = secret_scan.scan_content("x", _config(command="argv-scanner"), repo_root=tmp_path)
    assert "--config" in outcome.detail
    assert ".gitleaks.toml" in outcome.detail


def test_extra_args_are_forwarded(tmp_path: Path, scanner_path: Path) -> None:
    script = tmp_path / "bin" / "argv2-scanner"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nprint(' '.join(sys.argv))\nsys.exit(1)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    outcome = secret_scan.scan_content(
        "x",
        _config(command="argv2-scanner", extra_args=["--verbose"]),
        repo_root=tmp_path,
    )
    assert "--verbose" in outcome.detail


@pytest.mark.parametrize(
    ("target", "expected_suffix"),
    [("src/a.py", ".py"), ("notes.md", ".md"), (None, ".txt"), ("noext", ".txt")],
)
def test_temp_file_suffix_mirrors_target(
    tmp_path: Path, scanner_path: Path, target: str | None, expected_suffix: str
) -> None:
    script = tmp_path / "bin" / "suffix-scanner"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nprint(' '.join(sys.argv))\nsys.exit(1)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    outcome = secret_scan.scan_content(
        "x", _config(command="suffix-scanner"), repo_root=tmp_path, target=target
    )
    assert f"pending{expected_suffix}" in outcome.detail


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_finding_denies_with_actionable_reason(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=1, message="rule: aws-key")
    allowed, reason = secret_scan.evaluate(
        _payload("key = 'AKIA...'"), _config(command="stub-scanner"), repo_root=tmp_path
    )
    assert allowed is False
    assert "regex only — never allowlist by path" in reason


def test_clean_content_allows(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=0)
    allowed, _ = secret_scan.evaluate(
        _payload("x = 1"), _config(command="stub-scanner"), repo_root=tmp_path
    )
    assert allowed is True


def test_missing_scanner_warns_and_allows_by_default(tmp_path: Path) -> None:
    allowed, reason = secret_scan.evaluate(
        _payload("x = 1"), _config(command="absent-xyz"), repo_root=tmp_path
    )
    assert allowed is True
    assert reason == ""


def test_missing_scanner_denies_in_strict_mode(tmp_path: Path) -> None:
    allowed, reason = secret_scan.evaluate(
        _payload("x = 1"),
        _config(command="absent-xyz", strict=True),
        repo_root=tmp_path,
    )
    assert allowed is False
    assert "strict mode is enabled" in reason


def test_disabled_scanner_allows(tmp_path: Path) -> None:
    allowed, _ = secret_scan.evaluate(
        _payload("anything"), _config(enabled=False), repo_root=tmp_path
    )
    assert allowed is True


def test_payload_without_content_allows(tmp_path: Path) -> None:
    payload = {"tool_name": "Read", "tool_input": {"file_path": "a.py"}}
    assert secret_scan.evaluate(payload, _config(), repo_root=tmp_path)[0] is True


def test_oversized_content_is_skipped(tmp_path: Path, scanner_path: Path) -> None:
    _stub_scanner(tmp_path, exit_code=1, message="would have found something")
    allowed, _ = secret_scan.evaluate(
        _payload("x" * 5000),
        _config(command="stub-scanner", max_bytes=100),
        repo_root=tmp_path,
    )
    assert allowed is True


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


def test_main_denies_on_finding(tmp_path: Path, scanner_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _stub_scanner(tmp_path, exit_code=1, message="rule: token")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "workforce.yaml").write_text(
        "secret_scan:\n    command: stub-scanner\n", encoding="utf-8"
    )
    stdout = io.StringIO()
    code = secret_scan.main(
        stdin=io.StringIO(json.dumps(_payload("secret"))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    assert code == 0
    decision = json.loads(stdout.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"


def test_main_allows_silently_when_clean(tmp_path: Path, scanner_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _stub_scanner(tmp_path, exit_code=0)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "workforce.yaml").write_text(
        "secret_scan:\n    command: stub-scanner\n", encoding="utf-8"
    )
    stdout = io.StringIO()
    code = secret_scan.main(
        stdin=io.StringIO(json.dumps(_payload("x = 1"))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    assert code == 0
    assert stdout.getvalue() == ""


def test_main_denies_on_invalid_config(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "workforce.yaml").write_text("nope: 1\n", encoding="utf-8")
    stdout = io.StringIO()
    secret_scan.main(
        stdin=io.StringIO(json.dumps(_payload("x"))),
        stdout=stdout,
        env={"CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    decision = json.loads(stdout.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"


def test_main_allows_on_environment_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated environment failure")

    monkeypatch.setattr(secret_scan, "resolve_repo_root", _boom)
    stdout = io.StringIO()
    code = secret_scan.main(stdin=io.StringIO(json.dumps(_payload("x"))), stdout=stdout, env={})
    assert code == 0
    assert stdout.getvalue() == ""


def test_unpaired_surrogates_do_not_crash_the_scan(tmp_path: Path, scanner_path: Path) -> None:
    """A tool buffer can carry unpaired surrogates.

    A strict encode would raise out of the hook, which reads as a crashed hook
    and silently skips the scan for that edit — the gate must stay live.
    """
    _stub_scanner(tmp_path, exit_code=0)
    payload = _payload("lone surrogate: \ud800 tail")
    allowed, reason = secret_scan.evaluate(
        payload, _config(command="stub-scanner"), repo_root=tmp_path
    )
    assert allowed is True
    assert reason == ""


def test_surrogate_content_still_reports_a_finding(tmp_path: Path, scanner_path: Path) -> None:
    """Robust encoding must not turn a real finding into a false negative."""
    _stub_scanner(tmp_path, exit_code=1, message="rule: generic-api-key")
    allowed, _ = secret_scan.evaluate(
        _payload("key = 'x' \ud800"), _config(command="stub-scanner"), repo_root=tmp_path
    )
    assert allowed is False
