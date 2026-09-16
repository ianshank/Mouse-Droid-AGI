# tests/unit/tools/claude_hooks/test_post_edit_check.py
"""Unit tests for the advisory post-edit checks.

Contract: this hook always exits 0. ``PostToolUse`` fires after the write, so a
non-zero exit could not undo anything — it would only add noise.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from tools.claude_hooks import post_edit_check
from tools.claude_hooks.config import WorkforceConfig


def _config(**overrides: Any) -> WorkforceConfig:
    return WorkforceConfig.model_validate({"post_edit": overrides})


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def _stub_checker(tmp_path: Path, name: str, *, exit_code: int, message: str = "") -> Path:
    """Create a stub checker script and route `name` to it via _checker_base_argv."""
    script = tmp_path / "stubs" / f"{name}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"import sys\nprint({message!r})\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )
    _STUBS[name] = [sys.executable, str(script)]
    return script


#: name -> base argv, populated by _stub_checker and consulted by the fixture.
_STUBS: dict[str, list[str]] = {}


@pytest.fixture
def stub_bin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route checker resolution to per-test stubs instead of the real modules."""
    _STUBS.clear()

    def _fake_base_argv(name: str) -> list[str] | None:
        return _STUBS.get(name)

    monkeypatch.setattr(post_edit_check, "_checker_base_argv", _fake_base_argv)
    return tmp_path / "stubs"


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------


def test_clean_checker_reports_nothing(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=0)
    findings = post_edit_check.run_checks(target, _config(checks=["ruff"]), repo_root=repo)
    assert findings == []


def test_failing_checker_is_reported(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("import os\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="F401 unused import")
    findings = post_edit_check.run_checks(target, _config(checks=["ruff"]), repo_root=repo)
    assert len(findings) == 1
    assert findings[0][0] == "ruff"
    assert "F401" in findings[0][1]


def test_unknown_checker_is_skipped_not_fatal(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    # Forward compatibility: a config naming a newer checker must not crash.
    assert post_edit_check.run_checks(target, _config(checks=["nonexistent"]), repo_root=repo) == []


def test_uninstalled_checker_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(post_edit_check, "_checker_base_argv", lambda _name: None)
    assert post_edit_check.run_checks(target, _config(checks=["ruff"]), repo_root=repo) == []


def test_checkers_run_through_the_current_interpreter() -> None:
    """Repo convention (AGENTS.md): `python -m ruff`, never a bare PATH binary.

    A stray global ruff would report different findings than the pinned one the
    local gate and CI use.
    """
    base = post_edit_check._checker_base_argv("ruff")
    assert base is not None, "ruff should be importable in the dev environment"
    assert base[0] == sys.executable
    assert base[1] == "-m"


def test_unimportable_checker_resolves_to_none() -> None:
    assert post_edit_check._checker_base_argv("definitely_not_a_module_xyz") is None


def test_checker_timeout_is_survived(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    slow = tmp_path / "stubs" / "slow.py"
    slow.parent.mkdir(parents=True, exist_ok=True)
    slow.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    _STUBS["ruff"] = [sys.executable, str(slow)]
    findings = post_edit_check.run_checks(
        target, _config(checks=["ruff"], timeout_s=0.25), repo_root=repo
    )
    assert findings == []


def test_multiple_checkers_run_in_configured_order(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="ruff-finding")
    _stub_checker(tmp_path, "mypy", exit_code=1, message="mypy-finding")
    findings = post_edit_check.run_checks(target, _config(checks=["mypy", "ruff"]), repo_root=repo)
    assert [name for name, _ in findings] == ["mypy", "ruff"]


# ---------------------------------------------------------------------------
# mypy path profiles
# ---------------------------------------------------------------------------


def _profile_config(**profile: Any) -> WorkforceConfig:
    return _config(checks=["mypy"], mypy_profiles=[profile])


def test_profile_is_selected_by_repo_relative_glob() -> None:
    cfg = _profile_config(paths=["tools/**"], args=["--strict"], mypy_path=".")
    assert post_edit_check._mypy_profile(cfg, "tools/claude_hooks/config.py") is not None
    assert post_edit_check._mypy_profile(cfg, "src/mousedroid/x.py") is None


def test_unmatched_file_keeps_the_default_mypy_args(tmp_path: Path) -> None:
    """A non-matching path must not inherit another tree's flags.

    ``--explicit-package-bases`` is wrong for ``src/``: it makes mypy resolve the
    same file as both ``src.mousedroid`` and ``mousedroid`` and report nothing
    else, which is the exact failure the profiles exist to avoid.
    """
    repo = _repo(tmp_path)
    cfg = _profile_config(paths=["tools/**"], args=["--explicit-package-bases"], mypy_path=".")
    invocation = post_edit_check._checker_invocation(
        "mypy", cfg, repo / "src/x.py", rel_path="src/x.py", repo_root=repo
    )
    assert invocation is not None
    argv, env = invocation
    assert "--explicit-package-bases" not in argv
    assert argv[3:] == [*cfg.post_edit.mypy_args, str(repo / "src/x.py")]
    assert env == {}


def test_matched_file_replaces_args_rather_than_appending(tmp_path: Path) -> None:
    """A profile exists because the default set is wrong for the tree."""
    repo = _repo(tmp_path)
    cfg = _profile_config(paths=["tools/**"], args=["--strict"])
    invocation = post_edit_check._checker_invocation(
        "mypy", cfg, repo / "tools/a.py", rel_path="tools/a.py", repo_root=repo
    )
    assert invocation is not None
    argv, env = invocation
    assert argv[:3] == [sys.executable, "-m", "mypy"]
    assert argv[3:] == ["--strict", str(repo / "tools/a.py")]
    assert "--follow-imports=silent" not in argv
    assert env == {}


def test_profile_mypy_path_is_absolutised_against_the_repo_root(tmp_path: Path) -> None:
    """Config stays repo-relative (I-3); mypy receives a cwd-independent path."""
    repo = _repo(tmp_path)
    cfg = _profile_config(paths=["tools/**"], args=["--strict"], mypy_path=".")
    invocation = post_edit_check._checker_invocation(
        "mypy", cfg, repo / "tools/a.py", rel_path="tools/a.py", repo_root=repo
    )
    assert invocation is not None
    assert Path(invocation[1]["MYPYPATH"]).is_absolute()
    assert Path(invocation[1]["MYPYPATH"]) == repo


def test_profiles_do_not_apply_to_other_checkers(tmp_path: Path) -> None:
    """``ruff`` has no module resolution, so it must keep ``ruff_args``."""
    repo = _repo(tmp_path)
    cfg = _config(checks=["ruff"], mypy_profiles=[{"paths": ["tools/**"], "args": ["--strict"]}])
    invocation = post_edit_check._checker_invocation(
        "ruff", cfg, repo / "tools/a.py", rel_path="tools/a.py", repo_root=repo
    )
    assert invocation is not None
    argv, env = invocation
    assert argv[:3] == [sys.executable, "-m", "ruff"]
    assert argv[3:] == ["check", "--no-fix", str(repo / "tools/a.py")]
    assert env == {}


def test_file_outside_the_repo_matches_no_profile(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    cfg = _profile_config(paths=["**"], args=["--strict"], mypy_path=".")
    assert post_edit_check._mypy_profile(cfg, None) is None
    invocation = post_edit_check._checker_invocation(
        "mypy", cfg, repo / "a.py", rel_path=None, repo_root=repo
    )
    assert invocation is not None
    assert invocation[1] == {}


def test_first_matching_profile_wins() -> None:
    cfg = _config(
        checks=["mypy"],
        mypy_profiles=[
            {"paths": ["tools/**"], "args": ["--first"]},
            {"paths": ["tools/claude_hooks/**"], "args": ["--second"]},
        ],
    )
    profile = post_edit_check._mypy_profile(cfg, "tools/claude_hooks/config.py")
    assert profile is not None
    assert profile.args == ["--first"]


def test_profile_mypy_path_reaches_the_subprocess_environment(
    tmp_path: Path, stub_bin: Path
) -> None:
    """The behavioural proof: an argv assertion cannot show MYPYPATH was exported.

    The stub reports its own ``MYPYPATH`` and exits non-zero so the value comes
    back as a finding.
    """
    repo = _repo(tmp_path)
    target = repo / "tools" / "a.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n", encoding="utf-8")
    script = tmp_path / "stubs" / "echo_env.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        'import os, sys\nprint("MYPYPATH=" + os.environ.get("MYPYPATH", "<unset>"))\nsys.exit(1)\n',
        encoding="utf-8",
    )
    _STUBS["mypy"] = [sys.executable, str(script)]

    findings = post_edit_check.run_checks(
        target,
        _profile_config(paths=["tools/**"], args=[], mypy_path="."),
        repo_root=repo,
    )
    assert len(findings) == 1
    assert f"MYPYPATH={repo}" in findings[0][1]


def test_unmatched_file_does_not_export_mypy_path(tmp_path: Path, stub_bin: Path) -> None:
    """The negative half: the overlay must not leak to non-matching files."""
    repo = _repo(tmp_path)
    target = repo / "src" / "a.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n", encoding="utf-8")
    script = tmp_path / "stubs" / "echo_env.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        'import os, sys\nprint("MYPYPATH=" + os.environ.get("MYPYPATH", "<unset>"))\nsys.exit(1)\n',
        encoding="utf-8",
    )
    _STUBS["mypy"] = [sys.executable, str(script)]

    findings = post_edit_check.run_checks(
        target,
        _profile_config(paths=["tools/**"], args=[], mypy_path="."),
        repo_root=repo,
    )
    assert len(findings) == 1
    assert "MYPYPATH=<unset>" in findings[0][1]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _main(repo: Path, payload: dict[str, Any], stderr: io.StringIO) -> int:
    return post_edit_check.main(
        stdin=io.StringIO(json.dumps(payload)),
        stderr=stderr,
        env={"CLAUDE_PROJECT_DIR": str(repo)},
    )


def test_main_writes_findings_to_stderr(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("import os\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="F401 unused")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text(
        "post_edit:\n    checks:\n        - ruff\n", encoding="utf-8"
    )
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert "post-edit:ruff" in stderr.getvalue()


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_input": {}},
        {"tool_input": {"file_path": "does/not/exist.py"}},
    ],
)
def test_main_no_op_paths_exit_zero(
    tmp_path: Path, payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    assert _main(repo, payload, io.StringIO()) == 0


def test_main_skips_non_python_suffix(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "notes.md"
    target.write_text("# notes\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="should-not-run")
    stderr = io.StringIO()
    assert _main(repo, {"tool_input": {"file_path": str(target)}}, stderr) == 0
    assert stderr.getvalue() == ""


def test_main_skips_file_outside_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    stderr = io.StringIO()
    assert _main(repo, {"tool_input": {"file_path": str(outside)}}, stderr) == 0
    assert stderr.getvalue() == ""


def test_main_respects_disabled_flag(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="should-not-run")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text(
        "post_edit:\n    enabled: false\n", encoding="utf-8"
    )
    stderr = io.StringIO()
    assert _main(repo, {"tool_input": {"file_path": str(target)}}, stderr) == 0
    assert stderr.getvalue() == ""


def test_main_survives_invalid_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text("bogus: 1\n", encoding="utf-8")
    # Advisory hook: a bad config degrades to a no-op, never a failed turn.
    assert _main(repo, {"tool_input": {"file_path": "a.py"}}, io.StringIO()) == 0


def test_main_accepts_relative_target(tmp_path: Path, stub_bin: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("import os\n", encoding="utf-8")
    _stub_checker(tmp_path, "ruff", exit_code=1, message="F401")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "workforce.yaml").write_text(
        "post_edit:\n    checks:\n        - ruff\n", encoding="utf-8"
    )
    stderr = io.StringIO()
    assert _main(repo, {"tool_input": {"file_path": "a.py"}}, stderr) == 0
    assert "post-edit:ruff" in stderr.getvalue()
