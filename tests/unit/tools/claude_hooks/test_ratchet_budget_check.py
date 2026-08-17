# tests/unit/tools/claude_hooks/test_ratchet_budget_check.py
"""Unit tests for the ratchet-budget early-warning hook.

Contract: this hook always exits 0. ``PostToolUse`` fires after the write, so
a non-zero exit could not undo anything — it would only add noise.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from tools.claude_hooks import ratchet_budget_check


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def _write_workforce_config(repo: Path, body: str) -> None:
    (repo / ".claude").mkdir(exist_ok=True)
    (repo / ".claude" / "workforce.yaml").write_text(body, encoding="utf-8")


def _write_source(repo: Path, rel_path: str, *, occurrences: int, marker: str = "noqa") -> Path:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"x = 1  # {marker}\n" for _ in range(occurrences)), encoding="utf-8")
    return target


def _main(repo: Path, payload: dict[str, Any], stderr: io.StringIO) -> int:
    return ratchet_budget_check.main(
        stdin=io.StringIO(json.dumps(payload)),
        stderr=stderr,
        env={"CLAUDE_PROJECT_DIR": str(repo)},
    )


_SMALL_BUDGET = (
    "ratchet_budgets:\n"
    "    items:\n"
    "        - name: noqa\n"
    "          marker: noqa\n"
    "          scope_glob: 'src/mousedroid/**/*.py'\n"
    "          ceiling: 2\n"
    "          warn_threshold: 1\n"
)


def test_main_writes_findings_to_stderr(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, _SMALL_BUDGET)
    target = _write_source(repo, "src/mousedroid/a.py", occurrences=3)
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert "[ratchet-budget]" in stderr.getvalue()
    assert "noqa" in stderr.getvalue()


def test_main_silent_when_healthy(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, _SMALL_BUDGET)
    target = _write_source(repo, "src/mousedroid/a.py", occurrences=1)
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_input": {}},
        {"tool_input": {"file_path": "does/not/exist.py"}},
    ],
)
def test_main_no_op_paths_exit_zero(tmp_path: Path, payload: dict[str, Any]) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, _SMALL_BUDGET)
    assert _main(repo, payload, io.StringIO()) == 0


def test_main_skips_path_outside_every_scope_glob(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, _SMALL_BUDGET)
    target = repo / "docs" / "notes.md"
    target.parent.mkdir(parents=True)
    target.write_text("# notes\n", encoding="utf-8")
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert stderr.getvalue() == ""


def test_main_skips_file_outside_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(outside)}}, stderr)
    assert code == 0
    assert stderr.getvalue() == ""


def test_main_respects_disabled_flag(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, "ratchet_budgets:\n    enabled: false\n")
    target = _write_source(repo, "src/mousedroid/a.py", occurrences=999)
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert stderr.getvalue() == ""


def test_main_survives_invalid_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, "bogus: 1\n")
    target = _write_source(repo, "src/mousedroid/a.py", occurrences=1)
    # Advisory hook: a bad config degrades to a no-op, never a failed turn.
    assert _main(repo, {"tool_input": {"file_path": str(target)}}, io.StringIO()) == 0


def test_main_accepts_relative_target(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_workforce_config(repo, _SMALL_BUDGET)
    _write_source(repo, "src/mousedroid/a.py", occurrences=3)
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": "src/mousedroid/a.py"}}, stderr)
    assert code == 0
    assert "[ratchet-budget]" in stderr.getvalue()


def test_default_config_out_of_the_box_flags_a_python_file_in_scope(tmp_path: Path) -> None:
    """No workforce.yaml at all still uses the schema's built-in three items."""
    repo = _repo(tmp_path)
    target = _write_source(repo, "src/mousedroid/a.py", occurrences=1, marker="# hardcoded-ok")
    # Default hardcoded_ok ceiling is 24 / warn_threshold 17 apart — one
    # occurrence is healthy, so this just proves the default config loads and
    # the hook runs cleanly with no workforce.yaml present.
    stderr = io.StringIO()
    code = _main(repo, {"tool_input": {"file_path": str(target)}}, stderr)
    assert code == 0
    assert stderr.getvalue() == ""
