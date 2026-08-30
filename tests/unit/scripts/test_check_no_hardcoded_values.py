from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch


def _load_checker_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    script_path = repo_root / "scripts" / "check_no_hardcoded_values.py"

    spec = importlib.util.spec_from_file_location("check_no_hardcoded_values", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ast_detector_flags_numeric_literal_on_changed_line() -> None:
    checker = _load_checker_module()
    source = "timeout_s = 2.5\n"

    findings = checker._find_suspicious_literals(source, {1})

    assert findings == [(1, "2.5")]


def test_ast_detector_ignores_comments_and_string_numbers() -> None:
    checker = _load_checker_module()
    source = "message = 'port 9000'\n# timeout 5\n"

    findings = checker._find_suspicious_literals(source, {1, 2})

    assert findings == []


def test_ast_detector_ignores_allowed_literals_and_range_calls() -> None:
    checker = _load_checker_module()
    source = "for _ in range(5):\n    pass\nok = 1\n"

    findings = checker._find_suspicious_literals(source, {1, 3})

    assert findings == []


def test_ast_detector_respects_suppression_marker() -> None:
    checker = _load_checker_module()
    source = "workers = 16  # hardcoded-ok\n"

    findings = checker._find_suspicious_literals(source, {1})

    assert findings == []


def test_ast_detector_formats_negative_literals() -> None:
    checker = _load_checker_module()
    source = "threshold = -7\n"

    findings = checker._find_suspicious_literals(source, {1})

    assert findings == [(1, "-7")]


def test_hardcoded_value_dir_exemptions_are_pinned() -> None:
    """Growing ``ALLOWED_DIR_PREFIXES`` is a deliberate, reviewed decision.

    Each entry exists for a documented reason (see the constant's comment):
    either the package is inherently a defaults-only package
    (``config/schema/``), or it is pre-existing code relocated by a same-PR
    module split whose lines read as "new" only because a 1-file-to-many
    split has no git rename correspondence. A new entry silencing an
    unrelated finding should fail this test until deliberately added here.
    """
    checker = _load_checker_module()
    assert checker.ALLOWED_DIR_PREFIXES == (
        "src/mousedroid/config/schema/",
        "src/mousedroid/telemetry/metrics/",
        "src/mousedroid/telemetry/server/",
        "src/mousedroid/validation/runtime/",
        "src/mousedroid/factory/",
        "src/mousedroid/orchestrator/_",
    )


def _fake_run_factory(results: dict[tuple[str, ...], tuple[int, str, str]]) -> Any:
    def _fake_run(cmd: list[str]) -> Any:
        class _R:
            returncode: int
            stdout: str
            stderr: str

        key = tuple(cmd)
        code, out, err = results.get(key, (1, "", "unexpected"))
        r = _R()
        r.returncode = code
        r.stdout = out
        r.stderr = err
        return r

    return _fake_run


def test_git_base_candidates_prefers_origin_prefix() -> None:
    checker = _load_checker_module()
    with patch.dict(os.environ, {"GITHUB_BASE_REF": "main"}, clear=False):
        assert checker._git_base_candidates(None) == ["origin/main", "main"]


def test_git_base_candidates_passes_explicit_ref_unchanged() -> None:
    checker = _load_checker_module()
    assert checker._git_base_candidates("origin/feature") == ["origin/feature"]


def test_git_base_candidates_prefixes_slash_containing_branch_name() -> None:
    """A bare branch name containing '/' (e.g. a namespaced dev branch) must
    still be tried against origin/ first.

    Regression for a bug where the origin/ prefix was only added when the raw
    ref had no '/' at all, so a base branch like
    'claude/markdown-implementation-plan-aVJ2l' was never resolved against
    origin/ and the CI gate hard-failed with "base ref unresolved".
    """
    checker = _load_checker_module()
    with patch.dict(os.environ, {"GITHUB_BASE_REF": "claude/foo-bar"}, clear=False):
        assert checker._git_base_candidates(None) == [
            "origin/claude/foo-bar",
            "claude/foo-bar",
        ]


def test_first_valid_base_ref_returns_first_resolvable() -> None:
    checker = _load_checker_module()
    fake_run = _fake_run_factory(
        {
            ("git", "rev-parse", "--verify", "origin/main^{commit}"): (1, "", "bad"),
            ("git", "rev-parse", "--verify", "main^{commit}"): (0, "deadbeef", ""),
        }
    )
    with (
        patch.dict(os.environ, {"GITHUB_BASE_REF": "main"}, clear=False),
        patch.object(checker, "_run", fake_run),
    ):
        assert checker._first_valid_base_ref(None) == "main"


def test_first_valid_base_ref_none_when_unresolvable() -> None:
    checker = _load_checker_module()
    with patch.dict(os.environ, {"GITHUB_BASE_REF": ""}, clear=False):
        assert checker._first_valid_base_ref(None) is None


def test_main_fails_in_ci_when_base_ref_unresolved() -> None:
    checker = _load_checker_module()
    env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_BASE_REF", "CI"}}
    env["CI"] = "true"
    with patch.dict(os.environ, env, clear=True):
        assert checker.main([]) == 2
