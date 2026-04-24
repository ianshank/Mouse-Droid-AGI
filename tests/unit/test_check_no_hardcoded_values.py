from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_checker_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent.parent
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
