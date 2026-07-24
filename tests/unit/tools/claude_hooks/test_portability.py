# tests/unit/tools/claude_hooks/test_portability.py
"""Unit tests for the workforce portability rule.

False positives matter as much as false negatives here: this rule gates every
workforce asset, so legitimate repo-relative paths, globs and URLs must pass
cleanly or contributors will learn to ignore it.
"""

from __future__ import annotations

import pytest
from tools.claude_hooks.portability import find_absolute_paths


@pytest.mark.parametrize(
    "text",
    [
        "run /home/user/project/script.sh",
        "cd /Users/someone/repo",
        "path: /root/.config/thing.yaml",
        "log at /var/log/mousedroid.log",
        "installed to /opt/tools/bin",
        r"open C:\Users\me\repo\file.py",
        "open C:/Users/me/repo/file.py",
    ],
)
def test_absolute_paths_are_flagged(text: str) -> None:
    assert find_absolute_paths(text)


@pytest.mark.parametrize(
    "text",
    [
        "use $CLAUDE_PROJECT_DIR/tools/claude_hooks/freeze_gate.py",
        "see src/mousedroid/config/schema.py",
        "glob src/mousedroid/arm/**",
        "docs at https://example.com/etc/passwd",
        "the ratio is 3/4 and 5/6",
        "relative ./scripts/ci.sh",
        "no paths here at all",
        "",
    ],
)
def test_portable_references_are_not_flagged(text: str) -> None:
    assert find_absolute_paths(text) == []


def test_multiple_matches_are_returned_in_order() -> None:
    found = find_absolute_paths("first /home/a/x.py then /var/b/y.py")
    assert len(found) == 2
    assert found[0].startswith("/home/")
    assert found[1].startswith("/var/")
