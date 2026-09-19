"""Shared skip guard for tests that shell out to ``bash``.

Follows the repo's ``tests/_<name>.py`` shared-helper convention (cf.
``tests/_script_loader.py``, ``tests/_pyproject.py``).

**Why this exists as a helper rather than a copy-pasted marker.** The condition
has one non-obvious term, and the whole guard fails open without it:

    os.name == "nt" is NOT redundant with shutil.which("bash") is None

On the GitHub Windows runner ``bash`` resolves to the WSL shim. ``which()``
therefore *finds* it, the guard concludes bash is available, and every
invocation then exits 1 having printed "Windows Subsystem for Linux has no
installed distributions" on stdout — so the test fails with a confusing
assertion about a return code rather than skipping.

That has now bitten this repository at least twice. ``test_repin_tags.py:31-46``
carries a comment recording the first time ("the sibling had the guard, this
file did not"), and the second time was
``tests/regression/test_f051_{aqa,backwards_compat}.py``, which guarded on
``which("bash")`` alone and turned seven tests red on ``test-windows`` while
passing on Linux. A copy-pasted predicate that is *usually* right is worse than
a named one, because the missing term is invisible at the call site.

Six older modules still inline their own copy of this predicate
(``grep -rl 'os.name == "nt"' tests/``). Those copies carry the full, correct
condition, so they are left alone rather than churned; migrating them to this
helper is a safe follow-up whenever one of them is next edited.

Usage::

    from tests._bash import requires_bash

    pytestmark = requires_bash()                         # whole module
    ...
    @requires_bash("git", "tar")                         # extra tools
    def test_something() -> None: ...
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

__all__ = ["bash_is_usable", "requires_bash"]


def bash_is_usable() -> bool:
    """Is there a ``bash`` that will actually run a script?

    Returns:
        ``False`` on Windows (where ``bash`` is the WSL shim) or when no
        ``bash`` is on ``PATH``.
    """
    return os.name != "nt" and shutil.which("bash") is not None


def requires_bash(
    *also: str,
    paths: tuple[Path, ...] = (),
) -> pytest.MarkDecorator:
    """Skip unless ``bash``, every tool in ``also``, and every path in ``paths`` exist.

    Args:
        also: Extra executables the test shells out to, e.g. ``"git"``, ``"tar"``.
            Each is resolved with :func:`shutil.which`.
        paths: Scripts or fixtures the test needs on disk. A missing one skips
            rather than erroring, so a test does not turn red merely because a
            script was renamed — the rename is caught by the tests that assert
            the path, not by every consumer of it.

    Returns:
        A ``pytest.mark.skipif`` marker, usable as ``pytestmark`` or a decorator.
    """
    missing_tools = [tool for tool in also if shutil.which(tool) is None]
    missing_paths = [path.name for path in paths if not path.exists()]
    required = ", ".join(("bash", *also, *(path.name for path in paths)))
    return pytest.mark.skipif(
        not bash_is_usable() or bool(missing_tools) or bool(missing_paths),
        reason=f"requires {required}",
    )
