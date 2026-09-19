"""Tests for the shared ``bash`` skip guard (``tests/_bash.py``).

A guard that fails *open* is worse than no guard, because the tests it was
supposed to skip then fail with a confusing message instead. This helper's whole
reason to exist is one non-obvious term — ``os.name == "nt"`` — so that term is
the thing most worth pinning.

The bug being prevented, twice observed in this repository: on the GitHub Windows
runner ``bash`` resolves to the WSL shim. ``shutil.which("bash")`` therefore
finds it, a ``which``-only guard concludes bash is available, and every
invocation then exits 1 having printed "Windows Subsystem for Linux has no
installed distributions". Seven F-051 tests went red on ``test-windows`` that way
while passing on Linux.

**Every branch pins its host.** The first version of this module asserted the
positive branches ("does NOT skip") against the *real* host, which made them pass
on Linux and fail on ``test-windows`` — the same class of mistake, one level up.
So ``os.name`` and ``shutil.which`` are both patched in each test, and the
positive and negative branches are therefore checkable from either platform.
``TestAgainstTheRealHost`` keeps one unpinned assertion so the pinned ones cannot
all be true of a fiction; it is the only thing here that skips on Windows, and it
does so because on that host there is no positive answer to assert.

**Patching ``os.name`` has a second-order effect on ``pathlib``**, and that cost
a third round on ``test-windows``. ``Path()`` reads ``os.name`` at call time to
pick ``PosixPath`` or ``WindowsPath`` and refuses to instantiate the foreign one,
so a ``Path(...)`` built inside a patched block works on the platform the patch
names and raises ``NotImplementedError`` on the other. Executing the tests on
Linux cannot reveal it — patching to ``"posix"`` there is a no-op. So the path
fixtures are hoisted to import time, and
``TestNoPathIsBuiltUnderAPatchedHost`` pins that structurally, by reading this
module's own source. A structural check is the only kind that runs where the
mistake is *introduced* rather than where it fires.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from tests import _bash
from tests._bash import bash_is_usable, requires_bash

# Built at IMPORT time, on the native host, so they carry this platform's
# pathlib flavour. Constructing a ``Path`` *inside* a block that patches
# ``os.name`` makes pathlib select the OTHER platform's class, which it then
# refuses to instantiate ("cannot instantiate 'PosixPath' on your system" on
# Windows, and the ``WindowsPath`` mirror on Linux). That is how this module
# broke ``test-windows`` a second time, and a Linux run cannot reproduce it —
# Linux is perfectly happy with the ``PosixPath`` the patch selects.
_EXISTING_PATH = Path(__file__)
_MISSING_PATH = Path(__file__).with_name("definitely-not-on-disk.sh")


def _patched_which(result: str | None) -> object:
    """Patch ``shutil.which`` inside the helper module to return ``result``."""
    return mock.patch.object(_bash.shutil, "which", return_value=result)


def _skip_condition(marker: pytest.MarkDecorator) -> bool:
    """Extract the boolean a ``skipif`` marker was built with."""
    condition = marker.mark.args[0]
    assert isinstance(condition, bool), "requires_bash must resolve eagerly to a bool"
    return condition


class TestBashIsUsable:
    """The predicate itself, on a pinned host."""

    def test_true_on_a_posix_host_with_bash_present(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which("/usr/bin/bash"):
            assert bash_is_usable()

    def test_false_on_windows_even_when_bash_is_on_path(self) -> None:
        """The load-bearing term: `which` finds the WSL shim, which cannot run."""
        with mock.patch.object(_bash.os, "name", "nt"), _patched_which("C:\\Windows\\bash.EXE"):
            assert not bash_is_usable()

    def test_false_when_bash_is_absent(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which(None):
            assert not bash_is_usable()


class TestRequiresBash:
    """The marker factory, on a pinned host."""

    def test_does_not_skip_when_bash_is_usable(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which("/usr/bin/bash"):
            assert not _skip_condition(requires_bash())

    def test_skips_on_windows(self) -> None:
        with mock.patch.object(_bash.os, "name", "nt"), _patched_which("/usr/bin/bash"):
            assert _skip_condition(requires_bash())

    def test_skips_when_an_extra_tool_is_missing(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which(None):
            assert _skip_condition(requires_bash("some-tool"))

    def test_does_not_skip_for_a_tool_that_exists(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which("/usr/bin/git"):
            assert not _skip_condition(requires_bash("git"))

    def test_skips_when_a_required_path_is_missing(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which("/usr/bin/bash"):
            assert _skip_condition(requires_bash(paths=(_MISSING_PATH,)))

    def test_does_not_skip_for_a_path_that_exists(self) -> None:
        with mock.patch.object(_bash.os, "name", "posix"), _patched_which("/usr/bin/bash"):
            assert not _skip_condition(requires_bash(paths=(_EXISTING_PATH,)))

    def test_the_reason_names_everything_required(self) -> None:
        """A bare "bash required" tells a reader nothing about what was missing."""
        marker = requires_bash("git", "tar", paths=(_MISSING_PATH,))
        reason = marker.mark.kwargs["reason"]
        for expected in ("bash", "git", "tar", _MISSING_PATH.name):
            assert expected in reason, reason


class TestAgainstTheRealHost:
    """One unpinned check, so the pinned ones cannot all be true of a fiction.

    Skipped on Windows by the very condition under test — which is the honest
    thing to do, because on that host the correct answer *is* "not usable" and
    there is nothing positive to assert.
    """

    @pytest.mark.skipif(os.name == "nt", reason="the positive branch is POSIX-only by design")
    def test_the_helper_agrees_with_this_posix_box(self) -> None:
        assert shutil.which("bash") is not None, "premise: bash is on PATH here"
        assert bash_is_usable()
        assert not _skip_condition(requires_bash())


class TestTheF051FilesUseIt:
    """The two files whose half-guard caused the Windows failure.

    Pinned by import identity, not by a source substring: what matters is that
    they resolve the same predicate this module tests, so a future edit cannot
    quietly reintroduce a local ``which("bash")`` check.
    """

    @pytest.mark.parametrize(
        "module_name",
        [
            "tests.regression.test_f051_aqa",
            "tests.regression.test_f051_backwards_compat",
        ],
    )
    def test_the_module_imports_the_shared_guard(self, module_name: str) -> None:
        import importlib

        module = importlib.import_module(module_name)
        assert getattr(module, "requires_bash", None) is requires_bash, (
            f"{module_name} must use tests._bash.requires_bash, not a local "
            "shutil.which('bash') predicate that passes on the WSL shim"
        )


class TestNoPathIsBuiltUnderAPatchedHost:
    """Structural pin for the trap that broke ``test-windows`` a second time.

    ``pathlib.Path()`` reads ``os.name`` at call time to choose ``PosixPath`` or
    ``WindowsPath``, and refuses to instantiate the foreign one. So a ``Path(...)``
    inside a block that patches ``os.name`` is host-dependent by construction: it
    works on the platform the patch happens to name and raises
    ``NotImplementedError`` on the other.

    A Linux run cannot catch this by executing the tests — patching to ``"posix"``
    on Linux is a no-op. So it is caught structurally instead, by reading this
    module's own source. That is the only way the check runs where the bug is
    introduced rather than where it fires.
    """

    @staticmethod
    def _source() -> list[str]:
        return Path(__file__).read_text(encoding="utf-8").splitlines()

    def test_no_path_call_appears_inside_a_host_patched_block(self) -> None:
        offenders: list[str] = []
        patched_indent: int | None = None
        for line in self._source():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if patched_indent is not None and indent <= patched_indent:
                patched_indent = None
            if 'mock.patch.object(_bash.os, "name"' in stripped:
                patched_indent = indent
                continue
            if patched_indent is not None and "Path(" in stripped:
                offenders.append(stripped)
        assert offenders == [], (
            "build Path objects at import time (see _EXISTING_PATH / _MISSING_PATH); "
            "inside a patched-os.name block pathlib selects the foreign flavour and "
            f"raises on the other host: {offenders}"
        )

    def test_the_hoisted_fixtures_are_host_native(self) -> None:
        """They must carry whatever flavour this host's pathlib produces."""
        native = type(Path(__file__))
        assert type(_EXISTING_PATH) is native
        assert type(_MISSING_PATH) is native

    def test_the_missing_fixture_really_is_missing(self) -> None:
        """Otherwise the skip-on-missing-path test passes for the wrong reason."""
        assert not _MISSING_PATH.exists()
        assert _EXISTING_PATH.exists()
