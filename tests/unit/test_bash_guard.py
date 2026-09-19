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

These tests run on Linux, so the Windows branch is exercised by patching
``os.name`` rather than by being on Windows — which is exactly the point: the
term has to be checkable from the platform where it is *not* active, or it only
ever gets tested by breaking CI.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from tests import _bash
from tests._bash import bash_is_usable, requires_bash


def _skip_condition(marker: pytest.MarkDecorator) -> bool:
    """Extract the boolean a ``skipif`` marker was built with."""
    condition = marker.mark.args[0]
    assert isinstance(condition, bool), "requires_bash must resolve eagerly to a bool"
    return condition


class TestBashIsUsable:
    """The predicate itself."""

    def test_true_on_this_posix_box_with_bash_present(self) -> None:
        assert shutil.which("bash") is not None, "premise: bash is on PATH here"
        assert bash_is_usable()

    def test_false_on_windows_even_when_bash_is_on_path(self) -> None:
        """The load-bearing term: `which` finds the WSL shim, which cannot run."""
        with mock.patch.object(_bash.os, "name", "nt"):
            assert not bash_is_usable()

    def test_false_when_bash_is_absent(self) -> None:
        with mock.patch.object(_bash.shutil, "which", return_value=None):
            assert not bash_is_usable()


class TestRequiresBash:
    """The marker factory."""

    def test_does_not_skip_when_bash_is_usable(self) -> None:
        assert not _skip_condition(requires_bash())

    def test_skips_on_windows(self) -> None:
        with mock.patch.object(_bash.os, "name", "nt"):
            assert _skip_condition(requires_bash())

    def test_skips_when_an_extra_tool_is_missing(self) -> None:
        assert _skip_condition(requires_bash("definitely-not-a-real-executable"))

    def test_does_not_skip_for_a_tool_that_exists(self) -> None:
        assert shutil.which("git") is not None, "premise: git is on PATH here"
        assert not _skip_condition(requires_bash("git"))

    def test_skips_when_a_required_path_is_missing(self) -> None:
        marker = requires_bash(paths=(Path("/nonexistent/rover_wip_guard.sh"),))
        assert _skip_condition(marker)

    def test_does_not_skip_for_a_path_that_exists(self) -> None:
        assert not _skip_condition(requires_bash(paths=(Path(__file__),)))

    def test_the_reason_names_everything_required(self) -> None:
        """A bare "bash required" tells a reader nothing about which tool was missing."""
        marker = requires_bash("git", "tar", paths=(Path("/nope/deploy_remote.sh"),))
        reason = marker.mark.kwargs["reason"]
        for expected in ("bash", "git", "tar", "deploy_remote.sh"):
            assert expected in reason, reason


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
