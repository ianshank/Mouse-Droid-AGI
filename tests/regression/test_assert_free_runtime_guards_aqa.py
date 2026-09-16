"""Automated Quality Assurance (AQA) — runtime guards survive ``PYTHONOPTIMIZE=1``.

``Dockerfile.jetson`` sets ``ENV PYTHONOPTIMIZE=1``, the documented Jetson
runtime contract, which strips every ``assert`` from the shipped image. Until
this change ``pyproject.toml`` globally ignored ruff's ``S101``, so eight
``assert`` statements had accumulated in ``src/mousedroid`` with nothing flagging
them — four of them the *first statement* of a mission-lifecycle method, on the
mission / e-stop path, with no fallback. On the rover those guards did not exist:
``assert self._mission is not None`` vanished and the following
``self._mission.state`` raised ``AttributeError`` partway through a state
transition.

The codebase already knew the rule — ``world_model/checkpoint_migration.py``
carries the comment *"NOT assert: stripped under PYTHONOPTIMIZE=1"* and
``resilience/retry.py`` spells out the same reasoning — so this was drift, not a
policy disagreement.

Three independent pins, because each catches a different way the fix could rot:

* ``TestNoAssertsInRuntimeCode`` — the structural sweep. Catches a *new* assert.
* ``TestS101StaysEnforced`` — the lint configuration. Catches someone restoring
  the global ignore, which would let the sweep's subject reappear unflagged.
* ``TestGuardsSurviveOptimisedInterpreter`` — the actual behaviour, executed in a
  subprocess under ``-O``. Catches a "fix" that reintroduces an assert-shaped
  guard that merely *looks* explicit.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mousedroid.config.schema import MissionConfig
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleStateError,
)
from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "mousedroid"


def _cfg() -> MissionConfig:
    return MissionConfig(
        replan_enabled=False,
        success_threshold=0.9,
        stall_threshold=0.1,
        stall_window_ticks=3,
        max_replans_per_mission=2,
    )


class TestNoAssertsInRuntimeCode:
    """No ``assert`` statement anywhere under ``src/mousedroid``."""

    @staticmethod
    def _assert_sites() -> list[str]:
        sites: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            sites.extend(
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
            )
        return sites

    def test_sweep_actually_parsed_the_tree(self) -> None:
        """Guard the guard — an empty sweep would pass vacuously."""
        assert len(list(_SRC.rglob("*.py"))) > 300

    def test_no_assert_statements_in_src(self) -> None:
        sites = self._assert_sites()
        assert not sites, (
            "assert statements found in runtime code: "
            f"{sites}. Dockerfile.jetson sets PYTHONOPTIMIZE=1, which strips "
            "them, so each is a guard that does not exist on the rover. Raise "
            "an explicit exception instead - see "
            "orchestrator/mission_lifecycle.py::_require_mission."
        )


class TestS101StaysEnforced:
    """The lint rule that keeps the sweep above from silently reaccumulating."""

    @staticmethod
    def _ruff_lint_config() -> dict[str, object]:
        """Read ``[tool.ruff.lint]`` via the repo's shared pyproject loader.

        Uses ``tests._pyproject.load_pyproject`` rather than importing
        ``tomllib`` directly: that helper carries the ``tomli`` fallback the
        Python 3.10 CI leg needs (stdlib ``tomllib`` lands in 3.11), and it is
        the convention both sibling single-source tests already follow.
        """
        tool = load_pyproject()["tool"]
        assert isinstance(tool, dict)
        ruff = tool["ruff"]
        assert isinstance(ruff, dict)
        lint: dict[str, object] = ruff["lint"]
        return lint

    def test_s101_not_globally_ignored(self) -> None:
        ignore = self._ruff_lint_config()["ignore"]
        assert isinstance(ignore, list)
        assert "S101" not in ignore, (
            "S101 is globally ignored again, which un-flags every assert in "
            "src/. It belongs only in per-file-ignores for tests/**."
        )

    def test_s101_selected_via_the_s_ruleset(self) -> None:
        """``S101`` ships inside the ``S`` (bandit) ruleset rather than alone."""
        select = self._ruff_lint_config()["select"]
        assert isinstance(select, list)
        assert "S" in select

    def test_s101_still_ignored_for_tests(self) -> None:
        """Tests must keep using ``assert`` — it is their assertion mechanism.

        ``PYTHONOPTIMIZE`` is never set for the test run, so the stripping
        hazard does not apply there; removing this exemption would break the
        entire suite rather than harden anything.
        """
        per_file = self._ruff_lint_config()["per-file-ignores"]
        assert isinstance(per_file, dict)
        test_ignores = per_file["tests/**/*.py"]
        assert isinstance(test_ignores, list)
        assert "S101" in test_ignores

    @pytest.mark.parametrize("scope", ["tools/**/*.py", "scripts/**/*.py"])
    def test_s101_not_exempted_for_tools_or_scripts(self, scope: str) -> None:
        """Close the obvious escape hatch: exempting another tree instead of fixing.

        ``tools/**`` and ``scripts/**`` both already carry ``per-file-ignores``
        entries for other rules, so adding ``S101`` to one is a one-token way to
        silence a future assert rather than convert it. Neither tree contains an
        assert today and both are covered by CI's ``ruff check`` (the lint job
        runs ``scripts/`` as a separate invocation), so this keeps it that way.
        """
        per_file = self._ruff_lint_config()["per-file-ignores"]
        assert isinstance(per_file, dict)
        ignores = per_file.get(scope, [])
        assert isinstance(ignores, list)
        assert "S101" not in ignores, (
            f"S101 was exempted for {scope}. Convert the assert to an explicit "
            "raise instead of widening the exemption — tests/** is the only tree "
            "where assert is the contract."
        )


class TestMissionLifecycleGuardRaisesTypedError:
    """Each converted site raises the named error, not ``AttributeError``."""

    @pytest.mark.parametrize(
        "operation",
        ["transition", "transition_to_failed", "record_terminal_duration"],
    )
    def test_sync_guarded_methods_raise_typed_error(self, operation: str) -> None:
        lifecycle = MissionLifecycle(_cfg())
        # No mission started: every guarded method must refuse by name.
        with pytest.raises(MissionLifecycleStateError, match=operation):
            lifecycle._require_mission(operation)

    def test_error_is_a_runtime_error_subclass(self) -> None:
        """Additive typing: existing ``except RuntimeError`` handlers still catch."""
        assert issubclass(MissionLifecycleStateError, RuntimeError)

    def test_message_names_the_operation_for_debugging(self) -> None:
        lifecycle = MissionLifecycle(_cfg())
        with pytest.raises(MissionLifecycleStateError) as excinfo:
            lifecycle._require_mission("handle_stall")
        assert "handle_stall" in str(excinfo.value)
        assert "active mission" in str(excinfo.value)

    def test_returns_the_live_state_object_not_a_copy(self) -> None:
        """Identity, not equality — this is the load-bearing property.

        The conversion replaced ``self._mission.stall_counter = 0`` with
        ``mission.stall_counter = 0`` in ``_handle_stall``, so every mutation now
        goes through the returned reference. An implementation that returned a
        copy (or a ``model_copy()``, or a frozen view) would satisfy every other
        test in this file and in the backwards-compat pair while silently
        discarding all three of ``_handle_stall``'s writes.
        """
        lifecycle = MissionLifecycle(_cfg())
        lifecycle.start_mission("m-identity", "goal")
        returned = lifecycle._require_mission("transition")
        assert returned is lifecycle._mission

    def test_mutation_through_the_returned_reference_is_visible(self) -> None:
        """The same property from the caller's side, stated behaviourally."""
        lifecycle = MissionLifecycle(_cfg())
        lifecycle.start_mission("m-mutate", "goal")
        mission = lifecycle._require_mission("handle_stall")
        mission.stall_counter = 7
        assert lifecycle._require_mission("transition").stall_counter == 7

    def test_does_not_raise_while_a_mission_is_active(self) -> None:
        """The positive case — a guard that always raised would also pass above."""
        lifecycle = MissionLifecycle(_cfg())
        lifecycle.start_mission("m-active", "goal")
        for operation in ("transition", "transition_to_failed", "record_terminal_duration"):
            assert lifecycle._require_mission(operation) is not None


class TestGuardsSurviveOptimisedInterpreter:
    """The pin that the old ``assert`` form would fail: run under ``-O``."""

    _PROBE = """
import sys
from mousedroid.config.schema import MissionConfig
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleStateError,
)

# Proves asserts really are stripped in this interpreter, so a pass below
# cannot be explained by -O having had no effect.
stripped = True
try:
    assert False, "unreachable"
except AssertionError:
    stripped = False
if not stripped:
    print("ASSERTS_NOT_STRIPPED")
    sys.exit(2)

lifecycle = MissionLifecycle(
    MissionConfig(
        replan_enabled=False,
        success_threshold=0.9,
        stall_threshold=0.1,
        stall_window_ticks=3,
        max_replans_per_mission=2,
    )
)
try:
    lifecycle._require_mission("transition")
except MissionLifecycleStateError:
    print("GUARD_HELD")
    sys.exit(0)
except AttributeError:
    print("GUARD_STRIPPED_ATTRIBUTEERROR")
    sys.exit(3)
print("GUARD_MISSING")
sys.exit(4)
"""

    def test_guard_raises_under_pythonoptimize(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", self._PROBE],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, (
            f"guard did not survive -O: rc={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
        )
        assert "GUARD_HELD" in result.stdout
