"""AQA — F-053 Phase 1: the root ``AGENTS.md`` is loaded, and stays out of the wheel.

The defect this pins shut is silent. Claude Code reads ``AGENTS.md`` natively, but
only *"when you have no ``CLAUDE.md`` in your working directory or above it"* — and
the nested-subdirectory rule sits inside that same "when none count" block. This
repository has a root ``CLAUDE.md`` plus eight nested ones, so **no ``AGENTS.md``
anywhere in the tree is auto-loaded at any depth**. The only load path is an
explicit ``@AGENTS.md`` import from a ``CLAUDE.md`` in the same directory.

Before F-053 the repository carried a 403-line root ``AGENTS.md`` and **zero**
``@`` imports in any of its nine ``CLAUDE.md`` files. That file was maintained,
cited by tests and by docstrings across ``src/``, and read by nothing. Nothing
failed, because nothing checked.

Two properties are asserted here, and a third is asserted negatively:

1. Every ``AGENTS.md`` in the tree has a real load path — a same-directory
   ``CLAUDE.md`` that imports it with the bare ``@`` form.
2. Every agent-facing filename is excluded from the built wheel. These are
   internal instructions; ``pyproject.toml``'s own comment records that without
   the exclusion they ship to PyPI.
3. No ``AGENTS.md`` exists outside the repository root, because per the loading
   rule above one elsewhere would need a second file beside it to be read at all
   (F-053 design D-1/D-3).

Deliberately *not* asserted: any byte size. ``.gitattributes`` is absent, so a
CRLF checkout changes every file's byte count, and ``test-windows`` runs this
tier. Size is checked in lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._claude_md import (
    AGENT_FACING_WHEEL_PATTERNS,
    AGENTS_MD,
    CLAUDE_MD,
    discover_tracked,
    imported_paths,
    imports_target,
    repo_root,
)
from tests._pyproject import load_pyproject

_REPO_ROOT = repo_root()


def _read(relpath: Path) -> str:
    return (_REPO_ROOT / relpath).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tracked_claude_md() -> tuple[Path, ...]:
    return discover_tracked(CLAUDE_MD)


@pytest.fixture(scope="module")
def tracked_agents_md() -> tuple[Path, ...]:
    return discover_tracked(AGENTS_MD)


class TestEveryAgentsMdIsActuallyLoaded:
    """An ``AGENTS.md`` nothing imports is read by nothing. That is the defect."""

    def test_the_repository_has_at_least_one_agents_md(
        self, tracked_agents_md: tuple[Path, ...]
    ) -> None:
        """Premise: without this, the gate below passes vacuously."""
        assert tracked_agents_md, "expected at least the root AGENTS.md to be tracked"

    def test_every_agents_md_is_imported_by_a_sibling_claude_md(
        self, tracked_agents_md: tuple[Path, ...]
    ) -> None:
        unreachable: list[str] = []
        for agents_md in tracked_agents_md:
            sibling = agents_md.parent / CLAUDE_MD
            if not (_REPO_ROOT / sibling).is_file():
                unreachable.append(f"{agents_md}: no {sibling} to import it")
                continue
            if not imports_target(_read(sibling), AGENTS_MD):
                unreachable.append(f"{agents_md}: {sibling} does not import it")
        assert unreachable == [], (
            "these AGENTS.md files are read by nothing — with a CLAUDE.md at or "
            "above the working directory, Claude Code does not auto-load AGENTS.md "
            f"at any depth, so a bare `@{AGENTS_MD}` import is the only load "
            f"path: {unreachable}"
        )

    def test_a_backticked_mention_is_not_an_import(self) -> None:
        """The subtlety the gate turns on: import parsing skips code spans.

        A grep for the bare substring would pass on a file that only *mentions*
        the path, which would make this whole gate decorative.
        """
        assert imports_target(f"@{AGENTS_MD}", AGENTS_MD)
        assert not imports_target(f"see `@{AGENTS_MD}` for details", AGENTS_MD)
        assert not imports_target(f"```\n@{AGENTS_MD}\n```", AGENTS_MD)

    def test_no_import_chain_leaves_the_repository(
        self, tracked_claude_md: tuple[Path, ...]
    ) -> None:
        """An external import shows an approval dialog that silently disables it."""
        escaping: list[str] = []
        for claude_md in tracked_claude_md:
            for target in imported_paths(_read(claude_md)):
                if target.startswith(("/", "~")):
                    escaping.append(f"{claude_md} imports {target}")
        assert escaping == [], (
            "an import resolving outside the working directory prompts an approval "
            f"dialog, and stays disabled for anyone who declines: {escaping}"
        )


class TestTheWheelExcludesAgentInstructions:
    """Internal instructions are not part of the distribution."""

    def test_every_agent_facing_pattern_is_excluded(self) -> None:
        data = load_pyproject()
        tool = data.get("tool", {})
        assert isinstance(tool, dict)
        wheel = tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
        assert isinstance(wheel, dict), "expected [tool.hatch.build.targets.wheel]"
        exclude = wheel.get("exclude", [])
        missing = [p for p in AGENT_FACING_WHEEL_PATTERNS if p not in exclude]
        assert missing == [], (
            "these agent-facing patterns are not excluded from the wheel, so the "
            f"distribution ships internal instructions to PyPI: {missing}"
        )

    def test_the_pattern_list_has_no_silent_gaps(self) -> None:
        """Every agent-facing basename in the tree is covered by some pattern.

        Catches a future filename convention that nobody adds to the list — the
        way ``AGENTS.md`` itself was missing while ``CLAUDE.md`` and ``agent.md``
        were covered.
        """
        covered = {Path(p).name for p in AGENT_FACING_WHEEL_PATTERNS}
        present = {
            path.name
            for filename in (CLAUDE_MD, AGENTS_MD, "agent.md")
            for path in discover_tracked(filename)
        }
        assert present <= covered, f"agent-facing files with no wheel pattern: {present - covered}"


class TestTheBuiltWheelReallyExcludesThem:
    """Assert the behaviour, not only the declaration — with its limit stated.

    ``TestTheWheelExcludesAgentInstructions`` checks the patterns are *present*
    in ``pyproject.toml``. That cannot catch a pattern that is present but
    ineffective: a malformed glob, or a hatchling version treating ``**/``
    differently. This builds the wheel and looks inside.

    **What it cannot prove, and why that is recorded rather than hidden.** A
    pattern with no matching file in the tree is untestable by building. Proven
    by reverting: deleting ``**/AGENTS.md`` from the exclude list leaves this
    test **green**, because ``packages = ["src/mousedroid"]`` means only files
    under that directory are build candidates and the sole ``AGENTS.md`` is at
    the repository root. So this test covers ``CLAUDE.md`` and ``agent.md``,
    which have 22 files between them under ``src/``, and the ``AGENTS.md``
    pattern is defence-in-depth that ``test_every_agent_facing_pattern_is_excluded``
    is the real gate for.

    ``test_the_build_test_is_not_vacuous`` makes that limit explicit, so a reader
    cannot mistake this class for coverage it does not have.
    """

    @staticmethod
    def _build(destination: Path) -> Path:
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(destination)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        wheels = sorted(destination.glob("*.whl"))
        if not wheels:
            raise AssertionError("hatchling produced no wheel")
        return wheels[-1]

    def test_the_build_test_is_not_vacuous(self) -> None:
        """Name which patterns the build below can actually exercise.

        A build proves nothing about a pattern with no matching file. This keeps
        that fact visible: if a future change leaves *no* agent-facing file under
        ``src/``, the build test becomes decorative and this fails to say so.
        """
        candidates = {
            filename: [p for p in discover_tracked(filename) if p.parts[0] == "src"]
            for filename in (CLAUDE_MD, AGENTS_MD, "agent.md")
        }
        exercised = {name for name, paths in candidates.items() if paths}
        assert exercised, (
            "no agent-facing file exists under src/, so the wheel build below "
            "cannot prove any exclusion — it would pass with the patterns removed"
        )
        # Recorded, not asserted: AGENTS.md has no file under src/ by design
        # (F-053 keeps it at the root), so its pattern is not build-exercised.
        assert {CLAUDE_MD, "agent.md"} <= exercised, (
            f"expected CLAUDE.md and agent.md to be build-exercised; got {exercised}"
        )

    def test_no_agent_facing_file_reaches_the_wheel(self, tmp_path: Path) -> None:
        pytest.importorskip("hatchling", reason="wheel build needs hatchling")
        import zipfile

        agent_facing = {CLAUDE_MD, AGENTS_MD, "agent.md"}
        in_tree = [
            path
            for filename in agent_facing
            for path in discover_tracked(filename)
            if path.parts[0] == "src"
        ]
        assert in_tree, "premise: agent-facing files exist under src/ to be excluded"

        with zipfile.ZipFile(self._build(tmp_path)) as wheel:
            shipped = [name for name in wheel.namelist() if name.rsplit("/", 1)[-1] in agent_facing]
        assert shipped == [], (
            f"the wheel ships internal agent instructions to PyPI: {shipped} "
            f"(the exclude patterns in pyproject.toml are present but not working)"
        )


class TestAgentsMdStaysAtTheRoot:
    """D-1/D-3: elsewhere it needs a second file beside it to be read at all."""

    def test_no_agents_md_outside_the_repository_root(
        self, tracked_agents_md: tuple[Path, ...]
    ) -> None:
        nested = [str(p) for p in tracked_agents_md if p.parent != Path(".")]
        assert nested == [], (
            "F-053 keeps AGENTS.md at the root only: a nested one is not "
            "auto-loaded while a CLAUDE.md sits above it, so it would need its own "
            f"sibling CLAUDE.md importing it to be read: {nested}"
        )


class TestTheRootClaudeMdStaysWithinItsBudget:
    """The one doc budget this repository actually enforces."""

    def test_root_claude_md_is_inside_core_max_lines(self) -> None:
        from tools.claude_hooks.config import load_config

        budget = load_config().docs.core_max_lines
        actual = len(_read(Path(CLAUDE_MD)).splitlines())
        assert actual <= budget, (
            f"root {CLAUDE_MD} is {actual} lines against a configured budget of "
            f"{budget} (DocsConfig.core_max_lines); docs_trimmer fails local-gates"
        )
