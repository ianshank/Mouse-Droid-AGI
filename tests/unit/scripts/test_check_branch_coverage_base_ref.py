"""PR #105b B.4 — self-validation tests for ``scripts/check_branch_coverage.py``.

The script's base-ref autodetect was the silent footgun that bit PR #104:
without ``GITHUB_BASE_REF`` set, local invocations returned "no changed
src/mousedroid Python files detected" instead of the actual per-file
coverage data. The B.3 fix extended the candidate chain with three
local-dev fallbacks (upstream-tracking branch, ``origin/HEAD``,
``COVERAGE_FALLBACK_BASE_REF`` env-var default ``origin/main``). These
tests pin every leg of the chain via tmp-dir git repos so a future
refactor cannot silently re-introduce the gap.

Test isolation: each test stands up its own throwaway git repo inside
``tmp_path`` (pattern from ``tests/integration/test_sync_jetson_overlay.py``).
The repo's HEAD + remote branches are seeded with a minimal commit so the
candidate chain has something to resolve against. We never touch the real
host repo's git state.

Tier rationale (PR-105b harden gap-fix #8): the changes are pure
script-level utility code with no orchestrator / hardware / network
surface, so unit tests are the canonical tier. Integration / e2e /
property / hardware tiers are N/A — formally considered + declined.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest


@pytest.fixture(autouse=True)
def _isolate_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub inherited ``GIT_*`` context vars for every test in this module.

    A git hook (e.g. the repo's pre-commit branch-coverage gate) exports
    ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE`` into the environment
    of everything it runs — including this suite. If they leak into the
    sandbox-repo ``git`` subprocesses (here AND inside the script under test),
    ``git`` targets the REAL repository, which has no worktree in the hook
    context, and fails with ``fatal: this operation must be run in a work
    tree``. Scrubbing them keeps each throwaway repo self-contained whether
    the suite runs from a shell or inside a git hook. ``monkeypatch`` restores
    the original environment after each test automatically.
    """
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_SCRIPT_PATH: Final[Path] = _REPO_ROOT / "scripts" / "check_branch_coverage.py"

# ---------------------------------------------------------------------------
# Session-scoped autouse fixture — strips git plumbing env vars that are
# injected by the pre-commit hook so that sandbox git repos created in
# tmp_path are truly isolated from the outer repo's git state.
# ---------------------------------------------------------------------------
_GIT_PLUMBING_ENV_KEYS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)


@pytest.fixture(autouse=True)
def _clear_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset git plumbing env vars for every test in this module.

    When tests are invoked from inside a pre-commit hook the hook sets
    ``GIT_DIR`` (and sometimes ``GIT_WORK_TREE``) so that git commands run
    against the correct worktree.  Without clearing these, sandbox repos
    created via ``tmp_path`` inherit the outer repo's git state, causing
    ``git add / commit`` to operate on the wrong directory and causing
    ``_local_dev_base_candidates()`` to read the outer tracking branch
    instead of the sandbox's (empty) remote config.
    """
    for key in _GIT_PLUMBING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Module-loading helper — import the script as a Python module so we can
# unit-test private functions directly without spawning a subprocess.
# ---------------------------------------------------------------------------


def _load_coverage_script_module() -> object:
    """Spec-load ``scripts/check_branch_coverage.py`` as a Python module.

    The script lives outside ``src/`` so it's not on ``sys.path`` by
    default. ``spec_from_file_location`` lets us reach in without
    polluting the project's package layout. Mirrors the helper pattern
    in ``tests/unit/tools/test_dashboard_proxy.py``.
    """
    spec = importlib.util.spec_from_file_location("check_branch_coverage", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tmp-git-repo fixture — creates a self-contained git sandbox with
# configurable remote refs so each test can exercise one leg of the
# autodetect chain in isolation.
# ---------------------------------------------------------------------------


def _git_checked(args: list[str], cwd: Path) -> str:
    """Run a git subprocess in ``cwd`` and return stdout — raises on failure.

    Named ``_git_checked`` (not ``_git``) per PR-105b harden gap-fix #6:
    distinguishes this test helper's ``check=True`` exception-raising
    contract from the script-under-test's ``_run`` helper which uses
    ``check=False`` and returns the full ``CompletedProcess`` for caller
    inspection. Both wrap ``git``; only the error handling differs.

    The argv is a fixed list from the test (never user input); the
    ``shutil.which``-equivalent ``git`` resolution is the system's
    default. No noqa S603 needed — pytest already grants the test tree a
    broad subprocess waiver via ``pyproject.toml`` per-file-ignores.

    ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE`` are stripped from
    the subprocess environment so that ``git commit`` calls inside the sandbox
    repo do NOT inherit the outer repo's hook environment.  Without this,
    running the tests from inside a pre-commit hook (which sets ``GIT_DIR``)
    causes the sandbox's ``git commit`` to trigger the outer repo's
    ``scripts/check_branch_coverage.py`` hook, which fails because that script
    doesn't exist under the temp dir.
    """
    clean_env = {k: v for k, v in os.environ.items() if k not in _GIT_PLUMBING_ENV_KEYS}
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True, env=clean_env
    )
    return result.stdout


def _make_sandbox_repo(
    tmp_path: Path,
    *,
    with_upstream: bool = False,
    with_origin_head: bool = False,
    with_origin_main: bool = False,
) -> Path:
    """Build a throwaway git repo with optional remote refs.

    Args:
        tmp_path: pytest-supplied tmp dir (auto-cleaned).
        with_upstream: configure the current branch to track ``origin/<branch>``.
        with_origin_head: set ``refs/remotes/origin/HEAD`` symbolic ref.
        with_origin_main: create a literal ``origin/main`` remote branch.

    Returns:
        Path to the local repo root.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_checked(["init", "--initial-branch=feature"], repo)
    _git_checked(["config", "user.email", "test@example.invalid"], repo)
    _git_checked(["config", "user.name", "Test"], repo)
    # Throwaway repos must never sign: a host with global commit.gpgsign=true
    # (e.g. a managed CI/dev environment with a signing server) would otherwise
    # make these sandbox `git commit` calls fail with exit 128.
    _git_checked(["config", "commit.gpgsign", "false"], repo)
    (repo / "seed.txt").write_text("seed\n")
    _git_checked(["add", "seed.txt"], repo)
    _git_checked(["commit", "-m", "seed"], repo)

    if with_upstream or with_origin_head or with_origin_main:
        bare = tmp_path / "remote.git"
        _git_checked(["init", "--bare", str(bare)], repo)
        _git_checked(["remote", "add", "origin", str(bare)], repo)
        _git_checked(["push", "origin", "feature"], repo)

    if with_origin_main:
        _git_checked(["push", "origin", "feature:main"], repo)

    if with_origin_head:
        # Point origin/HEAD at the freshly-pushed branch so the script's
        # ``git symbolic-ref refs/remotes/origin/HEAD`` candidate resolves.
        # Operators usually get this automatically via ``git clone``; the
        # ``set-head -a`` here mimics that initial setup.
        _git_checked(["remote", "set-head", "origin", "feature"], repo)

    if with_upstream:
        _git_checked(["branch", "--set-upstream-to=origin/feature"], repo)

    return repo


# ---------------------------------------------------------------------------
# Direct-function tests (no subprocess; faster + easier failure attribution)
# ---------------------------------------------------------------------------


def test_local_dev_candidates_in_priority_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Upstream → origin/HEAD → origin/main fallback order is preserved.

    With ALL three legs configured, the script must return upstream-tracking
    first (the most-specific signal of "the branch this was created from").
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(
        tmp_path, with_upstream=True, with_origin_head=True, with_origin_main=True
    )
    monkeypatch.chdir(repo)
    candidates = mod._local_dev_base_candidates()  # type: ignore[attr-defined]
    # Upstream resolves first; origin/feature (mirror of HEAD) lands second;
    # origin/main appears last. Exact ordering matters because the script
    # picks the FIRST one ``git rev-parse --verify`` accepts.
    assert candidates[0] == "origin/feature"
    assert "origin/main" in candidates
    # No duplicates (dedupe preserves the first occurrence).
    assert len(candidates) == len(set(candidates))


def test_local_dev_candidates_without_remotes_yields_origin_main_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repo with NO remotes still returns ``origin/main`` as a final fallback.

    The literal ``"origin/main"`` is always present even when the prior
    legs returned nothing — the downstream ``_first_valid_base_ref`` then
    rejects it via ``git rev-parse --verify`` and the script falls
    through to its working-tree-diff path. This keeps the candidate list
    DETERMINISTIC for test assertion purposes.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)  # no remote
    monkeypatch.chdir(repo)
    candidates = mod._local_dev_base_candidates()  # type: ignore[attr-defined]
    assert candidates == ["origin/main"]


def test_cli_base_ref_takes_priority_over_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit ``--base-ref`` wins over ``GITHUB_BASE_REF`` env.

    The CLI value (and its ``origin/``-prefixed twin per the existing
    GH-Actions plain-name normalization) appear before any env-derived
    candidate. The env value is never consulted when the function arg
    is supplied — both pre-existing behaviour, this test pins it.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_BASE_REF", "env-value")
    candidates = mod._git_base_candidates("cli-value")  # type: ignore[attr-defined]
    # Both the prefixed and literal forms of the CLI value land first
    # (existing normalization at _git_base_candidates).
    assert candidates[:2] == ["origin/cli-value", "cli-value"]
    # Env value never consulted when CLI arg is non-empty.
    assert "env-value" not in candidates
    assert "origin/env-value" not in candidates


def test_env_base_ref_falls_back_to_origin_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plain branch name in ``GITHUB_BASE_REF`` gets the ``origin/`` prefix.

    GitHub Actions exposes the base ref WITHOUT the ``origin/`` prefix
    (it's just the branch name). The script's existing prefixing logic
    must survive the new local-dev candidate append.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)
    monkeypatch.chdir(repo)
    candidates = mod._git_base_candidates("main")  # type: ignore[attr-defined]
    # ``origin/main`` (the prefixed form) appears before the plain ``main``.
    assert candidates.index("origin/main") < candidates.index("main")


def test_first_valid_base_ref_logs_resolved_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script emits an stderr line naming the resolved base ref.

    The silent-resolution behaviour was the PR-104 footgun; the
    informational line is the operator-facing fix. We assert it appears
    on stderr (not stdout — stdout is reserved for coverage data).
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path, with_origin_head=True, with_origin_main=True)
    monkeypatch.chdir(repo)
    resolved = mod._first_valid_base_ref(None)  # type: ignore[attr-defined]
    assert resolved is not None
    out_err = capsys.readouterr()
    assert "resolved base ref:" in out_err.err
    assert resolved in out_err.err


def test_first_valid_base_ref_logs_when_none_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When NO candidate resolves the script still emits a diagnostic line.

    Repo with no remotes → ``origin/main`` candidate fails ``rev-parse
    --verify`` → script returns ``None`` AND prints the candidate list to
    stderr so the operator can see what was tried. Without this trace the
    script's silent fallback to working-tree diff is invisible.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)  # no remote
    monkeypatch.chdir(repo)
    resolved = mod._first_valid_base_ref(None)  # type: ignore[attr-defined]
    assert resolved is None
    out_err = capsys.readouterr()
    assert "no candidate base ref resolved" in out_err.err
    assert "origin/main" in out_err.err


# ---------------------------------------------------------------------------
# PR-105b harden-fix tests (gap-fixes #1, #2, #4)
# ---------------------------------------------------------------------------


def test_fallback_base_ref_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``COVERAGE_FALLBACK_BASE_REF`` env overrides the hardcoded fallback.

    PR-105b harden gap-fix #2: clones whose remote default branch is
    ``master`` / ``develop`` / ``trunk`` previously fell off the chain
    because ``origin/main`` was hardcoded. Operators now flip the env
    var; this test pins that path.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)  # no remote — only the fallback fires
    monkeypatch.chdir(repo)
    monkeypatch.setenv("COVERAGE_FALLBACK_BASE_REF", "origin/master")
    candidates = mod._local_dev_base_candidates()  # type: ignore[attr-defined]
    # The env-var value takes the slot the hardcoded default used to occupy.
    assert candidates == ["origin/master"]


def test_fallback_base_ref_defaults_to_origin_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without env override the chain still ends in ``origin/main``.

    Backwards-compatibility guard: legacy operators who never set
    ``COVERAGE_FALLBACK_BASE_REF`` get the same final-leg behaviour as
    before the harden-fix.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("COVERAGE_FALLBACK_BASE_REF", raising=False)
    candidates = mod._local_dev_base_candidates()  # type: ignore[attr-defined]
    assert candidates == ["origin/main"]


def test_script_tag_constant_used_by_stderr_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``_SCRIPT_TAG`` module-level constant prefixes every stderr line.

    PR-105b harden gap-fix #4: the previously-hardcoded
    ``"[check_branch_coverage]"`` literal appeared in two separate
    ``print`` calls. Centralising it lets renames/repurposes update one
    place. This test pins the constant is actually used.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path, with_origin_head=True, with_origin_main=True)
    monkeypatch.chdir(repo)
    tag = mod._SCRIPT_TAG  # type: ignore[attr-defined]
    assert tag == "[check_branch_coverage]"
    # Trigger both stderr-emitting branches in sequence + confirm the tag
    # appears in each.
    mod._first_valid_base_ref(None)  # type: ignore[attr-defined]  # resolved path
    out_err_ok = capsys.readouterr().err
    assert tag in out_err_ok

    # Now exercise the "no candidate resolves" path in a fresh repo +
    # assert the same tag prefixes that diagnostic too. ``_make_sandbox_repo``
    # expects its parent dir to exist (it ``mkdir``s ``<parent>/repo``
    # one level deeper) so create the bare-test parent explicitly.
    bare_parent = tmp_path / "bare"
    bare_parent.mkdir()
    bare_repo = _make_sandbox_repo(bare_parent)
    monkeypatch.chdir(bare_repo)
    mod._first_valid_base_ref(None)  # type: ignore[attr-defined]
    out_err_fail = capsys.readouterr().err
    assert tag in out_err_fail


def test_resolved_base_threaded_through_avoids_double_print(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``_changed_source_files`` + ``_changed_line_map`` accept a resolved base ref.

    PR-105b harden gap-fix #1: pre-harden, both helpers called
    ``_first_valid_base_ref`` internally, which emitted the
    ``resolved base ref`` stderr line TWICE per ``main()`` invocation.
    Now ``main()`` resolves once and threads via the ``resolved_base``
    kwarg, suppressing the second emission. This test exercises the
    threading path directly and confirms NO stderr line fires when the
    resolved value is passed.
    """
    mod = _load_coverage_script_module()
    repo = _make_sandbox_repo(tmp_path, with_origin_head=True, with_origin_main=True)
    monkeypatch.chdir(repo)

    # Resolve once + capture the stderr line.
    resolved = mod._first_valid_base_ref(None)  # type: ignore[attr-defined]
    first_log = capsys.readouterr().err
    assert "resolved base ref:" in first_log
    assert resolved is not None

    # Now call the threaded variant — must produce NO additional
    # ``resolved base ref:`` line because the helper short-circuits when
    # ``resolved_base`` is supplied.
    _ = mod._changed_source_files(None, resolved_base=resolved)  # type: ignore[attr-defined]
    second_log = capsys.readouterr().err
    assert "resolved base ref:" not in second_log


def test_branch_coverage_dir_exemptions_are_pinned() -> None:
    """Growing ``_ALLOWED_DIR_PREFIXES`` is a deliberate, reviewed decision.

    Mirrors ``test_hardcoded_value_dir_exemptions_are_pinned`` in
    ``tests/unit/scripts/test_check_no_hardcoded_values.py`` (same
    reasoning: a same-PR 1-file-to-many module split has no git rename
    correspondence, so every relocated line reads as newly "changed"
    against the pre-split base — here that meant a large file's blended
    branch-coverage average no longer hides an under-tested function once
    it lands in its own much-smaller file). A new entry silencing an
    unrelated finding should fail this test until deliberately added here.
    """
    mod = _load_coverage_script_module()
    assert mod._ALLOWED_DIR_PREFIXES == (  # type: ignore[attr-defined]
        "src/mousedroid/config/schema/",
        "src/mousedroid/telemetry/metrics/",
        "src/mousedroid/telemetry/server/",
        "src/mousedroid/validation/runtime/",
        "src/mousedroid/factory/",
        "src/mousedroid/orchestrator/_",
    )


def test_is_exempted_from_branch_gate_matches_prefix_precisely() -> None:
    """`_is_exempted_from_branch_gate` matches the split's own files, not siblings.

    `main()` uses this predicate to let an exempted path's real (possibly
    low) percentage still print while skipping it from `failures` --
    transparency over silence, unlike a bare path exclusion that would
    drop the file from the report entirely. This test only pins the
    predicate itself; ``factory/health.py`` and
    ``orchestrator/_lifecycle_mixin.py`` are exactly the two files
    ``check_branch_coverage.py --min 90`` flagged below threshold when
    this exemption was added (ADR-017), confirming the predicate covers
    the real gate failures it exists for.
    """
    mod = _load_coverage_script_module()
    assert mod._is_exempted_from_branch_gate(  # type: ignore[attr-defined]
        "src/mousedroid/factory/health.py"
    )
    assert mod._is_exempted_from_branch_gate(  # type: ignore[attr-defined]
        "src/mousedroid/orchestrator/_lifecycle_mixin.py"
    )
    assert not mod._is_exempted_from_branch_gate(  # type: ignore[attr-defined]
        "src/mousedroid/orchestrator/autonomous.py"
    )
    assert not mod._is_exempted_from_branch_gate(  # type: ignore[attr-defined]
        "src/mousedroid/orchestrator/orchestrator.py"
    )
    assert not mod._is_exempted_from_branch_gate("src/mousedroid/safety/monitor.py")  # type: ignore[attr-defined]
