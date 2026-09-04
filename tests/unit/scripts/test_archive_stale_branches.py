"""Unit tests for ``scripts/archive_stale_branches.sh``'s pin protection.

The script is destructive-but-reversible: it archives a stale branch as a tag,
then deletes the branch. Its safety net is the pin-carrier rule -- never delete
the last branch keeping a gate-critical commit reachable.

That rule originally read exactly ONE SHA, from ``deployments/jetson-image.json``.
Every ``implemented_in`` pin in ``features.yaml`` had no protection at all, so a
feature closed out on a branch and then squash-merged would lose its provenance
commit the moment the branch was cleaned up -- and the nightly
``validate.py --strict-git`` would start failing for a feature nobody touched.

These tests drive the real script in its default DRY-RUN mode against a
synthetic repo with a local bare remote. Nothing is deleted.

Also covers a second, independently-discovered bug in the same call sites:
``git branch -r --format='%(refname:short)'`` renders a configured
``refs/remotes/<remote>/HEAD`` symref as the BARE remote name ("origin"), not
"origin/HEAD" -- a real git behaviour any plain ``git clone`` produces, which
slips past a ``grep -v '^HEAD$'`` filter and reaches ``$REMOTE/$b`` as the
unresolvable "origin/origin", fatal under ``--push``. The ``repo_pair`` fixture
above happens not to trigger it (the pinned commit lives off the branch
``origin/HEAD`` resolves to); ``repo_with_stale_orphan_branch`` below does.

Skipped on Windows: the script is bash, and the fixture needs git.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "archive_stale_branches.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt"
    or shutil.which("bash") is None
    or shutil.which("git") is None
    or not _SCRIPT.exists(),
    reason="bash + git + scripts/archive_stale_branches.sh required",
)

_PINNED_BRANCH = "carries-the-feature-pin"

_FEATURES_TEMPLATE = """\
features:
  - id: "F-001"
    name: "A closed-out feature"
    status: "done"
    validation_command: "true"
    implemented_in: "{sha}"
"""

_FEATURES_NO_PIN = """\
features:
  - id: "F-001"
    name: "A feature with no pin"
    status: "todo"
    validation_command: "true"
    implemented_in: null
"""


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``, returning stdout and failing loudly on error."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=archive test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


@pytest.fixture
def repo_pair(tmp_path: Path) -> tuple[Path, str]:
    """A clone whose ``origin`` is a local bare repo.

    ``main`` carries ``features.yaml``; the pinned commit lives ONLY on
    ``carries-the-feature-pin``, so that branch is the sole thing keeping it
    reachable. Returns ``(clone, pinned_sha)``.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
        timeout=60,
    )

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "--initial-branch=main")
    (work / "README.md").write_text("seed\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "-u", "origin", "main")

    # The pinned commit exists only on this side branch.
    _git(work, "checkout", "-q", "-b", _PINNED_BRANCH)
    (work / "shipped.txt").write_text("the feature\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "the commit a feature pins")
    pinned_sha = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", "-u", "origin", _PINNED_BRANCH)

    # main declares the pin but does not contain the commit.
    _git(work, "checkout", "-q", "main")
    (work / "features.yaml").write_text(_FEATURES_TEMPLATE.format(sha=pinned_sha), encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "pin the feature")
    _git(work, "push", "-q", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    shutil.copytree(_SCRIPT.parent, clone / "scripts", dirs_exist_ok=True)
    return clone, pinned_sha


def _dry_run(clone: Path) -> str:
    """Invoke the script in its default (non-destructive) dry-run mode."""
    result = subprocess.run(
        ["bash", str(clone / "scripts" / _SCRIPT.name)],
        cwd=clone,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


class TestFeaturePinProtection:
    """features.yaml pins protect their carrier branches, same as the deploy pin."""

    def test_a_branch_carrying_a_feature_pin_is_protected(
        self, repo_pair: tuple[Path, str]
    ) -> None:
        clone, pinned_sha = repo_pair
        output = _dry_run(clone)
        assert f"pin {pinned_sha} is NOT reachable" in output, (
            "the features.yaml pin was not collected at all; only "
            f"deployments/jetson-image.json used to be.\n{output}"
        )
        assert f"PROTECTED (holds a gate-critical pin): {_PINNED_BRANCH}" in output, (
            "deleting the sole carrier of an implemented_in SHA breaks the "
            f"nightly --strict-git run for that feature.\n{output}"
        )

    def test_protection_is_driven_by_the_pin_not_the_branch_name(
        self, repo_pair: tuple[Path, str]
    ) -> None:
        """Drop the pin from features.yaml and the protection must lift.

        Without this, the assertion above would pass against a script that
        protects everything -- which is not protection, it is a no-op cleanup.
        """
        clone, _ = repo_pair
        (clone / "features.yaml").write_text(_FEATURES_NO_PIN, encoding="utf-8")
        output = _dry_run(clone)
        assert f"PROTECTED (holds a gate-critical pin): {_PINNED_BRANCH}" not in output, (
            "with no pin referencing it, that branch has nothing to protect:\n" + output
        )

    def test_a_remote_tag_lifts_the_protection(self, repo_pair: tuple[Path, str]) -> None:
        """The documented escape hatch: tag the SHA and the branch is free.

        This is the whole remediation path -- protection is contingent on the
        commit being unreachable any other way, so the tag ritual is what
        actually unblocks branch cleanup.
        """
        clone, pinned_sha = repo_pair
        _git(clone, "tag", "-a", "archive/pinned", pinned_sha, "-m", "keep reachable")
        _git(clone, "push", "-q", "origin", "archive/pinned")
        output = _dry_run(clone)
        assert f"pin {pinned_sha} is reachable from remote tag" in output, output
        assert f"PROTECTED (holds a gate-critical pin): {_PINNED_BRANCH}" not in output, (
            "once the SHA is reachable from a REMOTE tag the carrier no longer "
            "needs protecting:\n" + output
        )


_STALE_BRANCH = "truly-unrelated-history"


@pytest.fixture
def repo_with_stale_orphan_branch(tmp_path: Path) -> tuple[Path, Path]:
    """A clone with ``origin/HEAD`` configured, plus one genuinely stale branch.

    ``origin/HEAD`` is left for a plain ``git clone`` to set, the same way a
    real fresh clone would -- not hand-crafted -- so this fixture proves the
    bug against the actual condition, not a synthetic stand-in for it.

    The stale branch shares NO ancestry with ``main`` (built via
    ``checkout --orphan``), matching the script's own header comment about 74
    of the repo's real stale branches predating a history rewrite, where
    ``git branch --merged`` is structurally meaningless.

    Returns ``(clone, origin_bare_repo)``.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
        timeout=60,
    )

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "--initial-branch=main")
    (work / "README.md").write_text("seed\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "-u", "origin", "main")

    _git(work, "checkout", "-q", "--orphan", _STALE_BRANCH)
    _git(work, "rm", "-rf", "-q", ".")
    (work / "unrelated.txt").write_text("no shared history with main\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "unrelated root")
    _git(work, "push", "-q", "-u", "origin", _STALE_BRANCH)
    _git(work, "checkout", "-q", "main")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    shutil.copytree(_SCRIPT.parent, clone / "scripts", dirs_exist_ok=True)
    return clone, origin


def _assert_origin_head_is_configured(clone: Path) -> None:
    """Guard the fixture's own premise before trusting the tests below.

    If a future git version or clone flag stops setting this symref, these
    tests would pass vacuously -- for the same reason the pin-carrier tests
    above lift protection once a remote tag exists, not because the
    underlying bug is fixed.
    """
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
        cwd=clone,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    premise_failure = (
        "fixture premise violated: refs/remotes/origin/HEAD is not set the way "
        f"a plain `git clone` sets it (got {result.stdout!r}, rc={result.returncode}); "
        "the origin/HEAD regression tests would pass vacuously.\n"
        f"stderr: {result.stderr}"
    )
    assert result.returncode == 0, premise_failure
    assert result.stdout.strip() == "refs/remotes/origin/main", premise_failure


def _lists_bare_origin_as_branch(output: str) -> bool:
    """True if a printed branch line's first field is the bare "origin" token.

    Checking the first whitespace-separated field (rather than a substring or
    a date-shaped regex) is what makes this robust to the buggy line's second
    field being garbled or empty -- the date lookup for a non-existent
    "origin/origin" ref fails too, so the line the bug produces does not
    reliably contain a well-formed date to anchor a regex on.
    """
    return any(line.split()[:1] == ["origin"] for line in output.splitlines())


def _run_script(clone: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(clone / "scripts" / _SCRIPT.name), *extra_args],
        cwd=clone,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestOriginHeadSymrefHandling:
    """``git branch -r --format`` renders a configured origin/HEAD as "origin".

    Independently discovered while verifying the pin-protection fix above: a
    script modified in the same round that added ``features.yaml`` pin
    sourcing shipped this bug unnoticed, because the fixture used to develop
    that fix happens not to trigger it. Reproduced end-to-end (dry-run stray
    ``fatal:`` line, then ``--push`` exit 128 on ``git tag -f archive/origin
    origin/origin``) in disposable local fixtures before this permanent test
    was written; see the comment above ``_remote_branches()`` in the script.
    """

    def test_fixture_actually_configures_origin_head(
        self, repo_with_stale_orphan_branch: tuple[Path, Path]
    ) -> None:
        """Guard the premise: a plain `git clone` sets this symref.

        If this fails, every other test in this class passes vacuously.
        """
        clone, _origin = repo_with_stale_orphan_branch
        _assert_origin_head_is_configured(clone)

    def test_dry_run_does_not_list_the_bare_remote_name_as_a_branch(
        self, repo_with_stale_orphan_branch: tuple[Path, Path]
    ) -> None:
        clone, _origin = repo_with_stale_orphan_branch
        _assert_origin_head_is_configured(clone)
        result = _run_script(clone)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"dry run must exit 0:\n{combined}"
        assert "fatal:" not in combined, (
            "the bare 'origin' token from the HEAD symref reached a ref "
            f"resolution as 'origin/origin':\n{combined}"
        )
        assert not _lists_bare_origin_as_branch(result.stdout), (
            "'origin' (the HEAD symref, not a real branch) was listed as a "
            f"stale branch to archive:\n{result.stdout}"
        )
        assert _STALE_BRANCH in result.stdout, (
            f"the genuinely stale orphan branch should still be listed:\n{result.stdout}"
        )

    def test_push_completes_and_never_touches_the_bare_remote_name(
        self, repo_with_stale_orphan_branch: tuple[Path, Path]
    ) -> None:
        """The end-to-end proof: --push must actually archive and delete.

        Pre-fix this exits 128 on `git tag -f archive/origin origin/origin`
        before any real branch is touched -- the tool's entire reason for
        existing (bulk cleanup of genuinely stale branches).
        """
        clone, origin = repo_with_stale_orphan_branch
        _assert_origin_head_is_configured(clone)
        result = _run_script(clone, "--push")
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"--push must exit 0 against a clone with origin/HEAD configured "
            f"(rc={result.returncode}):\n{combined}"
        )
        # Scope this to the failure it exists to catch — `git tag -f
        # archive/origin origin/origin` aborting the run — rather than any
        # line containing "fatal:". git emits non-fatal diagnostics on that
        # stream: with `push.negotiate=true` set (a legitimate performance
        # setting, and the default in some managed environments) every push
        # prints "fatal: --negotiate-only needs one or more
        # --negotiation-tip=*" followed by "warning: push negotiation failed;
        # proceeding anyway with push" — and then succeeds. Matching the bare
        # substring made this test fail on developer machines while CI stayed
        # green, which is a false negative about the very tool it guards.
        # The return-code assertion above already covers "the run aborted".
        fatal_lines = [
            line
            for line in combined.splitlines()
            if line.startswith("fatal:") and "negotiat" not in line
        ]
        assert not fatal_lines, f"unexpected fatal git error(s): {fatal_lines}\n{combined}"
        assert "archive/origin" not in combined, (
            f"the bare remote-name token must never be archived as a branch:\n{combined}"
        )

        tags = subprocess.run(
            ["git", "ls-remote", "--tags", str(origin)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        assert f"refs/tags/archive/{_STALE_BRANCH}" in tags, (
            f"the genuinely stale branch should have been archived as a tag:\n{tags}"
        )
        assert "archive/origin" not in tags, tags

        heads = subprocess.run(
            ["git", "ls-remote", "--heads", str(origin)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        assert _STALE_BRANCH not in heads, (
            f"the genuinely stale branch should have been deleted after archiving:\n{heads}"
        )
        assert "refs/heads/main" in heads, "the default branch must never be deleted"


@pytest.fixture
def repo_with_orphaned_pin(tmp_path: Path) -> tuple[Path, str]:
    """A clone where a pinned SHA exists locally but has ZERO remote carriers.

    Built by pushing the pin commit on its own branch, then deleting that
    branch on the remote and running ``git fetch --prune`` locally -- the
    commit OBJECT survives (unreferenced objects are not immediately
    garbage-collected), but ``git branch -r --contains`` and
    ``git tag --contains`` both return nothing for it. This is the realistic
    trigger for the class of bug fixed alongside it: an already-orphaned pin
    is exactly the scenario ``pin-reachability-audit`` exists to catch, not a
    contrived edge case.

    Returns ``(clone, pinned_sha)``.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
        timeout=60,
    )

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "--initial-branch=main")
    (work / "README.md").write_text("seed\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "-u", "origin", "main")

    _git(work, "checkout", "-q", "-b", "soon-to-be-deleted")
    (work / "shipped.txt").write_text("the feature\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "the commit a feature pins")
    pinned_sha = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", "-u", "origin", "soon-to-be-deleted")

    _git(work, "checkout", "-q", "main")
    (work / "features.yaml").write_text(_FEATURES_TEMPLATE.format(sha=pinned_sha), encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "pin the feature")
    _git(work, "push", "-q", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    # The clone has fetched the pin commit (it exists locally) while it was
    # still on a real branch. Deleting that branch and pruning removes every
    # remote-tracking ref to it -- but not the object itself.
    _git(clone, "push", "-q", "origin", "--delete", "soon-to-be-deleted")
    _git(clone, "fetch", "-q", "--prune", "origin")

    shutil.copytree(_SCRIPT.parent, clone / "scripts", dirs_exist_ok=True)
    return clone, pinned_sha


class TestOrphanedPinDoesNotAbortTheScript:
    """A pin with zero surviving carriers must not crash the whole run.

    Under ``set -euo pipefail``, ``grep -Fvx -e "HEAD" -e "$REMOTE"`` exits 1
    when every input line is filtered out (including on completely empty
    input) -- and that pipe's exit status feeds a plain variable assignment
    (``carriers=$(... | grep ...)``), which DOES abort the script under
    ``set -e``, unlike a superficially similar ``for w in $(... | grep ...)``
    (word-splitting a command substitution is exempt from ``errexit`` --
    verified empirically; a variable assignment is not). An already-orphaned
    pin -- reachable from no remote branch and no remote tag -- produces
    exactly this all-filtered-out condition, and is a normal outcome this
    loop must process past, not a reason to abort before checking the
    remaining pins or reporting anything useful.
    """

    def test_dry_run_completes_and_reports_the_orphaned_pin(
        self, repo_with_orphaned_pin: tuple[Path, str]
    ) -> None:
        clone, pinned_sha = repo_with_orphaned_pin
        result = _run_script(clone)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"dry run must not abort just because a pin has zero surviving "
            f"carriers (rc={result.returncode}):\n{combined}"
        )
        assert f"pin {pinned_sha} is NOT reachable from any REMOTE tag" in result.stdout, (
            f"the script should still report on the orphaned pin, not exit silently:\n{combined}"
        )
