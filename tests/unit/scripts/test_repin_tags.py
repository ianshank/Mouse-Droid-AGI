"""Unit: ``scripts/repin_tags.sh`` creates the tags that protect pinned SHAs.

Mirrors ``test_archive_stale_branches.py``'s bare-remote fixture: a real local
bare repo as ``origin`` plus a clone, so the script's ``git ls-remote`` /
``git push`` paths are exercised for real rather than mocked. The two scripts
are two halves of one contract -- one refuses to delete a branch carrying an
unprotected pin, the other removes the need for that refusal -- so the last
test here drives both against the same synthetic repo.

Unit tier per ``.claude/skills/test-tier-mirror/SKILL.md``: this exercises one
script in isolation against a throwaway repo, with no application code and no
factory involved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
_SCRIPT_NAME = "repin_tags.sh"
_SIBLING_NAME = "archive_stale_branches.sh"
_PINNED_BRANCH = "carries-the-pin"

_FEATURES_TEMPLATE = """\
features:
  - id: "F-001"
    name: "A feature with a pin"
    status: "done"
    validation_command: "true"
    implemented_in: "{sha}"
"""

_DEPLOY_TEMPLATE = '{{\n  "platform": "jetson",\n  "sha": "{sha}"\n}}\n'


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``, returning stdout and failing loudly on error."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=repin test",
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


# The script creates ANNOTATED tags, and git refuses to write a tag object
# without a tagger identity ("Committer identity unknown"). ``_git`` supplies
# one via ``-c`` for the calls the test makes itself, but the script runs git
# in its own subprocess, so it needs the identity through the environment.
#
# Without this the suite passes on any developer machine with a global
# user.name and fails on a bare CI runner, which is exactly what happened:
# green locally, "fatal: empty ident name" on the GitHub runner. Reproduced
# with `HOME=<empty> GIT_CONFIG_GLOBAL=/dev/null` -- 4 tests red.
#
# Fixed here rather than in the script on purpose: needing an identity to
# write a tag object is correct git behaviour, and a real operator running
# --push has one configured. Having the script inject an identity would forge
# the tagger.
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "repin test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "repin test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def _run(
    clone: Path,
    *args: str,
    script: str = _SCRIPT_NAME,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke a script inside ``clone``; never raises on nonzero exit."""
    return subprocess.run(
        ["bash", str(clone / "scripts" / script), *args],
        cwd=clone,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **_GIT_IDENTITY, **(env or {})},
    )


def _remote_tags(clone: Path) -> set[str]:
    out = _git(clone, "ls-remote", "--tags", "origin")
    return {
        line.split("refs/tags/", 1)[1].removesuffix("^{}")
        for line in out.splitlines()
        if "refs/tags/" in line
    }


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    """A clone whose ``origin`` is a local bare repo, with one unprotected pin.

    ``main`` declares the pin in both pin files; the pinned commit itself lives
    ONLY on a side branch, so nothing but that branch keeps it reachable --
    the exact situation this script exists to fix. Returns ``(clone, sha)``.
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

    _git(work, "checkout", "-q", "-b", _PINNED_BRANCH)
    (work / "shipped.txt").write_text("the feature\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "the commit the pins point at")
    sha = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", "-u", "origin", _PINNED_BRANCH)

    _git(work, "checkout", "-q", "main")
    (work / "features.yaml").write_text(_FEATURES_TEMPLATE.format(sha=sha), encoding="utf-8")
    deployments = work / "deployments"
    deployments.mkdir()
    (deployments / "jetson-image.json").write_text(
        _DEPLOY_TEMPLATE.format(sha=sha), encoding="utf-8"
    )
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "declare the pins")
    _git(work, "push", "-q", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    shutil.copytree(_SCRIPTS, clone / "scripts", dirs_exist_ok=True)
    return clone, sha


def test_dry_run_reports_the_plan_and_changes_nothing(repo: tuple[Path, str]) -> None:
    """Default mode names the tags it would create without creating any."""
    clone, sha = repo
    before = _remote_tags(clone)

    result = _run(clone)

    assert result.returncode == 0, result.stderr
    assert f"deployments/jetson-image-{sha}" in result.stdout
    assert f"features/{sha}" in result.stdout
    assert "DRY RUN" in result.stdout
    # The load-bearing half: a dry run that quietly mutated the remote would
    # still print exactly the text asserted above.
    assert _remote_tags(clone) == before
    assert _git(clone, "tag", "--list") == "", "a local tag leaked from a dry run"


def test_push_creates_both_tag_families_on_the_remote(repo: tuple[Path, str]) -> None:
    """``--push`` puts real annotated tags on the remote, named by full SHA."""
    clone, sha = repo

    result = _run(clone, "--push")

    assert result.returncode == 0, result.stderr
    tags = _remote_tags(clone)
    assert f"deployments/jetson-image-{sha}" in tags
    assert f"features/{sha}" in tags
    # Annotated, not lightweight: the reason travels with the repo. `cat-file
    # -t` on the tag ref returns "tag" only for an annotated object.
    assert _git(clone, "cat-file", "-t", f"refs/tags/features/{sha}") == "tag"


def test_rerun_is_idempotent(repo: tuple[Path, str]) -> None:
    """A second run finds both pins already covered and creates nothing new."""
    clone, _sha = repo
    first = _run(clone, "--push")
    assert first.returncode == 0, first.stderr
    tags_after_first = _remote_tags(clone)

    second = _run(clone, "--push")

    assert second.returncode == 0, second.stderr
    assert "0 tag(s) to create" in second.stdout
    assert "already reachable from remote tag" in second.stdout
    assert _remote_tags(clone) == tags_after_first


def test_a_differently_named_existing_tag_counts_as_coverage(repo: tuple[Path, str]) -> None:
    """Coverage is about reachability, not about matching this script's naming.

    An operator who already tagged the commit by hand must not have a second,
    redundant tag created over the top of it.
    """
    clone, sha = repo
    _git(clone, "fetch", "-q", "origin", _PINNED_BRANCH)
    _git(clone, "tag", "-a", "someone-elses-name", sha, "-m", "manual")
    _git(clone, "push", "-q", "origin", "refs/tags/someone-elses-name")

    result = _run(clone)

    assert result.returncode == 0, result.stderr
    assert "0 tag(s) to create" in result.stdout
    assert "someone-elses-name" in result.stdout


@pytest.mark.parametrize(
    ("bad_sha", "why"),
    [
        ("not-a-sha", "not hex at all"),
        ("deadbeef", "hex but too short"),
        ("0123456789abcdef0123456789abcdef012345678", "41 chars"),
        ("0123456789ABCDEF0123456789abcdef01234567", "uppercase"),
    ],
)
def test_a_malformed_deploy_pin_is_rejected_by_the_format_check(
    repo: tuple[Path, str], bad_sha: str, why: str
) -> None:
    """A deploy pin that is not a full lowercase-hex SHA fails loudly.

    The deploy-pin extraction has no format guarantee -- python prints whatever
    the JSON value happens to be -- and that value flows into a tag name and a
    git argument. Silently skipping it would report a plan that looks complete
    while leaving the pin unprotected.

    Asserts the SPECIFIC format-check message rather than just a nonzero exit.
    A bare ``returncode != 0`` passed even with the format check deleted --
    these values also fail the later ``cat-file`` resolve check, so the weaker
    assertion proved nothing about the check it names. Verified by reverting
    ``_is_full_sha`` to ``return 0``: with the loose assertion 5 of 6 security
    tests stayed green; with this one they go red.
    """
    clone, _sha = repo
    (clone / "deployments" / "jetson-image.json").write_text(
        _DEPLOY_TEMPLATE.format(sha=bad_sha), encoding="utf-8"
    )

    result = _run(clone)

    assert result.returncode != 0, f"{why!r} was accepted: {result.stdout}"
    assert "not a 40-char lowercase hex SHA" in result.stderr, (
        f"{why!r} was rejected, but by the wrong check: {result.stderr}"
    )
    assert _remote_tags(clone) == set(), "a rejected run still touched the remote"


def test_an_empty_deploy_pin_is_rejected_before_the_format_check(
    repo: tuple[Path, str],
) -> None:
    """An empty ``sha`` value gets its own diagnostic, not the format one.

    Split from the parametrized cases above because it exits through a
    different branch: python prints an empty line, so the emptiness guard
    fires first and names the file rather than the format.
    """
    clone, _sha = repo
    (clone / "deployments" / "jetson-image.json").write_text(
        _DEPLOY_TEMPLATE.format(sha=""), encoding="utf-8"
    )

    result = _run(clone)

    assert result.returncode != 0
    assert 'could not read ["sha"]' in result.stderr


def test_a_deploy_pin_naming_a_branch_is_rejected_even_though_git_resolves_it(
    repo: tuple[Path, str],
) -> None:
    """The case format validation exists for, and that `cat-file -e` misses.

    ``git cat-file -e main`` succeeds -- git happily resolves branch names,
    ``HEAD``, and ``main~3``. So a reachability check alone would tag whatever
    that ref points at *as though it were the pinned commit*, quietly
    protecting the wrong object and leaving the real pin exposed. Only the
    40-hex format check catches it.
    """
    clone, _sha = repo
    (clone / "deployments" / "jetson-image.json").write_text(
        _DEPLOY_TEMPLATE.format(sha="main"), encoding="utf-8"
    )
    # Establish the premise rather than asserting it in the abstract: git
    # really does resolve this, so the format check is what does the work.
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", "main^{commit}"],
            cwd=clone,
            capture_output=True,
            timeout=60,
        ).returncode
        == 0
    )

    result = _run(clone)

    assert result.returncode != 0
    assert "not a 40-char lowercase hex SHA" in result.stderr
    assert _remote_tags(clone) == set()


def test_an_unresolvable_pin_fails_rather_than_reporting_success(
    repo: tuple[Path, str],
) -> None:
    """A well-formed SHA that no longer exists is an error, not a silent skip.

    ``0`` * 40 passes the format check but resolves to nothing. Reporting
    "0 tags to create" for it would read as success while the pin is, in fact,
    already unrecoverable.
    """
    clone, _sha = repo
    (clone / "deployments" / "jetson-image.json").write_text(
        _DEPLOY_TEMPLATE.format(sha="0" * 40), encoding="utf-8"
    )

    result = _run(clone)

    assert result.returncode != 0
    assert "does not resolve" in result.stderr


def test_remote_env_override_is_honoured(repo: tuple[Path, str], tmp_path: Path) -> None:
    """``REMOTE`` selects the remote, mirroring archive_stale_branches.sh."""
    clone, sha = repo
    second = tmp_path / "second.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(second)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    _git(clone, "remote", "add", "upstream", str(second))
    _git(clone, "push", "-q", "upstream", "main")

    result = _run(clone, "--push", env={"REMOTE": "upstream"})

    assert result.returncode == 0, result.stderr
    upstream_tags = _git(clone, "ls-remote", "--tags", "upstream")
    assert f"features/{sha}" in upstream_tags
    # origin must be untouched -- the override selected a different remote.
    assert _remote_tags(clone) == set()


def test_repin_then_archive_reports_the_pin_as_tag_protected(repo: tuple[Path, str]) -> None:
    """The cross-script contract: tagging frees the carrier branch for deletion.

    Before ``repin_tags.sh`` runs, ``archive_stale_branches.sh`` must protect
    the branch carrying the pin. After it runs, the same pin is reachable from
    a remote tag, so the sibling reports it as needing no carrier -- which is
    the entire point of this script existing.
    """
    clone, sha = repo

    before = _run(clone, script=_SIBLING_NAME)
    assert before.returncode == 0, before.stderr
    assert "NOT reachable from any REMOTE tag" in before.stdout

    repin = _run(clone, "--push")
    assert repin.returncode == 0, repin.stderr

    after = _run(clone, script=_SIBLING_NAME)
    assert after.returncode == 0, after.stderr
    assert "carriers not needed" in after.stdout
    assert sha[:8] in after.stdout
