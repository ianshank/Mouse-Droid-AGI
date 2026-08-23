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
