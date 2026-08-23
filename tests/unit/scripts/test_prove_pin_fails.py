"""Unit tests for ``scripts/prove_pin_fails.sh``.

The script's whole purpose is to prove a regression pin can fail *without*
losing the working tree, so its snapshot/restore path is the part that must be
tested rather than described. These tests drive the real script against a
synthetic git repository built in ``tmp_path``.

Why a synthetic repo: the script resolves its own repo root with
``cd "$(dirname "$0")/.."``, so copying it into ``<tmp>/scripts/`` points it at
the fixture repo with no test-only hook in the production script.

The headline case is two ``--paths`` entries that share a filename. Keying the
snapshot on ``$(basename)`` -- the original implementation -- silently made the
second entry overwrite the first, so restore wrote b's content over a. A tool
that corrupts the tree it promised to protect is worse than no tool.

Skipped on Windows: the script is bash, and the fixture needs git.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "prove_pin_fails.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt"
    or shutil.which("bash") is None
    or shutil.which("git") is None
    or not _SCRIPT.exists(),
    reason="bash + git + scripts/prove_pin_fails.sh required",
)

# Two files that share a basename in different directories -- the collision the
# snapshot keying must survive. Content markers are distinct so a cross-restore
# is detectable by value, not just by exit code.
_OLD = {"pkg_a/config.py": 'MARKER = "OLD_A"\n', "pkg_b/config.py": 'MARKER = "OLD_B"\n'}
_NEW = {"pkg_a/config.py": 'MARKER = "NEW_A"\n', "pkg_b/config.py": 'MARKER = "NEW_B"\n'}

# A pin over both files: red against _OLD, green against _NEW.
_PIN_TEST = """\
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def test_pkg_a_marker_is_new() -> None:
    assert "NEW_A" in (_ROOT / "pkg_a" / "config.py").read_text(encoding="utf-8")


def test_pkg_b_marker_is_new() -> None:
    assert "NEW_B" in (_ROOT / "pkg_b" / "config.py").read_text(encoding="utf-8")
"""

# A pin that passes no matter what the source says -- decoration, not a gate.
_TOOTHLESS_TEST = "def test_always_passes() -> None:\n    assert True\n"


def _git(repo: Path, *args: str) -> None:
    """Run a git command in ``repo``, failing loudly on a non-zero exit."""
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=prove-pin test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=30,
    )


def _write(repo: Path, files: dict[str, str]) -> None:
    """Write ``{relative path: content}`` into ``repo``, creating parents."""
    for rel, body in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A two-commit git repo: HEAD~1 holds _OLD, HEAD holds _NEW."""
    repo = tmp_path / "fixture-repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(_SCRIPT, repo / "scripts" / _SCRIPT.name)

    _git(repo, "init", "-q")
    _write(repo, _OLD)
    (repo / "test_pin.py").write_text(_PIN_TEST, encoding="utf-8")
    (repo / "test_toothless.py").write_text(_TOOTHLESS_TEST, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "old")

    _write(repo, _NEW)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "new")
    return repo


def _run(repo: Path, *, tests: str, paths: str = "pkg_a/config.py pkg_b/config.py"):
    """Invoke the script inside ``repo`` with a clean, inherited-python env."""
    env = dict(os.environ)
    # The nested pytest must use this interpreter; PYTEST_ADDOPTS from the
    # outer run would otherwise leak flags (e.g. -p plugins) into the fixture.
    env["MOUSEDROID_PYTHON"] = sys.executable
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [
            "bash",
            str(repo / "scripts" / _SCRIPT.name),
            "--from",
            "HEAD~1",
            "--paths",
            paths,
            "--tests",
            tests,
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _read_all(repo: Path) -> dict[str, str]:
    """Current on-disk content of the two fixture files."""
    return {rel: (repo / rel).read_text(encoding="utf-8") for rel in _NEW}


class TestSameBasenameRestore:
    """Two --paths entries sharing a filename must each restore their own bytes."""

    def test_proof_succeeds_and_both_files_survive(self, fixture_repo: Path) -> None:
        result = _run(fixture_repo, tests="test_pin.py")
        assert result.returncode == 0, (
            "the pin is red at HEAD~1 and green at HEAD, so the proof must "
            f"succeed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert _read_all(fixture_repo) == _NEW, (
            "each path must restore its OWN content. Keying the snapshot on "
            "$(basename) makes pkg_b/config.py overwrite pkg_a/config.py in "
            "the snapshot dir, and restore then writes b's bytes to both."
        )

    def test_restore_leaves_no_staged_revert(self, fixture_repo: Path) -> None:
        """`git checkout <ref> -- <paths>` stages too; restore must unstage.

        Without this the script reports a clean restore while `git status`
        shows MM, and the next commit silently ships the reverted source.
        """
        _run(fixture_repo, tests="test_pin.py")
        # Scoped to --paths on purpose: the claim is "the revert of those
        # paths is not left staged", not "the tree has no untracked files"
        # (the nested pytest run drops caches into the fixture repo).
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *_NEW],
            cwd=fixture_repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert status.stdout.strip() == "", (
            f"revert left staged/unstaged against --paths:\n{status.stdout}"
        )

    def test_toothless_pin_is_rejected_and_tree_still_restored(self, fixture_repo: Path) -> None:
        """A pin that passes against the reverted source exits 1 -- and the
        unconditional trap still restores both files."""
        result = _run(fixture_repo, tests="test_toothless.py")
        assert result.returncode == 1, (
            "a pin that passes against reverted source is decoration; the "
            f"script must reject it.\nstdout:\n{result.stdout}"
        )
        assert "PROVE-PIN FAIL" in result.stderr
        assert _read_all(fixture_repo) == _NEW, (
            "the EXIT trap must restore the tree even on the failure path"
        )


class TestInvocationGuards:
    """Bad invocations exit 2 rather than touching the tree."""

    def test_missing_arguments_exit_two(self, fixture_repo: Path) -> None:
        result = subprocess.run(
            ["bash", str(fixture_repo / "scripts" / _SCRIPT.name), "--from", "HEAD~1"],
            cwd=fixture_repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2
        assert "usage:" in result.stderr

    def test_unresolvable_ref_exits_two(self, fixture_repo: Path) -> None:
        env = dict(os.environ)
        env["MOUSEDROID_PYTHON"] = sys.executable
        result = subprocess.run(
            [
                "bash",
                str(fixture_repo / "scripts" / _SCRIPT.name),
                "--from",
                "no-such-ref-xyz",
                "--paths",
                "pkg_a/config.py",
                "--tests",
                "test_pin.py",
            ],
            cwd=fixture_repo,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2
        assert "does not resolve" in result.stderr

    def test_dirty_paths_are_refused(self, fixture_repo: Path) -> None:
        """Snapshot/restore over uncommitted work would discard it silently."""
        (fixture_repo / "pkg_a" / "config.py").write_text(
            'MARKER = "UNCOMMITTED"\n', encoding="utf-8"
        )
        result = _run(fixture_repo, tests="test_pin.py")
        assert result.returncode == 2
        assert "uncommitted changes" in result.stderr
        assert "UNCOMMITTED" in (fixture_repo / "pkg_a" / "config.py").read_text(
            encoding="utf-8"
        ), "refusing to run must leave the uncommitted edit untouched"
