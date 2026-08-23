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
import signal
import subprocess
import sys
import time
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

# Same claim as _PIN_TEST, but slow enough that a signal reliably lands while
# the reverted-source run is in flight -- the window where an interrupt used to
# leave the tree at the base ref.
_SLOW_PIN_TEST = """\
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def test_pkg_a_marker_is_new_slowly() -> None:
    time.sleep(4)
    assert "NEW_A" in (_ROOT / "pkg_a" / "config.py").read_text(encoding="utf-8")
"""


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
    (repo / "test_slow_pin.py").write_text(_SLOW_PIN_TEST, encoding="utf-8")
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
        result = _run(fixture_repo, tests="test_pin.py")
        assert result.returncode == 0, (
            "precondition: without this the assertion below is satisfied by a "
            "script that bails before the revert -- verified, a two-line "
            "`exit 2` stub passes it.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
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


class TestNothingIsLeftReverted:
    """Every abort path must leave the tree exactly as it was found.

    The tool's one non-negotiable property. Each case here reproduces a real
    way it was violated: a snapshot that silently did not happen, and a
    non-zero pytest exit that means "no verdict" rather than "the pin fired".
    """

    def test_a_directory_path_is_refused_before_anything_is_touched(
        self, fixture_repo: Path
    ) -> None:
        """`cp` without -r silently skipped the directory, then the revert ran.

        The reproduction: `--paths pkg_a` left `pkg_a/config.py` holding the
        base-ref content with `git status` showing ` M`, while the script
        printed "Pin failed as required" and exited 0. A safe-restore tool
        corrupting the tree and reporting success.
        """
        result = _run(fixture_repo, tests="test_pin.py", paths="pkg_a")
        assert result.returncode == 2, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "directories" in result.stderr
        assert _read_all(fixture_repo) == _NEW, "a refused invocation must not revert anything"

    def test_a_nonexistent_path_is_refused(self, fixture_repo: Path) -> None:
        result = _run(fixture_repo, tests="test_pin.py", paths="pkg_a/config.py no/such.py")
        assert result.returncode == 2
        assert "not an existing file" in result.stderr
        assert _read_all(fixture_repo) == _NEW

    def test_a_pytest_run_with_no_verdict_is_not_proof(self, fixture_repo: Path) -> None:
        """Only pytest rc=1 is evidence. rc=4/5 are non-zero for other reasons.

        `if [[ rc -eq 0 ]]` alone accepted a usage error, a collection error or
        an empty selection as "the pin fails without the change" -- the exact
        illusion the tool exists to dispel, in the tool itself.
        """
        result = _run(fixture_repo, tests="test_does_not_exist.py")
        assert result.returncode == 2, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "no verdict" in result.stderr
        assert _read_all(fixture_repo) == _NEW, "the tree must be restored on this path too"


class TestInterruptSafety:
    """Ctrl-C during the reverted-source run unwinds cleanly and says so.

    `restore` is reached from two traps. As an EXIT handler it must `return`
    the original status; as an INT/TERM handler it must `exit`. A handler that
    returns hands control back to the line after the interrupted command, so
    the script carries on with its snapshot already deleted.

    Measured both ways against this fixture: with `exit 130` the run reports
    130 and both files are restored; with `return 130` the files are *also*
    restored -- `restore` is idempotent and complete, so the tree is never the
    casualty -- but the resumed script reaches the post-verdict copy loop,
    finds no snapshot, and exits **3**, whose documented meaning is "restore
    failed, working tree needs manual attention". A clean tree reported as
    corrupt sends an operator hunting a problem that does not exist, so the
    exit code is what this test pins.
    """

    def test_sigint_mid_run_restores_the_tree_and_reports_130(self, fixture_repo: Path) -> None:
        env = dict(os.environ)
        env["MOUSEDROID_PYTHON"] = sys.executable
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        proc = subprocess.Popen(
            [
                "bash",
                str(fixture_repo / "scripts" / _SCRIPT.name),
                "--from",
                "HEAD~1",
                "--paths",
                "pkg_a/config.py pkg_b/config.py",
                "--tests",
                "test_slow_pin.py",
            ],
            cwd=fixture_repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert proc.stdout is not None
            deadline = time.monotonic() + 60
            for line in proc.stdout:
                if "expecting FAILURE" in line:
                    break
                if time.monotonic() > deadline:  # pragma: no cover - hang guard
                    pytest.fail("script never reached the reverted-source run")
            else:  # pragma: no cover - the script exited before reverting
                pytest.fail("script exited before the reverted-source run")
            # The revert has happened and pytest is sleeping inside it.
            time.sleep(1.0)
            proc.send_signal(signal.SIGINT)
            proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:  # pragma: no cover - only on an unexpected hang
                proc.kill()
                proc.communicate(timeout=30)

        assert _read_all(fixture_repo) == _NEW, (
            "an interrupt mid-revert must leave the tree restored"
        )
        assert proc.returncode == 130, (
            "an interrupted run must report SIGINT, not a restore failure. A "
            "signal handler that `return`s resumes the script, which then "
            f"finds no snapshot and exits 3; got {proc.returncode}"
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
