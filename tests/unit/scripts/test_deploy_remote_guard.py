"""Unit: the rover-WIP guard that stands in front of ``rsync -avz --delete``.

There is no shellcheck gate in CI, so shell safety is covered here instead —
with a REAL fixture, not string assertions about the script's source. Two
layers:

1. ``scripts/rover_wip_guard.sh`` driven directly against throwaway git
   repositories, exactly as ``test_repin_tags.py`` drives ``repin_tags.sh``
   against a bare-remote fixture.
2. ``scripts/deploy_remote.sh`` driven end-to-end with ``ssh``/``rsync``/``scp``
   shims on ``PATH``, so the wiring — refuse, or preserve *then* sync — is
   proven by whether the rsync shim was ever invoked. No rover, no SSH, no
   network, no Docker.

Unit tier per ``.claude/skills/test-tier-mirror/SKILL.md``: two scripts in
isolation against a throwaway checkout, no application code and no factory.

On the vacuous-assertion trap ``mouse-droid-deploy-repin/tasks.md:19-32``
records — 5 of 6 security tests there passed with the check deleted, because
they asserted only a nonzero exit — every refusal case below asserts the
*specific* message, and the whitespace-insensitivity case first proves its own
premise (a plain ``git diff`` of the same tree is NOT empty) so it cannot pass
by archiving nothing at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
_GUARD = _SCRIPTS / "rover_wip_guard.sh"
_DEPLOY = _SCRIPTS / "deploy_remote.sh"

# Byte-for-byte the guard ``test_repin_tags.py`` uses, and for the same reason:
# on the GitHub Windows runner ``bash`` resolves to the WSL shim, so which()
# finds it and every invocation then fails with "no installed distributions".
pytestmark = pytest.mark.skipif(
    os.name == "nt"
    or shutil.which("bash") is None
    or shutil.which("git") is None
    or shutil.which("tar") is None
    or not _GUARD.exists()
    or not _DEPLOY.exists(),
    reason="bash + git + tar + rover_wip_guard.sh + deploy_remote.sh required",
)

# The guard commits on the rover's behalf, and git refuses to write a commit
# object without an identity. Supplied through the environment because the
# script runs git in its own subprocess: without this the suite passes on any
# developer machine with a global user.name and fails on a bare CI runner.
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "wip guard test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "wip guard test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}

_WIP_DATE = "20260919"
_EXPECTED_BRANCH = f"rover/wip-{_WIP_DATE}"

# Documented in rover_wip_guard.sh's header as a contract deploy_remote.sh
# branches on, so the numbers are pinned here too.
_EXIT_USAGE = 2
_EXIT_DIRTY = 3
_EXIT_UNKNOWN = 4


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``, returning stdout and failing loudly on error."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=wip guard test",
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


def _run_guard(
    *args: str,
    env: dict[str, str] | None = None,
    stdout_to: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the guard; never raises on nonzero exit."""
    full_env = {**os.environ, **_GIT_IDENTITY, "MOUSEDROID_ROVER_WIP_DATE": _WIP_DATE}
    full_env.update(env or {})
    if stdout_to is not None:
        with stdout_to.open("wb") as handle:
            completed = subprocess.run(
                ["bash", str(_GUARD), *args],
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                env=full_env,
            )
        return subprocess.CompletedProcess(
            completed.args, completed.returncode, "", completed.stderr
        )
    return subprocess.run(
        ["bash", str(_GUARD), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=full_env,
    )


@pytest.fixture
def rover_repo(tmp_path: Path) -> Path:
    """A throwaway checkout shaped like the rover's: a committed tree with a
    subdirectory that stands in for the rsync destination."""
    repo = tmp_path / "opt-mousedroid"
    (repo / "src").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    (repo / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("rover\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "rover baseline")
    return repo


def _dirty(repo: Path) -> None:
    """Make the checkout dirty the way a rover is: an edited tracked file plus
    an untracked file that only exists on the rover."""
    (repo / "src" / "app.py").write_text("value = 1\nvalue = 2  # bench fix\n", encoding="utf-8")
    (repo / "src" / "rover_only.py").write_text("# only on the rover\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_passes_on_a_clean_target(rover_repo: Path) -> None:
    result = _run_guard("inspect", str(rover_repo / "src"))
    assert result.returncode == 0, result.stderr
    assert "target is clean" in result.stderr


def test_inspect_refuses_a_dirty_target_naming_the_files(rover_repo: Path) -> None:
    """The refusal must be identifiable, not just nonzero: a bare exit-code
    assertion would also pass if the guard died for an unrelated reason."""
    _dirty(rover_repo)
    result = _run_guard("inspect", str(rover_repo / "src"))
    assert result.returncode == _EXIT_DIRTY, result.stderr
    assert "DIRTY TARGET" in result.stderr
    assert "rsync --delete would destroy this" in result.stderr
    assert "src/app.py" in result.stderr
    assert "src/rover_only.py" in result.stderr


def test_inspect_treats_an_untracked_file_alone_as_dirty(rover_repo: Path) -> None:
    """Untracked files are the half a diff cannot carry and the half
    ``--delete`` removes, so they must count as work to preserve."""
    (rover_repo / "endurance_report.json").write_text("{}\n", encoding="utf-8")
    result = _run_guard("inspect", str(rover_repo / "src"))
    assert result.returncode == _EXIT_DIRTY, result.stderr
    assert "endurance_report.json" in result.stderr


def test_inspect_finds_the_enclosing_worktree_from_a_subdirectory(rover_repo: Path) -> None:
    """The rsync destination is a subdirectory of the checkout, so the guard has
    to walk up. Premise first: the target really is not the toplevel."""
    target = rover_repo / "src"
    toplevel = Path(_git(target, "rev-parse", "--show-toplevel"))
    assert toplevel.resolve() != target.resolve(), "fixture no longer exercises the walk-up"

    (rover_repo / "README.md").write_text("edited outside the sync target\n", encoding="utf-8")
    result = _run_guard("inspect", str(target))
    assert result.returncode == _EXIT_DIRTY, result.stderr
    assert "README.md" in result.stderr


def test_inspect_refuses_a_non_git_target_rather_than_calling_it_clean(tmp_path: Path) -> None:
    """Fail closed. The rover's known root-ownership drift makes ``git status``
    fail, and reading that as "nothing to preserve" is how work gets deleted."""
    plain = tmp_path / "not-a-checkout"
    (plain / "src").mkdir(parents=True)
    (plain / "src" / "precious.py").write_text("# unversioned rover work\n", encoding="utf-8")
    result = _run_guard("inspect", str(plain / "src"), env={"GIT_CEILING_DIRECTORIES": str(plain)})
    assert result.returncode == _EXIT_UNKNOWN, result.stderr
    assert "REFUSING" in result.stderr
    assert "cannot prove the target holds no uncommitted work" in result.stderr


def test_inspect_passes_when_the_target_does_not_exist_yet(tmp_path: Path) -> None:
    """A first-ever deploy has nothing to preserve and must not be blocked."""
    result = _run_guard("inspect", str(tmp_path / "never-deployed"))
    assert result.returncode == 0, result.stderr
    assert "nothing to preserve" in result.stderr


def test_unknown_command_and_bad_arity_are_usage_errors(rover_repo: Path) -> None:
    assert _run_guard("obliterate", str(rover_repo)).returncode == _EXIT_USAGE
    assert _run_guard("inspect").returncode == _EXIT_USAGE
    assert _run_guard("preserve", str(rover_repo)).returncode == _EXIT_USAGE
    # --archive is mandatory: preservation without an off-device artifact is
    # the failure mode 8.3b exists to close.
    assert _run_guard("preserve", str(rover_repo), "--to", "x").returncode == _EXIT_USAGE


# ---------------------------------------------------------------------------
# preserve
# ---------------------------------------------------------------------------


def test_preserve_commits_to_the_dated_wip_branch_and_leaves_a_clean_tree(
    rover_repo: Path, tmp_path: Path
) -> None:
    _dirty(rover_repo)
    archive = tmp_path / "wip.tar.gz"
    result = _run_guard("preserve", str(rover_repo / "src"), "--archive", str(archive))
    assert result.returncode == 0, result.stderr

    assert _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD") == _EXPECTED_BRANCH
    assert _git(rover_repo, "status", "--porcelain") == "", "tree must be clean after preservation"
    # The untracked rover file must be IN the commit, not merely archived --
    # otherwise the branch half of the preservation is a lie.
    tracked = _git(rover_repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src/rover_only.py" in tracked


def test_preserve_archives_status_diff_and_the_untracked_payload(
    rover_repo: Path, tmp_path: Path
) -> None:
    _dirty(rover_repo)
    archive = tmp_path / "wip.tar.gz"
    assert _run_guard("preserve", str(rover_repo / "src"), "--archive", str(archive)).returncode == 0

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        assert {
            "./status.txt",
            "./head.txt",
            "./diff-ignore-whitespace.patch",
            "./untracked.tar",
            "./untracked.txt",
        } <= names, names
        inner = tar.extractfile("./untracked.tar")
        assert inner is not None
        with tarfile.open(fileobj=inner) as untracked:
            assert "src/rover_only.py" in untracked.getnames()
            payload = untracked.extractfile("src/rover_only.py")
            assert payload is not None
            # The archive carries CONTENT, not just a filename list.
            assert payload.read().decode() == "# only on the rover\n"


def test_preserve_diff_ignores_whitespace(rover_repo: Path, tmp_path: Path) -> None:
    """Campaign step B1 asks for a whitespace-INSENSITIVE diff.

    Premise established inside the test: a plain ``git diff HEAD`` of this very
    tree is non-empty. Without that, an archived empty patch would also "pass"
    if the guard diffed nothing at all.
    """
    (rover_repo / "src" / "app.py").write_text("value    =     1\n", encoding="utf-8")
    plain_diff = _git(rover_repo, "diff", "HEAD")
    assert plain_diff, "premise: the tree really does differ from HEAD"

    archive = tmp_path / "wip.tar.gz"
    assert _run_guard("preserve", str(rover_repo / "src"), "--archive", str(archive)).returncode == 0
    with tarfile.open(archive, "r:gz") as tar:
        patch = tar.extractfile("./diff-ignore-whitespace.patch")
        assert patch is not None
        assert patch.read() == b"", "whitespace-only churn must not enter the patch"
        status = tar.extractfile("./status.txt")
        assert status is not None
        # ...but the change is still recorded, so nothing is silently dropped.
        assert "src/app.py" in status.read().decode()


def test_preserve_can_stream_the_archive_to_stdout(rover_repo: Path, tmp_path: Path) -> None:
    """``--archive -`` is how deploy_remote.sh gets the archive off the rover
    without it ever touching rover disk."""
    _dirty(rover_repo)
    streamed = tmp_path / "streamed.tar.gz"
    result = _run_guard(
        "preserve", str(rover_repo / "src"), "--archive", "-", stdout_to=streamed
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(streamed, "r:gz") as tar:
        assert "./status.txt" in tar.getnames()
    # Messages must not contaminate the tarball on stdout.
    assert "[wip-guard]" in result.stderr


def test_preserve_never_reuses_an_existing_wip_branch(rover_repo: Path, tmp_path: Path) -> None:
    """A second rescue on the same day must not land on top of the first."""
    _dirty(rover_repo)
    assert (
        _run_guard(
            "preserve", str(rover_repo / "src"), "--archive", str(tmp_path / "a.tar.gz")
        ).returncode
        == 0
    )
    first = _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD")
    (rover_repo / "src" / "second.py").write_text("# more bench work\n", encoding="utf-8")
    assert (
        _run_guard(
            "preserve", str(rover_repo / "src"), "--archive", str(tmp_path / "b.tar.gz")
        ).returncode
        == 0
    )
    second = _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert first == _EXPECTED_BRANCH
    assert second != first, "the second run must not reuse the first branch"
    assert second.startswith(_EXPECTED_BRANCH)
    # The first rescue is still reachable.
    assert _git(rover_repo, "rev-parse", "--verify", f"refs/heads/{first}")


def test_preserve_refuses_a_non_git_target(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-checkout"
    plain.mkdir()
    (plain / "precious.py").write_text("# unversioned\n", encoding="utf-8")
    result = _run_guard(
        "preserve",
        str(plain),
        "--archive",
        str(tmp_path / "x.tar.gz"),
        env={"GIT_CEILING_DIRECTORIES": str(plain)},
    )
    assert result.returncode == _EXIT_UNKNOWN, result.stderr
    assert not (tmp_path / "x.tar.gz").exists(), "no archive may be claimed for an unread tree"


# ---------------------------------------------------------------------------
# Source-level invariants
# ---------------------------------------------------------------------------


def test_neither_script_ever_runs_git_clean() -> None:
    """``git clean`` SHALL NEVER be used (delivery spec). Checked on both files
    because the guard is the only thing either script uses to touch rover git
    state."""
    for script in (_GUARD, _DEPLOY):
        text = script.read_text(encoding="utf-8")
        assert "git clean" not in text.replace("`git clean`", ""), (
            f"{script.name} must never run git clean"
        )


def test_the_guard_runs_before_the_destructive_rsync_in_source_order() -> None:
    """Ordering is the whole point: a preservation step after ``--delete``
    preserves the wreckage."""
    text = _DEPLOY.read_text(encoding="utf-8")
    guard_call = text.index("\n    preserve_remote_wip\n")
    rsync_delete = text.index("rsync -avz --delete")
    assert guard_call < rsync_delete, "preserve_remote_wip must precede rsync --delete"


# ---------------------------------------------------------------------------
# deploy_remote.sh end-to-end, with ssh/rsync/scp shimmed onto PATH
# ---------------------------------------------------------------------------

# The shim answers "ok" to every remote command EXCEPT the guard invocation,
# which it runs locally with stdin still carrying the piped guard script. So the
# real guard makes the real decision against a real checkout, while nothing
# else leaves the machine.
_SSH_SHIM = """#!/usr/bin/env bash
set -uo pipefail
cmd="${!#}"
case "$cmd" in
    *"bash -s --"*) exec bash -c "$cmd" ;;
esac
echo ok
exit 0
"""

_RSYNC_SHIM = """#!/usr/bin/env bash
set -uo pipefail
printf '%s\\n' "$@" >> "${RSYNC_MARKER}"
exit 0
"""

_SCP_SHIM = """#!/usr/bin/env bash
exit 0
"""


@pytest.fixture
def deploy_env(tmp_path: Path, rover_repo: Path) -> dict[str, str]:
    """PATH shims plus the env knobs that point deploy_remote.sh at the
    throwaway checkout instead of a rover."""
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    for name, body in (("ssh", _SSH_SHIM), ("rsync", _RSYNC_SHIM), ("scp", _SCP_SHIM)):
        path = shim_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    return {
        **os.environ,
        **_GIT_IDENTITY,
        "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "RSYNC_MARKER": str(tmp_path / "rsync-was-called.txt"),
        "MOUSEDROID_REMOTE_SRC": str(rover_repo / "src"),
        "MOUSEDROID_CONFIG_DIR": str(tmp_path / "remote-etc"),
        "MOUSEDROID_DEPLOY_ARCHIVE_DIR": str(tmp_path / "archives"),
        "MOUSEDROID_ROVER_WIP_DATE": _WIP_DATE,
    }


def _run_deploy(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_DEPLOY), "rover.invalid", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_deploy_refuses_a_dirty_rover_and_never_reaches_rsync(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    """The load-bearing case. Not "it printed a warning" — the rsync shim must
    never have been invoked at all."""
    _dirty(rover_repo)
    marker = Path(deploy_env["RSYNC_MARKER"])

    result = _run_deploy(deploy_env, "--code-only")

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "REFUSING" in combined
    assert "--confirm-dirty" in combined
    assert not marker.exists(), "rsync --delete ran against a dirty rover"
    archives = tmp_path / "archives"
    assert not archives.exists() or not list(archives.iterdir())
    # Nothing was mutated on the "rover" either: no rescue branch, work intact.
    assert _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (rover_repo / "src" / "rover_only.py").exists()


def test_deploy_with_confirmation_archives_off_device_then_syncs(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    _dirty(rover_repo)
    marker = Path(deploy_env["RSYNC_MARKER"])

    result = _run_deploy(deploy_env, "--code-only", "--confirm-dirty")

    assert result.returncode == 0, result.stdout + result.stderr
    # 1. The archive is OFF the rover -- on this machine, in the archive dir.
    archives = sorted((tmp_path / "archives").glob("rover-wip-*.tar.gz"))
    assert len(archives) == 1, archives
    with tarfile.open(archives[0], "r:gz") as tar:
        assert "./diff-ignore-whitespace.patch" in tar.getnames()
    # 2. The branch exists and the rover tree is clean.
    assert _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD") == _EXPECTED_BRANCH
    assert _git(rover_repo, "status", "--porcelain") == ""
    # 3. Only THEN did the destructive sync run.
    assert marker.exists(), "sync should proceed once the work is preserved"
    assert "--delete" in marker.read_text(encoding="utf-8")


def test_deploy_on_a_clean_rover_syncs_without_creating_an_archive(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    """Default behaviour on a clean target is unchanged -- the guard is a fence,
    not a new required ceremony."""
    marker = Path(deploy_env["RSYNC_MARKER"])

    result = _run_deploy(deploy_env, "--code-only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
    assert "--delete" in marker.read_text(encoding="utf-8")
    archives = tmp_path / "archives"
    assert not archives.exists() or not list(archives.iterdir())
    assert _git(rover_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
