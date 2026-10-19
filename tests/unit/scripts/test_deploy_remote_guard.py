"""Unit: the rover-WIP guard that stands in front of ``rsync -avz --delete``.

There is no shellcheck gate in CI, so shell safety is covered here instead —
with a REAL fixture, not string assertions about the script's source. Three
layers:

1. ``scripts/rover_wip_guard.sh`` driven directly against throwaway git
   repositories, exactly as ``test_repin_tags.py`` drives ``repin_tags.sh``
   against a bare-remote fixture.
2. ``scripts/deploy_remote.sh`` driven end-to-end with ``ssh``/``rsync``/``scp``
   shims on ``PATH``, so the wiring — refuse, or preserve *then* sync — is
   proven by whether the rsync shim was ever invoked, and the Docker-spine
   steps (restart the unit, check the container, config only after code) by
   the commands the ssh shim recorded. No rover, no SSH, no network, no Docker.
3. The rsync fence (``RSYNC_PROTECT_ARGS``): the args ``--print-rsync-args``
   emits, fed to REAL ``rsync -n --delete`` between two throwaway trees, so what
   the sync would delete is observed rather than inferred from the script's text.
   Skipped where rsync is absent.

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
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
_GUARD = _SCRIPTS / "rover_wip_guard.sh"
_DEPLOY = _SCRIPTS / "deploy_remote.sh"
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_DOCKER_UNIT = _SCRIPTS / "mousedroid-docker.service"

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
    result = _run_guard("preserve", str(rover_repo / "src"), "--archive", str(archive))
    assert result.returncode == 0, result.stderr

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
    result = _run_guard("preserve", str(rover_repo / "src"), "--archive", str(archive))
    assert result.returncode == 0, result.stderr
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
    result = _run_guard("preserve", str(rover_repo / "src"), "--archive", "-", stdout_to=streamed)
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
_SSH_SHIM_HEAD = """#!/usr/bin/env bash
set -uo pipefail
if [ -n "${SSH_LOG:-}" ]; then printf '%s\\n' "$*" >> "$SSH_LOG"; fi
cmd="${!#}"
case "$cmd" in
    *"bash -s --"*) exec bash -c "$cmd" ;;
esac
"""

_SSH_SHIM_TAIL = """echo ok
exit 0
"""


def _ssh_shim_body(fail_when_contains: str | None = None) -> str:
    """The ``ssh`` shim: ``ok`` for every remote command.

    With ``fail_when_contains`` any remote command whose arguments contain that text exits 1
    instead, which is how a test makes one probe (is the unit enabled? does the remote tree carry
    the safety floor?) come back negative while everything else keeps working. The match is on
    all the arguments, not the last: ``remote_sudo`` passes ``systemctl is-enabled <unit>`` as
    separate words, so the last one alone would be just the unit name. The guard's own
    invocation is matched first and never reaches that clause.
    """
    fail = (
        f'case "$*" in\n    *"{fail_when_contains}"*) exit 1 ;;\nesac\n'
        if fail_when_contains
        else ""
    )
    return _SSH_SHIM_HEAD + fail + _SSH_SHIM_TAIL


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
    throwaway checkout instead of a rover.

    The sync target is the checkout ROOT, as on the rover: ``REMOTE_SRC`` defaults to
    ``/opt/mousedroid``, not to the ``src`` directory below it. ``SSH_LOG`` records every
    command that crosses the ssh boundary, for the tests that assert on them.
    """
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    for name, body in (
        ("ssh", _ssh_shim_body()),
        ("rsync", _RSYNC_SHIM),
        ("scp", _SCP_SHIM),
    ):
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
        "SSH_LOG": str(tmp_path / "ssh.log"),
        "MOUSEDROID_REMOTE_SRC": str(rover_repo),
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


def _install_ssh_shim(tmp_path: Path, *, fail_when_contains: str | None = None) -> None:
    """Rewrite the ``ssh`` shim ``deploy_env`` installed.

    The shim directory is already first on ``PATH``, so replacing the file in place is all a test
    needs to make one class of remote command fail (see ``_ssh_shim_body``).
    """
    shim = tmp_path / "shims" / "ssh"
    shim.write_text(_ssh_shim_body(fail_when_contains), encoding="utf-8")
    shim.chmod(0o755)


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


# ---------------------------------------------------------------------------
# The Docker spine: restart its unit, check its container, config only after code
# ---------------------------------------------------------------------------
# The rover runs the bind-mounted checkout in a container under mousedroid-docker.service.
# A deploy that restarted only the bare-metal unit synced the tree and left the old code
# running, so the Docker unit is restarted when it is enabled and the container is checked
# afterwards. And config never lands ahead of the code that understands it: an overlay is
# written against the newest schema, and a tree that predates the plausibility floor
# (safety.battery_implausible_below_v, F-025) reads a missing battery voltage as 0.0 V and
# e-stops. Driven through the real script and the ssh shim, and asserted on what actually
# crossed the ssh boundary.


def _ssh_log(deploy_env: dict[str, str]) -> str:
    return Path(deploy_env["SSH_LOG"]).read_text(encoding="utf-8")


def test_code_only_restarts_the_docker_unit_when_enabled(deploy_env: dict[str, str]) -> None:
    result = _run_deploy(deploy_env, "--code-only")
    assert result.returncode == 0, result.stdout + result.stderr
    sent = _ssh_log(deploy_env)
    health = "docker exec -w /opt/mousedroid mousedroid python3 -m mousedroid.main --health-check"
    assert "systemctl restart mousedroid-docker" in sent
    assert health in sent
    # Checked AFTER the restart: a health check before it would be checking the old code.
    assert sent.index("systemctl restart mousedroid-docker") < sent.index(health)


def test_code_only_skips_the_docker_unit_when_absent(
    tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    _install_ssh_shim(tmp_path, fail_when_contains="is-enabled mousedroid-docker")
    result = _run_deploy(deploy_env, "--code-only")
    assert result.returncode == 0, result.stdout + result.stderr
    sent = _ssh_log(deploy_env)
    assert "is-enabled mousedroid-docker" in sent, "premise: the unit was probed"
    assert "restart mousedroid-docker" not in sent


def test_config_only_refuses_a_remote_tree_without_the_safety_floor(
    tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    """Review focus 3: an old image with a new YAML would e-stop on 0.0 V; code goes first."""
    _install_ssh_shim(tmp_path, fail_when_contains="grep -q battery_implausible_below_v")
    result = _run_deploy(deploy_env, "--config-only")
    assert result.returncode != 0
    assert "deploy code first" in result.stderr
    sent = _ssh_log(deploy_env)
    assert "battery_implausible_below_v" in sent, "premise: the remote tree was probed"
    assert "schema/reward_safety.py" in sent, "premise: the probe read the schema file"
    assert deploy_env["MOUSEDROID_CONFIG_DIR"] not in sent, "config was touched despite the refusal"


def test_config_only_installs_config_when_the_remote_tree_has_the_safety_floor(
    deploy_env: dict[str, str],
) -> None:
    """The control for the refusal above: the same run, floor present, does reach the config dir.

    Without it the refusal's "config was never touched" assertion could pass for the wrong reason.
    """
    result = _run_deploy(deploy_env, "--config-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert deploy_env["MOUSEDROID_CONFIG_DIR"] in _ssh_log(deploy_env)


def test_the_remote_container_is_the_one_compose_pins() -> None:
    """``docker exec`` needs the real name, so the constant is not a knob: compose fixes it."""
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    pinned = compose["services"]["mousedroid"]["container_name"]
    constant = re.search(r"^REMOTE_CONTAINER=(\S+)$", _DEPLOY.read_text(encoding="utf-8"), re.M)
    assert constant is not None, "deploy_remote.sh no longer declares REMOTE_CONTAINER"
    assert constant.group(1) == pinned


def test_the_unit_the_deploy_restarts_is_one_the_repo_ships() -> None:
    assert _DOCKER_UNIT.is_file()
    assert f"systemctl restart {_DOCKER_UNIT.stem}" in _DEPLOY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# One remote venv, the one the service runs (F-052 task 5.7)
# ---------------------------------------------------------------------------
# This script spelt /opt/mousedroid/venv out twice; it is one constant now. It
# is deliberately not a knob: scripts/mousedroid.service (which deploy_jetson.sh
# installs as-is) runs that venv from that directory, so a knob moving only the
# venv would reinstall and health-check one tree while the service restarted
# the other -- and report success. Asserted on the commands that actually
# crossed the ssh boundary, not on the script's text.

_SERVICE = _SCRIPTS / "mousedroid.service"


def _without_install_dir(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k != "MOUSEDROID_INSTALL_DIR"}


def _remote_commands(deploy_env: dict[str, str], tmp_path: Path, *args: str, **extra: str) -> str:
    ssh_log = tmp_path / "ssh.log"
    result = _run_deploy(
        {**deploy_env, "SSH_LOG": str(ssh_log), **extra}, *(args or ("--code-only",))
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return ssh_log.read_text(encoding="utf-8")


def _service_venv() -> str:
    """The venv the bare-metal unit runs, from its own ExecStart line."""
    match = re.search(r"^ExecStart=(\S+)/bin/python\b", _SERVICE.read_text(encoding="utf-8"), re.M)
    assert match is not None, "mousedroid.service no longer runs a venv python"
    return match.group(1)


def test_the_remote_venv_is_the_one_the_service_runs(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    venv = _service_venv()
    sent = _remote_commands(_without_install_dir(deploy_env), tmp_path)

    assert f"test -d {venv}" in sent
    assert f"{venv}/bin/pip install" in sent
    assert f"{venv}/bin/python -m mousedroid.main --health-check" in sent


def test_the_install_root_is_the_services_working_directory() -> None:
    working = re.search(r"^WorkingDirectory=(\S+)$", _SERVICE.read_text(encoding="utf-8"), re.M)
    constant = re.search(r"^REMOTE_INSTALL_DIR=(\S+)$", _DEPLOY.read_text(encoding="utf-8"), re.M)
    assert working is not None
    assert constant is not None

    assert constant.group(1) == working.group(1)
    assert _service_venv() == f"{working.group(1)}/venv"


def test_the_pcs_own_install_dir_never_retargets_the_rovers_venv(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    """``MOUSEDROID_INSTALL_DIR`` is for the scripts that run where it is set.

    docker_deploy.sh and deploy_jetson.sh read it on the machine they run on;
    an operator who set it on this PC for a local run must not move the
    rover's tree.
    """
    sent = _remote_commands(
        _without_install_dir(deploy_env), tmp_path, MOUSEDROID_INSTALL_DIR="/srv/elsewhere"
    )

    assert f"test -d {_service_venv()}" in sent
    assert "/srv/elsewhere" not in sent


def test_a_full_deploy_hands_deploy_jetson_no_install_root(
    rover_repo: Path, tmp_path: Path, deploy_env: dict[str, str]
) -> None:
    """deploy_jetson.sh keeps its own default -- the tree the unit it installs runs.

    Driven with this PC's own ``MOUSEDROID_INSTALL_DIR`` set, because that is
    the value a pass-through would carry to the rover.
    """
    sent = _remote_commands(
        _without_install_dir(deploy_env),
        tmp_path,
        "--full",
        MOUSEDROID_INSTALL_DIR="/srv/elsewhere",
    )

    assert "scripts/deploy_jetson.sh" in sent
    assert "MOUSEDROID_INSTALL_DIR" not in sent
    assert "/srv/elsewhere" not in sent


# ---------------------------------------------------------------------------
# Remote-path injection guard
# ---------------------------------------------------------------------------
# REMOTE_SRC and REMOTE_CONFIG became environment-overridable in this change so
# the WIP guard could be driven against a throwaway checkout instead of only on
# a rover. Before that they were literals. They are also interpolated into
# remote command STRINGS that run under sudo — `remote_sudo bash -c "mkdir -p
# ${REMOTE_SRC_Q} && chown ..."` and `remote_cmd "test -x ${REMOTE_SRC_Q}/..."`
# — so the override turned a safe interpolation into an injection point: a value
# like `/opt/mousedroid; rm -rf /` would run as root on the rover.
#
# The script validates both at startup, before any ssh, and quotes them with
# `printf %q` at every remote boundary. These tests drive the real script, so
# they fail if either defence is removed.

_INJECTION_PAYLOADS = [
    "/opt/mousedroid/src; touch /tmp/mousedroid-injection-canary",
    "/opt/x$(touch /tmp/mousedroid-injection-canary)",
    "/opt/x`touch /tmp/mousedroid-injection-canary`",
    "/opt/x|touch /tmp/mousedroid-injection-canary",
    "/opt/x&touch /tmp/mousedroid-injection-canary",
    "/opt/x>/tmp/mousedroid-injection-canary",
    '/opt/x"quoted',
    "/opt/x'quoted",
    "/opt/with space",
]

# NOT including "": `REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/...}"` uses `:-`,
# which substitutes the default when the variable is empty OR unset — so an empty
# override correctly means "use the default" and is not reachable as a bad value.
# The validator's own `-n` check stays as defence for a future refactor to `:=`
# or a direct assignment.
_NON_ABSOLUTE = ["relative/path", "opt/mousedroid", "./relative"]
_TRAVERSING = ["/opt/../etc", "/opt/mousedroid/.."]


def _deploy_help(remote_src: str) -> subprocess.CompletedProcess[str]:
    """Run ``deploy_remote.sh --help`` with ``MOUSEDROID_REMOTE_SRC`` set.

    ``--help`` is enough: the validator runs at script start, before any ssh, so
    a rejected path never reaches the network. That is the property under test.
    """
    env = dict(os.environ, MOUSEDROID_REMOTE_SRC=remote_src)
    return subprocess.run(
        ["bash", str(_DEPLOY), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_a_shell_metacharacter_in_the_remote_path_is_refused(payload: str) -> None:
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted an injectable path: {payload!r}"
    assert "unsafe for a remote command" in result.stderr, result.stderr


@pytest.mark.parametrize("payload", _NON_ABSOLUTE)
def test_a_non_absolute_remote_path_is_refused(payload: str) -> None:
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted a non-absolute path: {payload!r}"
    assert "absolute path" in result.stderr, result.stderr


@pytest.mark.parametrize("payload", _TRAVERSING)
def test_a_traversing_remote_path_is_refused(payload: str) -> None:
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted a traversing path: {payload!r}"
    assert "'..' segment" in result.stderr, result.stderr


def test_a_plain_absolute_path_is_still_accepted() -> None:
    """The guard must not break the legitimate override the tests rely on."""
    result = _deploy_help("/opt/mousedroid")
    assert result.returncode == 0, result.stderr


def test_no_payload_ever_executed() -> None:
    """Belt and braces: the canary the payloads try to create must not exist."""
    canary = Path("/tmp/mousedroid-injection-canary")
    assert not canary.exists(), "a payload executed — the remote-path validator did not hold"


def test_the_config_dir_override_is_validated_too() -> None:
    """``REMOTE_CONFIG`` lands in `remote_sudo mkdir -p` the same way."""
    env = dict(os.environ, MOUSEDROID_CONFIG_DIR="/etc/mousedroid; touch /tmp/x")
    result = subprocess.run(
        ["bash", str(_DEPLOY), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "MOUSEDROID_CONFIG_DIR" in result.stderr, result.stderr


def test_the_remote_boundaries_use_the_quoted_forms() -> None:
    """Validation plus quoting: neither alone is the whole defence."""
    source = _DEPLOY.read_text(encoding="utf-8")
    assert 'REMOTE_SRC_Q="$(printf ' in source
    assert 'REMOTE_CONFIG_Q="$(printf ' in source
    offenders = [
        line.strip()
        for line in source.splitlines()
        if ('remote_sudo bash -c "' in line or 'remote_cmd "' in line) and "${REMOTE_SRC}" in line
    ]
    assert offenders == [], (
        f"a remote command string interpolates the RAW path; use REMOTE_SRC_Q: {offenders}"
    )


# ---------------------------------------------------------------------------
# The reject set, widened after an audit found four ways through it
# ---------------------------------------------------------------------------
#
# The first version of the validator rejected the obvious metacharacters and
# stopped there. An adversarial read found four holes, each confirmed by driving
# the real script, and each is now a payload class below:
#
#   1. TAB, VT and FF were absent from the bracket class. Tab is an IFS
#      character, so at every argv-style boundary the value word-splits and
#      injects extra arguments into a command running under `sudo` on the rover.
#   2. The backslash arm was spelt *'\\'*. A QUOTED backslash pair in a case
#      pattern matches TWO literal backslashes, so a lone backslash passed and
#      survived into a boundary where the remote shell strips one more level.
#   3. Glob metacharacters are not shell metacharacters, so no arm caught them.
#      The remote shell pathname-expands an unquoted value, which turns
#      `mkdir -p /etc/mousedroid*` into whatever happens to match on the rover.
#   4. `REMOTE_CONFIG_Q` was computed and never used, while `${REMOTE_CONFIG}`
#      crossed raw at both config boundaries — so the "two defences" the header
#      promises were, for that variable, one.
#
# Hole 4 is the instructive one: the pre-existing quoting test asserted the
# pre-quoted form was *defined*, which the dead variable satisfied.
# `test_every_pre_quoted_form_is_used` asserts it is *used*, which is the
# property that was actually claimed.

_WORD_SPLITTING_PAYLOADS = ["/opt/x\tsplit", "/opt/x\vsplit", "/opt/x\fsplit"]
_BACKSLASH_PAYLOADS = ["/opt/x\\escaped", "/opt/\\etc"]
_GLOB_PAYLOADS = ["/opt/mousedroid*", "/opt/mousedroi?", "/opt/[me]ousedroid", "/opt/x]y"]


@pytest.mark.parametrize("payload", _WORD_SPLITTING_PAYLOADS)
def test_whitespace_that_is_not_a_space_is_refused(payload: str) -> None:
    """Tab/VT/FF split words in the remote shell exactly as a space does."""
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted a word-splitting path: {payload!r}"
    assert "unsafe for a remote command" in result.stderr, result.stderr


@pytest.mark.parametrize("payload", _BACKSLASH_PAYLOADS)
def test_a_single_backslash_is_refused(payload: str) -> None:
    """A lone backslash, not just a doubled one — the original arm missed it."""
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted a backslash path: {payload!r}"
    assert "backslash" in result.stderr, result.stderr


@pytest.mark.parametrize("payload", _GLOB_PAYLOADS)
def test_a_glob_metacharacter_is_refused(payload: str) -> None:
    """The REMOTE shell expands these; the local one never sees them."""
    result = _deploy_help(payload)
    assert result.returncode != 0, f"accepted a globbing path: {payload!r}"
    assert "glob characters" in result.stderr, result.stderr


def test_every_pre_quoted_form_is_used() -> None:
    """A `*_Q` variable that nothing reads is a defence that does not exist.

    `REMOTE_CONFIG_Q` was defined and never referenced while `${REMOTE_CONFIG}`
    crossed raw at two boundaries. `test_the_remote_boundaries_use_the_quoted_forms`
    passed throughout, because it only asserted the definition.
    """
    lines = _DEPLOY.read_text(encoding="utf-8").splitlines()
    declared = {
        line.split("=", 1)[0].strip()
        for line in lines
        if "=" in line and line.split("=", 1)[0].strip().endswith("_Q")
    }
    assert declared, "premise: the script declares at least one pre-quoted form"
    unused = sorted(
        name
        for name in declared
        if sum(1 for line in lines if name in line and not line.lstrip().startswith("#")) < 2
    )
    assert unused == [], (
        f"pre-quoted form(s) declared but never used, so the raw value is still "
        f"crossing the boundary they exist to protect: {unused}"
    )


def test_no_remote_command_string_interpolates_a_raw_validated_path() -> None:
    """Widened from `${REMOTE_SRC}` alone, which is how hole 4 stayed invisible."""
    source = _DEPLOY.read_text(encoding="utf-8")
    raw_forms = ("${REMOTE_SRC}", "${REMOTE_CONFIG}", "${REMOTE_INSTALL_DIR}", "${REMOTE_VENV}")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if ("remote_sudo " in line or "remote_cmd " in line or "rsync " in line)
        and any(form in line for form in raw_forms)
        and not line.lstrip().startswith("#")
    ]
    assert offenders == [], (
        f"a remote command interpolates a raw validated path; use the _Q form: {offenders}"
    )


# ---------------------------------------------------------------------------
# The rsync fence: --delete never reaches rover runtime state (F-059 R-2)
# ---------------------------------------------------------------------------
# `rsync -avz --delete` removes everything under the destination that the source
# lacks. The rover keeps state there that no checkout carries -- the venv, model
# weights, logs, smoke and endurance reports, the e-stop latch, the TensorRT
# engine cache -- so every deploy would wipe it unless a filter protects it.
#
# These cases run REAL rsync in dry-run mode between two throwaway trees, so they
# fail on what rsync would actually delete rather than on how the script is
# spelt. The args under test are the script's own (`--print-rsync-args`), and one
# more case proves the real sync passes that very list: one definition, not two
# that can drift apart.
#
# The destination here has no `.gitignore` of its own -- a dry run transfers
# none, and a first deploy has none -- which leaves the explicit list as the only
# thing protecting the rover. That is the layer worth pinning.
#
# Three further properties are pinned beyond "nothing is deleted". The runtime-root
# excludes are anchored to the transfer root, so a same-named directory deeper in the
# tree (docs/reports, src/mousedroid/models) is source and still ships. A tracked
# .gitkeep sentinel survives the .gitignore merge, which drops the negation lines that
# would otherwise re-include it. And a root that is itself a symlink (models/ pointing at
# NVMe) is held too: the excludes carry no trailing slash, because one would make them
# match directories only and rsync does not count a symlink as a directory.

_rsync_required = pytest.mark.skipif(
    shutil.which("rsync") is None,
    reason="real rsync needed; ubuntu-latest ships it",
)

# Top-level directories on the rover that hold state, not source.
_RUNTIME_ROOTS = (
    "venv",
    "weights",
    "models",
    "logs",
    "reports",
    "estop_latch",
    "tensorrt_cache",
)

# A directory that shares a runtime root's NAME but sits below the root is source, not
# state, and must ship. One per root, so un-anchoring any single exclude is caught.
_NESTED_SOURCE = (
    "docs/reports/x.md",
    "src/mousedroid/models/x.py",
    "src/mousedroid/venv/x.py",
    "src/mousedroid/weights/x.py",
    "src/mousedroid/logs/x.py",
    "src/mousedroid/estop_latch/x.py",
    "src/mousedroid/tensorrt_cache/x.py",
)


def _rsync_args() -> list[str]:
    """The fence exactly as the script defines it: one argument per output line."""
    result = subprocess.run(
        ["bash", str(_DEPLOY), "--print-rsync-args"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line]


@pytest.fixture
def pc_checkout(tmp_path: Path) -> Path:
    """The PC side of the sync: a git checkout whose ``.gitignore`` anchors every runtime
    root (as a root entry should be) and carries two negated ``.gitkeep`` sentinels, with
    same-named directories BELOW the root that are source and must ship."""
    src = tmp_path / "pc-checkout"
    src.mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    # The root entries are anchored (/models/, not models/): an unanchored one would also
    # ignore src/mousedroid/models/, which the nested-name test needs the checkout to track.
    ignored = (
        "/venv/",
        "/weights/",
        "/models/",
        "/logs/",
        "reports/endurance/*",
        "!reports/endurance/.gitkeep",
        "/estop_latch/",
        "/tensorrt_cache/",
        "__pycache__/",
        "cfg/runtime/*",
        "!cfg/runtime/.gitkeep",
    )
    tracked = (
        (".gitignore", "\n".join(ignored) + "\n"),
        ("src/mousedroid/__init__.py", ""),
        ("src/mousedroid/main.py", "print('v2')\n"),
        ("scripts/run.sh", "#!/bin/bash\n"),
        ("README.md", "rover\n"),
        ("reports/endurance/.gitkeep", ""),
        ("cfg/runtime/.gitkeep", ""),
        *((rel, "x\n") for rel in _NESTED_SOURCE),
    )
    for rel, body in tracked:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(body, encoding="utf-8")
    # -f: a developer's global excludes (a logs/ or venv/ line is common) must not drop
    # the nested same-named directories from this fixture.
    _git(src, "add", "-f", "-A")
    _git(src, "commit", "-qm", "baseline")
    # Generated beside the sentinel and ignored by `cfg/runtime/*`: it must never ship.
    (src / "cfg" / "runtime" / "generated.yml").write_text("generated: true\n", encoding="utf-8")
    return src


@pytest.fixture
def rover_tree(tmp_path: Path) -> Path:
    """The rover side: state under every runtime root, plus one source file the PC
    no longer has -- the one thing ``--delete`` is *supposed* to remove.

    ``models`` is a symlink to a directory outside the tree, as it is on a rover that keeps its
    GGUFs on NVMe. Where the platform cannot symlink it is a plain directory instead, and the
    one case that needs the link skips itself.
    """
    rover = tmp_path / "rover-tree"
    for rel in (
        "venv/bin/python",
        "weights/bdi/policy.onnx",
        "logs/mousedroid.log",
        "reports/jetson_smoke/2026-10-16/run.json",
        "reports/endurance/.gitkeep",
        "estop_latch/latched.json",
        "tensorrt_cache/engine.plan",
        "src/mousedroid/removed_module.py",
    ):
        (rover / rel).parent.mkdir(parents=True, exist_ok=True)
        (rover / rel).write_text("rover-only\n", encoding="utf-8")
    nvme = tmp_path / "nvme-models"
    nvme.mkdir()
    (nvme / "Phi-3-mini.gguf").write_text("rover-only\n", encoding="utf-8")
    try:
        (rover / "models").symlink_to(nvme, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(nvme, rover / "models")
    return rover


def _dry_run(src: Path, rover: Path) -> tuple[set[str], set[str]]:
    """What real rsync WOULD delete from ``rover`` and WOULD transfer to it."""
    args = _rsync_args()
    assert "--filter=:- .gitignore" in args
    result = subprocess.run(
        ["rsync", "-n", "-a", "--delete", "--itemize-changes", *args, f"{src}/", f"{rover}/"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    deleting = {line.split(None, 1)[1] for line in lines if line.startswith("*deleting")}
    transferred = {line.split(None, 1)[1] for line in lines if line.startswith(">f")}
    return deleting, transferred


@_rsync_required
def test_delete_never_touches_runtime_state(pc_checkout: Path, rover_tree: Path) -> None:
    """Pin 2 (R-2): the fence protects every runtime root on the rover."""
    deleting, _ = _dry_run(pc_checkout, rover_tree)
    assert "src/mousedroid/removed_module.py" in deleting, "premise: --delete is still live"
    at_risk = sorted(p for p in deleting if p.split("/")[0] in _RUNTIME_ROOTS)
    assert not at_risk, f"--delete would remove rover runtime state: {at_risk}"
    # The sentinel inside a fenced root stays where it is: never shipped (the transfer test
    # below pins that half), but never deleted from the rover either.
    assert "reports/endurance/.gitkeep" not in deleting, "the fenced-root sentinel was deleted"


@_rsync_required
def test_a_symlinked_runtime_root_is_held_too(pc_checkout: Path, rover_tree: Path) -> None:
    """``models`` as a symlink to NVMe is state, not source: ``--delete`` must leave the link.

    A trailing-slash exclude (``/models/``) matches directories only, and rsync does not count
    a symlink as one, so the link would be removed and the rover left without its models. The
    fence is spelt without the slash for exactly this case.
    """
    if not (rover_tree / "models").is_symlink():
        pytest.skip("this platform cannot create a symlink")
    deleting, _ = _dry_run(pc_checkout, rover_tree)
    assert "src/mousedroid/removed_module.py" in deleting, "premise: --delete is still live"
    assert "models" not in deleting, "--delete would remove the models symlink"


@_rsync_required
def test_every_tracked_file_outside_the_fenced_roots_is_transferred(
    pc_checkout: Path, rover_tree: Path
) -> None:
    """R3-4: the positive direction, the sentinel exception, and the documented loss.

    A fence that over-protects would "pass" the deletion case above while quietly
    shipping nothing, so every tracked file outside the fenced roots must still go.
    The ``-`` merge drops ``.gitignore``'s negation lines, so ``cfg/runtime/.gitkeep``
    (``cfg/runtime/*`` + ``!cfg/runtime/.gitkeep``) ships only because the include rule
    precedes the merge -- and only the sentinel does: the generated file beside it stays
    PC-side. ``reports/endurance/.gitkeep`` is the one documented loss: it sits under a
    fenced root, and rsync never descends into an excluded directory.
    """
    _, transferred = _dry_run(pc_checkout, rover_tree)
    tracked = set(_git(pc_checkout, "ls-files").splitlines())
    expected = {p for p in tracked if p.split("/")[0] not in _RUNTIME_ROOTS}
    assert "src/mousedroid/main.py" in expected
    assert expected <= transferred, sorted(expected - transferred)
    assert "cfg/runtime/.gitkeep" in tracked
    assert "cfg/runtime/.gitkeep" in transferred
    assert "cfg/runtime/generated.yml" not in transferred
    assert "reports/endurance/.gitkeep" in tracked
    assert "reports/endurance/.gitkeep" not in transferred


@_rsync_required
def test_runtime_root_excludes_are_anchored(pc_checkout: Path, rover_tree: Path) -> None:
    """A directory that merely shares a root's NAME is source, not state.

    ``/models`` is anchored to the transfer root; an unanchored ``models`` also drops
    ``src/mousedroid/models/`` and ``docs/reports/`` from the sync without a word, while
    the root-level state it exists to protect stays protected either way. One nested
    directory per root, so un-anchoring any single exclude turns this red.
    """
    assert {Path(p).parent.name for p in _NESTED_SOURCE} == set(_RUNTIME_ROOTS), (
        "premise: every runtime root has a same-named nested directory below it"
    )
    deleting, transferred = _dry_run(pc_checkout, rover_tree)
    tracked = set(_git(pc_checkout, "ls-files").splitlines())
    nested = set(_NESTED_SOURCE)
    assert nested <= tracked, "premise: the nested directories hold tracked source"
    assert nested <= transferred, sorted(nested - transferred)
    at_risk = sorted(p for p in deleting if p.split("/")[0] in _RUNTIME_ROOTS)
    assert not at_risk, f"anchoring must not unprotect the roots: {at_risk}"


def test_the_shim_recorded_the_same_rsync_args_the_script_prints(
    deploy_env: dict[str, str],
) -> None:
    """One definition: the args the real sync passes are the ones --print-rsync-args shows."""
    assert _run_deploy(deploy_env, "--code-only").returncode == 0
    recorded = Path(deploy_env["RSYNC_MARKER"]).read_text(encoding="utf-8").splitlines()
    printed = _rsync_args()
    assert printed, "premise: --print-rsync-args printed the fence"
    assert set(printed) <= set(recorded)
    assert "--delete" in recorded
    # Order matters to rsync (first matching rule wins), so the printed list must reach
    # the sync as one contiguous, in-order block -- not merely as a subset of it.
    start = recorded.index(printed[0])
    assert recorded[start : start + len(printed)] == printed
