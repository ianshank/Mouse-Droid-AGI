"""Unit: values compose splices into a mount spec are refused before compose runs.

``docker-compose.jetson.yml`` mounts the named volume ``mousedroid_tensorrt_cache``
at ``${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR:-...}``. Compose's short volume syntax
is ``VOLUME:TARGET[:MODE]``, so a value containing ``:`` injects a third field,
and a value such as ``/etc`` mounts the volume over an image directory. The cache
holds torch2trt engines that are unpickled at load time, so where it lands is not
cosmetic (F-052 task 5.6).

Compose is started from two places, and both are covered here, because guarding
only one would leave the other open: ``scripts/docker_deploy.sh`` and
``scripts/preflight_check.sh`` (the systemd unit's fatal ``ExecStartPre``). Both
call the one validator, ``scripts/validate_compose_env.sh``.

Real script, real fixture, specific messages — per ``test_deploy_remote_guard.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

from mousedroid.config import loader as config_loader
from mousedroid.config.schema.hardware import JetsonConfig
from mousedroid.config.schema.llm import LLMConfig
from mousedroid.config.schema.world_model import WorldModelConfig
from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
_VALIDATOR = _SCRIPTS / "validate_compose_env.sh"
_DEPLOY = _SCRIPTS / "docker_deploy.sh"
_PREFLIGHT = _SCRIPTS / "preflight_check.sh"
_UNIT = _SCRIPTS / "mousedroid-docker.service"
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_DOCKERFILE = _REPO_ROOT / "Dockerfile.jetson"
_KEY = "MOUSEDROID_JETSON__TENSORRT_CACHE_DIR"

pytestmark = requires_bash()


def _validate(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != _KEY}
    if value is not None:
        env[_KEY] = value
    return subprocess.run(
        ["bash", str(_VALIDATOR)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,  # unset: compose falls back to its own (valid) default
        "/opt/mousedroid/tensorrt_cache",  # the schema and compose default
        # One trailing '/' names the same directory. Refusing it would stop a
        # rover booting (the unit's preflight is fatal) on a value that worked
        # before this check existed.
        "/opt/mousedroid/tensorrt_cache/",
        "/opt/mousedroid/trt_cache",  # nested in the source bind mount, as the default is
        "/mnt/nvme/trt_cache",  # a relocation to a bigger disk
        "/data/cache",
        "/var/cache/mousedroid",
        "/home/jetson/trt_cache",  # a sibling of the experience volume, not inside it
        "/var/runner/cache",  # starts with "/var/run" as a string, not as a path
    ],
)
def test_safe_values_are_accepted(value: str | None) -> None:
    proc = _validate(value)

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("/x:ro", "only [A-Za-z0-9._/-] is allowed"),  # injects a :MODE field
        ("/x:/y", "only [A-Za-z0-9._/-] is allowed"),  # injects a third field
        ("/etc", "must not be the root or a top-level directory"),
        ("/opt", "must not be the root or a top-level directory"),
        ("/", "must not be the root or a top-level directory"),
        ("/etc/mousedroid", "must not be inside /etc"),
        ("/usr/lib/trt", "must not be inside /usr"),
        ("tensorrt_cache", "must be an absolute path"),
        ("/opt/../etc", "must not contain empty, '.' or '..' components"),
        ("/opt//x", "must not contain empty, '.' or '..' components"),
        ("/opt/x//", "must not contain empty, '.' or '..' components"),
        ("/opt/my cache", "only [A-Za-z0-9._/-] is allowed"),
        ("/opt/$HOME", "only [A-Za-z0-9._/-] is allowed"),
        ("/opt/caf\u00e9", "only [A-Za-z0-9._/-] is allowed"),
        ("/opt/" + "a" * 256, "must not have a component longer than 255 bytes"),
        # Links into /run: the runtime resolves them inside the container.
        ("/var/run/x", "must not be inside /var/run, a link into an OS-image tree"),
        ("/var/lock/x", "must not be inside /var/lock, a link into an OS-image tree"),
        # The service's other mounts: neither covered nor nested into.
        ("/opt/mousedroid", "would cover /opt/mousedroid, which compose also mounts"),
        ("/home/jetson", "would cover /home/jetson/mousedroid_experience, which compose"),
        (
            "/home/jetson/mousedroid_experience/trt",
            "must not be inside /home/jetson/mousedroid_experience, which compose",
        ),
        ("/var/lib/promtail", "would cover /var/lib/promtail, which compose also mounts"),
        # Inside the source bind mount, what the container runs from: neither
        # covered nor nested into, and the refusal says which.
        ("/opt/mousedroid/src", "would cover /opt/mousedroid/src, which the container"),
        ("/opt/mousedroid/src/mousedroid", "must not be inside /opt/mousedroid/src, which"),
        ("/opt/mousedroid/weights", "would cover /opt/mousedroid/weights, which"),
        # The relocation the template used to read as allowed.
        ("/opt/mousedroid/weights/trt_cache", "must not be inside /opt/mousedroid/weights"),
        ("/opt/mousedroid/config", "would cover /opt/mousedroid/config, which"),
        ("/opt/mousedroid/models/trt", "must not be inside /opt/mousedroid/models, which"),
    ],
)
def test_unsafe_values_are_refused_with_the_reason(value: str, reason: str) -> None:
    proc = _validate(value)

    assert proc.returncode == 1
    assert f"{_KEY} refused: {reason}" in proc.stderr


# Linux limits from <linux/limits.h>. The validator targets the Linux
# container whatever the host, so these are the spec, not the host's values.
_LINUX_PATH_MAX = 4096  # including the terminating NUL
_LINUX_NAME_MAX = 255


def _path_of_length(length: int) -> str:
    """A cache path breaking no rule but, possibly, its length: ``length`` bytes.

    Built from short components, so the per-component limit stays out of it.
    """
    path = "/data"
    while len(path) + 10 <= length:
        path += "/" + "a" * 9
    return path + "a" * (length - len(path))


def test_the_length_limits_are_linuxs_and_hold_at_the_edge() -> None:
    """``_PATH_MAX`` is exercised at its boundary, not merely declared."""
    limit = int(_script_scalar("_PATH_MAX"))
    longest = _path_of_length(limit)

    assert limit == _LINUX_PATH_MAX - 1
    assert int(_script_scalar("_NAME_MAX")) == _LINUX_NAME_MAX
    assert len(longest) == limit
    accepted = _validate(longest)
    assert accepted.returncode == 0, accepted.stderr
    over = _validate(_path_of_length(limit + 1))
    assert over.returncode == 1
    assert f"{_KEY} refused: must not be longer than {limit} bytes" in over.stderr


# The FHS trees a relocated cache must never enter. The validator may list
# more; it may not list fewer.
_REQUIRED_OS_TREES = frozenset(
    {"bin", "boot", "dev", "etc", "lib", "proc", "root", "run", "sbin", "sys", "usr"}
)


def test_every_os_owned_tree_is_refused_not_just_two() -> None:
    trees = _script_array("_OS_OWNED_TREES")
    wrong = {}
    for tree in trees:
        proc = _validate(f"/{tree}/mousedroid/trt_cache")
        reason = f"must not be inside /{tree}, which belongs to the OS image"
        if proc.returncode != 1 or reason not in proc.stderr:
            wrong[tree] = proc.stderr

    assert set(trees) >= _REQUIRED_OS_TREES, sorted(_REQUIRED_OS_TREES - set(trees))
    assert not wrong, wrong


def test_a_refusal_never_echoes_the_value() -> None:
    """On the boot path the value comes from systemd's EnvironmentFile parser.

    A value whose quote is left open (``KEY="/data/trt``) has the following
    lines joined onto it -- and in the shipped template those are the API key
    and telemetry token. An ExecStartPre's stderr goes to the journal.
    """
    sentinel = "do-not-echo-this-sentinel"
    merged = f"/data/trt\nANTHROPIC_API_KEY={sentinel}\nMOUSEDROID_TELEMETRY_TOKEN={sentinel}"

    proc = _validate(merged)

    assert proc.returncode == 1
    assert f"{_KEY} refused" in proc.stderr
    assert sentinel not in proc.stderr
    assert "/data/trt" not in proc.stderr


# ---------------------------------------------------------------------------
# The validator's lists, pinned to the files they describe
# ---------------------------------------------------------------------------


def _script_array(name: str) -> list[str]:
    match = re.search(rf"^{name}=\(([^)]*)\)", _VALIDATOR.read_text(encoding="utf-8"), re.M)
    assert match is not None, f"{name} not found in {_VALIDATOR.name}"
    return match.group(1).split()


def _script_scalar(name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", _VALIDATOR.read_text(encoding="utf-8"), re.M)
    assert match is not None, f"{name} not found in {_VALIDATOR.name}"
    return match.group(1)


_CACHE_VOLUME = "mousedroid_tensorrt_cache"


def _compose_volumes() -> list[tuple[str, str]]:
    """``(source, target)`` for each of the service's short-syntax volumes.

    Each ``${VAR:-default}`` is replaced by its default first -- what compose
    does with the variable unset -- because a default may itself contain ``:``.
    """
    service = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]["mousedroid"]
    pairs: list[tuple[str, str]] = []
    for entry in service["volumes"]:
        resolved = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}", r"\1", entry)
        source, target, *_ = resolved.split(":")
        pairs.append((source, target))
    return pairs


def test_the_mount_targets_are_exactly_the_composes_other_volumes() -> None:
    """A volume added to compose must be added to the validator too."""
    targets = {target for source, target in _compose_volumes() if source != _CACHE_VOLUME}

    assert set(_script_array("_COMPOSE_MOUNT_TARGETS")) == targets


def test_the_source_bind_mount_is_a_same_path_bind_in_compose() -> None:
    source_mount = _script_scalar("_SOURCE_BIND_MOUNT")

    assert (source_mount, source_mount) in _compose_volumes()


def test_the_paths_in_use_follow_the_image_and_the_schema() -> None:
    """Each reserved path is derived from where it actually comes from.

    The package root from the Dockerfile; the weights root from the world-model
    schema default (relative to WORKDIR); the base config directory from the
    config loader (repo-relative, so WORKDIR-relative in the image); the model
    directory from the LLM schema default.
    """
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    workdir = re.search(r"^WORKDIR (\S+)$", dockerfile, re.M)
    assert workdir is not None
    assert re.search(r"^COPY src/ \./src/$", dockerfile, re.M), "the image no longer copies src/"
    weights_root = PurePosixPath(WorldModelConfig().onnx_cache_dir).parts[0]
    config_dir = config_loader._DEFAULT_CONFIG_DIR.relative_to(_REPO_ROOT).as_posix()
    model_dir = PurePosixPath(LLMConfig().model_path).parent.as_posix()

    assert workdir.group(1) == _script_scalar("_SOURCE_BIND_MOUNT")
    assert set(_script_array("_SOURCE_PATHS_IN_USE")) == {
        f"{workdir.group(1)}/src",
        f"{workdir.group(1)}/{weights_root}",
        f"{workdir.group(1)}/{config_dir}",
        model_dir,
    }


# ---------------------------------------------------------------------------
# Entry point 1: docker_deploy.sh refuses before any compose command
# ---------------------------------------------------------------------------

_DOCKER_SHIM = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_LOG"
case "$1" in ps) echo "abc123" ;; exec) echo "OK" ;; esac
exit 0
"""


def _deploy(tmp_path: Path, cache_dir: str) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "docker"
    shim.write_text(_DOCKER_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    docker_log.write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(_DEPLOY), "--no-build"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "DOCKER_LOG": str(docker_log),
            "MOUSEDROID_INSTALL_DIR": str(tmp_path),
            "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
            "MOUSEDROID_DOCKER_ENV_FILE": str(tmp_path / "absent.env"),
            "MOUSEDROID_HEALTH_TIMEOUT": "1",
            _KEY: cache_dir,
        },
        cwd=str(_REPO_ROOT),
    )
    return proc, docker_log.read_text(encoding="utf-8")


def test_docker_deploy_refuses_before_compose_ever_runs(tmp_path: Path) -> None:
    """The refusal must precede every compose invocation, not just ``up``.

    ``pull`` and ``build`` parse the file too, so the docker shim's log is
    checked for *any* compose call rather than for ``up`` alone.
    """
    proc, docker_calls = _deploy(tmp_path, "/x:ro")

    assert proc.returncode != 0
    assert f"{_KEY} refused" in proc.stderr
    assert "compose" not in docker_calls, f"compose ran despite the refusal:\n{docker_calls}"


def test_docker_deploy_proceeds_to_compose_with_a_safe_value(tmp_path: Path) -> None:
    """...and the guard is not refusing everything: a safe value reaches compose.

    Without this, the test above would also pass if the validator rejected
    every input, or if the script died for an unrelated reason first.
    """
    _, docker_calls = _deploy(tmp_path, "/opt/mousedroid/tensorrt_cache")

    assert "compose" in docker_calls, "a safe value never reached compose"


# ---------------------------------------------------------------------------
# Compose interpolates from the checked environment and nothing else
# ---------------------------------------------------------------------------
# Without --env-file, compose also reads <project dir>/.env and COMPOSE_ENV_FILES
# for interpolation, and this check sees neither. deploy_remote.sh rsyncs the
# working tree, .env included, into the rover's checkout -- which is the
# project directory on both start paths.

_NO_ENV_FILE = "--env-file /dev/null -f "


def test_every_compose_call_on_the_deploy_path_reads_no_env_file(tmp_path: Path) -> None:
    _, docker_calls = _deploy(tmp_path, "/opt/mousedroid/tensorrt_cache")
    compose_calls = [line for line in docker_calls.splitlines() if line.startswith("compose")]

    assert compose_calls, "no compose call reached the shim"
    assert [c for c in compose_calls if not c.startswith(f"compose {_NO_ENV_FILE}")] == []


def test_every_compose_line_in_the_unit_reads_no_env_file() -> None:
    lines = [
        line.strip()
        for line in _UNIT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("Exec") and "docker compose" in line
    ]

    assert lines, "the unit no longer runs compose"
    assert [line for line in lines if f"docker compose {_NO_ENV_FILE}" not in line] == []


def _cache_mount(compose_args: list[str], env: dict[str, str]) -> tuple[str, bool]:
    """``(target, read_only)`` of the cache volume, as compose itself resolves it."""
    proc = subprocess.run(
        ["docker", "compose", *compose_args, "config"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    for volume in yaml.safe_load(proc.stdout)["services"]["mousedroid"]["volumes"]:
        if volume.get("source") == _CACHE_VOLUME:
            return volume["target"], bool(volume.get("read_only", False))
    raise AssertionError("the resolved config has no cache volume")


@pytest.mark.parametrize("source", ["project-dotenv", "COMPOSE_ENV_FILES"])
def test_the_premise_holds_against_compose_itself(tmp_path: Path, source: str) -> None:
    """Unasked, compose reads the env file; told ``--env-file /dev/null``, it does not.

    ``compose config`` resolves interpolation without a daemon. With the key
    absent from the environment, the planted value moves the cache onto
    ``/etc`` read-only -- an OS tree and an injected ``:ro``, both refused by
    the validator, which never saw it.
    """
    if (
        shutil.which("docker") is None
        or subprocess.run(
            ["docker", "compose", "version"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("needs the docker compose CLI (config runs without a daemon)")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    compose_file = checkout / _COMPOSE.name
    compose_file.write_text(_COMPOSE.read_text(encoding="utf-8"), encoding="utf-8")
    unset = {_KEY, "COMPOSE_ENV_FILES", "COMPOSE_FILE", "COMPOSE_PROJECT_NAME"}
    env = {k: v for k, v in os.environ.items() if k not in unset}
    planted = f"{_KEY}=/etc:ro\n"
    if source == "project-dotenv":
        (checkout / ".env").write_text(planted, encoding="utf-8")
    else:
        elsewhere = tmp_path / "elsewhere.env"
        elsewhere.write_text(planted, encoding="utf-8")
        env["COMPOSE_ENV_FILES"] = str(elsewhere)

    unasked = _cache_mount(["-f", str(compose_file)], env)
    guarded = _cache_mount([*_NO_ENV_FILE.split(), str(compose_file)], env)

    assert unasked == ("/etc", True), f"compose no longer reads {source}: {unasked}"
    assert guarded == (JetsonConfig().tensorrt_cache_dir.as_posix(), False)


# ---------------------------------------------------------------------------
# Entry point 2: the systemd boot path, via preflight_check.sh
# ---------------------------------------------------------------------------


def _preflight(cache_dir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_PREFLIGHT), "--skip-devices", "--skip-models"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, _KEY: cache_dir},
    )


def test_preflight_counts_an_unsafe_value_as_a_critical_failure(tmp_path: Path) -> None:
    proc = _preflight("/etc")
    output = proc.stdout + proc.stderr

    assert f"{_KEY} refused" in output
    assert "[FAIL] A value compose interpolates into a mount spec was refused" in output
    assert proc.returncode != 0


def test_preflight_passes_the_check_for_a_safe_value(tmp_path: Path) -> None:
    proc = _preflight("/opt/mousedroid/tensorrt_cache")

    assert "[OK] Compose-interpolated values are safe" in proc.stdout + proc.stderr


def _preflight_with(**env: str) -> str:
    proc = subprocess.run(
        ["bash", str(_PREFLIGHT), "--skip-models"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    return proc.stdout + proc.stderr


def test_preflight_never_prints_lines_joined_onto_a_value() -> None:
    """The same leak as the validator's, one step over in the same ExecStartPre.

    preflight prints device and config paths it reads from the environment;
    on the boot path systemd joins the next lines onto a value whose quote is
    left open, and the template's next lines down are the API key and token.
    """
    sentinel = "do-not-echo-this-sentinel"
    merged = f"/dev/ttyUSB1\nANTHROPIC_API_KEY={sentinel}"

    output = _preflight_with(MOUSEDROID_LIDAR_DEV=merged, MOUSEDROID_CONFIG=merged)

    assert sentinel not in output
    assert "LiDAR UART missing: <withheld: not a plain path>" in output
    assert "Config file missing: <withheld: not a plain path>" in output


def test_preflight_still_names_a_plain_path() -> None:
    """...and an ordinary value is still shown: it is what the operator needs."""
    output = _preflight_with(MOUSEDROID_LIDAR_DEV="/dev/does-not-exist-lidar")

    assert "LiDAR UART missing: /dev/does-not-exist-lidar" in output


def test_the_unit_runs_preflight_fatally_and_before_compose() -> None:
    """The boot-path coverage claim, pinned against the unit file itself.

    ``preflight_check.sh`` only protects the boot path if the unit runs it as a
    FATAL ``ExecStartPre`` (no ``-`` prefix, which systemd treats as "ignore
    failure") and before the first line that runs compose.
    """
    exec_lines = [
        line.strip()
        for line in _UNIT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("ExecStartPre=", "ExecStart="))
    ]
    preflight = [i for i, line in enumerate(exec_lines) if "preflight_check.sh" in line]
    compose = [i for i, line in enumerate(exec_lines) if "docker compose" in line]

    assert preflight, f"the unit no longer runs preflight_check.sh: {exec_lines}"
    first = exec_lines[preflight[0]]
    assert not first.startswith("ExecStartPre=-"), (
        f"preflight is non-fatal, so a refused value would not stop boot: {first}"
    )
    assert compose, exec_lines
    assert preflight[0] < compose[0], f"compose runs before preflight: {exec_lines}"
