"""Structural tests for the Jetson full smoke harness shell wrapper and its stage script.

Text pins, so they run on every platform. The behaviour behind them is driven for real in
``test_jetson_smoke_orchestrator.py`` (the stage script against a stubbed interpreter and a
fake ``serial``), which needs bash and so skips on Windows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HARNESS = Path("scripts/jetson_full_smoke_run.sh")
_SMOKE = Path("scripts/jetson_smoke_test.sh")


@pytest.fixture(scope="module")
def harness_text() -> str:
    if not _HARNESS.exists():
        pytest.skip(f"{_HARNESS} not present in this checkout")
    return _HARNESS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def smoke_text() -> str:
    if not _SMOKE.exists():
        pytest.skip(f"{_SMOKE} not present in this checkout")
    return _SMOKE.read_text(encoding="utf-8")


def _bash_function(text: str, name: str) -> str:
    """Code of the top-level shell function ``name``: its opening line to the first bare ``}``.

    Whole-line ``#`` comments are dropped, so prose that mentions a token cannot satisfy a pin
    on the code that must contain it.
    """
    lines = text.splitlines()
    start = lines.index(f"{name}() {{")
    body = lines[start : lines.index("}", start)]
    return "\n".join(line for line in body if not line.lstrip().startswith("#"))


def test_harness_defines_blocking_override_resolution(harness_text: str) -> None:
    assert "resolve_blocking()" in harness_text
    assert "MOUSEDROID_SMOKE_BLOCKING_${upper}" in harness_text
    assert "invalid; using default" in harness_text


def test_harness_exports_stage_timeout_to_wrapped_commands(harness_text: str) -> None:
    assert "MOUSEDROID_SMOKE_STAGE_TIMEOUT" in harness_text
    assert "timeout --signal=INT --kill-after=10" in harness_text
    assert "timeout --signal=INT --kill-after=5" in harness_text


def test_harness_marks_nonblocking_timeouts_as_expected_fail(harness_text: str) -> None:
    assert 'why="rc=${rc} (non-blocking)"' in harness_text
    assert 'why="rc=${rc} (timeout after ${tmo}s, non-blocking)"' in harness_text
    assert 'record "${label}" "EXPECTED-FAIL" "${why}"' in harness_text


def test_harness_summary_includes_stage_table(harness_text: str) -> None:
    assert 'echo "| Stage | Status | Note |"' in harness_text
    assert 'echo "|-------|--------|------|"' in harness_text
    assert 'for entry in "${RESULTS[@]}"; do' in harness_text


def test_harness_summary_includes_voice_remediation(harness_text: str) -> None:
    assert 'echo "## Rocky voice prerequisites"' in harness_text
    assert "voice_remediation" in harness_text
    assert "voice.tts_model_path" in harness_text


# ---------------------------------------------------------------------------
# F-059 Task 4.1: the serial stage is an exclusive probe
# ---------------------------------------------------------------------------


def test_serial_stage_opens_the_port_exclusively(smoke_text: str) -> None:
    """The stage's inline probe takes the lock the driver takes, so a held port FAILs fast.

    Without ``exclusive=True`` it opens a port the orchestrator already owns, writes into the
    shared UART and reads half of the frames: a live board then looks silent or garbled.
    """
    code = _bash_function(smoke_text, "test_serial")
    assert re.search(r"serial\.Serial\(.*\bexclusive=True\b", code), code


def test_serial_stage_fails_on_silence(smoke_text: str) -> None:
    """REVERSAL: a silent board used to record ``PASS: serial open/write (no response: ...)``.

    That pass could not fail: the stage opened the port, wrote, got nothing back and said the
    port opened, which is true of a board with no firmware, a mismatched baud or no board at
    all. A probe needs a reply, so silence is now a FAIL and the ``WARN:`` path is gone.
    """
    code = _bash_function(smoke_text, "test_serial")
    assert "FAIL:No response from ESP32" in code, code
    assert "WARN:" not in code, code
    assert "serial open/write" not in code, code
    assert code.count("record_pass") == 1, "only a reply may pass the serial stage"


# ---------------------------------------------------------------------------
# F-059 Task 4.2: warm/cold ESP32 modes, and the motion gate reaches the container
# ---------------------------------------------------------------------------

_ESP32_STAGES = ("serial", "motor", "power", "mcp_motor_smoke")
_GATE = "MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION"


def test_harness_defaults_the_esp32_mode_to_warm_and_rejects_anything_else(
    harness_text: str,
) -> None:
    assert 'ESP32_MODE="${MOUSEDROID_SMOKE_ESP32_MODE:-warm}"' in harness_text
    assert "warm|cold) ;;" in harness_text
    assert "MOUSEDROID_SMOKE_ESP32_MODE='${ESP32_MODE}' is invalid" in harness_text


@pytest.mark.parametrize("stage", _ESP32_STAGES)
def test_harness_starts_each_esp32_stage_through_run_esp32_stage(
    harness_text: str, stage: str
) -> None:
    """A stage started through plain run_stage would skip the warm refusal and the cold window."""
    assert f'run_esp32_stage "{stage}"' in harness_text, stage
    assert f'run_stage "{stage}"' not in harness_text, stage


def test_harness_forwards_the_motion_gate_into_both_wrappers_by_name(harness_text: str) -> None:
    """``-e NAME`` hands the container the host's value only when the host sets one.

    ``-e NAME=value`` (or an expansion of the variable) would send an unset gate over as an
    empty string and overwrite whatever the container's own environment says.
    """
    assert harness_text.count(f"-e {_GATE}") == 2, "the warm wrapper and the cold wrapper"
    assert f"-e {_GATE}=" not in harness_text


def test_harness_refuses_before_it_runs_and_honours_the_stage_blocking_flag(
    harness_text: str,
) -> None:
    stage = _bash_function(harness_text, "run_esp32_stage")
    assert "warm smoke refused" in stage
    assert stage.index("refuse_esp32_stage") < stage.index("run_stage")
    refuse = _bash_function(harness_text, "refuse_esp32_stage")
    assert 'record "${label}" "FAIL"' in refuse
    assert 'record "${label}" "EXPECTED-FAIL"' in refuse
    assert "OVERALL_FAIL=1" in refuse


def test_harness_arms_the_restart_trap_before_it_stops_the_container(harness_text: str) -> None:
    begin = _bash_function(harness_text, "esp32_cold_begin")
    assert begin.index("cold_dry_run") < begin.index("trap restart_container_on_exit EXIT")
    assert begin.index("trap restart_container_on_exit EXIT") < begin.index(
        'docker stop "${CONTAINER}"'
    )
    restore = _bash_function(harness_text, "restore_container")
    assert restore.index('docker rm -f "${COLD_CONTAINER}"') < restore.index(
        'docker restart "${CONTAINER}"'
    )
    for name in ("restart_container_on_exit", "esp32_cold_end"):
        assert "restore_container" in _bash_function(harness_text, name), name
    assert 'docker start "${CONTAINER}"' not in harness_text, "a start is a no-op during a stop"
    assert "SIGKILL" in harness_text, "the header owns up to what no trap can catch"


def test_harness_restores_the_container_before_it_logs(harness_text: str) -> None:
    """Whatever undoes the stop restores first and logs best effort; the stop logs before its trap.

    A dead stdout (a dropped non-tty ssh session) turns a log line into a failed write, and under
    ``set -e`` that ends the function: a restore queued behind it is lost and the brain stays
    stopped. The behavioural rows drive the exits a dead stdout can reach; the trap is the net under
    ``esp32_cold_end`` and the failed-stop branch, so they cannot see the ordering there, and this
    text pin holds it.
    """
    helper = next(
        (line for line in harness_text.splitlines() if line.startswith("log_best_effort() ")), ""
    )
    assert helper, "the best-effort log helper is missing"
    assert "2>/dev/null" in helper
    assert helper.endswith("|| true; }"), helper
    for name in ("restart_container_on_exit", "esp32_cold_end"):
        body = _bash_function(harness_text, name)
        assert "log_best_effort" in body, name
        assert body.index("restore_container") < body.index("log_best_effort"), name
        assert not re.search(r"(^|\s)log ", body), f"{name} logs without the best-effort guard"
    begin = _bash_function(harness_text, "esp32_cold_begin")
    armed = begin.index("trap restart_container_on_exit EXIT")
    stop = begin.index('docker stop "${CONTAINER}"')
    assert not re.search(r"\blog", begin[armed:stop]), "nothing that can die sits between them"
    assert begin.index("esp32_cold_end") < begin.index('log_best_effort "cold smoke: docker stop')


def test_harness_cold_wrapper_runs_the_compose_service_without_its_entrypoint(
    harness_text: str,
) -> None:
    for fragment in (
        'docker compose --env-file /dev/null -f "${REPO_DIR}/docker-compose.jetson.yml"',
        "run -T --rm --no-deps --name ${COLD_CONTAINER}",
        "-e MOUSEDROID_MOCK_HARDWARE=false",
        "-e MOUSEDROID_JETSON_CONFIGS",
        '-w "${REPO_DIR}"',
        "--entrypoint timeout mousedroid --signal=INT --kill-after=5",
        "--entrypoint python3 mousedroid",
        '--name "${COLD_CONTAINER}-dry" --entrypoint true mousedroid',
    ):
        assert fragment in harness_text, fragment


# A stand-in for mousedroid.validation.runtime, so the wrapper's ownership probe can be run for
# real: the settings it "loads" come from $FAKE_SETTINGS.
_FAKE_RUNTIME = """
import json
import os
import types


def load_runtime_settings():
    spec = json.loads(os.environ["FAKE_SETTINGS"])
    if spec.get("raises"):
        raise RuntimeError("settings did not load")
    esp32 = types.SimpleNamespace(enabled=spec["enabled"], protocol=spec["protocol"])
    return types.SimpleNamespace(esp32=esp32, mock_hardware=spec["mock"])
"""


def _run_ownership_probe(
    harness_text: str, tmp_path: Path, spec: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    """Run the python the wrapper sends into the container, against fake settings."""
    found = re.search(r"python3 -c '\n(.*?)\n' 2>/dev/null", harness_text, re.DOTALL)
    assert found, "the ownership probe is no longer a `docker exec ... python3 -c '...'`"
    package = tmp_path / "fake" / "mousedroid" / "validation"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime.py").write_text(_FAKE_RUNTIME, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "FAKE_SETTINGS": json.dumps(spec)}
    return subprocess.run(
        [sys.executable, "-c", found.group(1)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("enabled", "mock", "protocol", "verdict"),
    [
        (True, False, "serial", "esp32_owned=1"),
        (False, False, "serial", "esp32_owned=0"),
        (True, True, "serial", "esp32_owned=0"),
        (True, False, "wifi", "esp32_owned=0"),
    ],
    ids=["serial-enabled", "disabled", "mock-hardware", "wifi-board"],
)
def test_the_ownership_probe_counts_only_a_real_serial_driver(
    harness_text: str, tmp_path: Path, enabled: bool, mock: bool, protocol: str, verdict: str
) -> None:
    """The orchestrator holds the UART only when its settings build SerialESP32Driver.

    ``esp32.enabled`` with ``mock_hardware`` builds the mock, and a wifi board holds no serial
    port, so neither may make the wrapper refuse a warm run or stop the container for a cold one.
    """
    spec = {"enabled": enabled, "mock": mock, "protocol": protocol}
    proc = _run_ownership_probe(harness_text, tmp_path, spec)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == verdict, proc.stdout


def test_the_ownership_probe_prints_no_verdict_when_the_settings_do_not_load(
    harness_text: str, tmp_path: Path
) -> None:
    """A probe that did not decide must not read as "free": no sentinel, so the caller refuses."""
    spec = {"enabled": True, "mock": False, "protocol": "serial", "raises": True}
    proc = _run_ownership_probe(harness_text, tmp_path, spec)
    assert proc.returncode == 2
    assert "esp32_owned" not in proc.stdout


# --- behaviour: the real wrapper against a docker shim ----------------------

# Answers just enough of the docker CLI for the wrapper and writes every call to $SMOKE_EVENTS, in
# order, so a test reads what happened. State lives in $SMOKE_STATE: `running` while the container
# "runs", `stopping` while a `docker stop` is in flight, `containers/<name>` for transient
# containers that exist. Three Docker behaviours are modelled because the wrapper's safety rests
# on them (the review of Task 4.1+4.2 verified each on Docker 29.4 / Compose 5.1):
#   * `docker start` while a stop is in flight does nothing and the container then ends exited;
#     `docker restart` waits for the stop and starts it;
#   * `-e NAME` with NAME unset on the host REMOVES NAME from a `docker exec` environment (the
#     container's own value included), while `docker compose run` is left with the service's own;
#   * a second container under the same --name is a Conflict until the first is removed.
_FAKE_DOCKER = r"""#!/usr/bin/env bash
event() {
    local line="$*"
    printf '%s\n' "${line//$'\n'/ }" >> "${SMOKE_EVENTS}"
}

forwarded_env() {
    local verb="$1"
    shift
    local args=("$@") i spec
    for ((i = 0; i < ${#args[@]} - 1; i++)); do
        [[ "${args[i]}" == "-e" ]] || continue
        spec="${args[i + 1]}"
        if [[ "${spec}" == *=* ]]; then
            event "env ${spec}"
        elif [[ -n "${!spec+x}" ]]; then
            event "env ${spec}=${!spec}"
        elif [[ "${verb}" == "exec" ]]; then
            event "unset ${spec}"
        fi
    done
}

name_of() {
    local i j
    for ((i = 1; i < $#; i++)); do
        if [[ "${!i}" == "--name" ]]; then
            j=$((i + 1))
            printf '%s' "${!j}"
            return
        fi
    done
}

case "$1" in
    ps)
        if [[ -e "${SMOKE_STATE}/running" ]]; then echo "mousedroid"; fi
        ;;
    stop)
        event "docker $*"
        touch "${SMOKE_STATE}/stopping"
        sleep "${SHIM_STOP_SECONDS:-0}"
        if [[ "${SHIM_STOP_RC:-0}" != "0" ]]; then
            rm -f "${SMOKE_STATE}/stopping"
            exit "${SHIM_STOP_RC}"
        fi
        rm -f "${SMOKE_STATE}/running" "${SMOKE_STATE}/stopping"
        ;;
    start)
        event "docker $*"
        if [[ -e "${SMOKE_STATE}/stopping" ]]; then exit 0; fi
        if [[ "${SHIM_START_RC:-0}" != "0" ]]; then exit "${SHIM_START_RC}"; fi
        touch "${SMOKE_STATE}/running"
        ;;
    restart)
        event "docker $*"
        if [[ "${SHIM_RESTART_RC:-0}" != "0" ]]; then exit "${SHIM_RESTART_RC}"; fi
        for ((n = 0; n < 400; n++)); do
            [[ -e "${SMOKE_STATE}/stopping" ]] || break
            sleep 0.05
        done
        touch "${SMOKE_STATE}/running"
        ;;
    rm)
        event "docker $*"
        rm -f "${SMOKE_STATE}/containers/${!#}"
        ;;
    exec)
        if [[ "$*" == *load_runtime_settings* ]]; then
            event "probe"
            printf '%s\n' "${SHIM_PROBE_OUTPUT-esp32_owned=0}"
            exit 0
        fi
        event "docker $*"
        forwarded_env exec "$@"
        ;;
    compose)
        event "docker $*"
        if [[ "$*" == *"--entrypoint true"* ]]; then
            if [[ "${SHIM_DRY_RC:-0}" != "0" ]]; then
                printf '%s\n' "${SHIM_DRY_STDERR:-compose run failed}" >&2
                exit "${SHIM_DRY_RC}"
            fi
            exit 0
        fi
        forwarded_env compose "$@"
        name="$(name_of "$@")"
        runs=$(($(cat "${SMOKE_STATE}/compose_runs" 2>/dev/null || echo 0) + 1))
        echo "${runs}" > "${SMOKE_STATE}/compose_runs"
        if [[ -e "${SMOKE_STATE}/containers/${name}" ]]; then
            echo "Conflict. The container name \"/${name}\" is already in use" >&2
            exit 1
        fi
        case ",${SHIM_LEAK_RUNS:-}," in
            *",${runs},"*) touch "${SMOKE_STATE}/containers/${name}" ;;
        esac
        ;;
    inspect)
        health=0
        for arg in "$@"; do
            if [[ "${arg}" == "{{.State.Health.Status}}" ]]; then health=1; fi
        done
        if ((health)); then
            event "health-poll"
            IFS=',' read -r -a seq <<< "${SHIM_HEALTH:-healthy}"
            idx="$(cat "${SMOKE_STATE}/health_idx" 2>/dev/null || echo 0)"
            if ((idx >= ${#seq[@]})); then idx=$((${#seq[@]} - 1)); fi
            echo "${seq[idx]}"
            echo $((idx + 1)) > "${SMOKE_STATE}/health_idx"
        else
            echo "shim"
        fi
        ;;
    *)
        event "docker $*"
        ;;
esac
exit 0
"""

# Stand-in for scripts/jetson_smoke_test.sh: one event per stage, then one call through the python
# runtime the wrapper chose, so the shim sees what that runtime would run, and its exit status is
# the stage's. SMOKE_BLOCK_STAGE holds a stage until $SMOKE_RELEASE exists; SMOKE_FAIL_STAGES lists
# stages that exit 1; SMOKE_SKIP_STAGES and SMOKE_MIXED_STAGES print what the real stage script
# prints when everything skipped / when something passed and something skipped.
_STUB_STAGE = r"""#!/bin/bash
stage="${1:-}"
printf 'stage %s python=%s allow_motion=%s\n' "${stage}" "${MOUSEDROID_SMOKE_PYTHON:-}" \
    "${MOUSEDROID_SMOKE_ALLOW_MOTION:-}" >> "${SMOKE_EVENTS}"
case ",${SMOKE_SKIP_STAGES:-}," in
    *",${stage},"*) printf '  SKIP: %s smoke\n        pyserial not installed\n' "${stage}" ;;
esac
case ",${SMOKE_MIXED_STAGES:-}," in
    *",${stage},"*)
        printf '  PASS: %s smoke\n  SKIP: %s extra\n        optional check absent\n' \
            "${stage}" "${stage}"
        ;;
esac
rc=0
"${MOUSEDROID_SMOKE_PYTHON}" -c "print('${stage}')" > /dev/null 2>&1 || rc=$?
if [[ "${SMOKE_BLOCK_STAGE:-}" == "${stage}" ]]; then
    while [[ ! -e "${SMOKE_RELEASE}" ]]; do sleep 0.05; done
fi
case ",${SMOKE_FAIL_STAGES:-}," in
    *",${stage},"*) exit 1 ;;
esac
exit "${rc}"
"""

_STUB_NEW_FEATURES = r"""#!/bin/bash
printf 'stage new_features\n' >> "${SMOKE_EVENTS}"
"""

# The wrapper wraps every stage in GNU `timeout`, so the behavioural rows need it as well as bash.
_WRAPPER_NEEDS = requires_bash("timeout")

_OWNED = "esp32_owned=1"
_FREE = "esp32_owned=0"
_OWNED_REASON = (
    "warm smoke refused: the orchestrator in mousedroid owns the ESP32 port; "
    "re-run with MOUSEDROID_SMOKE_ESP32_MODE=cold"
)
_COLD_COMPOSE = "docker compose --env-file /dev/null -f"
_DRY_RUN = "--name mousedroid-cold-smoke-dry --entrypoint true mousedroid"


@dataclass(frozen=True)
class _Run:
    returncode: int
    output: str
    events: list[str]
    summary: str
    run_dir: Path
    repo: Path
    env: dict[str, str]


def _fake_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A throwaway checkout holding the real wrapper, stub stage scripts and a docker shim.

    Returns the wrapper path, the checkout root and the environment to run it under.
    """
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(_REPO_ROOT / _HARNESS, scripts / _HARNESS.name)
    (scripts / "jetson_smoke_test.sh").write_text(_STUB_STAGE, encoding="utf-8")
    (scripts / "jetson_new_features_probe.sh").write_text(_STUB_NEW_FEATURES, encoding="utf-8")
    (repo / "docker-compose.jetson.yml").write_text("services: {}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    state = tmp_path / "state"
    (state / "containers").mkdir(parents=True)
    (state / "running").touch()
    events = tmp_path / "events.log"
    events.touch()
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "MOUSEDROID_SMOKE_REPORT_ROOT": str(tmp_path / "reports"),
        "SMOKE_EVENTS": str(events),
        "SMOKE_STATE": str(state),
        "SMOKE_RELEASE": str(tmp_path / "release"),
    }
    return scripts / _HARNESS.name, repo, env


def _collect(tmp_path: Path, repo: Path, env: dict[str, str], returncode: int, output: str) -> _Run:
    events = Path(env["SMOKE_EVENTS"]).read_text(encoding="utf-8").splitlines()
    run_dirs = sorted((tmp_path / "reports").glob("*"))
    run_dir = run_dirs[-1] if run_dirs else tmp_path / "reports"
    summary_file = run_dir / "SUMMARY.md"
    summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else ""
    return _Run(returncode, output, events, summary, run_dir, repo, env)


def _run_wrapper(tmp_path: Path, **env_extra: str) -> _Run:
    script, repo, env = _fake_repo(tmp_path)
    env.update(env_extra)
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    return _collect(tmp_path, repo, env, proc.returncode, proc.stdout + proc.stderr)


def _patch_wrapper(script: Path, old: str, new: str) -> None:
    """Edit the throwaway COPY of the wrapper, never the real one: inject a fault to test around."""
    text = script.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must appear exactly once in the wrapper"
    script.write_text(text.replace(old, new), encoding="utf-8")


def _stages_run(events: list[str]) -> list[str]:
    return [line.split()[1] for line in events if line.startswith("stage ")]


def _container_moves(events: list[str]) -> list[str]:
    """Every stop, start, restart, remove or compose call: what warm mode must never make."""
    moves = ("docker stop", "docker start", "docker restart", "docker rm", "docker compose")
    return [event for event in events if event.startswith(moves)]


def _start(tmp_path: Path, **env_extra: str) -> tuple[subprocess.Popen[bytes], Path, Path, Path]:
    """Start the wrapper without waiting, to signal it; returns the process and its three files."""
    script, _, env = _fake_repo(tmp_path)
    env.update(env_extra)
    proc = subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, Path(env["SMOKE_EVENTS"]), Path(env["SMOKE_RELEASE"]), Path(env["SMOKE_STATE"])


def _wait_for_event(events_file: Path, prefix: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not any(
        line.startswith(prefix) for line in events_file.read_text(encoding="utf-8").splitlines()
    ):
        assert time.monotonic() < deadline, f"no {prefix!r} event within {timeout_s} s"
        time.sleep(0.05)


@_WRAPPER_NEEDS
def test_warm_mode_refuses_every_esp32_stage_while_the_orchestrator_owns_the_port(
    tmp_path: Path,
) -> None:
    """Warm smoke beside an orchestrator that holds the port is refused, stage by stage.

    serial and power are demoted so the run reaches all four stages; each is recorded as
    EXPECTED-FAIL with the refusal and never started, and the rest of the run still happens.
    """
    run = _run_wrapper(
        tmp_path,
        SHIM_PROBE_OUTPUT=_OWNED,
        MOUSEDROID_SMOKE_BLOCKING_SERIAL="no",
        MOUSEDROID_SMOKE_BLOCKING_POWER="no",
    )
    for stage in _ESP32_STAGES:
        row = f"| {stage} | EXPECTED-FAIL | {_OWNED_REASON} (non-blocking) |"
        assert row in run.summary, run.summary
    ran = _stages_run(run.events)
    assert not {"serial", "motor", "power"} & set(ran), ran
    assert {"system", "usbc", "gpio", "lidar", "camera", "e2e"} <= set(ran), ran
    assert not any("test_motor_smoke.py" in event for event in run.events), run.events
    assert _container_moves(run.events) == [], "warm mode never touches the container"
    assert run.events.count("probe") == 1, "ownership is asked once per run, not once per stage"
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_warm_mode_stops_the_run_at_a_refused_blocking_stage(tmp_path: Path) -> None:
    """serial is blocking, so its refusal is a FAIL that stops the run like any blocking failure."""
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=_OWNED)
    assert f"| serial | FAIL | {_OWNED_REASON} |" in run.summary, run.summary
    assert _stages_run(run.events) == ["system", "usbc", "gpio"], run.events
    assert "| new_features | SKIPPED |" in run.summary, run.summary
    assert _container_moves(run.events) == []
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
@pytest.mark.parametrize(
    "probe", ["", "Traceback (most recent call last):"], ids=["silent", "crash"]
)
def test_warm_mode_refuses_when_it_cannot_tell_who_owns_the_port(
    tmp_path: Path, probe: str
) -> None:
    """A probe that did not decide is not an answer of "free": the stage is refused, not run."""
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=probe)
    assert (
        "| serial | FAIL | warm smoke refused: could not tell whether the orchestrator in "
        "mousedroid owns the ESP32 port"
    ) in run.summary, run.summary
    assert "serial" not in _stages_run(run.events), run.events
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
def test_warm_mode_runs_the_esp32_stages_in_the_container_while_it_runs_the_mock(
    tmp_path: Path,
) -> None:
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=_FREE)
    warm = f"python={run.run_dir}/python3-in-container"
    for stage in ("serial", "motor", "power"):
        assert any(e.startswith(f"stage {stage} {warm}") for e in run.events), (stage, run.events)
    assert any("test_motor_smoke.py" in event for event in run.events), run.events
    assert _container_moves(run.events) == [], "warm mode never touches the container"
    assert "- ESP32 stages: warm" in run.summary, run.summary
    # mcp_motor_smoke has no bench overlay, so under esp32.enabled: false it measures the mock;
    # the row says so instead of reading as a pass on the board.
    assert "| mcp_motor_smoke | PASS | mock driver: " in run.summary, run.summary
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_cold_mode_runs_the_esp32_stages_in_a_transient_container_between_a_stop_and_a_restart(
    tmp_path: Path,
) -> None:
    """The window order; mcp_motor_smoke needs a window of its own while the brain owns the port."""
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_PROBE_OUTPUT=_OWNED)
    events = run.events
    assert run.returncode == 0, run.output
    assert "- ESP32 stages: cold" in run.summary, run.summary
    assert events.count("docker stop mousedroid") == 2, events
    assert events.count("docker restart mousedroid") == 2, events
    assert "docker start mousedroid" not in events, "a start is a no-op while a stop is in flight"

    cold = f"python={run.run_dir}/python3-in-cold-container"
    at = {
        name: next(i for i, e in enumerate(events) if e.startswith(f"stage {name} "))
        for name in ("serial", "motor", "power", "lidar")
    }
    dry = next(i for i, e in enumerate(events) if e.endswith(_DRY_RUN))
    stop = events.index("docker stop mousedroid")
    restart = events.index("docker restart mousedroid")
    removed = events.index("docker rm -f mousedroid-cold-smoke", at["power"])
    # The transient container is proven creatable BEFORE the brain stops; serial, motor and power
    # then share one window, and the container is put back before lidar needs it.
    assert dry < stop < at["serial"] < at["motor"] < at["power"] < removed < restart < at["lidar"]
    for name in ("serial", "motor", "power"):
        assert f"{cold} " in events[at[name]], events[at[name]]
    assert "python3-in-container" in events[at["lidar"]], events[at["lidar"]]
    # mcp_motor_smoke is its own window, closed before the stages that follow it.
    mcp = next(i for i, e in enumerate(events) if "test_motor_smoke.py" in e)
    stop2 = events.index("docker stop mousedroid", stop + 1)
    restart2 = events.index("docker restart mousedroid", restart + 1)
    assert stop2 < mcp < restart2, events
    assert events[mcp].startswith(_COLD_COMPOSE), "the stage runs in the transient container"

    expected = (
        f"{_COLD_COMPOSE} {run.repo}/docker-compose.jetson.yml run -T --rm --no-deps --name "
        "mousedroid-cold-smoke -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS "
        f"-e {_GATE} -w {run.repo} --entrypoint timeout mousedroid --signal=INT --kill-after=5 "
        "55 python3 -c print('serial')"
    )
    assert expected in events, events
    dry_run = (
        f"{_COLD_COMPOSE} {run.repo}/docker-compose.jetson.yml run -T --rm --no-deps {_DRY_RUN}"
    )
    assert dry_run in events, events
    assert sum(e.endswith(_DRY_RUN) for e in events) == 1, "the dry run is asked once per run"
    assert "docker rm -f mousedroid-cold-smoke-dry" in events, (
        "the throwaway cleans up after itself"
    )


@_WRAPPER_NEEDS
def test_cold_mode_reports_a_failed_restart_and_the_trap_retries_it(tmp_path: Path) -> None:
    """If the window cannot be closed the run says so, and the EXIT trap tries once more at exit."""
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_RESTART_RC="1")
    assert run.summary.count("| container_restart | FAIL | docker restart mousedroid failed") == 1
    # the failed close, any later close of the same window, and the trap's own attempt at exit
    assert run.events.count("docker restart mousedroid") >= 2, run.events
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
def test_cold_mode_runs_mcp_motor_smoke_warm_while_the_container_runs_the_mock(
    tmp_path: Path,
) -> None:
    """With nothing owning a port the stage measures the mock: no second window, and it says so."""
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_PROBE_OUTPUT=_FREE)
    events = run.events
    assert run.returncode == 0, run.output
    assert events.count("docker stop mousedroid") == 1, events
    assert events.count("docker restart mousedroid") == 1, events
    mcp = next(e for e in events if "test_motor_smoke.py" in e)
    assert mcp.startswith("docker exec "), "the stage runs in the running container"
    assert "| mcp_motor_smoke | PASS | mock driver: " in run.summary, run.summary
    assert events.count("probe") == 1, "cold mode asks once, for the stage that may skip its window"


@_WRAPPER_NEEDS
@pytest.mark.parametrize("way_out", ["failed-stage", "errexit-abort"])
def test_cold_mode_puts_the_container_back_with_docker_restart_on_every_way_out(
    tmp_path: Path, way_out: str
) -> None:
    """A failed stage, and a `set -e` abort between two stages of an open window, both end running.

    The abort is injected into the throwaway copy of the wrapper at top level, after serial's
    stage and before the window is closed, which is where an unguarded failure in the main flow
    would land (inside run_esp32_stage errexit is suspended by the caller's `|| true`).
    """
    script, _, env = _fake_repo(tmp_path)
    env["MOUSEDROID_SMOKE_ESP32_MODE"] = "cold"
    if way_out == "failed-stage":
        env["SMOKE_FAIL_STAGES"] = "serial"
    else:
        _patch_wrapper(
            script,
            'if [[ "${OVERALL_FAIL}" -eq 0 ]]; then\n    # Motor remains soft until M4',
            "false  # injected: a harness-logic bug aborting the run mid-window\n"
            'if [[ "${OVERALL_FAIL}" -eq 0 ]]; then\n    # Motor remains soft until M4',
        )
    proc = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    events = Path(env["SMOKE_EVENTS"]).read_text(encoding="utf-8").splitlines()
    assert events.index("docker stop mousedroid") < events.index("docker restart mousedroid")
    assert "docker start mousedroid" not in events, events
    assert (tmp_path / "state" / "running").exists(), "the container must be running again"
    assert proc.returncode == 1, proc.stdout


@_WRAPPER_NEEDS
def test_cold_mode_restarts_the_container_when_the_run_is_terminated(tmp_path: Path) -> None:
    """The EXIT trap: a SIGTERM mid-stage still removes the transient container and restarts."""
    proc, events_file, release, state = _start(
        tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SMOKE_BLOCK_STAGE="serial"
    )
    try:
        _wait_for_event(events_file, "stage serial")
        os.kill(proc.pid, signal.SIGTERM)
        time.sleep(0.5)
        release.write_text("", encoding="utf-8")  # let the blocked stub finish
        proc.wait(timeout=30)
    finally:
        release.write_text("", encoding="utf-8")
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    events = events_file.read_text(encoding="utf-8").splitlines()
    stop = events.index("docker stop mousedroid")
    serial = next(i for i, e in enumerate(events) if e.startswith("stage serial "))
    # the trap's own removal: after the stage began (the pre-stage removal comes before it)
    removed = events.index("docker rm -f mousedroid-cold-smoke", serial)
    restart = events.index("docker restart mousedroid")
    assert stop < serial < removed < restart, events
    assert "docker start mousedroid" not in events, events
    assert (state / "running").exists(), "the container must be running again"
    assert proc.returncode != 0


@_WRAPPER_NEEDS
def test_cold_mode_restart_wins_over_a_docker_stop_still_in_flight(tmp_path: Path) -> None:
    """An interrupt can land while `docker stop` is still running (it takes up to the grace period).

    `docker start` then does nothing and the container ends exited, which is the brain left
    down; `docker restart` waits for the stop and starts it again.
    """
    proc, events_file, release, state = _start(
        tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_STOP_SECONDS="2"
    )
    try:
        _wait_for_event(events_file, "docker stop mousedroid")
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
        deadline = time.monotonic() + 15
        while (state / "stopping").exists():  # the orphaned shim `docker stop` finishing
            assert time.monotonic() < deadline, "the in-flight stop never finished"
            time.sleep(0.05)
    finally:
        release.write_text("", encoding="utf-8")
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    events = events_file.read_text(encoding="utf-8").splitlines()
    assert "docker restart mousedroid" in events, events
    assert "docker start mousedroid" not in events, events
    assert (state / "running").exists(), "the brain was left down by a start during a stop"


@_WRAPPER_NEEDS
@pytest.mark.parametrize("way", ["sigpipe", "write-error", "failed-stop"])
def test_cold_mode_restores_the_container_before_it_writes_when_stdout_dies(
    tmp_path: Path, way: str
) -> None:
    """A dropped non-tty ssh session leaves the wrapper writing to a dead pipe, mid-window.

    Whichever write comes first, the container must be restored before anything is logged: a
    restore queued behind a log line dies with the failed write and the brain stays down. Each way
    reaches a different restore. ``sigpipe``: the shell is killed mid-window and the EXIT trap puts
    the container back. ``write-error`` (SIGPIPE ignored by the caller, so the write fails instead):
    ``esp32_cold_end`` closes the window. ``failed-stop`` (``docker stop`` fails): the failed-stop
    branch does. The last two run without the trap, which is the net under them and would hide a
    regression there.
    """
    script, _, env = _fake_repo(tmp_path)
    env.update(MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_STOP_SECONDS="1")
    if way == "failed-stop":
        env["SHIM_STOP_RC"] = "1"
    if way != "sigpipe":
        _patch_wrapper(
            script,
            "    trap restart_container_on_exit EXIT\n    CONTAINER_STOPPED_FOR_COLD=1\n",
            "    CONTAINER_STOPPED_FOR_COLD=1\n",
        )
    argv = ["bash", str(script)]
    if way == "write-error":
        # A signal ignored on entry stays ignored, so the writes fail instead of killing the shell.
        argv = ["bash", "-c", 'trap "" PIPE; exec "$0" "$@"', *argv]
    proc = subprocess.Popen(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True
    )
    events_file = Path(env["SMOKE_EVENTS"])
    try:
        _wait_for_event(events_file, "docker stop mousedroid")
        assert proc.stdout is not None
        proc.stdout.close()  # the reader goes away while `docker stop` is still running
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    events = events_file.read_text(encoding="utf-8").splitlines()
    assert "docker restart mousedroid" in events, events
    assert "docker start mousedroid" not in events, events
    assert (tmp_path / "state" / "running").exists(), "the brain was left stopped"


@_WRAPPER_NEEDS
def test_cold_mode_does_not_run_a_stage_when_the_container_will_not_stop(tmp_path: Path) -> None:
    """If docker stop fails the container may still own the port: refuse, and put it back."""
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_STOP_RC="1")
    assert (
        "| serial | FAIL | cold smoke aborted: docker stop mousedroid failed, so it may still "
        "own the ESP32 port |"
    ) in run.summary, run.summary
    assert "serial" not in _stages_run(run.events), run.events
    assert "docker restart mousedroid" in run.events, run.events
    assert "docker start mousedroid" not in run.events, run.events
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
@pytest.mark.parametrize(
    ("wrapper", "stage_timeout", "tail"),
    [
        ("python3-in-cold-container", "60", "--entrypoint timeout mousedroid --signal=INT"),
        ("python3-in-cold-container", "0", "--entrypoint python3 mousedroid -c print(1)"),
        ("python3-in-cold-container", "3", "--entrypoint timeout mousedroid --signal=INT"),
        ("python3-in-container", "60", "mousedroid timeout --signal=INT"),
    ],
    ids=["cold-60", "cold-none", "cold-3", "warm-60"],
)
def test_wrappers_run_python_under_an_inner_timeout_of_the_same_budget(
    tmp_path: Path, wrapper: str, stage_timeout: str, tail: str
) -> None:
    """Warm and cold alike: the stage budget less 5 s, inside the container (the budget if <= 5).

    Without it a stage that times out is cut off by the host-side `timeout`, whose `--kill-after`
    kills the docker CLI and leaves the transient container running; the next stage then dies on
    `Conflict. The container name ... already in use`.
    """
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_PROBE_OUTPUT=_FREE)
    events_file = Path(run.env["SMOKE_EVENTS"])
    events_file.write_text("", encoding="utf-8")
    env = {**run.env, "MOUSEDROID_SMOKE_STAGE_TIMEOUT": stage_timeout}
    proc = subprocess.run(
        [str(run.run_dir / wrapper), "-c", "print(1)"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    (event,) = [e for e in events_file.read_text(encoding="utf-8").splitlines() if " -c " in e]
    assert tail in event, event
    budget = {"60": "--kill-after=5 55 python3", "3": "--kill-after=5 3 python3"}.get(stage_timeout)
    if budget is not None:
        assert budget in event, event


@_WRAPPER_NEEDS
def test_cold_mode_removes_a_leftover_transient_container_before_each_stage(
    tmp_path: Path,
) -> None:
    """A transient container that outlived its stage (run 2, motor) must not fail the next one.

    Docker refuses a second container under the same --name, so without the removal before each
    stage the blocking `power` stage dies on the Conflict motor left behind.
    """
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_LEAK_RUNS="2")
    events = run.events
    for stage in ("serial", "motor", "power"):
        assert f"| {stage} | PASS |" in run.summary, run.summary
    stage_at = {
        s: next(i for i, e in enumerate(events) if e.startswith(f"stage {s} "))
        for s in ("serial", "motor", "power")
    }
    removals = [i for i, e in enumerate(events) if e == "docker rm -f mousedroid-cold-smoke"]
    previous = -1
    for stage in ("serial", "motor", "power"):
        assert any(previous < r < stage_at[stage] for r in removals), (stage, events)
        previous = stage_at[stage]
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_cold_mode_refuses_before_stopping_when_the_transient_container_cannot_be_created(
    tmp_path: Path,
) -> None:
    """compose run aborts on a docker.env the calling user cannot read; find out with the brain up.

    The dry run is the only safe moment to learn that. A refusal names compose's own message and
    leaves the container running: no stop, no restart, no stage.
    """
    message = "permission denied while trying to read /etc/mousedroid/docker.env"
    run = _run_wrapper(
        tmp_path,
        MOUSEDROID_SMOKE_ESP32_MODE="cold",
        SHIM_DRY_RC="1",
        SHIM_DRY_STDERR=message,
    )
    assert (
        "| serial | FAIL | cold smoke refused: the transient container cannot be created "
        f"({message}); mousedroid was left running |"
    ) in run.summary, run.summary
    assert any(e.endswith(_DRY_RUN) for e in run.events), run.events
    assert not [e for e in run.events if e.startswith(("docker stop", "docker restart"))], (
        run.events
    )
    assert "serial" not in _stages_run(run.events), run.events
    assert (tmp_path / "state" / "running").exists(), "the brain must be left running"
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
@pytest.mark.parametrize(
    ("health", "polls", "row"),
    [
        ("starting,starting,healthy", 3, "| container_health_wait | INFO | healthy"),
        ("starting", 3, "| container_health_wait | INFO | mousedroid not healthy after 3 polls"),
    ],
    ids=["becomes-healthy", "never-healthy"],
)
def test_cold_mode_waits_for_the_restarted_container_before_the_llm_probe(
    tmp_path: Path, health: str, polls: int, row: str
) -> None:
    """The blocking llm_probe loads a model; run beside a still-booting orchestrator it competes.

    The wait is bounded (retries x interval, both env-overridable) and never fails the run: past
    the bound it records that the container was not healthy and the probe runs anyway.
    """
    run = _run_wrapper(
        tmp_path,
        MOUSEDROID_SMOKE_ESP32_MODE="cold",
        SHIM_HEALTH=health,
        MOUSEDROID_SMOKE_HEALTH_RETRIES="3",
        MOUSEDROID_SMOKE_HEALTH_INTERVAL_S="0",
    )
    events = run.events
    polled = [i for i, e in enumerate(events) if e == "health-poll"]
    restart = max(i for i, e in enumerate(events) if e == "docker restart mousedroid")
    llm = next(i for i, e in enumerate(events) if "LLM_PROBE_READY" in e)
    assert len(polled) == polls, events
    assert restart < polled[0] < polled[-1] < llm, events
    assert row in run.summary, run.summary
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_warm_mode_does_not_wait_for_health(tmp_path: Path) -> None:
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=_FREE)
    assert "health-poll" not in run.events, run.events
    assert "container_health_wait" not in run.summary, run.summary


@_WRAPPER_NEEDS
@pytest.mark.parametrize("stage", ["serial", "motor", "power"])
def test_an_esp32_stage_that_only_skipped_records_skipped_never_pass(
    tmp_path: Path, stage: str
) -> None:
    """The stage script exits 0 when everything skipped (pyserial missing, overlay opted out).

    A clean exit is not a measurement: the row says SKIPPED and carries the reason, in the live
    log as well as in SUMMARY.md. A stage that passed something and skipped something stays a
    PASS, and stages that are not ESP32 stages are untouched.
    """
    run = _run_wrapper(
        tmp_path,
        SHIM_PROBE_OUTPUT=_FREE,
        SMOKE_SKIP_STAGES=f"{stage},system",
        SMOKE_MIXED_STAGES="gpio",
    )
    assert f"| {stage} | SKIPPED | pyserial not installed |" in run.summary, run.summary
    assert f"| {stage} | PASS |" not in run.summary, run.summary
    assert f"SKIPPED: {stage} pyserial not installed" in run.output, run.output
    assert "| gpio | PASS |" in run.summary, run.summary
    assert "| system | PASS |" in run.summary, "only ESP32 stages are held to a measurement"
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_an_unknown_esp32_mode_is_fatal_before_anything_runs(tmp_path: Path) -> None:
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="lukewarm")
    assert "MOUSEDROID_SMOKE_ESP32_MODE='lukewarm' is invalid" in run.output, run.output
    assert run.events == [], "no docker call, no stage"
    assert run.returncode == 2, run.output


@_WRAPPER_NEEDS
@pytest.mark.parametrize(
    ("wrapper", "mode", "unset_event"),
    [
        # exec: an unset host variable REMOVES the gate from the stage's environment, so only an
        # export on the host can open it, whatever the container's own docker.env says.
        ("python3-in-container", "warm", f"unset {_GATE}"),
        # compose run: nothing is passed, so the service's own environment (docker.env) stands.
        ("python3-in-cold-container", "cold", None),
    ],
    ids=["warm-wrapper", "cold-wrapper"],
)
def test_generated_wrappers_forward_the_motion_gate_by_name_only(
    tmp_path: Path, wrapper: str, mode: str, unset_event: str | None
) -> None:
    """Unset on the host, the gate must never arrive as an empty string; set, it arrives as set."""
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=_FREE, MOUSEDROID_SMOKE_ESP32_MODE=mode)
    events_file = Path(run.env["SMOKE_EVENTS"])

    def forwarded(**extra: str) -> list[str]:
        events_file.write_text("", encoding="utf-8")
        env = {key: value for key, value in run.env.items() if key != _GATE}
        proc = subprocess.run(
            [str(run.run_dir / wrapper), "-c", "print(1)"],
            capture_output=True,
            text=True,
            env={**env, **extra},
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        seen = events_file.read_text(encoding="utf-8").splitlines()
        return [e for e in seen if e.startswith((f"env {_GATE}", f"unset {_GATE}"))]

    assert forwarded() == ([unset_event] if unset_event else [])
    assert forwarded(**{_GATE: "true"}) == [f"env {_GATE}=true"]


@_WRAPPER_NEEDS
def test_an_esp32_stage_that_passed_something_stays_a_pass(tmp_path: Path) -> None:
    run = _run_wrapper(tmp_path, SHIM_PROBE_OUTPUT=_FREE, SMOKE_MIXED_STAGES="serial")
    assert "| serial | PASS |" in run.summary, run.summary
    assert "| serial | SKIPPED |" not in run.summary, run.summary
