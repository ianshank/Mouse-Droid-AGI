"""Structural tests for the Jetson full smoke harness shell wrapper and its stage script.

Text pins, so they run on every platform. The behaviour behind them is driven for real in
``test_jetson_smoke_orchestrator.py`` (the stage script against a stubbed interpreter and a
fake ``serial``), which needs bash and so skips on Windows.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
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
    assert begin.index("trap restart_container_on_exit EXIT") < begin.index(
        'docker stop "${CONTAINER}"'
    )
    handler = _bash_function(harness_text, "restart_container_on_exit")
    assert handler.index('docker rm -f "${COLD_CONTAINER}"') < handler.index(
        'docker start "${CONTAINER}"'
    )


def test_harness_cold_wrapper_runs_the_compose_service_without_its_entrypoint(
    harness_text: str,
) -> None:
    for fragment in (
        'docker compose --env-file /dev/null -f "${REPO_DIR}/docker-compose.jetson.yml"',
        "run -T --rm --no-deps --name ${COLD_CONTAINER}",
        "-e MOUSEDROID_MOCK_HARDWARE=false",
        "-e MOUSEDROID_JETSON_CONFIGS",
        '-w "${REPO_DIR}"',
        "--entrypoint python3",
    ):
        assert fragment in harness_text, fragment


# --- behaviour: the real wrapper against a docker shim ----------------------

# Answers just enough of the docker CLI for the wrapper, and writes every call to $SMOKE_EVENTS in
# order so a test reads what happened. $SMOKE_STATE/running exists while the container "runs".
# `-e NAME` on exec and compose run is modelled as Docker documents it: the value comes from this
# process's environment when NAME is set there (even to empty) and nothing is passed when it is
# not; `-e NAME=value` passes as given.
_FAKE_DOCKER = r"""#!/usr/bin/env bash
event() {
    local line="$*"
    printf '%s\n' "${line//$'\n'/ }" >> "${SMOKE_EVENTS}"
}

forwarded_env() {
    local args=("$@") i spec
    for ((i = 0; i < ${#args[@]} - 1; i++)); do
        [[ "${args[i]}" == "-e" ]] || continue
        spec="${args[i + 1]}"
        if [[ "${spec}" == *=* ]]; then
            event "env ${spec}"
        elif [[ -n "${!spec+x}" ]]; then
            event "env ${spec}=${!spec}"
        fi
    done
}

case "$1" in
    ps)
        if [[ -e "${SMOKE_STATE}/running" ]]; then echo "mousedroid"; fi
        ;;
    stop)
        event "docker $*"
        if [[ "${SHIM_STOP_RC:-0}" == "0" ]]; then rm -f "${SMOKE_STATE}/running"; fi
        exit "${SHIM_STOP_RC:-0}"
        ;;
    start)
        event "docker $*"
        if [[ "${SHIM_START_RC:-0}" == "0" ]]; then touch "${SMOKE_STATE}/running"; fi
        exit "${SHIM_START_RC:-0}"
        ;;
    exec)
        if [[ "$*" == *load_runtime_settings* ]]; then
            event "probe"
            printf '%s\n' "${SHIM_PROBE_OUTPUT-esp32_owned=0}"
            exit 0
        fi
        event "docker $*"
        forwarded_env "$@"
        ;;
    compose)
        event "docker $*"
        forwarded_env "$@"
        ;;
    inspect)
        echo "shim"
        ;;
    *)
        event "docker $*"
        ;;
esac
exit 0
"""

# Stand-in for scripts/jetson_smoke_test.sh: one event per stage, then one call through the python
# runtime the wrapper chose, so the shim sees what that runtime would run. SMOKE_BLOCK_STAGE holds a
# stage until $SMOKE_RELEASE exists; SMOKE_FAIL_STAGES lists stages that exit 1.
_STUB_STAGE = r"""#!/bin/bash
stage="${1:-}"
printf 'stage %s python=%s allow_motion=%s\n' "${stage}" "${MOUSEDROID_SMOKE_PYTHON:-}" \
    "${MOUSEDROID_SMOKE_ALLOW_MOTION:-}" >> "${SMOKE_EVENTS}"
"${MOUSEDROID_SMOKE_PYTHON}" -c "print('${stage}')" > /dev/null 2>&1
if [[ "${SMOKE_BLOCK_STAGE:-}" == "${stage}" ]]; then
    while [[ ! -e "${SMOKE_RELEASE}" ]]; do sleep 0.05; done
fi
case ",${SMOKE_FAIL_STAGES:-}," in
    *",${stage},"*) exit 1 ;;
esac
exit 0
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
    state.mkdir()
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


def _stages_run(events: list[str]) -> list[str]:
    return [line.split()[1] for line in events if line.startswith("stage ")]


def _container_moves(events: list[str]) -> list[str]:
    """Every stop, start, remove or compose call: what warm mode must never make."""
    moves = ("docker stop", "docker start", "docker rm", "docker compose")
    return [event for event in events if event.startswith(moves)]


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
    assert run.returncode == 0, run.output


@_WRAPPER_NEEDS
def test_cold_mode_runs_the_esp32_stages_in_a_transient_container_between_a_stop_and_a_start(
    tmp_path: Path,
) -> None:
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold")
    events = run.events
    assert run.returncode == 0, run.output
    assert "probe" not in events, "cold mode stops the container instead of asking who owns it"
    assert "- ESP32 stages: cold" in run.summary, run.summary
    assert events.count("docker stop mousedroid") == 2, events
    assert events.count("docker start mousedroid") == 2, events

    cold = f"python={run.run_dir}/python3-in-cold-container"
    at = {
        name: next(i for i, e in enumerate(events) if e.startswith(f"stage {name} "))
        for name in ("serial", "motor", "power", "lidar")
    }
    stop = events.index("docker stop mousedroid")
    start = events.index("docker start mousedroid")
    removed = events.index("docker rm -f mousedroid-cold-smoke")
    # serial, motor and power share one window: stop, then all three in the transient container,
    # then the leftover is removed and the container starts again before lidar needs it.
    assert stop < at["serial"] < at["motor"] < at["power"] < removed < start < at["lidar"], events
    for name in ("serial", "motor", "power"):
        assert f"{cold} " in events[at[name]], events[at[name]]
    assert "python3-in-container" in events[at["lidar"]], events[at["lidar"]]
    # mcp_motor_smoke is its own window, closed before the stages that follow it.
    mcp = next(i for i, e in enumerate(events) if "test_motor_smoke.py" in e)
    stop2 = events.index("docker stop mousedroid", stop + 1)
    start2 = events.index("docker start mousedroid", start + 1)
    assert stop2 < mcp < start2, events

    expected = (
        f"docker compose --env-file /dev/null -f {run.repo}/docker-compose.jetson.yml run -T --rm "
        "--no-deps --name mousedroid-cold-smoke -e MOUSEDROID_MOCK_HARDWARE=false "
        f"-e MOUSEDROID_JETSON_CONFIGS -e {_GATE} -w {run.repo} --entrypoint python3 mousedroid "
        "-c print('serial')"
    )
    assert expected in events, events


@_WRAPPER_NEEDS
def test_cold_mode_restarts_the_container_when_the_run_is_terminated(tmp_path: Path) -> None:
    """The EXIT trap: a SIGTERM mid-stage still removes the transient container and starts it."""
    script, _, env = _fake_repo(tmp_path)
    env.update(MOUSEDROID_SMOKE_ESP32_MODE="cold", SMOKE_BLOCK_STAGE="serial")
    events_file = Path(env["SMOKE_EVENTS"])
    release = Path(env["SMOKE_RELEASE"])
    proc = subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not any(
            line.startswith("stage serial") for line in events_file.read_text().splitlines()
        ):
            assert time.monotonic() < deadline, "the serial stage never started"
            time.sleep(0.05)
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
    removed = events.index("docker rm -f mousedroid-cold-smoke")
    start = events.index("docker start mousedroid")
    assert stop < removed < start, events
    assert (tmp_path / "state" / "running").exists(), "the container must be running again"
    assert proc.returncode != 0


@_WRAPPER_NEEDS
def test_cold_mode_does_not_run_a_stage_when_the_container_will_not_stop(tmp_path: Path) -> None:
    """If docker stop fails the container may still own the port: refuse, and put it back."""
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="cold", SHIM_STOP_RC="1")
    assert (
        "| serial | FAIL | cold smoke aborted: docker stop mousedroid failed, so it may still "
        "own the ESP32 port |"
    ) in run.summary, run.summary
    assert "serial" not in _stages_run(run.events), run.events
    assert "docker start mousedroid" in run.events, run.events
    assert run.returncode == 1, run.output


@_WRAPPER_NEEDS
def test_an_unknown_esp32_mode_is_fatal_before_anything_runs(tmp_path: Path) -> None:
    run = _run_wrapper(tmp_path, MOUSEDROID_SMOKE_ESP32_MODE="lukewarm")
    assert "MOUSEDROID_SMOKE_ESP32_MODE='lukewarm' is invalid" in run.output, run.output
    assert run.events == [], "no docker call, no stage"
    assert run.returncode == 2, run.output


@_WRAPPER_NEEDS
@pytest.mark.parametrize(
    ("wrapper", "mode"),
    [("python3-in-container", "warm"), ("python3-in-cold-container", "cold")],
    ids=["warm-wrapper", "cold-wrapper"],
)
def test_generated_wrappers_forward_the_motion_gate_by_name_only(
    tmp_path: Path, wrapper: str, mode: str
) -> None:
    """Unset on the host, the gate must not reach the container, not even as an empty string."""
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
        return events_file.read_text(encoding="utf-8").splitlines()

    unset = forwarded()
    assert any(f"-e {_GATE}" in event for event in unset), unset
    assert not [event for event in unset if event.startswith(f"env {_GATE}")], unset
    opened = forwarded(**{_GATE: "true"})
    assert [event for event in opened if event.startswith(f"env {_GATE}")] == [
        f"env {_GATE}=true"
    ], opened
