"""Unit: ``docker_deploy.sh`` probes the telemetry endpoint the rover actually serves.

The rover serves ``cfg.telemetry.port`` (compose runs ``network_mode: host``) at
``f"{cfg.telemetry.api_prefix}/health"``. The deploy script used to probe
``${MOUSEDROID_TELEMETRY_PORT:-8080}`` — but that is not a pydantic-settings key:
the nested delimiter is ``__``, so the real one is ``MOUSEDROID_TELEMETRY__PORT``.
That broke in both directions, and both are reproduced here against the shipped
script:

* moving the port the supported way (``MOUSEDROID_TELEMETRY__PORT=9191``) left
  the probe on ``8080``;
* setting the template's ``MOUSEDROID_TELEMETRY_PORT=7070`` moved only the probe.

The script now reads the endpoint from the running container's healthcheck env
file, which the entrypoint derives from the exact ``--config`` it hands to
``mousedroid.main`` (``print_healthcheck_env``).

Driven end-to-end with ``docker`` and ``curl`` shims on ``PATH``, following
``test_deploy_remote_guard.py``: the assertion is the URL the curl shim was
actually invoked with, never a grep of the script's source. Every refusal case
also asserts that curl was *not* invoked, because "printed an error" and
"probed anyway" are both possible and only one is correct.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY = _REPO_ROOT / "scripts" / "docker_deploy.sh"

pytestmark = requires_bash()

# The docker shim answers the three things health_check() asks of a container:
# is it running (``ps``), what did its entrypoint resolve (``exec ... cat`` of
# the healthcheck env file), and everything else (``exec`` of python probes),
# which must print OK so the import leg passes and the endpoint leg is reached.
_DOCKER_SHIM = """#!/usr/bin/env bash
case "$1" in
    ps) echo "abc123" ;;
    exec)
        if [[ "$*" == *HEALTHCHECK_ENV_FILE* ]]; then
            cat "${FAKE_CONTAINER_ENV:-/dev/null}"
        else
            echo "OK"
        fi
        ;;
    *) : ;;
esac
exit 0
"""

_CURL_SHIM = """#!/usr/bin/env bash
for arg in "$@"; do
    case "$arg" in
        http://*) printf '%s\\n' "$arg" >> "$CURL_LOG" ;;
    esac
done
exit 0
"""

# Shaped exactly as print_healthcheck_env writes it: KEY='value', one per line.
_BASE_KEYS = (
    "MOUSEDROID_HEARTBEAT_PATH='/tmp/mousedroid_heartbeat'",
    "MOUSEDROID_HEARTBEAT_STALE_S='30.000'",
)


def _container_env(tmp_path: Path, *, port: str | None, path: str | None) -> Path:
    """Write the healthcheck env file the fake container's entrypoint produced.

    ``None`` omits the key, which is what an image predating the resolved keys
    looks like.
    """
    lines = list(_BASE_KEYS)
    if port is not None:
        lines.append(f"MOUSEDROID_RESOLVED_TELEMETRY_PORT='{port}'")
    if path is not None:
        lines.append(f"MOUSEDROID_RESOLVED_HEALTH_PATH='{path}'")
    env_file = tmp_path / "container.env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


def _run(
    tmp_path: Path,
    container_env: Path,
    *args: str,
    **overrides: str,
) -> tuple[list[str], str]:
    """Run ``docker_deploy.sh --health-only`` against the shims.

    Returns:
        ``(probed_urls, combined_output)``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("docker", _DOCKER_SHIM), ("curl", _CURL_SHIM)):
        shim = bin_dir / name
        shim.write_text(body, encoding="utf-8")
        shim.chmod(0o755)
    curl_log = tmp_path / "curl.log"
    curl_log.write_text("", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "CURL_LOG": str(curl_log),
        "FAKE_CONTAINER_ENV": str(container_env),
        "MOUSEDROID_INSTALL_DIR": str(tmp_path),
        "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
        "MOUSEDROID_DOCKER_ENV_FILE": str(tmp_path / "absent.env"),
        **overrides,
    }
    # Keep a developer's own shell settings out of the fixture: each test states
    # exactly which telemetry/health keys are present.
    for key in (
        "MOUSEDROID_TELEMETRY_PORT",
        "MOUSEDROID_HEALTH_PORT",
        "MOUSEDROID_HEALTH_PATH",
    ):
        if key not in overrides:
            env.pop(key, None)

    proc = subprocess.run(
        ["bash", str(_DEPLOY), "--health-only", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(_REPO_ROOT),
    )
    probed = [line for line in curl_log.read_text(encoding="utf-8").splitlines() if line]
    return probed, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# The probe follows the rover
# ---------------------------------------------------------------------------


def test_the_probe_follows_the_port_the_rover_resolved(tmp_path: Path) -> None:
    """The supported way to move the port must move the probe too.

    Against the previous script this probes ``:8080`` — the regression this
    file exists for.
    """
    container = _container_env(tmp_path, port="9191", path="/api/v1/health")

    probed, _ = _run(tmp_path, container)

    assert probed == ["http://127.0.0.1:9191/api/v1/health"]


def test_the_probe_follows_a_non_default_api_prefix(tmp_path: Path) -> None:
    container = _container_env(tmp_path, port="8080", path="/rover/v2/health")

    probed, _ = _run(tmp_path, container)

    assert probed == ["http://127.0.0.1:8080/rover/v2/health"]


def test_the_template_telemetry_port_does_not_move_the_probe(tmp_path: Path) -> None:
    """``MOUSEDROID_TELEMETRY_PORT`` never moved the rover, so it must not move the probe.

    The template ships it, and its comment said it set the endpoint's port. The
    operator is told plainly, with both numbers and the key that does work.
    """
    container = _container_env(tmp_path, port="9191", path="/api/v1/health")

    probed, output = _run(tmp_path, container, MOUSEDROID_TELEMETRY_PORT="7070")

    assert probed == ["http://127.0.0.1:9191/api/v1/health"]
    assert "MOUSEDROID_TELEMETRY_PORT=7070 has no effect on the rover" in output
    assert "serves port 9191" in output
    assert "MOUSEDROID_TELEMETRY__PORT" in output


def test_no_warning_when_the_template_port_agrees(tmp_path: Path) -> None:
    """The common case — template ``8080``, rover on ``8080`` — must stay quiet.

    Every provisioned rover carries the template's value, so a warning that
    fired on agreement would fire on every deploy and teach operators to
    ignore it.
    """
    container = _container_env(tmp_path, port="8080", path="/api/v1/health")

    probed, output = _run(tmp_path, container, MOUSEDROID_TELEMETRY_PORT="8080")

    assert probed == ["http://127.0.0.1:8080/api/v1/health"]
    assert "has no effect on the rover" not in output


def test_explicit_overrides_win_over_the_resolved_endpoint(tmp_path: Path) -> None:
    container = _container_env(tmp_path, port="9191", path="/api/v1/health")

    probed, _ = _run(
        tmp_path,
        container,
        MOUSEDROID_HEALTH_PORT="7777",
        MOUSEDROID_HEALTH_PATH="/custom/health",
    )

    assert probed == ["http://127.0.0.1:7777/custom/health"]


# ---------------------------------------------------------------------------
# Backwards compatibility: an image that predates the resolved keys
# ---------------------------------------------------------------------------


def test_an_older_image_keeps_the_previous_behaviour_and_says_so(tmp_path: Path) -> None:
    """``--no-build`` can pull an image whose entrypoint never wrote the keys.

    That case reproduces the previous fallback exactly — it is the only
    behaviour available without the keys — and is announced, not silent.
    """
    container = _container_env(tmp_path, port=None, path=None)

    probed, output = _run(tmp_path, container)

    assert probed == ["http://127.0.0.1:8080/api/v1/health"]
    assert "Could not read the rover's resolved telemetry endpoint" in output


def test_an_older_image_still_honours_the_legacy_port_variable(tmp_path: Path) -> None:
    """...including the legacy variable, because that is what the old probe did."""
    container = _container_env(tmp_path, port=None, path=None)

    probed, _ = _run(tmp_path, container, MOUSEDROID_TELEMETRY_PORT="7070")

    assert probed == ["http://127.0.0.1:7070/api/v1/health"]


# ---------------------------------------------------------------------------
# Values read out of the container are validated before they reach a URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_port", ["80;id", "99999", "0", "8080x"])
def test_an_invalid_resolved_port_never_reaches_curl(tmp_path: Path, bad_port: str) -> None:
    container = _container_env(tmp_path, port=bad_port, path="/api/v1/health")

    probed, output = _run(tmp_path, container)

    assert probed == [], f"curl was invoked with an unvalidated port: {probed}"
    assert f"Telemetry health port is not a valid TCP port: {bad_port}" in output


@pytest.mark.parametrize("bad_path", ["api/v1/health", "/api v1/health", "/a?b=c"])
def test_an_invalid_resolved_path_never_reaches_curl(tmp_path: Path, bad_path: str) -> None:
    container = _container_env(tmp_path, port="8080", path=bad_path)

    probed, output = _run(tmp_path, container)

    assert probed == [], f"curl was invoked with an unvalidated path: {probed}"
    assert "Telemetry health path is not a safe absolute URL path" in output


def test_an_invalid_override_is_refused_the_same_way(tmp_path: Path) -> None:
    container = _container_env(tmp_path, port="8080", path="/api/v1/health")

    probed, output = _run(tmp_path, container, MOUSEDROID_HEALTH_PORT="not-a-port")

    assert probed == []
    assert "Telemetry health port is not a valid TCP port: not-a-port" in output
