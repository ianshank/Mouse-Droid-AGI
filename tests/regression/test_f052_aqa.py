"""AQA pins for F-052 Slice D: the promotion gate evaluates what the rover runs.

Task 5.1 asked for ``docker_deploy.sh``'s strict probe to resolve its overlay
through ``validation.runtime.resolve_runtime_config_paths`` so it would honour
the legacy ``MOUSEDROID_JETSON_CONFIG``. Verified against the tree, the premise
does not hold, and doing it would have created the defect it described:

* the rover runs ``mousedroid.main``, which loads ``load_settings(*args.config)``
  -- the explicit ``--config`` from compose's ``command:`` and nothing else;
* the strict probe reads ``MOUSEDROID_CONFIG``, which compose pins inline to the
  same file, and inline ``environment:`` beats ``env_file:``, so ``docker.env``
  cannot move it;
* ``MOUSEDROID_JETSON_CONFIG`` is therefore ignored by BOTH -- there is no
  "rover on the legacy key" on the Docker path;
* ``resolve_runtime_config_paths`` ranks the ``MOUSEDROID_CONFIGS`` CSV lists
  above ``MOUSEDROID_CONFIG``, and those can arrive through ``env_file:``. Routing
  the probe through it would let the probe evaluate an overlay the rover never
  loads.

So rather than change the probe, these tests pin the invariant 5.1 was reaching
for -- "the gate and the rover read the same config" -- at the places it can
actually break: the two independent pins of that path in compose, and the
entrypoint that hands one argument list to both the rover and the health-env
derivation the deploy probe now reads.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

from mousedroid.tools.print_healthcheck_env import main as print_healthcheck_env_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_ENTRYPOINT = _REPO_ROOT / "scripts" / "mousedroid_entrypoint.sh"
_SERVICE = "mousedroid"


def _service() -> dict[str, object]:
    data = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    service = data["services"][_SERVICE]
    assert isinstance(service, dict)
    return service


def _inline_environment(service: dict[str, object]) -> dict[str, str]:
    raw = service.get("environment", [])
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    assert isinstance(raw, list)
    return dict(str(item).split("=", 1) for item in raw if "=" in str(item))


def _command_config_paths(service: dict[str, object]) -> list[str]:
    command = service["command"]
    argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]
    assert "--config" in argv, f"compose command no longer passes --config: {argv}"
    paths: list[str] = []
    for arg in argv[argv.index("--config") + 1 :]:
        if arg.startswith("-"):
            break
        paths.append(arg)
    return paths


class TestTheProbeAndTheRoverReadTheSameConfig:
    def test_compose_pins_the_same_file_for_the_probe_and_the_rover(self) -> None:
        """The two pins must name one file, or the gate passes for the wrong config.

        The strict probe reads ``MOUSEDROID_CONFIG`` from the inline environment;
        the rover reads ``--config`` from ``command:``. Editing one without the
        other is exactly the silent divergence task 5.1 was worried about.
        """
        service = _service()
        probe_path = _inline_environment(service).get("MOUSEDROID_CONFIG")
        rover_paths = _command_config_paths(service)

        assert probe_path, "MOUSEDROID_CONFIG is no longer pinned inline in compose"
        assert rover_paths == [probe_path], (
            f"the strict probe evaluates {probe_path!r} but the rover runs {rover_paths!r}"
        )

    def test_the_probe_path_is_pinned_inline_not_left_to_env_file(self) -> None:
        """Inline ``environment:`` wins over ``env_file:`` -- that is the guarantee.

        If ``MOUSEDROID_CONFIG`` moved to ``docker.env`` only, an operator could
        point the probe somewhere the rover's ``--config`` does not follow.
        """
        assert "MOUSEDROID_CONFIG" in _inline_environment(_service())

    def test_the_entrypoint_hands_one_argument_list_to_both(self) -> None:
        """Health-env derivation and the rover must see the identical ``"$@"``.

        ``docker_deploy.sh`` now reads its probe endpoint from the env file
        ``print_healthcheck_env`` writes; that is only the rover's endpoint if
        both commands receive the same arguments.
        """
        lines = [
            line.strip()
            for line in _ENTRYPOINT.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        derive = [line for line in lines if "mousedroid.tools.print_healthcheck_env" in line]
        rover = [line for line in lines if "mousedroid.main" in line]
        assert len(derive) == 1, derive
        assert len(rover) == 1, rover
        assert rover[0].startswith("exec "), rover[0]

        derive_python, derive_args = _python_module_call(derive[0])
        rover_python, rover_args = _python_module_call(rover[0].removeprefix("exec "))

        # Presence of "$@" is not enough: `print_healthcheck_env --config x "$@"`
        # contains it, and the env file would then describe a different config
        # from the one the rover runs.
        assert derive_args == rover_args, (derive_args, rover_args)
        assert '"$@"' in rover_args, rover_args
        assert derive_python == rover_python, (derive_python, rover_python)


def _python_module_call(line: str) -> tuple[list[str], list[str]]:
    """Split ``<python...> -m <module> <args...> [> redirect]`` into its parts.

    Returns:
        ``(interpreter_tokens, argument_tokens)``, quoting kept as written so
        ``"$@"`` stays distinguishable from ``$@``.
    """
    tokens = shlex.split(line, posix=False)
    module_at = tokens.index("-m") + 1
    args: list[str] = []
    for token in tokens[module_at + 1 :]:
        if token.startswith((">", "<", "|", "&", ";")):
            break
        args.append(token)
    return tokens[: module_at - 1], args


class TestTheLegacyConfigKeysDoNotSteerTheRoverPath:
    """Why 5.1's premise is false, pinned as behaviour rather than prose.

    If the loader ever starts honouring these keys on the ``--config`` path, the
    reasoning above changes and the probe needs revisiting -- this fails first.
    """

    @pytest.mark.parametrize(
        "legacy_key",
        ["MOUSEDROID_JETSON_CONFIG", "MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS"],
    )
    def test_an_explicit_config_wins_over_every_legacy_selector(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        legacy_key: str,
    ) -> None:
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("telemetry:\n  port: 9191\n", encoding="utf-8")
        decoy = tmp_path / "decoy.yaml"
        decoy.write_text("telemetry:\n  port: 7272\n", encoding="utf-8")
        monkeypatch.delenv("MOUSEDROID_TELEMETRY__PORT", raising=False)
        monkeypatch.setenv(legacy_key, str(decoy))

        assert print_healthcheck_env_main(["--config", str(explicit)]) == 0

        out = capsys.readouterr().out
        assert "MOUSEDROID_RESOLVED_TELEMETRY_PORT='9191'" in out, (
            f"{legacy_key} steered the rover-path config resolution:\n{out}"
        )


class TestTheEnvTemplateOffersOnlyKeysThatWork:
    """F-052 task 5.5: document the deploy knobs, but never offer a trap.

    The task asked for every env key ``docker_deploy.sh`` and ``deploy_remote.sh``
    read that the template lacked (it said 13; it is 14). Checked one by one
    against whether setting it in ``/etc/mousedroid/docker.env`` actually works,
    nine do not -- and offering them would repeat the ``MOUSEDROID_TELEMETRY_PORT``
    mistake, a key that reads as settable and does nothing. The comment block in
    the template explains each exclusion; these tests stop the list being
    "completed" later by someone who did not read it.
    """

    _TEMPLATE = _REPO_ROOT / "config" / "docker.env.example"

    # Keys the template offers as an assignment, commented or not.
    @classmethod
    def _offered(cls) -> set[str]:
        text = cls._TEMPLATE.read_text(encoding="utf-8")
        return set(re.findall(r"^\s*#?\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))

    @pytest.mark.parametrize(
        "key",
        [
            "MOUSEDROID_HEALTH_PORT",
            "MOUSEDROID_HEALTH_PATH",
            "MOUSEDROID_HEALTH_TIMEOUT",
            "MOUSEDROID_STRICT_HEALTH",
            "MOUSEDROID_DEPLOY_RECORD",
            # The key that actually moves the rover's port, beside the one that
            # does not.
            "MOUSEDROID_TELEMETRY__PORT",
        ],
    )
    def test_a_working_knob_is_documented(self, key: str) -> None:
        assert key in self._offered(), f"{key} works from docker.env but is undocumented"

    @pytest.mark.parametrize(
        ("key", "why"),
        [
            ("MOUSEDROID_CONTAINER", "compose hardcodes container_name: mousedroid"),
            ("MOUSEDROID_COMPOSE_FILE", "the systemd unit reads COMPOSE_FILE instead"),
            ("MOUSEDROID_DOCKER_ENV_FILE", "it is the path of this file: circular"),
            ("MOUSEDROID_REMOTE_SRC", "deploy_remote.sh runs on the PC, not the rover"),
            ("MOUSEDROID_REMOTE_USER", "deploy_remote.sh runs on the PC, not the rover"),
            # Added with the review fix that gave the remote venv its own knob.
            ("MOUSEDROID_REMOTE_INSTALL_DIR", "deploy_remote.sh runs on the PC, not the rover"),
            ("MOUSEDROID_DEPLOY_ARCHIVE_DIR", "deploy_remote.sh runs on the PC, not the rover"),
            ("MOUSEDROID_DEPLOY_CONFIRM_DIRTY", "deploy_remote.sh runs on the PC, not the rover"),
            ("MOUSEDROID_ROVER_WIP_BRANCH", "deploy_remote.sh runs on the PC, not the rover"),
            ("MOUSEDROID_ROVER_WIP_DATE", "deploy_remote.sh runs on the PC, not the rover"),
        ],
    )
    def test_a_key_that_would_not_work_is_never_offered(self, key: str, why: str) -> None:
        assert key not in self._offered(), f"{key} is offered in docker.env but {why}"

    def test_the_new_knobs_are_all_commented(self) -> None:
        """An UNcommented key would make every provisioned rover warn.

        ``host_env_keys`` compares the template's uncommented key set against the
        deployed file. The defaults are already correct, so none of these may
        join that set.
        """
        text = self._TEMPLATE.read_text(encoding="utf-8")
        live = set(re.findall(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))
        added = {
            "MOUSEDROID_HEALTH_PORT",
            "MOUSEDROID_HEALTH_PATH",
            "MOUSEDROID_HEALTH_TIMEOUT",
            "MOUSEDROID_STRICT_HEALTH",
            "MOUSEDROID_DEPLOY_RECORD",
            "MOUSEDROID_TELEMETRY__PORT",
        }
        assert not (added & live), f"uncommented, so every rover would warn: {added & live}"
