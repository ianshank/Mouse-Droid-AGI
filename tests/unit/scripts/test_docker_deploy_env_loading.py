"""Unit: ``scripts/docker_deploy.sh`` treats its env file as data, not as code.

``/etc/mousedroid/docker.env`` is an *environment file*. Its three other
consumers all parse it — ``scripts/mousedroid-docker.service`` and
``scripts/mousedroid-trend.service`` via ``EnvironmentFile=``, and
``docker-compose.jetson.yml`` via ``env_file:``. ``docker_deploy.sh`` was the
lone exception: it dot-sourced the file under ``set -a``, and its documented
invocation is ``sudo bash scripts/docker_deploy.sh``, so every line of the
documented home of ``MOUSEDROID_TELEMETRY_TOKEN`` and ``ANTHROPIC_API_KEY``
executed as root.

Two layers, following ``test_deploy_remote_guard.py``'s shape — a REAL fixture
driven through ``bash``, never a string assertion about the script's source:

1. ``_load_env_file_as_data`` extracted from the script and driven against
   throwaway env files, including payloads that would run if the file were
   sourced.
2. The whole script driven end-to-end with ``docker`` / ``curl`` / ``systemctl``
   shims on ``PATH``, so the mode of the env file it *creates* is proven by
   stat-ing the real artefact rather than by grepping for a ``chmod`` line.

On the vacuous-assertion trap ``mouse-droid-deploy-repin/tasks.md:19-32``
records: a sentinel-file test like ``test_a_command_substitution_...`` below
passes trivially if the payload could never have run in the first place. So
``test_the_premise_holds_...`` dot-sources the *same fixture* and asserts the
sentinel IS created — if that ever stops being true, the security test above it
is proving nothing and says so out loud.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY = _REPO_ROOT / "scripts" / "docker_deploy.sh"
_TEMPLATE = _REPO_ROOT / "config" / "docker.env.example"

_LOADER = "_load_env_file_as_data"

pytestmark = requires_bash()


def _extract_loader() -> str:
    """Return the loader function's source, lifted out of the deploy script.

    The script is not sourceable: past the function it parses arguments and
    performs a deployment. So the function under test is extracted by name and
    run on its own, which also keeps the test honest about *which* code it
    exercises — if the function is renamed or deleted, this fails loudly rather
    than silently testing nothing.

    Returns:
        The ``bash`` source of the loader function.

    Raises:
        AssertionError: If the function cannot be located in the script.
    """
    src = _DEPLOY.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(_LOADER)}\(\) \{{.*?^\}}", src, re.DOTALL | re.MULTILINE)
    assert match is not None, (
        f"{_LOADER}() not found in {_DEPLOY}. If it was renamed, update this test — "
        "do not delete it: it is the only proof the env file is not executed."
    )
    return match.group(0)


def _load(env_file: Path, *report: str) -> tuple[dict[str, str], str, int]:
    """Run the extracted loader against ``env_file`` and report chosen keys.

    Args:
        env_file: The environment file to parse.
        report: Variable names to echo back after loading.

    Returns:
        ``(values, stderr, returncode)``. ``values`` maps each requested name to
        its loaded value, or to ``"<unset>"`` when the loader did not set it.
    """
    probe = "; ".join(f'printf "%s\\037%s\\036" "{name}" "${{{name}-<unset>}}"' for name in report)
    script = f'{_extract_loader()}\n{_LOADER} "$1"\n{probe}\n'
    proc = subprocess.run(
        ["bash", "-c", script, "bash", str(env_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    values: dict[str, str] = {}
    for record in proc.stdout.split("\036"):
        if "\037" in record:
            key, _, value = record.partition("\037")
            values[key] = value
    return values, proc.stderr, proc.returncode


# ---------------------------------------------------------------------------
# Layer 1 — the loader does not execute what it reads
# ---------------------------------------------------------------------------


def test_a_command_substitution_in_a_value_is_not_executed(tmp_path: Path) -> None:
    sentinel = tmp_path / "EXECUTED"
    env = tmp_path / "docker.env"
    env.write_text(f"MOUSEDROID_PAYLOAD=$(touch {sentinel})\n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_PAYLOAD")

    assert rc == 0
    assert not sentinel.exists(), "the loader executed the env file's contents"
    # The literal text survives, which is also what systemd would hand the unit.
    assert values["MOUSEDROID_PAYLOAD"] == f"$(touch {sentinel})"


def test_the_premise_holds_dot_sourcing_the_same_fixture_does_execute_it(tmp_path: Path) -> None:
    """The payload above is genuinely dangerous, so the test above is not vacuous.

    This is the guard against the trap the repo has already been bitten by: a
    sentinel that could never have been created makes its own security test
    pass for free. Here the *old* mechanism — ``set -a`` plus a dot-source, what
    ``docker_deploy.sh`` did before this change — is run against the identical
    fixture, and the sentinel must appear.
    """
    sentinel = tmp_path / "EXECUTED"
    env = tmp_path / "docker.env"
    env.write_text(f"MOUSEDROID_PAYLOAD=$(touch {sentinel})\n", encoding="utf-8")

    subprocess.run(
        ["bash", "-c", 'set -a; . "$1"; set +a', "bash", str(env)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert sentinel.exists(), (
        "dot-sourcing this fixture no longer executes it, so the sibling "
        "security test can pass without proving anything. Fix the fixture."
    )


def test_a_backtick_in_a_value_is_not_executed(tmp_path: Path) -> None:
    sentinel = tmp_path / "EXECUTED_BACKTICK"
    env = tmp_path / "docker.env"
    env.write_text(f"MOUSEDROID_PAYLOAD=`touch {sentinel}`\n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_PAYLOAD")

    assert rc == 0
    assert not sentinel.exists()
    assert values["MOUSEDROID_PAYLOAD"] == f"`touch {sentinel}`"


def test_a_tilde_and_a_variable_reference_stay_literal(tmp_path: Path) -> None:
    """No expansion, matching systemd — the script and the units must agree."""
    env = tmp_path / "docker.env"
    env.write_text("MOUSEDROID_A=~/relative\nMOUSEDROID_B=$HOME/x\n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_A", "MOUSEDROID_B")

    assert rc == 0
    assert values["MOUSEDROID_A"] == "~/relative"
    assert values["MOUSEDROID_B"] == "$HOME/x"


# ---------------------------------------------------------------------------
# Layer 1 — and it still loads everything the old mechanism loaded
# ---------------------------------------------------------------------------


def test_plain_key_value_pairs_load(tmp_path: Path) -> None:
    env = tmp_path / "docker.env"
    env.write_text("MOUSEDROID_MOCK_HARDWARE=false\nMOUSEDROID_TELEMETRY_PORT=8080\n", "utf-8")

    values, _, rc = _load(env, "MOUSEDROID_MOCK_HARDWARE", "MOUSEDROID_TELEMETRY_PORT")

    assert rc == 0
    assert values == {"MOUSEDROID_MOCK_HARDWARE": "false", "MOUSEDROID_TELEMETRY_PORT": "8080"}


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Both comment markers systemd honours, including an indented one.

    The ``;`` form matters beyond tidiness: it is a *syntax error* in bash, so
    the previous dot-source aborted the whole file on one, silently dropping
    every key below it.
    """
    env = tmp_path / "docker.env"
    env.write_text(
        "# MOUSEDROID_KEY=from_a_hash_comment\n"
        "   ; MOUSEDROID_KEY=from_a_semicolon_comment\n\n   \n"
        "MOUSEDROID_KEY=real\n",
        encoding="utf-8",
    )

    values, stderr, rc = _load(env, "MOUSEDROID_KEY")

    assert rc == 0
    assert values["MOUSEDROID_KEY"] == "real"
    assert "unparseable" not in stderr


def test_one_layer_of_matching_quotes_is_stripped(tmp_path: Path) -> None:
    env = tmp_path / "docker.env"
    env.write_text(
        'MOUSEDROID_D="two words"\nMOUSEDROID_S=\'two words\'\nMOUSEDROID_N="outer\'inner"\n',
        encoding="utf-8",
    )

    values, _, rc = _load(env, "MOUSEDROID_D", "MOUSEDROID_S", "MOUSEDROID_N")

    assert rc == 0
    assert values["MOUSEDROID_D"] == "two words"
    assert values["MOUSEDROID_S"] == "two words"
    assert values["MOUSEDROID_N"] == "outer'inner"


def test_an_export_prefix_is_accepted(tmp_path: Path) -> None:
    env = tmp_path / "docker.env"
    env.write_text("export MOUSEDROID_CONTAINER=mousedroid\n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_CONTAINER")

    assert rc == 0
    assert values["MOUSEDROID_CONTAINER"] == "mousedroid"


def test_an_empty_value_stays_empty_and_set(tmp_path: Path) -> None:
    """``MOUSEDROID_TELEMETRY_TOKEN=`` ships uncommented-and-empty on purpose.

    ``config/docker.env.example`` says so in the template: the key stays present
    with an empty value so the ``host_env_keys`` preflight check can see it. An
    empty value must therefore round-trip as *set and empty*, never as unset.
    """
    env = tmp_path / "docker.env"
    env.write_text("MOUSEDROID_TELEMETRY_TOKEN=\n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_TELEMETRY_TOKEN")

    assert rc == 0
    assert values["MOUSEDROID_TELEMETRY_TOKEN"] == ""


def test_trailing_whitespace_is_not_part_of_an_unquoted_value(tmp_path: Path) -> None:
    env = tmp_path / "docker.env"
    env.write_text("MOUSEDROID_ESP32_DEV=/dev/ttyUSB0   \n", encoding="utf-8")

    values, _, rc = _load(env, "MOUSEDROID_ESP32_DEV")

    assert rc == 0
    assert values["MOUSEDROID_ESP32_DEV"] == "/dev/ttyUSB0"


def test_a_quoted_value_followed_by_whitespace_loses_its_quotes(tmp_path: Path) -> None:
    """`KEY="v"   ` must yield `v`, not `"v"`.

    Regression for a real bug in the first cut of this parser: the quote test
    ran *before* trailing blanks were stripped, so the anchored pattern did not
    match a line with a trailing space and the quote characters were exported
    literally. That reintroduced the exact script-vs-systemd disagreement the
    parser exists to remove, for a line shape systemd accepts.
    """
    env = tmp_path / "docker.env"
    env.write_text(
        'MOUSEDROID_A="value"   \nMOUSEDROID_B="inner   "\nMOUSEDROID_C=plain   \n',
        encoding="utf-8",
    )

    values, _, rc = _load(env, "MOUSEDROID_A", "MOUSEDROID_B", "MOUSEDROID_C")

    assert rc == 0
    assert values["MOUSEDROID_A"] == "value", "outer quotes leaked into the value"
    # Blanks *inside* the quotes are part of the value and must survive.
    assert values["MOUSEDROID_B"] == "inner   "
    assert values["MOUSEDROID_C"] == "plain"


def test_the_warning_never_echoes_the_line(tmp_path: Path) -> None:
    """A malformed secret assignment must not put the secret in stderr.

    The file holds `ANTHROPIC_API_KEY` and `MOUSEDROID_TELEMETRY_TOKEN`, and a
    typo in one of those is precisely the line that fails to parse — so echoing
    the offending line writes the credential to stderr and the journal. The
    warning reports the file and line number only.
    """
    secret = "sk-ant-thisvaluemustnotappearinstderr"  # noqa: S105 - fixture, not a credential
    env = tmp_path / "docker.env"
    env.write_text(f"ANTHROPIC_API_KEY {secret}\n", encoding="utf-8")

    _, stderr, rc = _load(env)

    assert rc == 0
    assert secret not in stderr, "the parser echoed a credential into stderr"
    assert "ANTHROPIC_API_KEY" not in stderr, "the parser echoed the key name too"
    # Still actionable: the operator gets a location to go and look at.
    assert "docker.env:1: ignoring unparseable line" in stderr


def test_the_reported_line_number_is_the_real_one(tmp_path: Path) -> None:
    """...and the location is correct, or it is not actionable.

    Counting must include comments and blank lines, otherwise the number sends
    the operator to the wrong line of a file they cannot have echoed.
    """
    env = tmp_path / "docker.env"
    env.write_text("# comment\n\nGOOD=1\nbroken line here\n", encoding="utf-8")

    _, stderr, rc = _load(env, "GOOD")

    assert rc == 0
    assert "docker.env:4: ignoring unparseable line" in stderr


def test_an_unparseable_line_is_warned_about_and_does_not_lose_other_keys(
    tmp_path: Path,
) -> None:
    """Warn-and-skip, as systemd does — a malformed line must not be fatal.

    Failing closed here would brick an upgrade on any rover whose file has an
    unrelated stray line, and the keys that *do* parse include the telemetry
    token.
    """
    env = tmp_path / "docker.env"
    env.write_text(
        "MOUSEDROID_BEFORE=1\nthis is not an assignment\nMOUSEDROID_AFTER=2\n", encoding="utf-8"
    )

    values, stderr, rc = _load(env, "MOUSEDROID_BEFORE", "MOUSEDROID_AFTER")

    assert rc == 0
    assert values == {"MOUSEDROID_BEFORE": "1", "MOUSEDROID_AFTER": "2"}
    # Location, not contents — see test_the_warning_never_echoes_the_line.
    assert "docker.env:2: ignoring unparseable line" in stderr
    assert "this is not an assignment" not in stderr


def test_a_lowercase_or_dashed_key_is_rejected_rather_than_assigned(tmp_path: Path) -> None:
    """The key pattern is what makes ``export "$key=$value"`` safe to run."""
    env = tmp_path / "docker.env"
    env.write_text("not-an-identifier=x\nMOUSEDROID_OK=y\n", encoding="utf-8")

    values, stderr, rc = _load(env, "MOUSEDROID_OK")

    assert rc == 0
    assert values["MOUSEDROID_OK"] == "y"
    assert "docker.env:1: ignoring unparseable line" in stderr
    assert "not-an-identifier" not in stderr


def test_the_shipped_template_parses_with_no_warnings() -> None:
    """Backwards compatibility: the real template must load cleanly.

    This is the regression half. ``config/docker.env.example`` is what
    ``host_bootstrap.sh`` and this script seed a rover from, so if the parser
    rejects any line in it, provisioning silently drops that key.
    """
    _, stderr, rc = _load(_TEMPLATE, "MOUSEDROID_TELEMETRY_TOKEN")

    assert rc == 0
    assert "unparseable" not in stderr, f"the shipped template does not parse cleanly:\n{stderr}"


def test_every_template_key_the_deploy_path_reads_is_actually_loaded() -> None:
    """...and the clean parse above is not clean because nothing was read.

    A parser that skipped every line would emit no warnings at all, so the
    no-warnings assertion needs a companion that counts what arrived. Only the
    names this script or compose's interpolation read are exported (Layer 4);
    the rest of the template is for the container, which compose gives it
    through ``env_file:``. That remainder is pinned, so a new one is a decision.
    """
    keys = {
        match.group(1)
        for match in re.finditer(
            r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=",
            _TEMPLATE.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    }
    assert keys, "no uncommented keys found in the template — check the regex"
    exported = {key for key in keys if _exported().fullmatch(key)}

    values, _, rc = _load(_TEMPLATE, *sorted(exported))

    assert rc == 0
    unset = sorted(name for name, value in values.items() if value == "<unset>")
    assert not unset, f"template keys the loader failed to set: {unset}"
    assert keys - exported == {"NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"}


# ---------------------------------------------------------------------------
# Layer 2 — the env file the script CREATES is not world-readable
# ---------------------------------------------------------------------------


def _make_shim(directory: Path, name: str) -> None:
    """Drop a no-op executable called ``name`` into ``directory``."""
    shim = directory / name
    shim.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    shim.chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_the_created_env_file_is_not_world_readable(tmp_path: Path) -> None:
    """The template copy must land 0600, as ``host_bootstrap.sh:99,103`` does.

    ``cp`` carries no mode from the tracked template, so on a stock umask the
    operator's ``MOUSEDROID_TELEMETRY_TOKEN`` / ``ANTHROPIC_API_KEY`` would land
    world-readable. Two scripts create this same file; the other one already
    tightens it.

    The script is driven for real with shims, and the assertion is a stat of the
    artefact — so deleting the ``chmod`` line reds this test, which a grep of the
    source could not guarantee.
    """
    install_dir = tmp_path / "opt"
    config_dir = tmp_path / "etc"
    bin_dir = tmp_path / "bin"
    install_dir.mkdir()
    bin_dir.mkdir()
    (install_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    for tool in ("docker", "curl", "systemctl"):
        _make_shim(bin_dir, tool)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "MOUSEDROID_INSTALL_DIR": str(install_dir),
        "MOUSEDROID_CONFIG_DIR": str(config_dir),
        "MOUSEDROID_DOCKER_ENV_FILE": str(config_dir / "docker.env"),
        "MOUSEDROID_HEALTH_TIMEOUT": "1",
    }
    # The exit status is deliberately not asserted: the later steps talk to a
    # shimmed Docker and a health endpoint that does not exist. What matters is
    # the mode of the artefact step 2 created on the way past.
    subprocess.run(
        ["bash", str(_DEPLOY), "--no-build"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(_REPO_ROOT),
    )

    created = config_dir / "docker.env"
    assert created.is_file(), "step 2 did not seed docker.env — the fixture no longer reaches it"
    mode = created.stat().st_mode & 0o777
    assert mode == 0o600, f"docker.env landed {mode:#o}, expected 0o600 (holds API credentials)"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_an_existing_world_readable_env_file_is_repaired(tmp_path: Path) -> None:
    """Re-running the deploy must tighten a file it did not create.

    Every rover seeded by an earlier version of this script has `0644`
    credentials on disk right now, and nothing else ever revisits them. A fix
    that only covered the creation path would leave the whole existing fleet
    exposed while looking remediated, so the `chmod` runs on every deploy.

    The fixture's env file deliberately sets no `MOUSEDROID_*_DIR` keys: the env
    file legitimately overrides the process environment (as the old
    dot-source also did), so a fixture seeded from the real template would
    redirect the script at `/etc/mousedroid` and never reach this step.
    """
    install_dir = tmp_path / "opt"
    config_dir = tmp_path / "etc"
    bin_dir = tmp_path / "bin"
    for directory in (install_dir, config_dir, bin_dir):
        directory.mkdir()
    (install_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    for tool in ("docker", "curl", "systemctl"):
        _make_shim(bin_dir, tool)

    existing = config_dir / "docker.env"
    existing.write_text("MOUSEDROID_TELEMETRY_TOKEN=abc\n", encoding="utf-8")
    existing.chmod(0o644)

    subprocess.run(
        ["bash", str(_DEPLOY), "--no-build"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "MOUSEDROID_INSTALL_DIR": str(install_dir),
            "MOUSEDROID_CONFIG_DIR": str(config_dir),
            "MOUSEDROID_HEALTH_TIMEOUT": "1",
        },
        cwd=str(_REPO_ROOT),
    )

    mode = existing.stat().st_mode & 0o777
    assert mode == 0o600, (
        f"a pre-existing docker.env stayed {mode:#o}; re-running the deploy must repair "
        "the fleet the old script left world-readable, not only new installs"
    )
    # And the file was not clobbered while being tightened.
    assert "MOUSEDROID_TELEMETRY_TOKEN=abc" in existing.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_the_config_directory_is_deliberately_left_group_readable() -> None:
    """``mousedroid.service`` runs as ``User=jetson`` and reads this directory.

    Recorded as a test rather than a comment because "harden the directory too"
    is the obvious next change, and it would break the venv deployment path:
    ``scripts/mousedroid.service:23`` drops to ``User=jetson`` and ``:26`` points
    ``MOUSEDROID_CONFIG`` at a YAML inside this same directory. The secret lives
    in one file, and that file is what gets tightened.
    """
    unit = (_REPO_ROOT / "scripts" / "mousedroid.service").read_text(encoding="utf-8")
    assert "User=jetson" in unit, (
        "mousedroid.service no longer drops privileges. If every reader of "
        "the config directory is now root, tightening the directory itself "
        "becomes safe — revisit docker_deploy.sh's mkdir comment."
    )
    assert "/etc/mousedroid/" in unit, "mousedroid.service no longer reads the config directory"
    assert shutil.which("bash") is not None  # keeps the module's guard honest


# ---------------------------------------------------------------------------
# Layer 3 — values from that file must not become Docker arguments
# ---------------------------------------------------------------------------


def _run_deploy(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    """Run the deploy script with a throwaway install/config root.

    No shims: the guards under test sit before the Docker prerequisite check, so
    a refusal here is the guard and nothing else.

    Args:
        tmp_path: pytest temporary directory.
        overrides: Extra ``MOUSEDROID_*`` environment entries.

    Returns:
        The completed process.
    """
    env = {
        **os.environ,
        "MOUSEDROID_INSTALL_DIR": str(tmp_path / "opt"),
        "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
        "MOUSEDROID_DOCKER_ENV_FILE": str(tmp_path / "absent.env"),
        **overrides,
    }
    return subprocess.run(
        ["bash", str(_DEPLOY), "--health-only"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def test_a_container_name_beginning_with_a_dash_is_refused(tmp_path: Path) -> None:
    """``docker exec -i "$NAME" ...`` would read a leading dash as a flag.

    The specific message is asserted, not merely a nonzero exit: the trap
    recorded in ``mouse-droid-deploy-repin/tasks.md:19-32`` is that this script
    has many ways to exit nonzero, so an exit-code-only assertion passes with the
    guard deleted.
    """
    proc = _run_deploy(tmp_path, MOUSEDROID_CONTAINER="-uroot")

    assert proc.returncode != 0
    assert "container name is not a valid Docker name: -uroot" in proc.stderr


def test_the_weak_pattern_would_have_admitted_that_name() -> None:
    """Why the first character is anchored rather than just the character class.

    The plan for this task specified ``^[A-Za-z0-9_.-]+$``. ``-`` is a member of
    that class, so ``-uroot`` satisfies it and the guard would have passed the
    very value it exists to reject. This pins the reasoning so the pattern is not
    "simplified" back later.
    """
    weak = re.compile(r"^[A-Za-z0-9_.-]+$")
    strong = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    assert weak.match("-uroot"), "the weak pattern no longer admits it; this note is stale"
    assert not strong.match("-uroot")
    # Both must still accept the real default and ordinary names.
    for good in ("mousedroid", "mousedroid_2", "mouse-droid.v1"):
        assert weak.match(good)
        assert strong.match(good)


def test_an_ordinary_container_name_is_not_refused(tmp_path: Path) -> None:
    """The guard must not reject the values the repo actually ships."""
    proc = _run_deploy(tmp_path, MOUSEDROID_CONTAINER="mousedroid")

    assert "is not a valid Docker name" not in proc.stderr


def test_a_relative_deploy_record_is_refused(tmp_path: Path) -> None:
    """A relative path resolves against the operator's cwd, silently."""
    proc = _run_deploy(tmp_path, MOUSEDROID_DEPLOY_RECORD="deployments/jetson-image.json")

    assert proc.returncode != 0
    assert "deploy record must be an absolute path" in proc.stderr


def test_an_absolute_deploy_record_is_not_refused(tmp_path: Path) -> None:
    proc = _run_deploy(tmp_path, MOUSEDROID_DEPLOY_RECORD=str(tmp_path / "rec.json"))

    assert "deploy record must be an absolute path" not in proc.stderr


def test_a_container_name_from_the_env_file_is_validated_too(tmp_path: Path) -> None:
    """The guard has to sit after the env file is read, not before it.

    This is the case that ties Layer 3 back to Layer 1: the value's dangerous
    source is the env file, so validating only the process environment would
    leave the actual attack path open.
    """
    env_file = tmp_path / "docker.env"
    env_file.write_text("MOUSEDROID_CONTAINER=-uroot\n", encoding="utf-8")

    proc = _run_deploy(
        tmp_path,
        MOUSEDROID_DOCKER_ENV_FILE=str(env_file),
        MOUSEDROID_CONTAINER="",
    )

    assert proc.returncode != 0
    assert "container name is not a valid Docker name: -uroot" in proc.stderr


# ---------------------------------------------------------------------------
# Layer 4 — only what the deploy path reads is exported
# ---------------------------------------------------------------------------
#
# Exporting runs in the script's OWN shell, so every exported key used to be a
# chance to overwrite the loader's locals or the script's globals -- and every
# process hook (BASH_ENV, LD_*, PATH, DOCKER_CONFIG, CURL_HOME, ...) reached
# every child. Each case below ran code as the deploy user against the
# export-everything loader; reproduced, then pinned.


def _exported() -> re.Pattern[str]:
    """The loader's own allow-list, read from the script (not restated here)."""
    match = re.search(r"local exported='([^']+)'", _extract_loader())
    assert match is not None, "the loader's export allow-list was not found"
    return re.compile(match.group(1))


@pytest.mark.parametrize(
    "name",
    [
        "MOUSEDROID_ANYTHING",
        "COMPOSE_FILE",
        "GCP_CREDENTIALS_FILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ],
)
def test_names_the_deploy_path_reads_are_exported(tmp_path: Path, name: str) -> None:
    env = tmp_path / "docker.env"
    env.write_text(f"{name}=/wanted/value\n", encoding="utf-8")

    values, _, rc = _load(env, name)

    assert rc == 0
    assert values[name] == "/wanted/value"


_NOT_SETTINGS = {
    # Process hooks: each would run code in the next child.
    "BASH_ENV": "/nonexistent/hook.sh",
    "ENV": "/nonexistent/hook.sh",
    "SHELLOPTS": "xtrace",
    "BASHOPTS": "extglob",
    "PS4": "$(touch /nonexistent/x)",
    "PROMPT_COMMAND": "touch /nonexistent/x",
    "LD_PRELOAD": "/nonexistent/evil.so",
    "LD_AUDIT": "/nonexistent/evil.so",
    "LD_LIBRARY_PATH": "/nonexistent/lib",
    "PATH": "/nonexistent/bin",
    "DOCKER_CONFIG": "/nonexistent/docker",
    "CURL_HOME": "/nonexistent/curl",
    "HOME": "/nonexistent/home",
    "PYTHONPATH": "/nonexistent/py",
    # This script's own names: a global, and the loader's locals.
    "IFS": "x",
    "SCRIPT_DIR": "/nonexistent/scripts",
    "INSTALL_DIR": "/nonexistent/opt",
    "exported": ".*",
    "lineno": "7",
    # For the container only; compose hands it over through env_file.
    "ANTHROPIC_API_KEY": "placeholder-value",
}


def test_nothing_else_is_exported(tmp_path: Path) -> None:
    env = tmp_path / "docker.env"
    env.write_text(
        "".join(f"{name}={value}\n" for name, value in _NOT_SETTINGS.items())
        + "MOUSEDROID_KEPT=1\n",
        encoding="utf-8",
    )

    values, stderr, rc = _load(env, *_NOT_SETTINGS, "MOUSEDROID_KEPT")

    assert rc == 0
    assert values["MOUSEDROID_KEPT"] == "1", "leaving one name out must not cost the others"
    leaked = sorted(name for name, value in _NOT_SETTINGS.items() if values[name] == value)
    assert leaked == [], f"exported from the env file: {leaked}"
    assert "unparseable" not in stderr


def _run_deploy_with_env_file(tmp_path: Path, env_file: Path) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    return subprocess.run(
        ["bash", str(_DEPLOY), "--health-only"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **clean,
            "MOUSEDROID_INSTALL_DIR": str(tmp_path / "opt"),
            "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
            "MOUSEDROID_DOCKER_ENV_FILE": str(env_file),
        },
        cwd=str(_REPO_ROOT),
    )


def test_bash_env_in_the_env_file_does_not_run_when_the_deploy_starts_bash(
    tmp_path: Path,
) -> None:
    """The reviewed reproduction, as a test.

    ``docker_deploy.sh`` runs ``bash validate_compose_env.sh`` after loading
    docker.env, so an exported BASH_ENV ran its file as root.
    """
    sentinel = tmp_path / "HOOK_RAN"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"touch '{sentinel}'\n", encoding="utf-8")
    clean = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}

    # Premise: a bash child started with this BASH_ENV does run the hook --
    # otherwise the assertion below would pass for the wrong reason.
    subprocess.run(["bash", "-c", "true"], env={**clean, "BASH_ENV": str(hook)}, check=True)
    assert sentinel.exists(), "premise failed: BASH_ENV did not run the hook"
    sentinel.unlink()

    env_file = tmp_path / "docker.env"
    env_file.write_text(f"BASH_ENV={hook}\n", encoding="utf-8")
    _run_deploy_with_env_file(tmp_path, env_file)

    assert not sentinel.exists(), "BASH_ENV from docker.env ran as the deploy started bash"


@pytest.mark.parametrize(
    "lines",
    [
        # The loader's line counter is evaluated arithmetically on the next
        # line, and bash runs a `$(...)` inside an array subscript there.
        pytest.param(
            ["lineno=BASH_VERSINFO[$(touch {sentinel})0]", "OK=1"], id="lineno-arithmetic"
        ),
        # Overwrite what a deny-list would have consulted, then use the hook.
        pytest.param(["IFS=", "BASH_ENV={hook}"], id="ifs-then-bash-env"),
        pytest.param(["exported=.*", "BASH_ENV={hook}"], id="allow-list-name-then-bash-env"),
        # Redirect the validator this script runs as root.
        pytest.param(["SCRIPT_DIR={evil_dir}"], id="script-dir"),
    ],
)
def test_the_env_file_cannot_reach_the_scripts_own_names(tmp_path: Path, lines: list[str]) -> None:
    sentinel = tmp_path / "PWNED"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"touch '{sentinel}'\n", encoding="utf-8")
    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    planted = evil_dir / "validate_compose_env.sh"
    planted.write_text(f"#!/usr/bin/env bash\ntouch '{sentinel}'\n", encoding="utf-8")
    planted.chmod(0o755)
    env_file = tmp_path / "docker.env"
    env_file.write_text(
        "".join(
            line.format(sentinel=sentinel, hook=hook, evil_dir=evil_dir) + "\n" for line in lines
        ),
        encoding="utf-8",
    )

    _run_deploy_with_env_file(tmp_path, env_file)

    assert not sentinel.exists(), f"a docker.env line ran code: {lines}"


_UNITS = (
    _REPO_ROOT / "scripts" / "mousedroid-docker.service",
    _REPO_ROOT / "scripts" / "mousedroid-trend.service",
)


def test_the_allow_list_covers_everything_compose_interpolates() -> None:
    """The reason the deploy path exports anything: compose's ${VAR}s.

    A new ${VAR} in the compose file that the allow-list does not admit would
    interpolate differently here than on the boot path, where systemd hands
    compose the whole file.
    """
    compose = (_REPO_ROOT / "docker-compose.jetson.yml").read_text(encoding="utf-8")
    interpolated = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", compose)) - {"VAR"}
    assert interpolated, "no interpolations found -- check the regex"

    assert sorted(name for name in interpolated if not _exported().fullmatch(name)) == []


@pytest.mark.parametrize("unit", _UNITS, ids=lambda path: path.name)
def test_the_boot_path_unsets_the_shell_startup_hooks(unit: Path) -> None:
    """Both units read docker.env and start bash; neither may hand bash a hook.

    The deploy path exports nothing outside its allow-list, which covers these
    and more. The units cannot use an allow-list (systemd's EnvironmentFile has
    none), so they unset the names that make a starting shell or the dynamic
    loader run code; the file's root-only mode remains the control there.
    """
    text = unit.read_text(encoding="utf-8")
    assert "EnvironmentFile=-/etc/mousedroid/docker.env" in text
    lines = [line for line in text.splitlines() if line.startswith("UnsetEnvironment=")]
    assert len(lines) == 1, f"{unit.name}: expected one UnsetEnvironment= line, got {lines}"
    unset = set(lines[0].removeprefix("UnsetEnvironment=").split())

    assert {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4", "LD_PRELOAD", "LD_AUDIT"} <= unset
    assert not any(_exported().fullmatch(name) for name in unset)
