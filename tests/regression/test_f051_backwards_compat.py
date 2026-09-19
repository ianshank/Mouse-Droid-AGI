"""Backwards compatibility — F-051 must be additive only.

Delivery hardening lands on live operator tooling: a rover mid-bring-up, a
half-provisioned host, a ``docker compose config`` lint run with no
``/etc/mousedroid`` at all. So every default here has to behave exactly as it
did before, and the new behaviour has to be opt-in at every seam.

Pinned:

* **Design D-10** — ``scripts/docker_deploy.sh`` without ``--strict-health`` is
  unchanged: strict defaults to off, the telemetry leg keeps its exact warn
  wording (including the "normal if telemetry is disabled" line that bring-up
  flows rely on), the two new probe legs never run, and step 6 keeps its
  softening.
* **``scripts/deploy_remote.sh``** — the three modes, the host-resolution order
  and the remote path defaults are untouched; the newly env-overridable
  ``REMOTE_SRC`` / ``REMOTE_CONFIG`` keep their previous literal values as
  defaults.
* **No new YAML key** — every shipped ``config/*.yaml`` still loads, none gains
  a ``world_model:`` block (design D-7) and none gains a
  ``jetson.tensorrt_cache_dir`` override: the cache directory is a schema
  default plus an optional env lever, so ``check_config_compat.py`` has nothing
  new to validate against the pinned schema.
* **``config/docker.env.example``** — the new key stays COMMENTED, so the
  preflight ``host_env_keys`` check does not start warning on every rover that
  was provisioned before this change.
* **Compose** — the cache volume is additive: the F-014 ``env_file`` contract
  and every pre-existing mount survive.
* **The deploy record** — still satisfies the keys and the 40-hex SHA that
  ``config-compat`` worktrees.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema.hardware import JetsonConfig
from mousedroid.validation.preflight import _parse_env_keys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_DOCKER_DEPLOY = _REPO_ROOT / "scripts" / "docker_deploy.sh"
_DEPLOY_REMOTE = _REPO_ROOT / "scripts" / "deploy_remote.sh"
_ENV_EXAMPLE = _CONFIG_DIR / "docker.env.example"
_DEPLOY_RECORD = _REPO_ROOT / "deployments" / "jetson-image.json"

_CACHE_ENV_VAR = "MOUSEDROID_JETSON__TENSORRT_CACHE_DIR"

# Mounts that existed before the cache volume and must all still be there.
_PREEXISTING_MOUNTS = (
    "/opt/mousedroid:/opt/mousedroid",
    "/etc/mousedroid:/etc/mousedroid:ro",
    "/tmp/argus_socket:/tmp/argus_socket",
    "/dev/serial:/dev/serial:ro",
    "mousedroid_experience:/home/jetson/mousedroid_experience",
    "/sys/devices/virtual/thermal:/sys/devices/virtual/thermal:ro",
    "/sys/devices/platform:/sys/devices/platform:ro",
    "promtail_positions:/var/lib/promtail",
)

_bash_required = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def _service() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    svc: dict[str, Any] = raw["services"]["mousedroid"]
    return svc


def _config_yaml_files() -> list[Path]:
    return sorted(p for p in _CONFIG_DIR.rglob("*.yaml") if p.is_file())


# ---------------------------------------------------------------------------
# docker_deploy.sh — the default path is untouched (design D-10)
# ---------------------------------------------------------------------------


def test_strict_health_defaults_to_off() -> None:
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "*)                   STRICT_HEALTH=false ;;" in text, (
        "anything other than an explicit 1/true/yes must read as off, so a typo "
        "fails open into today's behaviour instead of breaking a bring-up"
    )


def test_the_telemetry_warn_wording_survives_for_non_strict_runs() -> None:
    """Operators and runbooks read these two lines verbatim during bring-up."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert 'warn "  Telemetry health endpoint: not responding (${health_url})"' in text
    assert 'warn "  (This is normal if telemetry is disabled or still starting)"' in text, (
        "the bring-up reassurance line must stay on the default path"
    )


def test_the_new_probe_legs_are_gated_behind_strict_health() -> None:
    """Gated, not merely non-fatal: a default run must not gain an extra
    ``docker exec``, extra output or a new way to fail."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    probe_call = text.index("strict_promotion_probe || failures=$((failures + 1))")
    guard = text.rindex('if [ "$STRICT_HEALTH" = true ]; then', 0, probe_call)
    between = text[guard:probe_call]
    assert "\nhealth_check" not in between, (
        "the probe call must sit directly inside the strict-mode branch"
    )


def test_the_pre_existing_flags_and_env_knobs_are_unchanged() -> None:
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    for flag in ("--service)", "--no-build)", "--health-only)", "--help|-h)"):
        assert flag in text, f"{flag} must keep working"
    for knob in (
        "MOUSEDROID_INSTALL_DIR:-/opt/mousedroid",
        "MOUSEDROID_CONFIG_DIR:-/etc/mousedroid",
        "MOUSEDROID_CONTAINER:-mousedroid",
        "MOUSEDROID_HEALTH_TIMEOUT:-30",
    ):
        assert knob in text, f"{knob} default must be unchanged"


@_bash_required
def test_docker_deploy_help_still_exits_zero_without_docker() -> None:
    result = subprocess.run(
        ["bash", str(_DOCKER_DEPLOY), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


@_bash_required
def test_docker_deploy_still_rejects_an_unknown_flag() -> None:
    result = subprocess.run(
        ["bash", str(_DOCKER_DEPLOY), "--no-such-flag"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 1
    assert "Unknown option" in result.stderr


# ---------------------------------------------------------------------------
# deploy_remote.sh — defaults and modes unchanged
# ---------------------------------------------------------------------------


def test_remote_path_defaults_are_the_previous_literals() -> None:
    """The paths became env-overridable; their values did not change."""
    text = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    assert 'REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}"' in text
    assert 'REMOTE_CONFIG="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"' in text
    assert 'REMOTE_USER="${MOUSEDROID_REMOTE_USER:-jetson}"' in text


def test_the_three_deploy_modes_and_host_resolution_are_unchanged() -> None:
    text = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    for mode in ("--full)", "--code-only)", "--config-only)"):
        assert mode in text
    assert 'DEPLOY_MODE="code-only"' in text, "code-only must remain the default mode"
    assert "${HOME}/.mousedroid/jetson_host" in text
    assert "jetson_discover.sh" in text


def test_confirm_dirty_defaults_to_off() -> None:
    text = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    assert "*)                   CONFIRM_DIRTY=false ;;" in text


def test_secrets_are_not_transferred_by_the_sync() -> None:
    """``/etc/mousedroid/docker.env`` is preserved and never transferred: the
    rsync destination is the source tree, and the config copy only touches
    ``*.yaml`` with ``cp -n``."""
    text = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    assert "docker.env" not in text, "the deploy script must not touch the host secret file"
    assert 'for cfg_file in "${config_dir}"/*.yaml' in text
    assert "cp -n" in text, "config copies must never clobber an operator's file"


@_bash_required
def test_deploy_remote_help_still_exits_zero() -> None:
    result = subprocess.run(
        ["bash", str(_DEPLOY_REMOTE), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "--code-only" in result.stdout


# ---------------------------------------------------------------------------
# No new YAML key, and no new required env key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _config_yaml_files(), ids=lambda p: p.name)
def test_every_shipped_overlay_still_loads(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data is None or isinstance(data, dict)


def test_default_settings_still_resolve_with_no_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache directory is a schema default; nothing has to be set for a
    checkout to load."""
    monkeypatch.delenv(_CACHE_ENV_VAR, raising=False)
    cfg = load_settings()
    assert cfg.jetson.tensorrt_cache_dir == JetsonConfig().tensorrt_cache_dir


def test_no_tracked_overlay_carries_world_model_or_a_cache_dir_override() -> None:
    """Design D-7: ``WorldModelConfig`` is ``extra="ignore"`` at the pinned SHA,
    so a new ``world_model:`` key would pass ``config-compat`` and then be
    silently dropped. And the TRT cache dir must stay a schema default plus an
    env lever, not a YAML key."""
    for path in _config_yaml_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "world_model" not in data, f"{path.name} must not carry a world_model: block"
        jetson = data.get("jetson") or {}
        assert "tensorrt_cache_dir" not in jetson, (
            f"{path.name} must not pin jetson.tensorrt_cache_dir; it is a schema "
            "default plus an optional env override"
        )


def test_the_new_env_knob_stays_commented_in_the_operator_template() -> None:
    """Parsed with the preflight's own parser, because that is the code whose
    behaviour matters: an UNCOMMENTED key here makes ``host_env_keys`` warn on
    every rover provisioned before this change."""
    keys = _parse_env_keys(_ENV_EXAMPLE.read_text(encoding="utf-8"))
    assert _CACHE_ENV_VAR not in keys, f"{_CACHE_ENV_VAR} must stay commented in docker.env.example"
    assert _CACHE_ENV_VAR in _ENV_EXAMPLE.read_text(encoding="utf-8"), (
        "...but it must still be documented there for operators who relocate the cache"
    )
    # The keys that must keep reaching the container are untouched.
    assert {"MOUSEDROID_TELEMETRY_TOKEN", "MOUSEDROID_MOCK_HARDWARE"} <= keys


# ---------------------------------------------------------------------------
# Compose — additive only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mount", _PREEXISTING_MOUNTS)
def test_pre_existing_compose_mounts_survive(mount: str) -> None:
    assert mount in [str(entry) for entry in _service()["volumes"]], (
        f"{mount} disappeared — the cache volume must be purely additive"
    )


def test_the_f014_env_file_contract_survives() -> None:
    """The crash-loop fix: ``env_file`` present with ``required: false``, and the
    two keys that must come from it stay out of the inline block."""
    svc = _service()
    entries = [e for e in svc.get("env_file", []) if isinstance(e, dict)]
    docker_env = next(
        (e for e in entries if "/etc/mousedroid/docker.env" in str(e.get("path", ""))), None
    )
    assert docker_env is not None
    assert docker_env.get("required") is False
    inline = [str(e) for e in svc.get("environment", [])]
    assert not [e for e in inline if "MOUSEDROID_TELEMETRY_TOKEN" in e]
    assert not [e for e in inline if "MOUSEDROID_MOCK_HARDWARE" in e]


def test_the_cache_volume_did_not_displace_the_gpu_reservation() -> None:
    reservations = _service()["deploy"]["resources"]["reservations"]["devices"]
    assert reservations[0]["driver"] == "nvidia"
    assert reservations[0]["capabilities"] == ["gpu"]


# ---------------------------------------------------------------------------
# Deploy record — still valid for the config-compat gate
# ---------------------------------------------------------------------------


def test_the_extended_record_still_satisfies_the_config_compat_gate() -> None:
    record = json.loads(_DEPLOY_RECORD.read_text(encoding="utf-8"))
    for key in ("sha", "platform", "image_tag"):
        assert key in record
    assert record["platform"] == "jetson"
    assert record["image_tag"] == "mousedroid:jetson"
    sha = record["sha"]
    assert len(sha) == 40, sha
    assert all(char in "0123456789abcdef" for char in sha), sha


def test_the_record_sha_was_not_repinned_in_this_change() -> None:
    """Re-pinning to a feature-branch SHA reproduces the ``9c31968`` failure that
    killed the ``config-compat`` gate repo-wide. The re-pin is post-merge only,
    so this change leaves the SHA alone."""
    record = json.loads(_DEPLOY_RECORD.read_text(encoding="utf-8"))
    assert record["sha"] == "032942b50ab71abf285282a4de7333193f208c38", (
        "the deploy-record SHA must be re-pinned post-merge, to the squash-merge "
        "trunk commit — never to a feature-branch SHA"
    )
