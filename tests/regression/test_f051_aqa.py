"""AQA — F-051 PC-to-rover delivery hardening.

Pins the four things this feature adds, so none of them can silently regress:

* **TensorRT cache volume** (task 8.1) — a named volume beside
  ``mousedroid_experience`` / ``promtail_positions``, whose container path comes
  from ``cfg.jetson.tensorrt_cache_dir`` rather than a literal. Cross-checked
  against the Pydantic schema here, which is the only assertion that can catch
  the two drifting apart.
* **Resource-budget decision** (task 8.0) — ``memory: 6G`` with swap, shm and
  pids deliberately unbounded, recorded in the compose file with the trigger
  that flips it. Pinned as the *absence* of those three keys, so adding one is
  a conscious decision rather than a diff nobody read.
* **``--strict-health``** (task 8.2) — a promotion gate that fails on a dead
  telemetry endpoint, an unexpected ORT provider or a digest mismatch, while
  the default permissive path stays exactly as it was (design D-10; the
  unchanged half is pinned in ``test_f051_backwards_compat.py``).
* **Rover-WIP preservation** (tasks 8.3 / 8.3b) — refusal plus a
  ``rover/wip-<date>`` branch plus an off-device archive, ahead of
  ``rsync -avz --delete``. The *behaviour* is proven against a real throwaway
  checkout in ``tests/unit/scripts/test_deploy_remote_guard.py``; what is
  pinned here is that the wiring and the deploy record's extension points stay
  in place.

No rover, no SSH, no Docker, no network — the on-rover promotion and the
rollback drill are operator steps, recorded in
``docs/runbooks/pc-to-jetson-promotion.md``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from mousedroid.config.schema import Settings
from mousedroid.config.schema.hardware import JetsonConfig
from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_DOCKER_DEPLOY = _REPO_ROOT / "scripts" / "docker_deploy.sh"
_DEPLOY_REMOTE = _REPO_ROOT / "scripts" / "deploy_remote.sh"
_WIP_GUARD = _REPO_ROOT / "scripts" / "rover_wip_guard.sh"
_DEPLOY_RECORD = _REPO_ROOT / "deployments" / "jetson-image.json"
_DOCS_README = _REPO_ROOT / "docs" / "README.md"
_RUNBOOKS = _REPO_ROOT / "docs" / "runbooks"

_CACHE_ENV_VAR = "MOUSEDROID_JETSON__TENSORRT_CACHE_DIR"
_CACHE_VOLUME = "mousedroid_tensorrt_cache"

_NEW_RUNBOOKS = ("jetson-onnx-benchmark.md", "pc-to-jetson-promotion.md")

# Shared guard, not a local predicate: `which("bash")` alone is NOT enough —
# on the GitHub Windows runner bash resolves to the WSL shim, which which()
# finds and which then exits 1 on every invocation. See tests/_bash.py.
_bash_required = requires_bash()


def _compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return data


def _service() -> dict[str, Any]:
    svc: dict[str, Any] = _compose()["services"]["mousedroid"]
    return svc


def _volume_mounts() -> list[str]:
    return [str(entry) for entry in _service().get("volumes", [])]


# ---------------------------------------------------------------------------
# 8.1 — TensorRT engine/timing cache volume
# ---------------------------------------------------------------------------


def test_compose_declares_the_named_tensorrt_cache_volume() -> None:
    """Beside the two volumes that already persist state. Nothing persisted a
    TRT engine/timing cache before this."""
    volumes = _compose()["volumes"]
    assert _CACHE_VOLUME in volumes, (
        f"{_CACHE_VOLUME} must be declared beside mousedroid_experience and "
        f"promtail_positions; got {sorted(volumes)}"
    )
    assert volumes[_CACHE_VOLUME] == {"driver": "local"}, (
        "the cache volume must match its siblings' driver declaration"
    )


def test_the_cache_mount_path_is_interpolated_not_a_literal() -> None:
    """The directory comes from ``cfg.jetson.tensorrt_cache_dir``. A literal
    here is the failure mode: compose and the Python config would then be two
    sources of truth for one path."""
    mounts = [m for m in _volume_mounts() if m.startswith(f"{_CACHE_VOLUME}:")]
    assert len(mounts) == 1, f"expected exactly one {_CACHE_VOLUME} mount; got {mounts}"
    container_path = mounts[0].split(":", 1)[1]
    assert container_path.startswith("${" + _CACHE_ENV_VAR), (
        f"the cache mount must interpolate {_CACHE_ENV_VAR} (the env name "
        f"pydantic-settings reads for cfg.jetson.tensorrt_cache_dir); got {container_path!r}"
    )


def test_the_compose_fallback_equals_the_schema_default() -> None:
    """The one assertion that catches drift between the mount and the schema.

    ``MOUSEDROID_`` + ``env_nested_delimiter="__"`` means the compose variable
    and the Pydantic field are the same knob, and the ``:-`` fallback only
    restates the schema default for an un-provisioned host — so if the schema
    default moves and the compose fallback does not, the cache silently lands
    somewhere the runtime is not looking.
    """
    mounts = [m for m in _volume_mounts() if m.startswith(f"{_CACHE_VOLUME}:")]
    container_path = mounts[0].split(":", 1)[1]
    fallback = container_path.split(":-", 1)[1].rstrip("}")
    schema_default = JetsonConfig().tensorrt_cache_dir
    assert fallback == str(schema_default), (
        f"compose fallback {fallback!r} must equal JetsonConfig."
        f"tensorrt_cache_dir {str(schema_default)!r}"
    )


def test_no_weights_volume_and_weights_persist_transitively() -> None:
    """For the record (task 8.1): weights are NOT mounted. They persist because
    ``onnx_cache_dir`` is *relative* and resolves under ``WORKDIR
    /opt/mousedroid``, which is inside the bind mount. An absolute default, or a
    weights volume shadowing that directory, would break the claim."""
    mounts = _volume_mounts()
    assert any(m.startswith("/opt/mousedroid:/opt/mousedroid") for m in mounts), (
        "the /opt/mousedroid bind mount is what makes the weights claim true"
    )
    assert not [m for m in mounts if "weights" in m], (
        f"no weights mount may exist — weights persist transitively; got {mounts}"
    )
    onnx_cache_dir = Settings(mock_hardware=True).world_model.onnx_cache_dir
    assert not Path(onnx_cache_dir).is_absolute(), (
        "world_model.onnx_cache_dir must stay relative so it resolves inside "
        f"the bind mount; got {onnx_cache_dir!r}"
    )


# ---------------------------------------------------------------------------
# 8.0 — recorded container resource budget
# ---------------------------------------------------------------------------


def test_the_memory_limit_is_unchanged_and_swap_stays_unbounded_by_decision() -> None:
    """Task 8.0's decision, pinned as facts rather than as prose: the 6 GiB cap
    stays, and swap/shm/pids stay unbounded until the change that first enables
    on-device engine builds bounds all three together. Adding one of these keys
    should turn this red and send the author to the recorded rationale in
    docker-compose.jetson.yml."""
    svc = _service()
    assert svc["deploy"]["resources"]["limits"]["memory"] == "6G"
    for key in ("memswap_limit", "shm_size", "pids_limit"):
        assert key not in svc, (
            f"{key} appeared: see the RESOURCE BUDGET DECISION note in "
            "docker-compose.jetson.yml — bound swap, shm and pids together, in "
            "the same change that enables on-device TensorRT engine builds"
        )


# ---------------------------------------------------------------------------
# 8.2 — --strict-health
# ---------------------------------------------------------------------------


def test_docker_deploy_offers_strict_health_by_flag_and_by_env() -> None:
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "--strict-health) STRICT_HEALTH=true" in text, "the flag must be parsed"
    assert "MOUSEDROID_STRICT_HEALTH" in text, "an env lever must exist too"
    assert "--strict-health Promotion gate" in text, "the flag must be documented in --help"


def test_all_three_strict_legs_are_present_and_fail_closed() -> None:
    """Telemetry, provider and digest — the three the spec names."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "--strict-health: a promotion requires a live telemetry endpoint" in text
    assert "ORT provider downgrade" in text, "the provider leg must exist"
    assert "model digest mismatch or artifact missing" in text, "the digest leg must exist"
    # Fail closed: under engine=onnx_trt an absent expectation is a failure, not
    # a pass. This is the assertion that stops the probe degrading into theatre.
    assert "cannot prove the artifact under engine=onnx_trt" in text
    assert "onnxruntime is not importable but engine=onnx_trt" in text


def test_the_provider_leg_reuses_the_runtime_provider_chain() -> None:
    """No second source of truth for the provider order: the probe imports
    ``DEFAULT_ORT_PROVIDERS`` and ``resolve_providers`` from the module the
    runtime itself uses, so a provider literal cannot drift into the script."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "from mousedroid.common.onnx_session import" in text
    assert "DEFAULT_ORT_PROVIDERS" in text
    assert "resolve_providers" in text
    assert "TensorrtExecutionProvider" not in text, (
        "provider names belong in DEFAULT_ORT_PROVIDERS, never inline in the script"
    )


def test_the_digest_leg_reuses_verify_sha256_and_the_deploy_record() -> None:
    """Not a shell ``sha256sum`` re-implementation, and not a new manifest."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "from mousedroid.utils.weights_manager import verify_sha256" in text
    assert "MOUSEDROID_DEPLOY_RECORD" in text
    assert "model_sha256" in text


def test_step_six_is_fatal_only_under_strict_health() -> None:
    """The softening at the end of the deploy is what the flag removes."""
    text = _DOCKER_DEPLOY.read_text(encoding="utf-8")
    assert "if ! health_check; then" in text
    assert "promotion aborted" in text
    assert 'health_check || warn "Some health checks failed (non-fatal for deployment)"' in text, (
        "the non-strict branch must keep today's exact softening"
    )


# ---------------------------------------------------------------------------
# 8.3 / 8.3b — rover WIP preservation
# ---------------------------------------------------------------------------


def test_the_wip_guard_ships_and_is_invoked_before_the_destructive_rsync() -> None:
    assert _WIP_GUARD.is_file(), "scripts/rover_wip_guard.sh must exist"
    text = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    assert "rover_wip_guard.sh" in text, "deploy_remote.sh must use the guard"
    assert text.index("\n    preserve_remote_wip\n") < text.index("rsync -avz --delete"), (
        "preservation after the delete preserves the wreckage"
    )


def test_preservation_produces_both_a_branch_and_an_off_device_archive() -> None:
    """A branch alone does not survive a disk failure (task 8.3b), and an
    unverified transfer is not an archive."""
    guard = _WIP_GUARD.read_text(encoding="utf-8")
    deploy = _DEPLOY_REMOTE.read_text(encoding="utf-8")
    assert 'name="rover/wip-${stamp}"' in guard, "the dated branch name is a contract"
    assert "--ignore-all-space" in guard, "the archived diff must be whitespace-insensitive"
    assert "MOUSEDROID_DEPLOY_ARCHIVE_DIR" in deploy, "archives land on the operator's machine"
    assert "is empty or corrupt" in deploy, "the transferred archive must be verified"
    assert "--confirm-dirty" in deploy, "the operator confirmation lever must exist"


def test_git_clean_is_never_used_by_either_script() -> None:
    for script in (_WIP_GUARD, _DEPLOY_REMOTE):
        text = script.read_text(encoding="utf-8").replace("`git clean`", "")
        assert "git clean" not in text, f"{script.name} must never run git clean"


@_bash_required
@pytest.mark.parametrize("script", [_DOCKER_DEPLOY, _DEPLOY_REMOTE, _WIP_GUARD])
def test_shell_scripts_parse_and_are_strict_mode(script: Path) -> None:
    """There is no shellcheck gate in CI, so the minimum bar is asserted here:
    the file parses, and it runs under ``set -euo pipefail`` so a failed guard
    cannot be walked past."""
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert "set -euo pipefail" in script.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 8.6 — the deploy record is extended, not duplicated
# ---------------------------------------------------------------------------


def test_the_deploy_record_carries_the_model_digest_and_provider_fields() -> None:
    """Present, and allowed to be null: no ``.onnx`` is published and no
    provider has been proven on hardware, so an invented value would be worse
    than an explicit absence. ``--strict-health`` fails closed on the null."""
    record = json.loads(_DEPLOY_RECORD.read_text(encoding="utf-8"))
    assert "model_sha256" in record, "the record must carry the model digest field"
    assert "ort_provider" in record, "the record must carry the proven-provider field"
    for key in ("model_sha256", "ort_provider"):
        value = record[key]
        assert value is None or (isinstance(value, str) and value), (
            f"{key} must be null or a non-empty string, never an empty placeholder"
        )
    assert record.get("model_provenance_note"), "null values must be explained in the record"


def test_no_parallel_deploy_manifest_was_introduced() -> None:
    """``deployments/<platform>-image.json`` is the extension point; a second
    manifest format is what the spec forbids."""
    manifests = sorted(p.name for p in (_REPO_ROOT / "deployments").glob("*"))
    assert manifests == ["jetson-image.json"], (
        f"deployments/ must hold only the per-platform image records; got {manifests}"
    )


# ---------------------------------------------------------------------------
# 8.8 — runbooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NEW_RUNBOOKS)
def test_the_new_runbooks_exist_with_an_h1(name: str) -> None:
    path = _RUNBOOKS / name
    assert path.is_file(), f"docs/runbooks/{name} must exist"
    assert path.read_text(encoding="utf-8").lstrip().startswith("# "), (
        f"{name} must open with an H1 (test_runbooks_structure.py enforces this too)"
    )


@pytest.mark.parametrize("name", _NEW_RUNBOOKS)
def test_the_new_runbooks_are_in_the_canonical_index(name: str) -> None:
    """``docs/README.md`` declares itself the canonical runbook index, so a
    runbook that is not listed there is one nobody finds."""
    index = _DOCS_README.read_text(encoding="utf-8")
    assert f"runbooks/{name}" in index, f"docs/README.md must link runbooks/{name}"


def test_the_promotion_runbook_records_the_unrun_rollback_drill() -> None:
    """Task 8.7 is an operator drill on real hardware. It has NOT been run, and
    the runbook must say so rather than implying a passed drill."""
    text = (_RUNBOOKS / "pc-to-jetson-promotion.md").read_text(encoding="utf-8")
    assert "mousedroid:jetson-rollback-" in text, "the established tag anchor must be documented"
    assert "NOT been run" in text or "not been run" in text, (
        "the drill's unrun status must be explicit"
    )


def test_the_promotion_runbook_flags_the_overlapping_deployment_docs() -> None:
    """Rather than becoming a fifth parallel deploy path."""
    text = (_RUNBOOKS / "pc-to-jetson-promotion.md").read_text(encoding="utf-8")
    for overlap in (
        "docs/deployment.md",
        "jetson-full-bringup.md",
        "jetson-claude-pilot-deploy.md",
        "JETSON_DEPLOY_RUNBOOK.md",
    ):
        assert overlap in text, f"the overlap with {overlap} must be named, not ignored"
