"""Regression: the artifact-integrity playbook deletes a cached artifact and nothing else.

``docs/playbooks/artifact-integrity-fail.md`` has the operator paste the path
from the ``refusing to load`` line into a ``sudo rm``. Its ``case`` guard is all
that stands between a mistyped path and root deleting it, and the ORDER of its
arms is the guard: the ``..`` arm must come before the weights-prefix arm, or
``weights/../src`` matches the prefix and is deleted. A dry run of the first
draft did exactly that.

The snippet runs as documented: extracted from the playbook, with ``sudo``
shimmed to record what it was asked to do. The tree it may delete from is
derived from where both caches actually live, so neither side can drift alone.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path, PurePosixPath

import pytest
import yaml

from mousedroid.config.schema.cognitive import CognitiveConfig
from mousedroid.config.schema.world_model import WorldModelConfig
from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYBOOK = _REPO_ROOT / "docs" / "playbooks" / "artifact-integrity-fail.md"
_DOCKERFILE = _REPO_ROOT / "Dockerfile.jetson"
_OVERLAY = _REPO_ROOT / "config" / "jetson_production.yaml"
_ASSIGNMENT = re.compile(r"^ARTIFACT=.*$", re.M)

pytestmark = requires_bash()

_SUDO_SHIM = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SUDO_LOG"
"""


def _weights_root() -> str:
    """The container's weights tree: WORKDIR plus the caches' shared root.

    Both caches default to a path relative to the process directory, which in
    the image is WORKDIR -- the ``/opt/mousedroid`` bind mount, so the path the
    refusal line prints is valid on the host too.
    """
    workdir = re.search(r"^WORKDIR (\S+)$", _DOCKERFILE.read_text(encoding="utf-8"), re.M)
    assert workdir is not None, "Dockerfile.jetson no longer sets WORKDIR"
    roots = {
        PurePosixPath(WorldModelConfig().onnx_cache_dir).parts[0],
        PurePosixPath(CognitiveConfig().weights_dir.as_posix()).parts[0],
    }
    assert len(roots) == 1, f"the two caches no longer share a root: {roots}"
    return f"{workdir.group(1)}/{roots.pop()}"


def _delete_snippet() -> str:
    """The playbook's delete step, dedented, exactly as an operator pastes it."""
    blocks = re.findall(r"```bash\n(.*?)```", _PLAYBOOK.read_text(encoding="utf-8"), re.S)
    matching = [textwrap.dedent(block) for block in blocks if 'case "$ARTIFACT"' in block]
    assert len(matching) == 1, "expected exactly one delete snippet in the playbook"
    return matching[0]


def _run(tmp_path: Path, artifact: str) -> tuple[list[str], str]:
    """Run the snippet with ``ARTIFACT`` set as the operator would set it.

    Returns:
        ``(sudo_invocations, stderr)``.
    """
    snippet, count = _ASSIGNMENT.subn('ARTIFACT="$TEST_ARTIFACT"', _delete_snippet())
    assert count == 1, "the snippet no longer starts by assigning ARTIFACT"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "sudo"
    shim.write_text(_SUDO_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    sudo_log = tmp_path / "sudo.log"
    sudo_log.write_text("", encoding="utf-8")
    proc = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "SUDO_LOG": str(sudo_log),
            "TEST_ARTIFACT": artifact,
        },
    )
    assert proc.returncode == 0, proc.stderr
    return sudo_log.read_text(encoding="utf-8").splitlines(), proc.stderr


def test_the_guard_names_the_tree_both_caches_live_in() -> None:
    """Including where the production overlay moves the BDI weights."""
    root = _weights_root()
    overlay = yaml.safe_load(_OVERLAY.read_text(encoding="utf-8"))
    overlay_dir = overlay.get("cognitive", {}).get("weights_dir")

    assert f"{root}/*)" in _delete_snippet()
    if overlay_dir is not None:
        assert PurePosixPath(overlay_dir).is_relative_to(root), overlay_dir


@pytest.mark.parametrize("relative", ["dual_stream_rssm/model.onnx", "bdi/policy.npz"])
def test_a_cached_artifact_is_deleted_then_the_service_restarted(
    tmp_path: Path, relative: str
) -> None:
    artifact = f"{_weights_root()}/{relative}"

    calls, _ = _run(tmp_path, artifact)

    assert calls == [f"rm -v -- {artifact}", "systemctl restart mousedroid-docker"]


@pytest.mark.parametrize(
    ("relative", "why"),
    [
        # Inside the prefix as a string, outside it as a path.
        ("/../src/mousedroid/main.py", "climbs out with '..'"),
        ("/..", "climbs out with '..'"),
        ("/bdi/../../config/default.yaml", "climbs out with '..'"),
        # A sibling that shares the prefix as a string.
        ("X/model.onnx", "not under"),
        # The tree itself, not a file in it.
        ("", "not under"),
    ],
)
def test_a_path_that_leaves_the_weights_tree_is_never_deleted(
    tmp_path: Path, relative: str, why: str
) -> None:
    calls, stderr = _run(tmp_path, f"{_weights_root()}{relative}")

    assert not [call for call in calls if call.startswith("rm")], calls
    assert why in stderr


@pytest.mark.parametrize("artifact", ["/etc/passwd", "/opt/mousedroid/src/x.py", "weights/x"])
def test_a_path_elsewhere_is_never_deleted(tmp_path: Path, artifact: str) -> None:
    calls, stderr = _run(tmp_path, artifact)

    assert not [call for call in calls if call.startswith("rm")], calls
    assert "not under" in stderr
