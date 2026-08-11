"""Unit tests for OpenShell CLI wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.security.openshell import (
    generate_openshell_policy,
    invoke_openshell_sandbox,
    write_policy_to_disk,
)

_SUBPROCESS_PATCH = (
    "mousedroid.security.openshell.asyncio.create_subprocess_exec"
)

# ------------------------------------------------------------------
# generate_openshell_policy
# ------------------------------------------------------------------


def test_generate_openshell_policy_basic() -> None:
    """Policy has expected structure and sorted tool names."""
    skill = MagicMock()
    skill.name = "nav_skill"
    skill.tool_names = frozenset(["look", "move", "analyze"])

    policy = generate_openshell_policy(skill, allowed_actuation=True)

    assert policy["metadata"]["name"] == "policy_nav_skill"
    assert policy["rules"]["allowed_tools"] == ["analyze", "look", "move"]
    assert policy["rules"]["allow_actuation"] is True


def test_generate_openshell_policy_no_actuation() -> None:
    """allow_actuation=False is reflected in the policy."""
    skill = MagicMock()
    skill.name = "observe_only"
    skill.tool_names = frozenset(["look"])

    policy = generate_openshell_policy(skill, allowed_actuation=False)

    assert policy["rules"]["allow_actuation"] is False


def test_generate_openshell_policy_empty_tools() -> None:
    """Empty tool set produces an empty allowed_tools list."""
    skill = MagicMock()
    skill.name = "empty"
    skill.tool_names = frozenset()

    policy = generate_openshell_policy(skill, allowed_actuation=True)

    assert policy["rules"]["allowed_tools"] == []


# ------------------------------------------------------------------
# write_policy_to_disk
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_policy_to_disk(tmp_path: Path) -> None:
    """Policy is written as valid JSON to the specified path."""
    policy: dict[str, Any] = {
        "metadata": {"name": "test"},
        "rules": {"allowed_tools": ["look"], "allow_actuation": True},
    }
    out = tmp_path / "policies" / "test.json"

    await write_policy_to_disk(policy, out)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == policy


@pytest.mark.asyncio
async def test_write_policy_creates_parents(tmp_path: Path) -> None:
    """Intermediate directories are created automatically."""
    policy: dict[str, Any] = {"metadata": {"name": "deep"}, "rules": {}}
    out = tmp_path / "a" / "b" / "c" / "deep.json"

    await write_policy_to_disk(policy, out)

    assert out.exists()


# ------------------------------------------------------------------
# invoke_openshell_sandbox
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_openshell_sandbox_success() -> None:
    """Successful invocation returns exit code 0."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        _SUBPROCESS_PATCH, return_value=mock_proc
    ) as mock_exec:
        rc = await invoke_openshell_sandbox(
            Path("/tmp/policy.json"), ["echo", "hi"], log_only=False
        )

    assert rc == 0
    # Verify the command was constructed correctly
    call_args = mock_exec.call_args[0]
    assert "openshell" in call_args
    assert "--policy" in call_args
    assert "--" in call_args


@pytest.mark.asyncio
async def test_invoke_openshell_sandbox_audit_mode() -> None:
    """Audit mode passes --audit flag."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch(
        _SUBPROCESS_PATCH, return_value=mock_proc
    ) as mock_exec:
        await invoke_openshell_sandbox(
            Path("/tmp/policy.json"), ["echo", "test"], log_only=True
        )

    call_args = mock_exec.call_args[0]
    assert "--audit" in call_args


@pytest.mark.asyncio
async def test_invoke_openshell_sandbox_failure() -> None:
    """Non-zero exit code is returned and logged."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error details"))

    with patch(
        _SUBPROCESS_PATCH, return_value=mock_proc
    ):
        rc = await invoke_openshell_sandbox(
            Path("/tmp/policy.json"), ["bad_cmd"]
        )

    assert rc == 1


@pytest.mark.asyncio
async def test_invoke_openshell_sandbox_binary_missing() -> None:
    """Returns 127 when openshell is not installed."""
    with patch(
        "mousedroid.security.openshell.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("openshell not found"),
    ):
        rc = await invoke_openshell_sandbox(
            Path("/tmp/policy.json"), ["echo", "test"]
        )

    assert rc == 127
