"""OpenShell CLI wrappers and policy generation for NemoClaw sandbox integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import SkillSpec

_log = get_logger(__name__)


def generate_openshell_policy(skill: SkillSpec, allowed_actuation: bool) -> dict[str, Any]:
    """Map a SkillSpec to an OpenShell policy dictionary.

    Args:
        skill: The skill specification containing the tool whitelist.
        allowed_actuation: Global baseline config allowing/denying actuation.

    Returns:
        A dictionary representing the OpenShell JSON/YAML policy.
    """
    allowed_tools = sorted(skill.tool_names)

    policy = {
        "metadata": {
            "name": f"policy_{skill.name}",
            "description": f"Auto-generated policy for {skill.name}",
        },
        "rules": {
            "allowed_tools": allowed_tools,
            "allow_actuation": allowed_actuation,
        },
    }
    return policy


async def write_policy_to_disk(policy: dict[str, Any], path: Path) -> None:
    """Write the policy to disk for openshell consumption."""

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy, indent=2))

    await asyncio.to_thread(_write)


async def invoke_openshell_sandbox(
    policy_path: Path, command: list[str], *, log_only: bool = False
) -> int:
    """Invoke the openshell CLI binary.

    Args:
        policy_path: Path to the generated OpenShell policy.
        command: The command to execute inside the sandbox.
        log_only: If True, openshell runs in audit mode (no enforcement).

    Returns:
        Exit code of the process.
    """
    args = ["openshell", "exec", "--policy", str(policy_path)]
    if log_only:
        args.append("--audit")

    # double dash separates openshell args from the inner command
    args.append("--")
    args.extend(command)

    _log.debug("invoking_openshell", args=args, log_only=log_only)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            _log.warning(
                "openshell_invocation_failed",
                returncode=proc.returncode,
                stderr=stderr.decode(errors="replace"),
            )
        return proc.returncode or 0
    except FileNotFoundError:
        _log.warning("openshell_binary_not_found")
        return 127
