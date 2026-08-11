#!/usr/bin/env python3
"""Audit soak test script for NemoClaw/OpenShell sandbox (Phase 4 Spike).

Usage::

    python scripts/nemoclaw_audit_spike.py [--policy-dir /tmp/openshell]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from mousedroid.logging.setup import get_logger
from mousedroid.security.openshell import (
    generate_openshell_policy,
    invoke_openshell_sandbox,
    write_policy_to_disk,
)
from mousedroid.skills.protocol import SkillSpec

_log = get_logger(__name__)

_DEFAULT_POLICY_DIR = os.environ.get(
    "MOUSEDROID_OPENSHELL_POLICY_DIR",
    str(Path.home() / ".mousedroid" / "openshell_policies"),
)


async def run_audit_spike(policy_dir: Path) -> None:
    """Execute the OpenShell audit spike.

    Args:
        policy_dir: Directory to write generated policy files.
    """
    _log.info("starting_openshell_audit_spike", policy_dir=str(policy_dir))

    # 1. Create a dummy skill specification to map
    skill = SkillSpec(
        name="test_skill",
        description="A test skill for the audit spike",
        tool_names=frozenset(["move", "look", "analyze"]),
    )

    # 2. Map to policy
    policy = generate_openshell_policy(skill, allowed_actuation=False)

    # 3. Write policy to disk
    policy_path = policy_dir / "openshell_test_policy.json"
    await write_policy_to_disk(policy, policy_path)
    _log.info("wrote_policy", path=str(policy_path))

    # 4. Invoke sandbox in log-only mode (audit soak)
    _log.info("invoking_sandbox_audit_mode")
    command_to_run = ["echo", "sandbox_test"]

    exit_code = await invoke_openshell_sandbox(
        policy_path,
        command_to_run,
        log_only=True,
    )

    if exit_code == 127:
        _log.warning(
            "openshell_not_installed",
            msg="The openshell binary was not found. Expected in environments without NemoClaw.",
        )
    else:
        _log.info("openshell_completed", exit_code=exit_code)

    _log.info("audit_spike_finished")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="NemoClaw OpenShell audit spike")
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path(_DEFAULT_POLICY_DIR),
        help="Directory to write generated policy files (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(run_audit_spike(args.policy_dir))


if __name__ == "__main__":
    main()
