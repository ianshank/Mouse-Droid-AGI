"""Curated MCP prompts.

Static prompt templates surfaced via MCP's prompts capability. They are
plain strings that the connected client can render against its own LLM —
the MouseDroid stack does not invoke an LLM here.

Defined as a small registry rather than hardcoded literals at the call
site so the orchestrator can extend / override per-platform from
:class:`MCPConfig` in a future revision without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPPrompt:
    """A single curated MCP prompt.

    Attributes:
        name: Stable prompt identifier (slug-cased).
        description: One-line description shown by clients.
        template: Prompt body the client renders to its LLM.
    """

    name: str
    description: str
    template: str


_DEFAULT_PROMPTS: tuple[MCPPrompt, ...] = (
    MCPPrompt(
        name="diagnose-last-failure",
        description="Walk through the last safety violation or error in recent telemetry.",
        template=(
            "Read the resources mousedroid://telemetry/recent and "
            "mousedroid://logs/tail. Identify the most recent safety "
            "violation or error event. Summarise what happened, the "
            "likely cause, and a single suggested next diagnostic step."
        ),
    ),
    MCPPrompt(
        name="summarise-recent-telemetry",
        description="Concise health summary derived from the last telemetry window.",
        template=(
            "Read mousedroid://telemetry/recent. Produce a 5-bullet "
            "summary covering: forward clearance, battery, loop time, "
            "lidar minimum distance, and any anomalies."
        ),
    ),
    MCPPrompt(
        name="arm-task-status",
        description="Status read for the current robot-arm task (if active).",
        template=(
            "Read mousedroid://config/redacted to determine the active "
            "platform and arm task. Then read mousedroid://logs/tail "
            "and report whether the task appears to be making progress, "
            "stuck, or in error."
        ),
    ),
)


def default_prompts() -> tuple[MCPPrompt, ...]:
    """Return the immutable default prompt set.

    Returns:
        Tuple of curated prompts, safe to expose verbatim.
    """
    return _DEFAULT_PROMPTS
