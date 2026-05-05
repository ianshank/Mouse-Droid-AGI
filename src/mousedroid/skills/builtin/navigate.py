"""``mousedroid-navigate`` — NL goal → MCTS-planned waypoint route.

Actuation skill: requires both
:attr:`MCPConfig.expose_actuation_tools` and
:attr:`OpenClawConfig.require_actuation_ack` to be True before the
mission dispatcher accepts the call. The whitelist is intentionally
narrow — only ``set_velocity`` (the planned waypoint actuator) plus the
read-only world-model pose query.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mousedroid.skills.protocol import SkillSpec


class NavigateInput(BaseModel):
    """Input schema for a navigation request."""

    nl_command: str = Field(..., min_length=1, description="Natural-language navigation command.")
    max_speed: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional normalised speed cap (0-1). None = config default.",
    )


class NavigateOutput(BaseModel):
    """Output schema for a navigation request."""

    trace_id: str = Field(..., description="Dispatcher correlation id.")
    vx: float = Field(..., description="Forward velocity target (-1..1).")
    vy: float = Field(..., description="Lateral velocity target (-1..1).")
    omega: float = Field(..., description="Angular velocity target (-1..1).")


SYSTEM_PROMPT = (
    "You are the navigate skill of a Star Wars MSE-6 Mouse Droid. "
    "Translate the operator's NL command into a normalised velocity target. "
    "Refuse missions that would breach the configured patrol area. "
    "Always return JSON with keys vx, vy, omega in [-1, 1]."
)


SPEC = SkillSpec(
    name="mousedroid-navigate",
    description=(
        "Translate an NL navigation command into a velocity target via the "
        "mission dispatcher and the existing world model / MCTS planner."
    ),
    tool_names=frozenset({"set_velocity", "query_world_model_pose"}),
    system_prompt=SYSTEM_PROMPT,
    schema_in=NavigateInput,
    schema_out=NavigateOutput,
    source="builtin",
    metadata={"actuation": True, "channel": ("rest", "mcp"), "version": "1.0.0"},
)


__all__ = ["SPEC", "NavigateInput", "NavigateOutput"]
