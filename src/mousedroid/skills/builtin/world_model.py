"""``mousedroid-world-model`` — RSSM latent state + recent episodic summary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mousedroid.skills.protocol import SkillSpec


class WorldModelInput(BaseModel):
    """Input schema for a world-model query."""

    include_belief: bool = Field(True, description="Include RSSM latent belief summary.")
    include_pose: bool = Field(True, description="Include current pose estimate.")
    episodic_window: int = Field(
        16,
        ge=0,
        le=512,
        description="Number of recent episodic samples to include in the report.",
    )


class WorldModelOutput(BaseModel):
    """Output schema for a world-model query."""

    belief: dict[str, float] | None = Field(
        None, description="Latent state summary (norms, entropy, etc.)."
    )
    pose: dict[str, float] | None = Field(None, description="Pose estimate.")
    episodic_recent: list[dict[str, object]] = Field(
        default_factory=list, description="Recent episodic experiences (sanitised)."
    )


SYSTEM_PROMPT = (
    "You are the world-model skill of a Star Wars MSE-6 Mouse Droid. "
    "Produce a concise human-readable summary of the robot's current world "
    "model and recent experience. Never call actuation tools."
)


SPEC = SkillSpec(
    name="mousedroid-world-model",
    description=(
        "Dump RSSM latent state, pose estimate, and recent episodic "
        "experiences as a structured summary for OpenClaw consumption."
    ),
    tool_names=frozenset(
        {
            "query_world_model_belief",
            "query_world_model_pose",
            "episodic_recent_summary",
        }
    ),
    system_prompt=SYSTEM_PROMPT,
    schema_in=WorldModelInput,
    schema_out=WorldModelOutput,
    source="builtin",
    metadata={"actuation": False, "channel": ("rest", "mcp"), "version": "1.0.0"},
)


__all__ = ["SPEC", "WorldModelInput", "WorldModelOutput"]
