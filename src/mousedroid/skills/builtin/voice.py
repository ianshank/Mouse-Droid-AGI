"""``mousedroid-voice`` — trigger a Piper TTS phrase from the phrase bank.

Wraps :meth:`RockyVoiceEngine.speak` (event-driven phrase) and
:meth:`RockyVoiceEngine.play_phrase` (free-form text) so the OpenClaw
agent can either invoke a canonical event (``"obstacle_detected"``,
``"greeting"``, ...) or play arbitrary copy.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from mousedroid.skills.protocol import SkillSpec


class VoiceInput(BaseModel):
    """Input schema for a voice request.

    Exactly one of ``event`` or ``text`` must be supplied. ``event``
    routes through the personality-mapped phrase bank; ``text`` plays
    the supplied string as-is via the speaker.
    """

    event: str | None = Field(
        None,
        description="Phrase-bank event (e.g. 'greeting', 'obstacle_detected').",
    )
    text: str | None = Field(
        None,
        max_length=512,
        description="Free-form text for direct synthesis (alternative to event).",
    )
    valence: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Optional valence context for personality-driven inflection.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> VoiceInput:
        provided = sum(1 for v in (self.event, self.text) if v)
        if provided != 1:
            msg = "exactly one of 'event' or 'text' must be provided"
            raise ValueError(msg)
        return self


class VoiceOutput(BaseModel):
    """Output schema for a voice request."""

    queued: bool = Field(..., description="True if the request was accepted.")
    samples_written: int | None = Field(
        None,
        description="For text=..., samples played by the speaker (None for queued events).",
    )


SYSTEM_PROMPT = (
    "You are the voice skill of a Star Wars MSE-6 Mouse Droid. "
    "Produce short, character-consistent vocalisations. "
    "Refuse text that is offensive, defamatory, or longer than 512 characters."
)


SPEC = SkillSpec(
    name="mousedroid-voice",
    description=(
        "Trigger a Piper TTS phrase from the phrase bank or play "
        "operator-supplied text. Non-actuation, but still rate-limited."
    ),
    tool_names=frozenset({"speak_event", "play_phrase"}),
    system_prompt=SYSTEM_PROMPT,
    schema_in=VoiceInput,
    schema_out=VoiceOutput,
    source="builtin",
    metadata={"actuation": False, "channel": ("rest", "mcp"), "version": "1.0.0"},
)


__all__ = ["SPEC", "VoiceInput", "VoiceOutput"]
