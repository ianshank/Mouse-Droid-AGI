"""Body schema for ``POST /api/v1/mission``.

Split out of the former monolithic ``telemetry/server.py`` so
``MissionRequest`` — a plain Pydantic model with no server dependencies —
can be imported without pulling in aiohttp or any of the
``TelemetryServer`` mixins.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MissionRequest(BaseModel):
    """Body schema for ``POST /api/v1/mission``.

    Defined at module scope so OpenClaw clients (and the test suite) can
    import it without instantiating the whole server.

    ``channel`` is constrained to :data:`Literal["rest"]` AND ignored by
    the handler — defence-in-depth. A REST client cannot smuggle
    ``channel="mcp"`` past the dispatcher's
    :class:`OpenClawConfig.allowed_channels` gate either by Pydantic
    validation (this constraint) or by handler logic (the call site
    hard-codes the channel string).

    ``idempotency_key`` is bounded by length and charset so an attacker
    cannot inflate the in-memory dedup map with arbitrary-length keys
    or smuggle log-corrupting bytes into the structured logs.
    """

    nl_command: str = Field(..., description="Natural language mission command")
    idempotency_key: str | None = Field(
        None,
        # 128 chars is generous: a UUID4 hex is 32, a UUIDv7+suffix
        # comfortably fits below 64. The bound is on the schema (so 400
        # is returned before the dedup map is touched) and the charset
        # regex blocks log-injection / unicode shenanigans.
        max_length=128,
        pattern=r"^[A-Za-z0-9_\-:.]+$",
        description=(
            "Optional dedup token; replays within the window return 202 "
            "with cached body. ASCII alphanumeric + ``_-:.`` only, "
            "max_length=128."
        ),
    )
    channel: Literal["rest"] = Field(
        "rest",
        description=(
            "Channel marker. Constrained to 'rest' on this endpoint; clients "
            "cannot spoof a different channel to bypass allowed_channels."
        ),
    )
