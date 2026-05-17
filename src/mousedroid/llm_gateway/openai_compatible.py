"""Tier C2.3: OpenAI-compatible HTTP LLM gateway.

Talks to ``{base_url}/v1/chat/completions`` over HTTP. Conforms to
:class:`LLMGatewayProtocol` so the existing factory + adapter wiring
treats it identically to the in-process llama-cpp gateway. The same
endpoint is served by Ollama (0.1.18+), LM Studio (0.2.x+), OpenAI,
and most local-LLM tooling — operators swap deployments by changing
``cfg.llm.base_url`` only.

Architecture invariants (per CLAUDE.md):

* Asyncio-only — uses ``aiohttp.ClientSession``. No blocking I/O.
* Structured logging via ``mousedroid.logging.setup.get_logger``.
* No hardcoded URLs / model names / timeouts — every tunable comes from
  :class:`LLMConfig`.
* Never raises on the happy path; on any failure mode the gateway logs
  a structured event, sets ``_degraded=True`` if persistent, and
  returns a neutral :class:`GoalVector` so the orchestrator never
  crashes on a misbehaving LLM.
* API key is stored as ``SecretStr`` on the config and only resolved
  via ``get_secret_value()`` inside the HTTP layer — never logged.
"""

from __future__ import annotations

import asyncio
import json
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import aiohttp

from mousedroid.constants import MILLISECONDS_PER_SECOND
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import LLMConfig

_log = get_logger(__name__)

# OpenAI-compatible REST endpoint paths. Constants so a future spec
# revision (e.g. a hypothetical ``/v2/chat/completions``) lands in one
# place rather than across three call sites. Matches the
# "no hardcoded values" CLAUDE.md invariant — the host portion already
# lives on ``LLMConfig.base_url`` and these path tails are part of the
# OpenAI REST contract, not operator-tunable knobs.
_HEALTH_PATH = "/v1/models"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
# Authorization header format for bearer-token auth (RFC 6750).
_BEARER_PREFIX = "Bearer "
# ``GoalVector`` velocity-axis bounds. Mirrors the clamp the legacy
# in-process :class:`LLMGateway._parse_response` applies so both
# backends produce equivalent ``GoalVector`` output for the same LLM
# response. Not config-driven because the bounds are part of the
# ``GoalVector`` semantic contract (normalised in ``[-1, 1]``), not an
# operator-tunable knob.
_GOAL_VECTOR_MIN = -1.0
_GOAL_VECTOR_MAX = 1.0


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` to the GoalVector ``[-1, 1]`` velocity range."""
    return max(_GOAL_VECTOR_MIN, min(_GOAL_VECTOR_MAX, value))


class OpenAICompatibleLLMGateway:
    """HTTP-backed LLM gateway hitting ``/v1/chat/completions``.

    Conforms structurally to :class:`LLMGatewayProtocol`. Production
    deployments wire this via :func:`build_llm_gateway` when
    ``cfg.llm.backend == "openai_compatible"``.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self._ready = False
        self._degraded = False

    @property
    def is_ready(self) -> bool:
        """True iff ``start()`` confirmed the server is reachable."""
        return self._ready

    @property
    def is_degraded(self) -> bool:
        """True iff a persistent transport-level failure was seen during ``start()``."""
        return self._degraded

    async def start(self) -> None:
        """Open a session and probe ``GET {base_url}/v1/models``.

        On any transport-level failure the gateway logs a structured
        ``llm_gateway_http_degraded`` event and sets ``_degraded=True``;
        :meth:`translate_mission` returns neutral GoalVectors until
        :meth:`start` is retried.
        """
        if not self._cfg.enabled:
            _log.info("llm_gateway_disabled")
            return

        # Idempotent session creation: a second ``start()`` (operator
        # retry / reconnect) must not leak the prior ``ClientSession``.
        # The session is only torn down by :meth:`stop` (which sets
        # ``_session = None``), so reusing the existing handle here is
        # safe and avoids socket-handle leaks under flaky network
        # conditions where ``start()`` is invoked multiple times.
        if self._session is None:
            self._session = self._build_session()
        try:
            async with self._session.get(
                f"{self._cfg.base_url}{_HEALTH_PATH}",
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_s),
            ) as resp:
                if resp.status == HTTPStatus.OK:
                    self._ready = True
                    _log.info(
                        "llm_gateway_http_started",
                        base_url=self._cfg.base_url,
                        model=self._cfg.model_name,
                    )
                else:
                    self._degraded = True
                    _log.warning(
                        "llm_gateway_http_health_non_200",
                        status=resp.status,
                        base_url=self._cfg.base_url,
                    )
        except aiohttp.ClientConnectionError as exc:
            self._degraded = True
            _log.warning(
                "llm_gateway_http_degraded",
                error=f"{type(exc).__name__}:{exc}",
                base_url=self._cfg.base_url,
            )
        except asyncio.TimeoutError:
            self._degraded = True
            _log.warning(
                "llm_gateway_http_health_timeout",
                base_url=self._cfg.base_url,
                timeout_s=self._cfg.request_timeout_s,
            )

    def _build_session(self) -> aiohttp.ClientSession:
        """Construct the aiohttp session (extracted for test patching)."""
        return aiohttp.ClientSession()

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """POST to ``/v1/chat/completions`` and parse a GoalVector from the body.

        Returns a neutral :class:`GoalVector` on any failure path
        (gateway not started, network error, non-200, non-JSON content,
        missing fields). Never raises.
        """
        if self._session is None or not self._ready:
            _log.warning(
                "llm_gateway_http_not_started",
                ready=self._ready,
                session=self._session is not None,
            )
            return GoalVector()

        payload: dict[str, Any] = {
            "model": self._cfg.model_name,
            "messages": [
                {"role": "system", "content": self._cfg.system_prompt},
                {"role": "user", "content": nl_command},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._cfg.api_key is not None:
            headers["Authorization"] = f"{_BEARER_PREFIX}{self._cfg.api_key.get_secret_value()}"

        start = time.monotonic()
        try:
            async with self._session.post(
                f"{self._cfg.base_url}{_CHAT_COMPLETIONS_PATH}",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_s),
            ) as resp:
                if resp.status != HTTPStatus.OK:
                    _log.warning(
                        "llm_gateway_http_non_200",
                        status=resp.status,
                        elapsed_ms=(time.monotonic() - start) * MILLISECONDS_PER_SECOND,
                    )
                    return GoalVector()
                body = await resp.json()
        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as exc:
            # ``resp.json()`` raises ``json.JSONDecodeError`` when the
            # body bytes don't decode as JSON even if the server set
            # ``Content-Type: application/json`` — caught here so the
            # docstring's "never raises" invariant holds on misbehaving
            # upstreams (a half-buffered Ollama response, for instance).
            _log.warning(
                "llm_gateway_http_error",
                error=f"{type(exc).__name__}:{exc}",
            )
            return GoalVector()

        content = self._extract_message_content(body)
        if content is None:
            return GoalVector()
        return self._parse_goal_vector(content)

    @staticmethod
    def _extract_message_content(body: dict[str, Any]) -> str | None:
        """Pull ``choices[0].message.content`` defensively."""
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            _log.warning("llm_gateway_http_malformed_body")
            return None

    @staticmethod
    def _parse_goal_vector(content: str) -> GoalVector:
        """Parse ``content`` as JSON and build a clamped :class:`GoalVector`.

        Keys (``"vx"``, ``"vy"``, ``"omega"``) and the ``[-1, 1]`` clamp
        mirror the legacy in-process :class:`LLMGateway._parse_response`
        contract so swapping ``cfg.llm.backend`` between ``llama_cpp``
        and ``openai_compatible`` produces equivalent ``GoalVector``
        output for a given LLM response. The legacy parser's key choice
        matches the canonical ``LLMConfig.system_prompt`` instruction
        ("output a JSON object with keys 'vx', 'vy', 'omega'").

        Returns a neutral :class:`GoalVector` when ``content`` is not
        valid JSON, when the decoded payload isn't a JSON object, or
        when any field is non-numeric — keeping the "never raises"
        invariant intact.
        """
        try:
            doc = json.loads(content)
        except json.JSONDecodeError:
            _log.warning("llm_gateway_http_non_json_content")
            return GoalVector()
        if not isinstance(doc, dict):
            # Some LLMs occasionally emit a top-level list or scalar
            # instead of the requested object — surface as a neutral
            # GoalVector rather than blowing up the parser.
            _log.warning("llm_gateway_http_non_object_content")
            return GoalVector()
        try:
            return GoalVector(
                vx_target=_clamp_unit(float(doc.get("vx", 0.0))),
                vy_target=_clamp_unit(float(doc.get("vy", 0.0))),
                omega_target=_clamp_unit(float(doc.get("omega", 0.0))),
            )
        except (TypeError, ValueError):
            # Defends against non-numeric fields (e.g. ``{"vx": "fast"}``).
            _log.warning("llm_gateway_http_non_numeric_fields")
            return GoalVector()

    async def stop(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._ready = False
        _log.info("llm_gateway_http_stopped")


__all__ = ["OpenAICompatibleLLMGateway"]
