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
from mousedroid.llm_gateway._telemetry import extract_token_pair, record_round_trip_metrics
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import LLMConfig
    from mousedroid.security.injection_filter import PromptInjectionFilterProtocol
    from mousedroid.telemetry.metrics import MetricsRegistry

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

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        injection_filter: PromptInjectionFilterProtocol | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Construct the HTTP gateway.

        Args:
            cfg: LLM gateway configuration (``base_url`` / ``model_name`` /
                ``request_timeout_s`` / prompts).
            injection_filter: Optional shared
                :class:`PromptInjectionFilterProtocol`. When supplied (the
                production default via :func:`build_orchestrator`),
                ``translate_mission`` calls ``injection_filter.sanitize(nl)``
                before sending the user content to the upstream LLM —
                mirroring the local llama-cpp gateway's behaviour at
                :meth:`LLMGateway._sanitize_command` so both backends apply
                the same guardrails. The previous Tier-C2.3 implementation
                discarded this argument on the HTTP path (factory.py:627-629
                commented "upstream provider expected to enforce its own
                guardrails"); the f006-remote-llm sprint closes that gap so
                operator-supplied mission text from probes / dashboards /
                voice intent can't bypass the local rejection envelope.
                When ``None`` (legacy default), the gateway skips local
                sanitisation — backwards-compatible.
            metrics: Optional shared :class:`MetricsRegistry`. When supplied,
                every ``/v1/chat/completions`` round-trip records latency,
                token usage (from the response ``usage`` block), and a
                latency-budget-exceeded counter. ``None`` (the default) makes
                each metric call a no-op so existing callers are unaffected.
        """
        self._cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self._ready = False
        self._degraded = False
        self._injection_filter = injection_filter
        self._metrics = metrics

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

        # Reset both flags at the top of every ``start()`` so retries
        # cleanly reflect the latest health-probe outcome (Copilot HIGH).
        # Without this:
        #   * A previously-successful run leaves ``_ready=True`` even
        #     after a later non-200 / connection-refused, so
        #     ``translate_mission`` would happily POST to a downed
        #     daemon.
        #   * A previously-degraded run leaves ``_degraded=True`` even
        #     after a recovery, hiding the fact that the gateway is
        #     usable again from operator dashboards.
        # The two flags are now mutually exclusive on every exit path
        # (success → _ready=True, _degraded=False; failure → reverse).
        self._ready = False
        self._degraded = False

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

        Applies the prompt-injection filter (when one was injected) BEFORE
        the command leaves the rover, mirroring the local llama-cpp gateway's
        ``_sanitize_command`` (``gateway.py:148``) so operator-supplied
        mission text from probes / dashboards / voice intent cannot bypass
        the local rejection envelope. Backwards-compat: when no filter was
        injected the raw ``nl_command`` is sent through unchanged. On any
        sanitiser exception the gateway short-circuits to a neutral
        :class:`GoalVector` WITHOUT touching the upstream LLM (a misbehaving
        filter must never DoS the host).

        Returns a neutral :class:`GoalVector` on any failure path
        (gateway not started, network error, non-200, non-JSON content,
        missing fields). Never raises.
        """
        if self._injection_filter is not None:
            try:
                nl_command = self._injection_filter.sanitize(nl_command)
            except Exception as exc:  # boundary catch — never crash orchestrator
                _log.warning(
                    "llm_gateway_http_sanitize_failed",
                    error=f"{type(exc).__name__}:{exc}",
                )
                return GoalVector()

        content = await self._chat_completion(
            self._cfg.system_prompt, nl_command, self._cfg.max_tokens
        )
        if content is None:
            return GoalVector()
        return self._parse_goal_vector(content)

    async def answer_query(self, query: str) -> str:
        """Answer a free-text operator query with prose (NOT a GoalVector).

        The conversational sibling of :meth:`translate_mission`: same HTTP
        endpoint and telemetry, driven with ``cfg.query_system_prompt`` and
        ``cfg.query_max_tokens`` so the response is prose rather than the JSON
        GoalVector. Runs OUTSIDE the 30 Hz control loop — operator Q&A only.

        Unlike :meth:`translate_mission` (which now sanitises NL mission
        text through the injected prompt-injection filter before egress),
        the operator Q&A path applies no local injection filter — it
        carries free-text questions, not actuation commands, and mirrors
        the local llama-cpp gateway which likewise only sanitises the
        mission-translation path.

        Args:
            query: Natural language question. Must be non-empty.

        Returns:
            The model's free-text answer, or ``""`` on any failure path
            (not started / network error / non-200 / malformed body) — the
            neutral result mirroring the all-zero GoalVector.

        Raises:
            ValueError: If ``query`` is empty.
        """
        if not query.strip():
            msg = "query must be non-empty"
            raise ValueError(msg)
        content = await self._chat_completion(
            self._cfg.query_system_prompt, query, self._cfg.query_max_tokens
        )
        return "" if content is None else content.strip()

    async def _chat_completion(
        self, system_prompt: str, user_content: str, max_tokens: int
    ) -> str | None:
        """POST one chat completion, record telemetry, return the message text.

        Shared by :meth:`translate_mission` and :meth:`answer_query` so both
        paths get identical latency / token / budget instrumentation. Returns
        ``None`` on any failure path (the caller maps that to its own neutral
        result) so the "never raises on backend failure" invariant holds.
        """
        if self._session is None or not self._ready:
            _log.warning(
                "llm_gateway_http_not_started",
                ready=self._ready,
                session=self._session is not None,
            )
            return None

        payload: dict[str, Any] = {
            "model": self._cfg.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": max_tokens,
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
                    return None
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
            return None

        elapsed_ms = (time.monotonic() - start) * MILLISECONDS_PER_SECOND
        input_tokens, output_tokens = extract_token_pair(
            body.get("usage"), input_key="prompt_tokens", output_key="completion_tokens"
        )
        record_round_trip_metrics(
            self._metrics,
            model=self._cfg.model_name,
            elapsed_ms=elapsed_ms,
            over_budget=elapsed_ms > self._cfg.latency_target_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return self._extract_message_content(body)

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
